"""Make Qwen3.5's Gated-DeltaNet layers correct (and fast) on packed sequences.

verl's remove-padding path packs every sequence of a micro-batch into one row
of shape (1, total_tokens) and marks the boundaries only through position_ids,
which restart at 0 for each sequence. The full-attention layers handle this
through flash-attention's varlen kernel, but transformers' Qwen3_5GatedDeltaNet
(5.6) ignores position_ids entirely: its causal conv1d runs with seq_idx=None
and its chunked delta rule with no cu_seqlens, so the conv window and the
recurrent state would flow from one sequence into the next. That is why the
current training runs one sequence per micro-batch (micro=1, no dynamic bsz).

This module patches, at import time and without editing transformers or verl:
  * Qwen3_5DecoderLayer.forward -- derives cu_seqlens from text position_ids
    when the batch is a single packed row and hands it to the linear-attention
    layer;
  * Qwen3_5GatedDeltaNet.forward -- with cu_seqlens (no cache), runs
    flash-linear-attention's varlen causal_conv1d and chunk_gated_delta_rule so
    every sequence starts from a zero conv window and a zero recurrent state.
    Every other call (generation with a cache, unpacked batches) goes to the
    original forward unchanged.

Load it in every verl worker with VERL_USE_EXTERNAL_MODULES=ha_verl.qwen35_packing
(or actor_rollout_ref.model.external_lib), with the fla install
(/users/tguo2/envs/qwen35_fast) on PYTHONPATH.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5 as _m

_ORIG_GDN_FORWARD = _m.Qwen3_5GatedDeltaNet.forward
_ORIG_LAYER_FORWARD = _m.Qwen3_5DecoderLayer.forward
_cache: dict = {"key": None, "cu": None}


def cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor | None:
    """(1, T) position ids restarting per sequence -> int32 cu_seqlens, else None."""
    if position_ids is None or position_ids.ndim != 2 or position_ids.shape[0] != 1:
        return None
    key = (position_ids.data_ptr(), position_ids.shape, position_ids._version)
    if _cache["key"] == key:
        return _cache["cu"]
    p = position_ids[0]
    total = p.numel()
    starts = torch.nonzero(p[1:] <= p[:-1]).flatten() + 1
    cu = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=p.device),
        starts.to(torch.int32),
        torch.full((1,), total, dtype=torch.int32, device=p.device),
    ])
    _cache["key"], _cache["cu"] = key, cu
    return cu


def _gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, cu_seqlens=None):
    if cu_seqlens is None or cache_params is not None:
        return _ORIG_GDN_FORWARD(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)

    hidden_states = _m.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = self.in_proj_qkv(hidden_states)                      # [1, T, D]
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # depthwise causal conv, window reset at every sequence start
    mixed_qkv, _ = _fla_causal_conv1d(
        x=mixed_qkv,
        weight=self.conv1d.weight.squeeze(1),
        bias=self.conv1d.bias,
        activation=self.activation,
        cu_seqlens=cu_seqlens,
    )

    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, _ = _fla_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
    )

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def _layer_forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                   past_key_values=None, **kwargs):
    if self.layer_type != "linear_attention":
        return _ORIG_LAYER_FORWARD(self, hidden_states, position_embeddings, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    cu = None
    if past_key_values is None and hidden_states.shape[0] == 1:
        cu = cu_seqlens_from_position_ids(position_ids)
    hidden_states = self.linear_attn(
        hidden_states=hidden_states,
        cache_params=past_key_values,
        attention_mask=attention_mask,
        cu_seqlens=cu,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states


_m.Qwen3_5GatedDeltaNet.forward = _gdn_forward
_m.Qwen3_5DecoderLayer.forward = _layer_forward
