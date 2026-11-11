"""CPU integration against installed veRL 0.8 (opt in with RUN_VERL_TRANSPORT=1).

Uses the real Qwen tokenizer, ToolAgentLoop state machine, v09 execute wrapper,
AgentLoopWorker postprocessing, DataProto serialization and NaiveRewardManager.
Only model generation, external tool execution and Ray RPC are local substitutes.
No model weights, API requests, GPU, task fixtures or scientific rewards are used.
"""
import asyncio
import os
import pickle
from types import SimpleNamespace
from unittest.mock import patch

if os.environ.get("RUN_VERL_TRANSPORT") != "1":
    import unittest
    raise unittest.SkipTest("requires installed veRL/Qwen CPU integration environment")

import unittest
import faulthandler

if __name__ == "__main__":
    faulthandler.dump_traceback_later(60, repeat=True)

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap, ToolListWrap
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput

from guidance_rl.agent_loop import GuidanceToolAgentLoop
from ha_verl.tool_v08 import HarnessAuditToolV08
from ha_verl.tool_v09 import HarnessAuditToolV09

MODEL = "/groups/xzhang33/tguo2taicheng/workAmazon/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CALL = "<tool_call>\n<function=transport_probe>\n</function>\n</tool_call>"
BAIT = "OBSERVATION_ONLY_BAIT_DO_NOT_SCORE_AS_FINAL"


def run_cpu(coro):
    # The restricted local runner can lose selector self-pipe notifications even
    # for asyncio.to_thread(lambda: 3). A timer bounds the selector wait without
    # replacing any veRL async operation or its thread executor.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def heartbeat():
        while True:
            await asyncio.sleep(.05)
    pulse = loop.create_task(heartbeat())
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        pulse.cancel()
        loop.run_until_complete(asyncio.gather(pulse, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)


class ScriptedGenerator:
    def __init__(self, tokenizer, turns):
        self.tokenizer, self.turns, self.request_ids = tokenizer, iter(turns), []

    async def generate(self, *, request_id, **kwargs):
        self.request_ids.append(request_id)
        await asyncio.sleep(0)  # allow eight same-task trajectories to interleave
        return TokenOutput(token_ids=self.tokenizer.encode(next(self.turns), add_special_tokens=False))


def make_loop(tokenizer, turns, *, response_length=2048, max_turns=8, thinking=False):
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {
        "prompt_length": 4096, "response_length": response_length,
        "multi_turn": {"max_user_turns": 8, "max_assistant_turns": max_turns,
                       "max_parallel_calls": 1, "max_tool_response_length": 1024,
                       "tool_response_truncate_side": "right", "format": "qwen3_coder"}}}})
    schema = OpenAIFunctionToolSchema.model_validate({"type": "function", "function": {
        "name": "transport_probe", "description": "Local transport test only",
        "parameters": {"type": "object", "properties": {}, "required": []}}})
    server = ScriptedGenerator(tokenizer, turns)
    loop = GuidanceToolAgentLoop(
        trainer_config=DictConfigWrap(config), server_manager=server, tokenizer=tokenizer,
        processor=None, dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(OmegaConf.create({"apply_chat_template_kwargs": {"enable_thinking": thinking}})),
        tools=ToolListWrap([HarnessAuditToolV09({"type": "native"}, schema)]))
    return loop, server


async def local_tool(self, instance_id, parameters, agent_data=None, **kwargs):
    # v09.execute itself is NOT patched: it must supply ha_request_id.
    return ToolResponse(text=BAIT), 0.0, {}


async def transport(tokenizer, loop, turns_kwargs, checker):
    output = await asyncio.wait_for(loop.run({}, **turns_kwargs), timeout=20)
    manager = NaiveRewardManager(OmegaConf.create({}), tokenizer, checker)

    class LocalRPC:
        async def remote(self, data):
            # Materialize the same real DataProto across an isolation boundary.
            # Only RPC scheduling is replaced; production builds/merges extra_info.
            restored = pickle.loads(pickle.dumps(data))
            return await manager.run_single(restored)

    worker = object.__new__(AgentLoopWorker)
    worker.tokenizer, worker.processor = tokenizer, None
    worker.rollout_config = loop.rollout_config
    worker.mm_processor_kwargs = {}
    worker.distillation_enabled = False
    worker.reward_loop_worker_handles = [SimpleNamespace(compute_score=LocalRPC())]
    processed = await asyncio.wait_for(worker._agent_loop_postprocess(output, validate=False, **turns_kwargs), timeout=20)
    return output, processed, worker


def task_kwargs(index=0):
    return {"raw_prompt": [{"role": "user", "content": "Answer the same task."}],
            "data_source": "harnessaudit", "reward_model": {"ground_truth": "same-task"},
            "extra_info": {"task_id": "same-task", "index": index}}


def check_eight_concurrent_tool_trajectories_transport_exact_final_and_request(tokenizer):
    async def scenario():
        received = {}
        async def one(i):
            answer = f"FINAL_ANSWER_{i}"
            loop, server = make_loop(tokenizer, [CALL, f"<think>PRIVATE_REASONING_{i}</think>{answer}<|im_end|>"])
            def checker(data_source, solution_str, ground_truth, extra_info):
                assert data_source == "harnessaudit" and ground_truth == "same-task"
                assert BAIT in solution_str  # legacy flattened decode includes observations
                final = extra_info["guidance_final"]
                assert final["text"] == answer
                assert final["status"] == "final" and final["extraction_version"] == 1
                assert extra_info["ha_request_id"] == extra_info["guidance_rollout_id"]
                assert extra_info["ha_request_id"] == server.request_ids[0]
                assert len(set(server.request_ids)) == 1
                received[i] = extra_info["ha_request_id"]
                return {"score": (i + 1) / 10, "transport_index": i}
            return await transport(tokenizer, loop, task_kwargs(i), checker)
        with patch.object(HarnessAuditToolV08, "execute", local_tool):
            rows = await asyncio.gather(*(one(i) for i in range(8)))
        assert len(set(received.values())) == 8
        batch = rows[0][2]._postprocess([row[1] for row in rows])
        batch = pickle.loads(pickle.dumps(batch))
        assert all(abs(a-b) < 1e-6 for a,b in zip(batch.batch["rm_scores"].sum(-1).tolist(), [.1,.2,.3,.4,.5,.6,.7,.8]))
        for i in range(8):
            assert batch.non_tensor_batch["guidance_final"][i]["text"] == f"FINAL_ANSWER_{i}"
            assert batch.non_tensor_batch["guidance_rollout_id"][i] == received[i]
        assert not HarnessAuditToolV08._pending
    run_cpu(scenario())


def check_no_tool_reasoning_and_termination_transport(tokenizer, turns, response_length, max_turns,
                                                  thinking, expected_text, expected_status):
    async def scenario():
        loop, server = make_loop(tokenizer, turns, response_length=response_length, max_turns=max_turns, thinking=thinking)
        def checker(data_source, solution_str, ground_truth, extra_info):
            final = extra_info["guidance_final"]
            assert final["text"] == expected_text
            assert final["status"] == expected_status
            assert extra_info["guidance_rollout_id"] == server.request_ids[0]
            assert "ha_request_id" not in extra_info
            return 0.25
        _, processed, worker = await transport(tokenizer, loop, task_kwargs(), checker)
        assert worker._postprocess([processed]).batch["rm_scores"].sum().item() == .25
    run_cpu(scenario())


class InstalledVerlTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    def test_eight_concurrent_tool_trajectories(self):
        check_eight_concurrent_tool_trajectories_transport_exact_final_and_request(self.tokenizer)

    def test_no_tool(self):
        check_no_tool_reasoning_and_termination_transport(self.tokenizer, ["NO_TOOL_FINAL<|im_end|>"], 2048, 8, False, "NO_TOOL_FINAL", "final")

    def test_prompt_open_thinking(self):
        check_no_tool_reasoning_and_termination_transport(self.tokenizer, ["private reasoning</think>VISIBLE_FINAL<|im_end|>"], 2048, 8, True, "VISIBLE_FINAL", "final")

    def test_turn_limit_does_not_call_tool(self):
        check_no_tool_reasoning_and_termination_transport(self.tokenizer, [CALL], 2048, 1, False, "", "tool_call")

    def test_length_limit(self):
        text = "token " * 32
        length = len(self.tokenizer.encode(text, add_special_tokens=False))
        check_no_tool_reasoning_and_termination_transport(self.tokenizer, [text], length, 8, False, text.strip(), "length")


if __name__ == "__main__":
    unittest.main(verbosity=2)
