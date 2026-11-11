"""Capture current assistant output before veRL flattens tool observations."""
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop,AgentState
from guidance_rl.pipeline import extract_final

class GuidanceToolAgentLoop(ToolAgentLoop):
    async def _handle_generating_state(self,agent_data,sampling_params,ignore_termination=False):
        # Decode only the current prompt suffix; Qwen may pre-open its reasoning block.
        prefix=self.tokenizer.decode(agent_data.prompt_ids[-64:],skip_special_tokens=False)
        thinking_open=prefix.rfind('<think>')>prefix.rfind('</think>')
        state=await super()._handle_generating_state(agent_data,sampling_params,ignore_termination)
        raw=self.tokenizer.decode(agent_data.response_ids,skip_special_tokens=False)
        agent_data.extra_fields['guidance_rollout_id']=str(agent_data.request_id)
        if state==AgentState.TERMINATED:
            if not ignore_termination and len(agent_data.response_mask)>=self.response_length:why='length'
            elif (self.max_assistant_turns and agent_data.assistant_turns>=self.max_assistant_turns) or (self.max_user_turns and agent_data.user_turns>=self.max_user_turns):why='turn_limit'
            else:why='final'
            agent_data.extra_fields['guidance_final']=extract_final(raw,why,thinking_open=thinking_open)
        return state

    async def _handle_processing_tools_state(self,agent_data):
        state=await super()._handle_processing_tools_state(agent_data)
        if state==AgentState.TERMINATED:
            agent_data.extra_fields['guidance_final']=extract_final('','tool_overflow')
        return state
