"""DeepSeek-Powered Autonomous LLM Architecture Design Agent.

Integrated closed-loop agent:
  1. Observes FDIR Dual Performance metrics (Model Capacity & Hardware Roofline)
  2. Queries DeepSeek LLM for architecture optimization reasoning
  3. Extracts structured Mutation Action JSON
  4. Executes mutation on DesignAgentInterface and updates the computation graph
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Any, Optional
from .design_agent import DesignAgentInterface, MutationAction, MutationType, Observation
from .llm_client import DeepSeekClient


class LLMDesignAgent:
    """Autonomous LLM Agent driven by DeepSeek for closed-loop FDIR architecture search."""

    def __init__(self, agent_env: DesignAgentInterface,
                 llm_client: Optional[DeepSeekClient] = None,
                 system_prompt_addon: str = ""):
        self.env = agent_env
        self.client = llm_client or DeepSeekClient()
        self.system_prompt_addon = system_prompt_addon

    def run_optimization_step(self, verbose: bool = True) -> Tuple[Observation, MutationAction, str]:
        """Run a single closed-loop optimization step using DeepSeek LLM.

        Returns:
          (new_observation, action_taken, llm_reasoning_text)
        """
        obs = self.env.observe()
        available_actions = self.env.get_available_mutations()

        prompt_messages = self._construct_prompt(obs, available_actions)
        reasoning_response = self.client.chat_completion(prompt_messages, temperature=0.2)

        action = self._parse_llm_action(reasoning_response, available_actions)

        if verbose:
            print(f"\n[Agent Optimization Step {obs.step + 1}]")
            print(f"  Reasoning: {reasoning_response.strip().splitlines()[0]}")
            print(f"  Chosen Action: {action}")

        new_obs = self.env.mutate(action)
        return new_obs, action, reasoning_response

    def _construct_prompt(self, obs: Observation, available_actions: List[MutationAction]) -> List[Dict[str, str]]:
        system_content = (
            "You are an expert AI Compiler and Architecture Search Agent optimizing a Feynman Diagrammatic IR (FDIR) computational graph.\n"
            "Your goal is to optimize both Model Capacity (parameters, depth, expressiveness) and Infra Efficiency (FLOPs, memory traffic, execution time, bottleneck).\n\n"
            "Respond in JSON format with two keys:\n"
            '1. "reasoning": Brief explanation of your architectural decision.\n'
            '2. "action": An object containing "mutation_type" and "params".\n\n'
            "Valid Mutation Types:\n"
            "  - FUSE_ATTENTION\n"
            "  - SWAP_NORM_TYPE\n"
            "  - MODIFY_TILE_CONFIG (params e.g. {\"tile_m\": 256, \"tile_n\": 128, \"tile_k\": 64})\n"
            "  - REMOVE_VERTEX (params {\"vertex_id\": \"...\"})\n"
            "  - ADD_RESIDUAL_BYPASS (params {\"from_vertex\": \"...\", \"to_before_vertex\": \"...\"})\n"
        )
        if self.system_prompt_addon:
            system_content += f"\nAdditional Instructions: {self.system_prompt_addon}\n"

        user_content = obs.to_prompt_context() + "\n\nAvailable Suggested Actions:\n"
        for idx, a in enumerate(available_actions, 1):
            user_content += f"  {idx}. {a}\n"

        user_content += "\nAnalyze the current bottleneck and output your chosen action JSON object."

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def _parse_llm_action(self, llm_response: str, available_actions: List[MutationAction]) -> MutationAction:
        """Parse structured JSON from LLM output, with fallback to default action."""
        try:
            # Extract JSON block if surrounded by markdown code fences
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
            json_str = match.group(1) if match else llm_response

            data = json.loads(json_str)
            act_data = data.get("action", {})
            m_type_str = act_data.get("mutation_type", "").upper()
            params = act_data.get("params", {})

            m_type = MutationType[m_type_str]
            return MutationAction(m_type, params)
        except Exception:
            # Fallback if parsing fails: pick first available mutation
            if available_actions:
                return available_actions[0]
            return MutationAction(MutationType.FUSE_ATTENTION)
