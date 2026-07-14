"""Live DeepSeek LLM Autonomous Design Agent Optimization Demo.

Demonstrates closed-loop architecture optimization powered by live DeepSeek API calls.
Reads DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL from .env file.
"""

import sys
import os

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Feynman.fdir import FormulaMapper, HardwareSpec
from Feynman.agent import DesignAgentInterface, DeepSeekClient, LLMDesignAgent


def main():
    print("==========================================================================")
    print(" FDIR Autonomous Design Agent Demo — Live DeepSeek LLM Optimization")
    print("==========================================================================\n")

    # 1. Check for .env file and initialize DeepSeek client
    try:
        client = DeepSeekClient()
        print(f"[*] Successfully connected to DeepSeek API endpoint: '{client.base_url}'")
        print(f"    Model target: '{client.model}'\n")
    except Exception as e:
        print(f"[!] Initialization Error: {e}")
        return

    # 2. Build initial FDIR diagram
    env_dim = {"B": 4, "S": 256, "D": 768}
    hw = HardwareSpec(name="NVIDIA H100 SXM", peak_tflops_fp16=989.4, hbm_bandwidth_tb_s=3.35)
    diagram = FormulaMapper.attention_formula_to_diagram(B=4, S=256, D=768)

    agent_env = DesignAgentInterface(diagram, env=env_dim, hardware=hw)
    llm_agent = LLMDesignAgent(agent_env, llm_client=client)

    print("[*] Initial FDIR Computational Graph & Dual Performance Metrics:")
    initial_obs = agent_env.observe()
    print(initial_obs.performance)
    print("-" * 74)

    # 3. Run closed-loop LLM optimization iterations
    num_steps = 3
    print(f"[*] Launching {num_steps}-Step Closed-Loop Optimization Driven by DeepSeek LLM...\n")

    for i in range(num_steps):
        new_obs, action, reasoning = llm_agent.run_optimization_step(verbose=True)
        print(f"    -> Updated Execution Time Estimate: {new_obs.performance.infra.estimated_time_ms:.4f} ms")
        print(f"    -> Current Bottleneck:             {new_obs.performance.infra.bottleneck}")
        print("-" * 74)

    print("\n[*] Final Optimized Diagram State & Performance Metrics:")
    final_obs = agent_env.observe()
    print(final_obs.performance)

    print("==========================================================================")
    print(" [OK] Live DeepSeek LLM Autonomous Agent Optimization Complete!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
