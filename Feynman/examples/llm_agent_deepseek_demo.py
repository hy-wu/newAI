"""Live DeepSeek LLM Autonomous Design Agent Optimization Demo with GPU Telemetry.

Demonstrates closed-loop architecture optimization powered by live DeepSeek API calls,
incorporating physical hardware GPU profiling metrics (latency and memory).
"""

import sys
import os
import torch

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Feynman.fdir import FormulaMapper, HardwareSpec
from Feynman.agent import DesignAgentInterface, DeepSeekClient, LLMDesignAgent


def main():
    print("==========================================================================")
    print(" FDIR Autonomous Design Agent Demo — Live DeepSeek & GPU Telemetry")
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

    # 3. Create physical inputs for hardware profiling on the GPU
    x_val = torch.randn(4, 256, 768)
    wq_val = torch.randn(768, 768)
    wk_val = torch.randn(768, 768)
    wv_val = torch.randn(768, 768)
    inputs = [x_val, wq_val, wk_val, wv_val]

    # Instantiate design environment with GPU profiling inputs
    agent_env = DesignAgentInterface(diagram, env=env_dim, hardware=hw, profiling_inputs=inputs)
    llm_agent = LLMDesignAgent(agent_env, llm_client=client)

    print("[*] Initial FDIR Computational Graph & Dual Performance Metrics:")
    initial_obs = agent_env.observe()
    print(initial_obs.performance)
    
    if initial_obs.physical_gpu_metrics:
        m = initial_obs.physical_gpu_metrics
        print("=== Initial Physical GPU Profile ===")
        print(f"  Device Name:            {m.get('cuda_device')}")
        print(f"  Real Kernel Latency:    {m.get('latency_ms', 0.0):.4f} ms")
        print(f"  Real Peak Memory usage: {m.get('peak_memory_mb', 0.0):.2f} MB")
    print("-" * 74)

    # 4. Run closed-loop LLM optimization iterations
    num_steps = 3
    print(f"[*] Launching {num_steps}-Step Closed-Loop Optimization Driven by DeepSeek LLM...\n")

    for i in range(num_steps):
        new_obs, action, reasoning = llm_agent.run_optimization_step(verbose=True)
        if new_obs.physical_gpu_metrics:
            m = new_obs.physical_gpu_metrics
            print(f"    -> Real Latency:                   {m.get('latency_ms', 0.0):.4f} ms")
            print(f"    -> Real Memory:                    {m.get('peak_memory_mb', 0.0):.2f} MB")
        print(f"    -> Est. Loop Execution Bottleneck: {new_obs.performance.infra.bottleneck}")
        print("-" * 74)

    print("\n[*] Final Optimized Diagram State & Performance Metrics:")
    final_obs = agent_env.observe()
    print(final_obs.performance)
    if final_obs.physical_gpu_metrics:
        m = final_obs.physical_gpu_metrics
        print("=== Final Physical GPU Profile ===")
        print(f"  Device Name:            {m.get('cuda_device')}")
        print(f"  Real Kernel Latency:    {m.get('latency_ms', 0.0):.4f} ms")
        print(f"  Real Peak Memory usage: {m.get('peak_memory_mb', 0.0):.2f} MB")

    print("==========================================================================")
    print(" [OK] Live DeepSeek LLM & GPU Telemetry Optimization Complete!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
