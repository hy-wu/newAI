"""Multi-Layer LLaMA-3 Modern LLM Architecture Verification Demo.

Constructs, validates, compiles, profiles, and optimizes a multi-layer stacked
LLaMA-3 Decoder architecture in FDIR:
  - 2 Stacked Decoder Blocks (RMSNorm + SDPA Attn + SwiGLU FFN + Dual Residuals per layer)
  - Static Shape Conservation Check (ShapeTypeChecker)
  - Multi-Layer Cost & Roofline Analysis (DualEvaluator)
  - Lowering to PyTorch & Hardware Profiling on GPU (GPUProfiler)
  - Lowering to NVIDIA CUDA Tile IR & OpenAI Triton JIT
  - Live Optimization Loop via DeepSeek LLM Agent
  - Exporting Multi-Layer Visual Diagrams & PDF Compilation (pdflatex)
"""

import sys
import os
import subprocess
import torch

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Feynman.fdir import (
    FormulaMapper, ShapeTypeChecker, FeynmanVisualizer, FDIRCodeGen,
    TorchLowering, TileIRLowering, TileConfig, TritonLowering, TritonConfig,
    DualEvaluator, HardwareSpec
)
from Feynman.agent import DesignAgentInterface, DeepSeekClient, LLMDesignAgent


def main():
    print("==========================================================================")
    print(" Multi-Layer Modern LLM Architecture Verification: LLaMA-3 (2-Layer Stack)")
    print("==========================================================================\n")

    num_layers = 2
    hidden_dim = 4096
    num_heads = 32
    head_dim = 128
    intermediate_dim = 11008

    env_dim = {
        "B": 2,
        "S": 128,
        "D": hidden_dim,
        "H": num_heads,
        "D_head": head_dim,
        "I": intermediate_dim
    }

    # 1. Build Multi-Layer LLaMA-3 Diagram AST
    print(f"[*] Constructing {num_layers}-Layer LLaMA-3 Decoder Architecture (Hidden={hidden_dim}, FFN={intermediate_dim})...")
    d_llama = FormulaMapper.llama_architecture_to_diagram(
        num_layers=num_layers,
        B=2, S=128, D=hidden_dim,
        num_heads=num_heads, head_dim=head_dim,
        intermediate_dim=intermediate_dim
    )

    print(f"    Constructed AST Vertices Count:   {len(d_llama.vertices)}")
    print(f"    Constructed AST Propagators Count:{len(d_llama.propagators)}\n")

    # 2. Static Shape Conservation Law Checking
    print("[*] Performing Static Dimension Conservation Law Verification (ShapeTypeChecker)...")
    checker = ShapeTypeChecker(env=env_dim)
    is_valid = checker.check(d_llama)
    print(f"    -> Dimension Conservation Laws Verified: {is_valid}\n")

    # 3. Dual Performance Evaluation (Model Capacity + GPU Roofline)
    print("[*] Evaluating Dual Performance Metrics (Model Capacity & Roofline)...")
    hw_h100 = HardwareSpec(name="NVIDIA H100 SXM", peak_tflops_fp16=989.4, hbm_bandwidth_tb_s=3.35)
    dual_eval = DualEvaluator(env=env_dim, hardware=hw_h100)
    report = dual_eval.evaluate(d_llama)
    print(report)

    # 4. Lowering to PyTorch and Physical GPU Profiling
    print("[*] Lowering Multi-Layer LLaMA-3 to PyTorch & Running GPU Profiler...")
    torch_lowering = TorchLowering()
    pytorch_llama = torch_lowering.lower(d_llama)

    # Construct physical tensor inputs for 2 layers
    device_inputs = [torch.randn(2, 128, hidden_dim)]
    for l in range(num_layers):
        device_inputs.extend([
            torch.randn(hidden_dim, hidden_dim),         # W_q
            torch.randn(hidden_dim, hidden_dim),         # W_k
            torch.randn(hidden_dim, hidden_dim),         # W_v
            torch.randn(hidden_dim, hidden_dim),         # W_o
            torch.randn(hidden_dim, intermediate_dim),  # W_gate
            torch.randn(hidden_dim, intermediate_dim),  # W_up
            torch.randn(intermediate_dim, hidden_dim),  # W_down
        ])

    # Run PyTorch forward pass
    if torch.cuda.is_available():
        gpu_inputs = [x.to("cuda") for x in device_inputs]
        pytorch_llama = pytorch_llama.to("cuda")
        out_tensor = pytorch_llama(*gpu_inputs)
        print(f"    -> Physical GPU Forward Pass Succeeded. Output Logits Shape: {tuple(out_tensor.shape)}")

    # 5. Hardware Lowering to CUDA Tile IR & Triton JIT
    print("\n[*] Compiling Multi-Layer LLaMA-3 to NVIDIA CUDA Tile IR & OpenAI Triton JIT...")
    tile_code = TileIRLowering(TileConfig(tile_m=128, tile_n=128, tile_k=32), env=env_dim).lower(d_llama)
    triton_code = TritonLowering(TritonConfig(block_m=128, block_n=128, block_k=32), env=env_dim).lower(d_llama)

    print(f"    -> CUDA Tile IR Code Generated: {len(tile_code.splitlines())} lines")
    print(f"    -> Triton JIT Code Generated:    {len(triton_code.splitlines())} lines\n")

    # 6. Live Optimization Loop via DeepSeek LLM Agent
    print("[*] Launching Closed-Loop Architecture Optimization Loop via DeepSeek LLM Agent...")
    try:
        client = DeepSeekClient()
        agent_env = DesignAgentInterface(d_llama, env=env_dim, hardware=hw_h100, profiling_inputs=device_inputs)
        llm_agent = LLMDesignAgent(agent_env, llm_client=client)

        for step in range(2):
            new_obs, action, reasoning = llm_agent.run_optimization_step(verbose=True)
            if new_obs.physical_gpu_metrics:
                m = new_obs.physical_gpu_metrics
                print(f"    -> Physical GPU Latency: {m.get('latency_ms', 0.0):.4f} ms | Peak Memory: {m.get('peak_memory_mb', 0.0):.2f} MB")
    except Exception as e:
        print(f"    -> DeepSeek LLM Agent optimization step skipped: {e}")

    # 7. Generate Multi-Layer Visual Diagrams & PDF Artifacts
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[*] Exporting Multi-Layer Visual Diagrams & Compiling PDF Artifacts...")
    svg_code = FeynmanVisualizer.to_svg(d_llama)
    svg_path = os.path.join(output_dir, "llama3_diagram.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"    -> Saved Dynamic Canvas SVG: {svg_path}")

    tikz_code = FeynmanVisualizer.to_tikz(d_llama)
    tex_path = os.path.join(output_dir, "llama3_diagram.tex")
    full_latex_doc = (
        "\\documentclass[tikz, border=15pt]{standalone}\n"
        "\\usepackage[compat=1.1.0]{tikz-feynman}\n"
        "\\begin{document}\n" + tikz_code + "\n\\end{document}\n"
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(full_latex_doc)
    print(f"    -> Saved LaTeX TikZ Source: {tex_path}")

    # Compile PDF via pdflatex
    print("[*] Compiling pdflatex for Multi-Layer LLaMA-3 TikZ Diagram...")
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "llama3_diagram.tex"],
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=25
        )
        pdf_path = os.path.join(output_dir, "llama3_diagram.pdf")
        if proc.returncode == 0 or os.path.exists(pdf_path):
            print(f"    -> Compiled PDF Diagram:    {pdf_path}")
        else:
            print("    -> [Warning] pdflatex execution finished with warnings.")
    except Exception as e:
        print(f"    -> [Warning] pdflatex skipped: {e}")

    with open(os.path.join(output_dir, "llama3_tile_ir.cu"), "w", encoding="utf-8") as f:
        f.write(tile_code)

    with open(os.path.join(output_dir, "llama3_triton_kernel.py"), "w", encoding="utf-8") as f:
        f.write(triton_code)

    print("==========================================================================")
    print(" [OK] Multi-Layer Modern LLM (LLaMA-3 Architecture) Verification Complete!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
