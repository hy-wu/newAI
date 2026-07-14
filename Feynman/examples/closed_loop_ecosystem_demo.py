"""End-to-End Demonstration of the FDIR Closed-Loop Matrix Ecosystem.

Demonstrates all 5 core mapping paths and the dual-evaluation agent loop:
  1. Formula Parsing: Einstein Notation / Formula -> FDIR AST & LaTeX Generation
  2. Visual Feynman Diagram Generation: TikZ-Feynman & Color-Coded SVG
  3. DSL Serialization: FDIR AST <-> Python DSL Code Roundtrip
  4. Heterogeneous GPU IR Compilation: PyTorch, CUDA Tile IR, OpenAI Triton
  5. Dual Evaluation & Autonomous Design Agent Optimization Loop
"""

import sys
import os
import torch

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Feynman.fdir import (
    Diagram, TensorType, Shape, DType,
    FormulaMapper, FeynmanVisualizer, FDIRCodeGen,
    TorchLowering, TileIRLowering, TileConfig, TritonLowering, TritonConfig,
    DualEvaluator, HardwareSpec,
    DesignAgentInterface, MutationAction, MutationType
)


def main():
    print("==========================================================================")
    print(" FDIR Closed-Loop Matrix Ecosystem: Full Multimodal Pipeline Demo")
    print("==========================================================================\n")

    # --------------------------------------------------------------------
    # 1. Math Formula -> FDIR AST -> LaTeX Conversion
    # --------------------------------------------------------------------
    print("[1] Formula Mapping (Math Formula <-> FDIR AST)")
    einsum_expr = "ik,kj->ij; ij,jl->il"
    d_formula = FormulaMapper.einsum_to_diagram(einsum_expr, name="chained_gemm")
    latex_str = FormulaMapper.diagram_to_latex(d_formula)

    print(f"    Input Einsum Chain: '{einsum_expr}'")
    print(f"    Constructed AST:    {d_formula}")
    print(f"    Exported LaTeX:     {latex_str}\n")

    # Build Attention Block Diagram from Formula
    env = {"B": 4, "S": 256, "D": 768}
    d_attn = FormulaMapper.attention_formula_to_diagram(B=4, S=256, D=768)

    # --------------------------------------------------------------------
    # 2. Visual Machine Learning Feynman Diagram Rendering
    # --------------------------------------------------------------------
    print("[2] Visual Feynman Diagram Rendering (Visualizer)")
    tikz_code = FeynmanVisualizer.to_tikz(d_attn)
    svg_code = FeynmanVisualizer.to_svg(d_attn, width=800, height=360)
    html_code = FeynmanVisualizer.to_html(d_attn)

    print(f"    Generated LaTeX tikz-feynman lines: {len(tikz_code.splitlines())} lines")
    print(f"    Generated SVG markup size:           {len(svg_code)} characters")
    print(f"    Generated HTML interactive doc:      {len(html_code)} characters\n")

    # Save SVG visual diagram artifact to current working dir
    svg_path = os.path.join(os.path.dirname(__file__), "feynman_diagram_attn.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"    [*] Saved SVG Visualization: {svg_path}\n")

    # --------------------------------------------------------------------
    # 3. FDIR AST <-> Python DSL Code Roundtrip
    # --------------------------------------------------------------------
    print("[3] Code Generator (FDIR AST <-> Python DSL Code)")
    dsl_code = FDIRCodeGen.ast_to_code(d_attn)
    d_reconstructed = FDIRCodeGen.code_to_ast(dsl_code)

    print(f"    Generated Python DSL Code lines:    {len(dsl_code.splitlines())} lines")
    print(f"    Reconstructed AST Vertices Count:   {len(d_reconstructed.vertices)}\n")

    # --------------------------------------------------------------------
    # 4. Heterogeneous GPU IR Compilation (PyTorch, CUDA Tile IR, Triton)
    # --------------------------------------------------------------------
    print("[4] Heterogeneous GPU IR Compilers Lowering")

    # 4a. PyTorch nn.Module Lowering
    torch_lowering = TorchLowering()
    pytorch_module = torch_lowering.lower(d_attn)

    x = torch.randn(4, 256, 768)
    wq = torch.randn(768, 768)
    wk = torch.randn(768, 768)
    wv = torch.randn(768, 768)
    y = pytorch_module(x, wq, wk, wv)
    print(f"    [PyTorch Backend] Lowered & Executed OK. Output shape: {tuple(y.shape)}")

    # 4b. NVIDIA CUDA Tile IR Lowering
    tile_lowering = TileIRLowering(config=TileConfig(tile_m=128, tile_n=128, tile_k=32), env=env)
    cuda_tile_code = tile_lowering.lower(d_attn)
    print(f"    [CUDA Tile IR Backend] Generated 'cuda::tile' C++ Code ({len(cuda_tile_code.splitlines())} lines)")

    # 4c. OpenAI Triton JIT Lowering
    triton_lowering = TritonLowering(config=TritonConfig(block_m=128, block_n=128, block_k=32), env=env)
    triton_code = triton_lowering.lower(d_attn)
    print(f"    [OpenAI Triton Backend] Generated '@triton.jit' Kernel ({len(triton_code.splitlines())} lines)\n")

    # --------------------------------------------------------------------
    # 5. Dual Performance Evaluator (Model + Infra Feedback)
    # --------------------------------------------------------------------
    print("[5] Dual Performance Evaluation Engine (Model + Infra Metrics)")
    hw_h100 = HardwareSpec(name="NVIDIA H100 SXM", peak_tflops_fp16=989.4, hbm_bandwidth_tb_s=3.35)
    dual_eval = DualEvaluator(env=env, hardware=hw_h100)

    report = dual_eval.evaluate(d_attn)
    print(report)

    # --------------------------------------------------------------------
    # 6. Autonomous Design Agent Optimization Loop
    # --------------------------------------------------------------------
    print("[6] Autonomous Design Agent Optimization Loop")
    agent_env = DesignAgentInterface(d_attn, env=env, hardware=hw_h100)

    obs_start = agent_env.observe()
    print(f"    [*] Step 0 State: {obs_start.diagram_summary['num_vertices']} Vertices, Est Time: {obs_start.performance.infra.estimated_time_ms:.4f} ms")

    # Agent Action 1: Apply Attention Fusion Rewrite
    obs_step1 = agent_env.mutate(MutationAction(MutationType.FUSE_ATTENTION))
    print(f"    [*] Agent Action 1 (Fuse Attention): Vertices={obs_step1.diagram_summary['num_vertices']}, Est Time: {obs_step1.performance.infra.estimated_time_ms:.4f} ms")

    # Agent Action 2: Swap Normalization Layer to RMSNorm
    obs_step2 = agent_env.mutate(MutationAction(MutationType.SWAP_NORM_TYPE))
    print(f"    [*] Agent Action 2 (Swap to RMSNorm): Vertices={obs_step2.diagram_summary['num_vertices']}, History={obs_step2.mutation_history[-1]}")

    # Agent Action 3: Adjust Hardware Block Tile Configuration
    obs_step3 = agent_env.mutate(MutationAction(
        MutationType.MODIFY_TILE_CONFIG,
        {"tile_m": 256, "tile_n": 128, "tile_k": 64}
    ))
    print(f"    [*] Agent Action 3 (Modify Tile Config): {obs_step3.performance.infra.tile_config_summary}\n")

    print("==========================================================================")
    print(" [OK] Full Closed-Loop Matrix Ecosystem Demo Executed Successfully!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
