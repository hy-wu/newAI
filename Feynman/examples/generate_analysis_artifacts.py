"""Artifacts Generator for FDIR Analysis.

Runs the FDIR pipeline and generates a complete suite of analytical artifacts:
  - feynman_diagram.tex & feynman_diagram.pdf (TikZ LaTeX)
  - feynman_diagram.svg & feynman_diagram.html (SVG/HTML vector rendering)
  - formula_math.txt (LaTeX formula representation and Einsum subscript chain)
  - fdir_dsl_code.py (FDIR Python DSL serialization)
  - tile_ir_code.cu (NVIDIA CUDA Tile IR template code)
  - triton_kernel_code.py (OpenAI Triton JIT code)
  - performance_analysis.txt (Model Capacity + GPU Roofline + Physical GPU Telemetry)
"""

import sys
import os
import subprocess
import torch

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Feynman.fdir import (
    FormulaMapper, FeynmanVisualizer, FDIRCodeGen,
    TileIRLowering, TileConfig, TritonLowering, TritonConfig,
    DualEvaluator, HardwareSpec
)
from Feynman.agent import DesignAgentInterface


def main():
    print("==========================================================================")
    print(" FDIR Artifacts Generator: Building Visualizations, Codes & Reports")
    print("==========================================================================\n")

    # 1. Output directory setup
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Output directory configured at: '{output_dir}'")

    # 2. Build FDIR target diagram (Transformer block)
    env_dim = {"B": 4, "S": 256, "D": 768}
    hw = HardwareSpec(name="NVIDIA H100 SXM", peak_tflops_fp16=989.4, hbm_bandwidth_tb_s=3.35)
    d = FormulaMapper.attention_formula_to_diagram(B=4, S=256, D=768)

    # Prepare physical inputs for hardware profiling on GPU
    x_val = torch.randn(4, 256, 768)
    wq_val = torch.randn(768, 768)
    wk_val = torch.randn(768, 768)
    wv_val = torch.randn(768, 768)
    inputs = [x_val, wq_val, wk_val, wv_val]

    # 3. Generate Math Formulas
    print("[*] Generating formulas...")
    latex_formula = FormulaMapper.diagram_to_latex(d)
    einsum_chain = FormulaMapper.diagram_to_einsum_chain(d)
    
    formula_path = os.path.join(output_dir, "formula_math.txt")
    with open(formula_path, "w", encoding="utf-8") as f:
        f.write(f"LaTeX Representation:\n{latex_formula}\n\n")
        f.write(f"Einstein Notation Chain:\n{einsum_chain}\n")
    print(f"    -> Saved: {formula_path}")

    # 4. Generate Visual Feynman Diagrams
    print("[*] Generating visual diagrams...")
    tikz_code = FeynmanVisualizer.to_tikz(d)
    svg_code = FeynmanVisualizer.to_svg(d, width=900, height=400)
    html_code = FeynmanVisualizer.to_html(d)

    # Save Tex, SVG, HTML
    tex_path = os.path.join(output_dir, "feynman_diagram.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        # Wrap TikZ code in a full standalone LaTeX document for compilation
        full_latex_doc = (
            "\\documentclass[tikz, border=10pt]{standalone}\n"
            "\\usepackage[compat=1.1.0]{tikz-feynman}\n"
            "\\begin{document}\n" + tikz_code + "\n\\end{document}\n"
        )
        f.write(full_latex_doc)
    print(f"    -> Saved LaTeX TikZ: {tex_path}")

    svg_path = os.path.join(output_dir, "feynman_diagram.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"    -> Saved SVG:        {svg_path}")

    html_path = os.path.join(output_dir, "feynman_diagram.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"    -> Saved HTML:       {html_path}")

    # Compile LaTeX into PDF (using pdflatex)
    print("[*] Attempting pdflatex compilation of tikz-feynman diagram...")
    try:
        # Run pdflatex in non-interactive mode
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "feynman_diagram.tex"],
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15
        )
        if proc.returncode == 0:
            print(f"    -> Compiled PDF:     {os.path.join(output_dir, 'feynman_diagram.pdf')}")
        else:
            print("    -> [Warning] pdflatex compilation finished with errors (check if tikz-feynman is installed).")
    except Exception as e:
        print(f"    -> [Warning] pdflatex execution skipped: {e}")

    # 5. Generate Code Representations
    print("[*] Generating code modules...")
    dsl_code = FDIRCodeGen.ast_to_code(d)
    dsl_path = os.path.join(output_dir, "fdir_dsl_code.py")
    with open(dsl_path, "w", encoding="utf-8") as f:
        f.write(dsl_code)
    print(f"    -> Saved DSL:        {dsl_path}")

    # 6. Generate GPU IR Codes
    tile_lowering = TileIRLowering(config=TileConfig(tile_m=128, tile_n=128, tile_k=32), env=env_dim)
    tile_code = tile_lowering.lower(d)
    tile_path = os.path.join(output_dir, "tile_ir_code.cu")
    with open(tile_path, "w", encoding="utf-8") as f:
        f.write(tile_code)
    print(f"    -> Saved Tile IR:    {tile_path}")

    triton_lowering = TritonLowering(config=TritonConfig(block_m=128, block_n=128, block_k=32), env=env_dim)
    triton_code = triton_lowering.lower(d)
    triton_path = os.path.join(output_dir, "triton_kernel_code.py")
    with open(triton_path, "w", encoding="utf-8") as f:
        f.write(triton_code)
    print(f"    -> Saved Triton IR:  {triton_path}")

    # 7. Generate Performance & Profiling Reports
    print("[*] Generating performance reports...")
    agent_env = DesignAgentInterface(d, env=env_dim, hardware=hw, profiling_inputs=inputs)
    obs = agent_env.observe()

    analysis_path = os.path.join(output_dir, "performance_analysis.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(obs.to_prompt_context())
    print(f"    -> Saved Metrics:    {analysis_path}\n")

    print("==========================================================================")
    print(f" [OK] Artifact generation complete. All files are located in:")
    print(f"      {output_dir}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
