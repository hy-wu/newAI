"""Comprehensive Unit Test Suite for the FDIR Closed-Loop Matrix Ecosystem.

Covers:
- FormulaMapper: Formula ↔ AST ↔ LaTeX / Einsum
- FeynmanVisualizer: TikZ, SVG, and HTML rendering
- FDIRCodeGen: AST ↔ Python DSL Code roundtrip
- TileIRLowering: NVIDIA CUDA Tile IR code generation
- TritonLowering: OpenAI Triton JIT code generation
- DualEvaluator: Model & Infra evaluation metrics
- DesignAgentInterface: Observe-mutate cycle & mutation operators
"""

import pytest
import torch
from Feynman.fdir import (
    Diagram, TensorType, Shape, DType,
    FormulaMapper, FeynmanVisualizer, FDIRCodeGen,
    TorchLowering, TileIRLowering, TileConfig, TritonLowering, TritonConfig,
    DualEvaluator, ModelPerformanceEvaluator, InfraPerformanceEvaluator, HardwareSpec,
    DesignAgentInterface, MutationAction, MutationType
)


# ============================================================
# 1. Formula Mapper Tests
# ============================================================

class TestFormulaMapper:
    def test_einsum_to_diagram_basic(self):
        einsum_str = "ik,kj->ij"
        d = FormulaMapper.einsum_to_diagram(einsum_str)
        assert len(d.vertices) == 4  # 2 inputs, 1 contraction, 1 output
        assert len(d.propagators) == 3

    def test_einsum_chain_multi_step(self):
        einsum_chain = "ik,kj->ij; ij,jl->il"
        d = FormulaMapper.einsum_to_diagram(einsum_chain)
        assert len(d.vertices) == 6  # 3 inputs, 2 contractions, 1 output

    def test_diagram_to_latex(self):
        d = FormulaMapper.attention_formula_to_diagram(B=2, S=64, D=128)
        latex = FormulaMapper.diagram_to_latex(d)
        assert "\\mathbf{x}" in latex or "\\mathbf{" in latex
        assert "\\text{Softmax}" in latex or "\\text{" in latex

    def test_diagram_to_einsum_chain(self):
        d = FormulaMapper.einsum_to_diagram("ik,kj->ij")
        chain = FormulaMapper.diagram_to_einsum_chain(d)
        assert "->i,j" in chain or "->" in chain


# ============================================================
# 2. Visualizer Tests
# ============================================================

class TestFeynmanVisualizer:
    def test_to_tikz(self):
        d = FormulaMapper.attention_formula_to_diagram()
        tikz = FeynmanVisualizer.to_tikz(d)
        assert "\\begin{tikzpicture}" in tikz
        assert "\\begin{feynman}" in tikz
        assert "\\end{feynman}" in tikz

    def test_to_svg(self):
        d = FormulaMapper.attention_formula_to_diagram()
        svg = FeynmanVisualizer.to_svg(d)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "sdpa_attention" in svg

    def test_to_html(self):
        d = FormulaMapper.attention_formula_to_diagram()
        html_str = FeynmanVisualizer.to_html(d)
        assert "<!DOCTYPE html>" in html_str
        assert "<svg" in html_str


# ============================================================
# 3. Code Generator Tests
# ============================================================

class TestFDIRCodeGen:
    def test_roundtrip_code_serialization(self):
        d = FormulaMapper.attention_formula_to_diagram(B=2, S=64, D=128)
        code = FDIRCodeGen.ast_to_code(d)
        assert "def build_diagram()" in code

        d_reconstructed = FDIRCodeGen.code_to_ast(code)
        assert len(d_reconstructed.vertices) == len(d.vertices)
        assert len(d_reconstructed.propagators) == len(d.propagators)


# ============================================================
# 4. Hardware Lowering Tests (CUDA Tile IR & Triton)
# ============================================================

class TestHardwareLowering:
    def test_tileir_lowering_code_gen(self):
        d = FormulaMapper.attention_formula_to_diagram()
        lowering = TileIRLowering(config=TileConfig(tile_m=64, tile_n=64, tile_k=32))
        cuda_code = lowering.lower(d)
        assert "using namespace cuda::tile;" in cuda_code
        assert "__global__ void" in cuda_code
        assert "mma(" in cuda_code or "load(" in cuda_code

    def test_triton_lowering_code_gen(self):
        d = FormulaMapper.attention_formula_to_diagram()
        lowering = TritonLowering(config=TritonConfig(block_m=64, block_n=64, block_k=32))
        triton_code = lowering.lower(d)
        assert "@triton.jit" in triton_code
        assert "tl.load" in triton_code
        assert "tl.dot" in triton_code
        assert "launch_" in triton_code


# ============================================================
# 5. Dual Evaluator Tests
# ============================================================

class TestDualEvaluator:
    def test_dual_evaluator_metrics(self):
        d = FormulaMapper.attention_formula_to_diagram(B=4, S=256, D=768)
        evaluator = DualEvaluator(env={"B": 4, "S": 256, "D": 768})
        report = evaluator.evaluate(d)

        assert report.model.total_parameters > 0
        assert report.model.depth > 0
        assert report.infra.total_flops > 0
        assert report.infra.arithmetic_intensity > 0
        assert report.infra.estimated_time_ms > 0

        # Export dict
        d_dict = report.to_dict()
        assert "model" in d_dict
        assert "infra" in d_dict


# ============================================================
# 6. Autonomous Design Agent Interface Tests
# ============================================================

class TestDesignAgentInterface:
    def test_agent_observe_mutate_loop(self):
        d = FormulaMapper.attention_formula_to_diagram(B=2, S=64, D=128)
        agent_env = DesignAgentInterface(d, env={"B": 2, "S": 64, "D": 128})

        obs0 = agent_env.observe()
        assert obs0.step == 0
        assert "transformer_attention_formula" in obs0.diagram_summary["name"]

        # Mutate: swap norm type
        obs1 = agent_env.mutate(MutationAction(MutationType.SWAP_NORM_TYPE))
        assert obs1.step == 1
        assert len(obs1.mutation_history) == 1

        # Mutate: modify tile config
        obs2 = agent_env.mutate(MutationAction(
            MutationType.MODIFY_TILE_CONFIG,
            {"tile_m": 256, "tile_n": 128, "tile_k": 32}
        ))
        assert obs2.step == 2
        assert "TILE_M=256" in obs2.performance.infra.tile_config_summary

        # Available mutations
        mutations = agent_env.get_available_mutations()
        assert len(mutations) > 0

        # Reset
        obs_reset = agent_env.reset()
        assert obs_reset.step == 0
        assert len(obs_reset.mutation_history) == 0

    def test_gpu_profiler_telemetry(self):
        from Feynman.agent.profiler_runner import GPUProfiler
        d = FormulaMapper.attention_formula_to_diagram(B=2, S=16, D=64)
        
        # Test basic availability check
        is_avail = GPUProfiler.is_gpu_available()
        assert isinstance(is_avail, bool)

        # Profile lowered module
        module = TorchLowering().lower(d)
        x = torch.randn(2, 16, 64)
        wq = torch.randn(64, 64)
        wk = torch.randn(64, 64)
        wv = torch.randn(64, 64)
        inputs = [x, wq, wk, wv]

        res = GPUProfiler.profile_module(module, inputs, num_warmup=2, num_steps=5)
        assert isinstance(res, dict)
        if is_avail:
            assert "latency_ms" in res
            assert "peak_memory_mb" in res
            assert res["latency_ms"] >= 0.0
        else:
            assert "error" in res

