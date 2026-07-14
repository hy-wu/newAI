"""Comprehensive test suite for FDIR (Feynman Diagrammatic IR).

Covers:
- Shape conservation law enforcement (C2)
- Contraction axis validation (C1)
- Rewriter actual graph mutation (C3)
- Lowering correctness (C4, D5)
- Cost model liveness-based peak memory (D4)
- Propagator type validation (D1)
"""

import pytest
import torch
import torch.nn.functional as F
from Feynman.fdir import (
    Diagram, TensorType, Shape, DType, Layout,
    InputVertex, OutputVertex, ContractionVertex, AttentionVertex,
    PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex,
    ShapeTypeChecker, ShapeMismatchError, CostModel, PerformanceReport,
    RewriteEngine, TorchLowering
)


# ====================================================================
# Helper: build standard Transformer block diagram
# ====================================================================

def build_transformer_block() -> Diagram:
    """Builds a Self-Attention + Residual + LayerNorm diagram."""
    d = Diagram("transformer_block")

    x_type = TensorType(shape=Shape(("B", "S", "D")), dtype=DType.FLOAT32)
    w_type = TensorType(shape=Shape(("D", "D")), dtype=DType.FLOAT32)

    d.add_vertex(InputVertex("x_in", x_type))
    d.add_vertex(InputVertex("w_q", w_type))
    d.add_vertex(InputVertex("w_k", w_type))
    d.add_vertex(InputVertex("w_v", w_type))

    d.add_vertex(ContractionVertex("proj_q"))
    d.add_vertex(ContractionVertex("proj_k"))
    d.add_vertex(ContractionVertex("proj_v"))

    d.connect("x_in", "proj_q", x_type)
    d.connect("w_q", "proj_q", w_type)
    d.connect("x_in", "proj_k", x_type)
    d.connect("w_k", "proj_k", w_type)
    d.connect("x_in", "proj_v", x_type)
    d.connect("w_v", "proj_v", w_type)

    d.add_vertex(AttentionVertex("sdpa_attn", num_heads=12, head_dim=64))
    d.connect("proj_q", "sdpa_attn", x_type)
    d.connect("proj_k", "sdpa_attn", x_type)
    d.connect("proj_v", "sdpa_attn", x_type)

    d.add_vertex(ResidualVertex("res_add_1"))
    d.connect("x_in", "res_add_1", x_type, label="bypass_1")
    d.connect("sdpa_attn", "res_add_1", x_type, label="attn_out")

    d.add_vertex(NormVertex("norm_1"))
    d.connect("res_add_1", "norm_1", x_type)

    d.add_vertex(OutputVertex("out"))
    d.connect("norm_1", "out", x_type)

    return d


# ====================================================================
# Shape & Type Checker Tests
# ====================================================================

class TestShapeTypeChecker:
    def test_valid_transformer_passes(self):
        diagram = build_transformer_block()
        checker = ShapeTypeChecker(env={"B": 2, "S": 64, "D": 768})
        assert checker.check(diagram) is True

    def test_residual_shape_mismatch_concrete(self):
        """C2: Concrete dim mismatch on residual bypass must raise."""
        d = Diagram("bad_residual")
        t1 = TensorType(shape=Shape((2, 64, 768)))
        t2 = TensorType(shape=Shape((2, 64, 512)))

        d.add_vertex(InputVertex("in1", t1))
        d.add_vertex(InputVertex("in2", t2))
        d.add_vertex(ResidualVertex("res"))

        d.connect("in1", "res", t1)
        d.connect("in2", "res", t2)

        checker = ShapeTypeChecker()
        with pytest.raises(ShapeMismatchError, match="conservation law violated"):
            checker.check(d)

    def test_residual_symbolic_mismatch(self):
        """C2: Different symbolic dim names must be incompatible."""
        d = Diagram("bad_symbolic_residual")
        t1 = TensorType(shape=Shape(("B", "S", "D")))
        t2 = TensorType(shape=Shape(("B", "S", "H")))  # H != D

        d.add_vertex(InputVertex("in1", t1))
        d.add_vertex(InputVertex("in2", t2))
        d.add_vertex(ResidualVertex("res"))

        d.connect("in1", "res", t1)
        d.connect("in2", "res", t2)

        checker = ShapeTypeChecker()
        with pytest.raises(ShapeMismatchError):
            checker.check(d)

    def test_contraction_axis_mismatch_concrete(self):
        """C1: MatMul with mismatched contraction axis must raise."""
        d = Diagram("bad_contraction")
        t1 = TensorType(shape=Shape((2, 64, 768)))
        t2 = TensorType(shape=Shape((512, 768)))  # dim -2 = 512 != 768

        d.add_vertex(InputVertex("in1", t1))
        d.add_vertex(InputVertex("in2", t2))
        d.add_vertex(ContractionVertex("mm"))

        d.connect("in1", "mm", t1)
        d.connect("in2", "mm", t2)

        checker = ShapeTypeChecker()
        with pytest.raises(ShapeMismatchError, match="conservation law violated"):
            checker.check(d)

    def test_contraction_axis_mismatch_symbolic(self):
        """C1: MatMul with different symbolic contraction axis must raise."""
        d = Diagram("bad_contraction_sym")
        t1 = TensorType(shape=Shape(("B", "S", "D")))
        t2 = TensorType(shape=Shape(("H", "D")))  # H != D at dim -2

        d.add_vertex(InputVertex("in1", t1))
        d.add_vertex(InputVertex("in2", t2))
        d.add_vertex(ContractionVertex("mm"))
        d.add_vertex(OutputVertex("out"))

        d.connect("in1", "mm", t1)
        d.connect("in2", "mm", t2)
        d.connect("mm", "out", t1)

        checker = ShapeTypeChecker()
        with pytest.raises(ShapeMismatchError, match="conservation law violated"):
            checker.check(d)


# ====================================================================
# Rewrite Engine Tests
# ====================================================================

class TestRewriteEngine:
    def _build_attention_pattern_diagram(self) -> Diagram:
        """Builds a Contraction(Q, K^T) -> Softmax -> Contraction(scores, V) pattern."""
        d = Diagram("attn_pattern")
        t = TensorType(shape=Shape((2, 8, 64)))

        d.add_vertex(InputVertex("Q", t))
        d.add_vertex(InputVertex("K", t))
        d.add_vertex(InputVertex("V", t))

        d.add_vertex(ContractionVertex("qk_matmul", transpose_b=True))
        d.connect("Q", "qk_matmul", t)
        d.connect("K", "qk_matmul", t)

        scores_type = TensorType(shape=Shape((2, 8, 8)))
        d.add_vertex(PointwiseVertex("softmax", sub_op="Softmax"))
        d.connect("qk_matmul", "softmax", scores_type)

        d.add_vertex(ContractionVertex("scores_v_matmul"))
        d.connect("softmax", "scores_v_matmul", scores_type)
        d.connect("V", "scores_v_matmul", t)

        d.add_vertex(OutputVertex("out"))
        d.connect("scores_v_matmul", "out", t)

        return d

    def test_attention_fusion_reduces_vertex_count(self):
        """C3: Rewriter must actually remove old nodes and insert fused AttentionVertex."""
        d = self._build_attention_pattern_diagram()
        original_count = len(d.vertices)

        rw = RewriteEngine()
        new_d = rw.fuse_attention_pattern(d)

        # 3 nodes replaced by 1 -> net reduction of 2
        assert len(new_d.vertices) == original_count - 2

        # The fused vertex must exist and be an AttentionVertex
        attn_vertices = [v for v in new_d.vertices.values() if isinstance(v, AttentionVertex)]
        assert len(attn_vertices) == 1

    def test_attention_fusion_preserves_connectivity(self):
        """C3: Fused diagram must still be valid (topo sort succeeds, can lower)."""
        d = self._build_attention_pattern_diagram()
        rw = RewriteEngine()
        new_d = rw.fuse_attention_pattern(d)

        # Must not raise
        sorted_v = new_d.topological_sort()
        assert len(sorted_v) == len(new_d.vertices)

    def test_double_transpose_cancellation(self):
        """C3: T_p(T_p(x)) should be eliminated when compose(p, p) = identity."""
        d = Diagram("double_transpose")
        t = TensorType(shape=Shape((2, 3, 4)))

        d.add_vertex(InputVertex("x", t))

        # Swap last two dims: (0, 2, 1) applied twice = identity
        d.add_vertex(TransposeVertex("t1", perm=(0, 2, 1)))
        d.connect("x", "t1", t)

        t_transposed = TensorType(shape=Shape((2, 4, 3)))
        d.add_vertex(TransposeVertex("t2", perm=(0, 2, 1)))
        d.connect("t1", "t2", t_transposed)

        d.add_vertex(OutputVertex("out"))
        d.connect("t2", "out", t)

        rw = RewriteEngine()
        new_d = rw.cancel_double_transpose(d)

        # Both transpose vertices should be removed
        assert "t1" not in new_d.vertices
        assert "t2" not in new_d.vertices

        # x should be directly connected to out
        sorted_v = new_d.topological_sort()
        assert len(sorted_v) == 2  # just InputVertex and OutputVertex


# ====================================================================
# Cost Model Tests
# ====================================================================

class TestCostModel:
    def test_basic_cost_positive(self):
        diagram = build_transformer_block()
        cost_model = CostModel(env={"B": 2, "S": 128, "D": 768})
        report = cost_model.evaluate(diagram)

        assert report.total_flops > 0
        assert report.hbm_bytes > 0
        assert report.kernel_launches > 0

    def test_peak_memory_liveness(self):
        """D4: Peak memory must account for simultaneously live tensors,
        not just the single largest tensor."""
        d = Diagram("liveness_test")

        # Two inputs that are both consumed at the same step
        t1 = TensorType(shape=Shape((1024, 1024)))
        t2 = TensorType(shape=Shape((1024, 1024)))

        d.add_vertex(InputVertex("a", t1))
        d.add_vertex(InputVertex("b", t2))
        d.add_vertex(ResidualVertex("add"))
        d.add_vertex(OutputVertex("out"))

        d.connect("a", "add", t1)
        d.connect("b", "add", t2)
        d.connect("add", "out", t1)

        cost = CostModel(env={})
        report = cost.evaluate(d)

        single_size = t1.size_in_bytes()
        # Peak should be >= 2 * single_size (both inputs live at the add step)
        assert report.peak_memory_bytes >= 2 * single_size


# ====================================================================
# Lowering Tests
# ====================================================================

class TestTorchLowering:
    def test_transformer_block_execution(self):
        """C4: Lowered module must produce correct output tensor."""
        diagram = build_transformer_block()
        lowering = TorchLowering()
        module = lowering.lower(diagram)

        B, S, D = 2, 16, 64
        x = torch.randn(B, S, D)
        w_q = torch.randn(D, D)
        w_k = torch.randn(D, D)
        w_v = torch.randn(D, D)

        output = module(x, w_q, w_k, w_v)
        assert output.shape == (B, S, D)
        assert not torch.isnan(output).any()

    def test_rmsnorm_lowering(self):
        """D5: NormVertex with norm_type='RMSNorm' must execute correctly."""
        d = Diagram("rmsnorm_test")
        t = TensorType(shape=Shape((2, 4, 8)))

        d.add_vertex(InputVertex("x", t))
        d.add_vertex(NormVertex("rms", norm_type="RMSNorm"))
        d.add_vertex(OutputVertex("out"))

        d.connect("x", "rms", t)
        d.connect("rms", "out", t)

        module = TorchLowering().lower(d)
        x = torch.randn(2, 4, 8)
        output = module(x)

        assert output.shape == (2, 4, 8)
        assert not torch.isnan(output).any()
        # RMSNorm output should have roughly unit RMS per last dim
        rms = torch.sqrt(torch.mean(output * output, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_lowering_too_few_inputs_raises(self):
        """C4: Must raise clear error when not enough inputs are provided."""
        d = Diagram("two_inputs")
        t = TensorType(shape=Shape((2, 4)))

        d.add_vertex(InputVertex("a", t))
        d.add_vertex(InputVertex("b", t))
        d.add_vertex(ResidualVertex("add"))
        d.add_vertex(OutputVertex("out"))

        d.connect("a", "add", t)
        d.connect("b", "add", t)
        d.connect("add", "out", t)

        module = TorchLowering().lower(d)
        with pytest.raises(RuntimeError, match="expects at least"):
            module(torch.randn(2, 4))  # Only 1 input, need 2
