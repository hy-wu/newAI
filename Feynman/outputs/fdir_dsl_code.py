# Auto-generated Python DSL code from FDIR Diagram
from Feynman.fdir import (
    Diagram, TensorType, Shape, DType, Layout,
    InputVertex, OutputVertex, ContractionVertex, AttentionVertex,
    PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex
)

def build_diagram() -> Diagram:
    d = Diagram("transformer_attention_formula")

    t_x = TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32)
    d.add_vertex(InputVertex("x", t_x))
    t_W_q = TensorType(shape=Shape((768, 768)), dtype=DType.FLOAT32)
    d.add_vertex(InputVertex("W_q", t_W_q))
    t_W_k = TensorType(shape=Shape((768, 768)), dtype=DType.FLOAT32)
    d.add_vertex(InputVertex("W_k", t_W_k))
    t_W_v = TensorType(shape=Shape((768, 768)), dtype=DType.FLOAT32)
    d.add_vertex(InputVertex("W_v", t_W_v))
    d.add_vertex(ContractionVertex("proj_Q", transpose_b=False))
    d.add_vertex(ContractionVertex("proj_K", transpose_b=False))
    d.add_vertex(ContractionVertex("proj_V", transpose_b=False))
    d.add_vertex(AttentionVertex("sdpa_attention", num_heads=12, head_dim=64))
    d.add_vertex(ResidualVertex("residual_add"))
    d.add_vertex(NormVertex("layer_norm", norm_type="LayerNorm", eps=1e-05))
    d.add_vertex(OutputVertex("out"))

    # Propagators
    d.connect("x", "proj_Q", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="edge_1")
    d.connect("W_q", "proj_Q", TensorType(shape=Shape((768, 768)), dtype=DType.FLOAT32), label="edge_2")
    d.connect("x", "proj_K", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="edge_3")
    d.connect("W_k", "proj_K", TensorType(shape=Shape((768, 768)), dtype=DType.FLOAT32), label="edge_4")
    d.connect("x", "proj_V", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="edge_5")
    d.connect("W_v", "proj_V", TensorType(shape=Shape((768, 768)), dtype=DType.FLOAT32), label="edge_6")
    d.connect("proj_Q", "sdpa_attention", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="edge_7")
    d.connect("proj_K", "sdpa_attention", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="edge_8")
    d.connect("proj_V", "sdpa_attention", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="edge_9")
    d.connect("x", "residual_add", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="bypass")
    d.connect("sdpa_attention", "residual_add", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="attn_branch")
    d.connect("residual_add", "layer_norm", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="edge_12")
    d.connect("layer_norm", "out", TensorType(shape=Shape((4, 256, 768)), dtype=DType.FLOAT32), label="edge_13")

    return d

diagram = build_diagram()