"""Math Formula ↔ FDIR AST Bidirectional Conversion.

Provides bidirectional conversion between mathematical representations
(LaTeX equations and Einstein notation strings) and FDIR Diagram ASTs.
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Optional
from .diagram import Diagram
from .types import TensorType, Shape, DType
from .nodes import (
    Vertex, InputVertex, OutputVertex, ContractionVertex,
    AttentionVertex, PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex
)


class FormulaMapper:
    """Bidirectional mapper between Mathematical Formulas and FDIR Diagram AST."""

    @staticmethod
    def einsum_to_diagram(einsum_str: str, name: str = "einsum_diagram",
                          env: Optional[Dict[str, int]] = None) -> Diagram:
        """Parse Einstein notation (e.g. 'ik,kj->ij' or 'bsh,hd->bsd') into an FDIR Diagram.

        Supports multi-step chains separated by semicolons (e.g. 'ik,kj->ij; ij,jl->il').
        """
        diagram = Diagram(name)
        steps = [s.strip() for s in einsum_str.split(";") if s.strip()]

        current_input_var: Optional[str] = None

        for step_idx, step in enumerate(steps):
            if "->" not in step:
                raise ValueError(f"Invalid einsum expression (missing '->'): '{step}'")

            inputs_part, output_part = step.split("->")
            input_subscripts = [s.strip() for s in inputs_part.split(",")]
            out_subscript = output_part.strip()

            if len(input_subscripts) < 2:
                raise ValueError(f"Contraction step requires at least 2 operands, got: '{step}'")

            # Resolve dimension names
            lhs_indices = tuple(list(input_subscripts[0]))
            rhs_indices = tuple(list(input_subscripts[1]))
            out_indices = tuple(list(out_subscript))

            # Input Tensors
            lhs_name = f"x_{step_idx}" if step_idx > 0 else "input_lhs"
            rhs_name = f"W_{step_idx}"

            lhs_type = TensorType(shape=Shape(lhs_indices), indices=lhs_indices)
            rhs_type = TensorType(shape=Shape(rhs_indices), indices=rhs_indices)
            out_type = TensorType(shape=Shape(out_indices), indices=out_indices)

            if step_idx == 0:
                diagram.add_vertex(InputVertex(lhs_name, lhs_type))

            diagram.add_vertex(InputVertex(rhs_name, rhs_type))

            contract_v_id = f"contract_{step_idx}"
            diagram.add_vertex(ContractionVertex(contract_v_id))

            src_lhs = current_input_var if current_input_var else lhs_name
            diagram.connect(src_lhs, contract_v_id, lhs_type)
            diagram.connect(rhs_name, contract_v_id, rhs_type)

            current_input_var = contract_v_id

        # Add Output vertex
        out_v_id = "output"
        diagram.add_vertex(OutputVertex(out_v_id))
        diagram.connect(current_input_var, out_v_id, out_type)

        return diagram

    @staticmethod
    def attention_formula_to_diagram(B: Union[int, str] = "B",
                                     S: Union[int, str] = "S",
                                     D: Union[int, str] = "D",
                                     num_heads: int = 12,
                                     head_dim: int = 64) -> Diagram:
        """Build a standard Transformer SDPA Attention block from formula specifications."""
        d = Diagram("transformer_attention_formula")

        x_type = TensorType(shape=Shape((B, S, D)), dtype=DType.FLOAT32)
        w_type = TensorType(shape=Shape((D, D)), dtype=DType.FLOAT32)

        # 1. External Inputs
        d.add_vertex(InputVertex("x", x_type))
        d.add_vertex(InputVertex("W_q", w_type))
        d.add_vertex(InputVertex("W_k", w_type))
        d.add_vertex(InputVertex("W_v", w_type))

        # 2. Linear Projections (Contractions)
        d.add_vertex(ContractionVertex("proj_Q"))
        d.add_vertex(ContractionVertex("proj_K"))
        d.add_vertex(ContractionVertex("proj_V"))

        d.connect("x", "proj_Q", x_type)
        d.connect("W_q", "proj_Q", w_type)
        d.connect("x", "proj_K", x_type)
        d.connect("W_k", "proj_K", w_type)
        d.connect("x", "proj_V", x_type)
        d.connect("W_v", "proj_V", w_type)

        # 3. Attention Interaction Vertex
        d.add_vertex(AttentionVertex("sdpa_attention", num_heads=num_heads, head_dim=head_dim))
        d.connect("proj_Q", "sdpa_attention", x_type)
        d.connect("proj_K", "sdpa_attention", x_type)
        d.connect("proj_V", "sdpa_attention", x_type)

        # 4. Residual Addition
        d.add_vertex(ResidualVertex("residual_add"))
        d.connect("x", "residual_add", x_type, label="bypass")
        d.connect("sdpa_attention", "residual_add", x_type, label="attn_branch")

        # 5. LayerNorm
        d.add_vertex(NormVertex("layer_norm", norm_type="LayerNorm"))
        d.connect("residual_add", "layer_norm", x_type)

        # 6. Output
        d.add_vertex(OutputVertex("out"))
        d.connect("layer_norm", "out", x_type)

        return d

    @staticmethod
    def llama_architecture_to_diagram(num_layers: int = 2,
                                      B: Union[int, str] = "B",
                                      S: Union[int, str] = "S",
                                      D: Union[int, str] = 4096,
                                      num_heads: int = 32,
                                      head_dim: int = 128,
                                      intermediate_dim: int = 11008) -> Diagram:
        """Build a multi-layer stacked LLaMA-style Decoder architecture Diagram AST.

        Each layer contains:
          - Input RMSNorm
          - Q, K, V Projections & SDPA Attention Interaction
          - Output Projection & Residual Bypass
          - Post-Attention RMSNorm
          - SwiGLU FFN (Gate + Up Projection, SiLU activation, Mul, Down Projection)
          - Post-FFN Residual Bypass
        """
        d = Diagram(f"llama3_{num_layers}layer_architecture")

        x_type = TensorType(shape=Shape((B, S, D)), dtype=DType.FLOAT32)
        w_attn_type = TensorType(shape=Shape((D, D)), dtype=DType.FLOAT32)
        w_ffn_type = TensorType(shape=Shape((D, intermediate_dim)), dtype=DType.FLOAT32)
        w_down_type = TensorType(shape=Shape((intermediate_dim, D)), dtype=DType.FLOAT32)

        # External Initial Input X
        d.add_vertex(InputVertex("x_input", x_type))
        current_state_id = "x_input"

        for l in range(num_layers):
            pfx = f"layer_{l}"

            # Weights for layer l
            d.add_vertex(InputVertex(f"{pfx}_W_q", w_attn_type))
            d.add_vertex(InputVertex(f"{pfx}_W_k", w_attn_type))
            d.add_vertex(InputVertex(f"{pfx}_W_v", w_attn_type))
            d.add_vertex(InputVertex(f"{pfx}_W_o", w_attn_type))

            d.add_vertex(InputVertex(f"{pfx}_W_gate", w_ffn_type))
            d.add_vertex(InputVertex(f"{pfx}_W_up", w_ffn_type))
            d.add_vertex(InputVertex(f"{pfx}_W_down", w_down_type))

            # --- Sub-Block 1: Self Attention ---
            # 1.1 Input RMSNorm
            norm1_id = f"{pfx}_rms_norm1"
            d.add_vertex(NormVertex(norm1_id, norm_type="RMSNorm"))
            d.connect(current_state_id, norm1_id, x_type)

            # 1.2 Q, K, V Projections
            q_id, k_id, v_id = f"{pfx}_proj_Q", f"{pfx}_proj_K", f"{pfx}_proj_V"
            d.add_vertex(ContractionVertex(q_id))
            d.add_vertex(ContractionVertex(k_id))
            d.add_vertex(ContractionVertex(v_id))

            d.connect(norm1_id, q_id, x_type)
            d.connect(f"{pfx}_W_q", q_id, w_attn_type)
            d.connect(norm1_id, k_id, x_type)
            d.connect(f"{pfx}_W_k", k_id, w_attn_type)
            d.connect(norm1_id, v_id, x_type)
            d.connect(f"{pfx}_W_v", v_id, w_attn_type)

            # 1.3 SDPA Attention Interaction
            attn_id = f"{pfx}_attention"
            d.add_vertex(AttentionVertex(attn_id, num_heads=num_heads, head_dim=head_dim))
            d.connect(q_id, attn_id, x_type)
            d.connect(k_id, attn_id, x_type)
            d.connect(v_id, attn_id, x_type)

            # 1.4 Output Projection
            proj_o_id = f"{pfx}_proj_O"
            d.add_vertex(ContractionVertex(proj_o_id))
            d.connect(attn_id, proj_o_id, x_type)
            d.connect(f"{pfx}_W_o", proj_o_id, w_attn_type)

            # 1.5 Residual Addition (Attn Bypass)
            res1_id = f"{pfx}_res_attn"
            d.add_vertex(ResidualVertex(res1_id))
            d.connect(current_state_id, res1_id, x_type, label="bypass")
            d.connect(proj_o_id, res1_id, x_type, label="attn_branch")

            # --- Sub-Block 2: SwiGLU FFN ---
            # 2.1 Post-Attn RMSNorm
            norm2_id = f"{pfx}_rms_norm2"
            d.add_vertex(NormVertex(norm2_id, norm_type="RMSNorm"))
            d.connect(res1_id, norm2_id, x_type)

            # 2.2 Gate & Up Projections
            gate_id, up_id = f"{pfx}_proj_gate", f"{pfx}_proj_up"
            d.add_vertex(ContractionVertex(gate_id))
            d.add_vertex(ContractionVertex(up_id))

            ffn_inter_type = TensorType(shape=Shape((B, S, intermediate_dim)), dtype=DType.FLOAT32)
            d.connect(norm2_id, gate_id, x_type)
            d.connect(f"{pfx}_W_gate", gate_id, w_ffn_type)
            d.connect(norm2_id, up_id, x_type)
            d.connect(f"{pfx}_W_up", up_id, w_ffn_type)

            # 2.3 SiLU Activation on Gate & Multiplication (SwiGLU)
            silu_id = f"{pfx}_silu"
            d.add_vertex(PointwiseVertex(silu_id, sub_op="GELU"))
            d.connect(gate_id, silu_id, ffn_inter_type)

            swiglu_mul_id = f"{pfx}_swiglu_mul"
            d.add_vertex(PointwiseVertex(swiglu_mul_id, sub_op="Mul"))
            d.connect(silu_id, swiglu_mul_id, ffn_inter_type)
            d.connect(up_id, swiglu_mul_id, ffn_inter_type)

            # 2.4 Down Projection
            down_id = f"{pfx}_proj_down"
            d.add_vertex(ContractionVertex(down_id))
            d.connect(swiglu_mul_id, down_id, ffn_inter_type)
            d.connect(f"{pfx}_W_down", down_id, w_down_type)

            # 2.5 Post-FFN Residual Addition
            res2_id = f"{pfx}_res_ffn"
            d.add_vertex(ResidualVertex(res2_id))
            d.connect(res1_id, res2_id, x_type, label="bypass")
            d.connect(down_id, res2_id, x_type, label="ffn_branch")

            current_state_id = res2_id

        # Final Normalization and Output
        d.add_vertex(NormVertex("final_rms_norm", norm_type="RMSNorm"))
        d.connect(current_state_id, "final_rms_norm", x_type)

        d.add_vertex(OutputVertex("logits_out"))
        d.connect("final_rms_norm", "logits_out", x_type)

        return d


    @staticmethod
    def diagram_to_latex(diagram: Diagram) -> str:
        """Convert an FDIR Diagram AST into an explicit LaTeX math string."""
        sorted_vertices = diagram.topological_sort()
        var_expressions: Dict[str, str] = {}

        for v in sorted_vertices:
            if isinstance(v, InputVertex):
                var_expressions[v.id] = f"\\mathbf{{{v.id}}}"

            elif isinstance(v, OutputVertex):
                in_props = diagram.get_input_propagators(v.id)
                if in_props:
                    src_id = in_props[0].src_vertex_id
                    var_expressions[v.id] = var_expressions.get(src_id, f"\\mathbf{{{src_id}}}")

            elif isinstance(v, ContractionVertex):
                in_props = diagram.get_input_propagators(v.id)
                if len(in_props) >= 2:
                    lhs = var_expressions.get(in_props[0].src_vertex_id, in_props[0].src_vertex_id)
                    rhs = var_expressions.get(in_props[1].src_vertex_id, in_props[1].src_vertex_id)
                    transpose_b = v.attributes.get("transpose_b", False)
                    rhs_str = f"{rhs}^T" if transpose_b else rhs
                    var_expressions[v.id] = f"{lhs} \\cdot {rhs_str}"
                else:
                    var_expressions[v.id] = f"\\text{{Contract}}({v.id})"

            elif isinstance(v, AttentionVertex):
                in_props = diagram.get_input_propagators(v.id)
                if len(in_props) >= 3:
                    q = var_expressions.get(in_props[0].src_vertex_id, "Q")
                    k = var_expressions.get(in_props[1].src_vertex_id, "K")
                    v_mat = var_expressions.get(in_props[2].src_vertex_id, "V")
                    var_expressions[v.id] = f"\\text{{Softmax}}\\left(\\frac{{{q} {k}^T}}{{\\sqrt{{d_k}}}}\\right) {v_mat}"
                else:
                    var_expressions[v.id] = f"\\text{{Attention}}({v.id})"

            elif isinstance(v, PointwiseVertex):
                sub_op = v.attributes.get("sub_op", "Add")
                in_props = diagram.get_input_propagators(v.id)
                if in_props:
                    arg0 = var_expressions.get(in_props[0].src_vertex_id, "x")
                    if sub_op in ("ReLU", "GELU", "Softmax", "Sigmoid", "Tanh"):
                        var_expressions[v.id] = f"\\text{{{sub_op}}}\\left({arg0}\\right)"
                    elif sub_op in ("Add", "Mul", "Sub") and len(in_props) >= 2:
                        arg1 = var_expressions.get(in_props[1].src_vertex_id, "y")
                        op_symbol = "+" if sub_op == "Add" else ("\\times" if sub_op == "Mul" else "-")
                        var_expressions[v.id] = f"({arg0} {op_symbol} {arg1})"
                    else:
                        var_expressions[v.id] = f"\\text{{{sub_op}}}({arg0})"

            elif isinstance(v, ResidualVertex):
                in_props = diagram.get_input_propagators(v.id)
                if len(in_props) >= 2:
                    bypass = var_expressions.get(in_props[0].src_vertex_id, "x")
                    branch = var_expressions.get(in_props[1].src_vertex_id, "\\mathcal{F}(x)")
                    var_expressions[v.id] = f"{bypass} + {branch}"
                else:
                    var_expressions[v.id] = f"x + \\mathcal{{F}}(x)"

            elif isinstance(v, NormVertex):
                norm_type = v.attributes.get("norm_type", "LayerNorm")
                in_props = diagram.get_input_propagators(v.id)
                arg0 = var_expressions.get(in_props[0].src_vertex_id, "x") if in_props else "x"
                var_expressions[v.id] = f"\\text{{{norm_type}}}\\left({arg0}\\right)"

            elif isinstance(v, TransposeVertex):
                perm = v.attributes.get("perm", ())
                in_props = diagram.get_input_propagators(v.id)
                arg0 = var_expressions.get(in_props[0].src_vertex_id, "x") if in_props else "x"
                var_expressions[v.id] = f"\\text{{Transpose}}_{{{perm}}}\\left({arg0}\\right)"

        output_var = diagram.outputs[0] if diagram.outputs else sorted_vertices[-1].id
        final_expr = var_expressions.get(output_var, "\\text{Output}")
        return f"y = {final_expr}"

    @staticmethod
    def diagram_to_einsum_chain(diagram: Diagram) -> str:
        """Convert ContractionVertex operations in Diagram back to Einstein notation string."""
        einsum_steps = []
        for v in diagram.topological_sort():
            if isinstance(v, ContractionVertex):
                in_props = diagram.get_input_propagators(v.id)
                out_props = diagram.get_output_propagators(v.id)
                if len(in_props) >= 2 and out_props:
                    t1_idx = "".join(str(d).lower() for d in in_props[0].tensor_type.shape.dims)
                    t2_idx = "".join(str(d).lower() for d in in_props[1].tensor_type.shape.dims)
                    out_idx = "".join(str(d).lower() for d in out_props[0].tensor_type.shape.dims)
                    einsum_steps.append(f"{t1_idx},{t2_idx}->{out_idx}")
        return "; ".join(einsum_steps)
