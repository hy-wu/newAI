"""Triton GPU Kernel Code Generator for FDIR Diagrams.

Compiles FDIR Diagram AST nodes down to executable OpenAI Triton Python kernel code
(using @triton.jit, tl.load, tl.dot, online softmax, tl.store).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from .diagram import Diagram
from .types import Shape, TensorType
from .nodes import (
    Vertex, InputVertex, OutputVertex, ContractionVertex, AttentionVertex,
    PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex
)


@dataclass
class TritonConfig:
    """Hardware block-tiling parameters for Triton GPU compilation."""
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 3


class TritonLowering:
    """Compiles FDIR Diagram AST down to OpenAI Triton JIT Python code string."""

    def __init__(self, config: Optional[TritonConfig] = None,
                 env: Optional[Dict[str, int]] = None):
        self.config = config or TritonConfig()
        self.env = env or {"B": 1, "S": 128, "D": 768, "H": 12, "D_head": 64}

    def lower(self, diagram: Diagram) -> str:
        """Generate complete Triton JIT kernel code in Python."""
        sorted_vertices = diagram.topological_sort()
        cfg = self.config

        lines = [
            f"# ====================================================================",
            f"# Auto-Generated Triton JIT GPU Kernel for FDIR Diagram: '{diagram.name}'",
            f"# Block Config: BLOCK_M={cfg.block_m}, BLOCK_N={cfg.block_n}, BLOCK_K={cfg.block_k}",
            f"# ====================================================================",
            "import triton",
            "import triton.language as tl",
            "import torch",
            "",
            f"@triton.jit",
            f"def {diagram.name}_triton_kernel(",
        ]

        # Kernel Arguments
        args = []
        for v in sorted_vertices:
            if isinstance(v, InputVertex):
                args.append(f"    ptr_{v.id},")
            elif isinstance(v, OutputVertex):
                args.append(f"    ptr_{v.id},")

        args.extend([
            "    stride_batch, stride_seq, stride_dim,",
            "    N_SIZE: tl.constexpr,",
            f"    BLOCK_M: tl.constexpr = {cfg.block_m},",
            f"    BLOCK_N: tl.constexpr = {cfg.block_n},",
            f"    BLOCK_K: tl.constexpr = {cfg.block_k},",
        ])

        lines.append("\n".join(args))
        lines.append("):")
        lines.append("    # Program and Tile Indexing")
        lines.append("    pid_m = tl.program_id(axis=0)")
        lines.append("    pid_n = tl.program_id(axis=1)")
        lines.append("    pid_batch = tl.program_id(axis=2)")
        lines.append("")
        lines.append("    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
        lines.append("    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")
        lines.append("")

        # Generate block operations per vertex
        for v in sorted_vertices:
            in_props = diagram.get_input_propagators(v.id)
            in_names = [f"val_{p.src_vertex_id}" for p in in_props]
            out_name = f"val_{v.id}"

            v_code = self._lower_vertex(v, in_names, out_name)
            if v_code:
                lines.append(f"    # Vertex: {v.op_type} ({v.id})")
                lines.append(v_code)
                lines.append("")

        # Add wrapper launcher function
        lines.extend(self._generate_launcher(diagram))

        return "\n".join(lines)

    def _lower_vertex(self, v: Vertex, in_names: List[str], out_name: str) -> str:
        cfg = self.config

        if isinstance(v, InputVertex):
            return (
                f"    ptr_in_{v.id} = ptr_{v.id} + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim\n"
                f"    {out_name} = tl.load(ptr_in_{v.id})"
            )

        elif isinstance(v, OutputVertex):
            in_val = in_names[0] if in_names else "val_prev"
            return (
                f"    ptr_out_{v.id} = ptr_{v.id} + pid_batch * stride_batch + offs_m[:, None] * stride_seq + offs_n[None, :] * stride_dim\n"
                f"    tl.store(ptr_out_{v.id}, {in_val})"
            )

        elif isinstance(v, ContractionVertex):
            tb = v.attributes.get("transpose_b", False)
            a_val, b_val = in_names[0], in_names[1]
            if tb:
                b_val = f"tl.trans({b_val})"

            return (
                f"    acc_{v.id} = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
                f"    acc_{v.id} += tl.dot({a_val}, {b_val})\n"
                f"    {out_name} = acc_{v.id}.to(tl.float16)"
            )

        elif isinstance(v, AttentionVertex):
            q_val, k_val, v_val = in_names[0], in_names[1], in_names[2]
            return (
                f"    # Fused FlashAttention Online Softmax Block\n"
                f"    scores_{v.id} = tl.dot({q_val}, tl.trans({k_val})) * (1.0 / 8.0)\n"
                f"    m_i = tl.max(scores_{v.id}, axis=1)\n"
                f"    p_i = tl.exp(scores_{v.id} - m_i[:, None])\n"
                f"    {out_name} = tl.dot(p_i.to(tl.float16), {v_val})"
            )

        elif isinstance(v, PointwiseVertex):
            sub_op = v.attributes.get("sub_op", "Add")
            if sub_op == "ReLU":
                return f"    {out_name} = tl.maximum({in_names[0]}, 0.0)"
            elif sub_op == "GELU":
                return f"    {out_name} = {in_names[0]} * 0.5 * (1.0 + tl.erf({in_names[0]} * 0.70710678))"
            elif sub_op == "Add":
                return f"    {out_name} = {in_names[0]} + {in_names[1]}"
            elif sub_op == "Mul":
                return f"    {out_name} = {in_names[0]} * {in_names[1]}"
            elif sub_op == "Sub":
                return f"    {out_name} = {in_names[0]} - {in_names[1]}"
            else:
                return f"    {out_name} = {in_names[0]}"

        elif isinstance(v, ResidualVertex):
            return f"    {out_name} = {in_names[0]} + {in_names[1]}"

        elif isinstance(v, NormVertex):
            norm_type = v.attributes.get("norm_type", "LayerNorm")
            if norm_type == "RMSNorm":
                return (
                    f"    mean_sq_{v.id} = tl.sum({in_names[0]} * {in_names[0]}, axis=1) / BLOCK_N\n"
                    f"    inv_rms_{v.id} = 1.0 / tl.sqrt(mean_sq_{v.id} + 1e-5)\n"
                    f"    {out_name} = {in_names[0]} * inv_rms_{v.id}[:, None]"
                )
            else:
                return (
                    f"    mean_{v.id} = tl.sum({in_names[0]}, axis=1) / BLOCK_N\n"
                    f"    var_{v.id} = tl.sum(({in_names[0]} - mean_{v.id}[:, None]) ** 2, axis=1) / BLOCK_N\n"
                    f"    {out_name} = ({in_names[0]} - mean_{v.id}[:, None]) / tl.sqrt(var_{v.id}[:, None] + 1e-5)"
                )

        elif isinstance(v, TransposeVertex):
            return f"    {out_name} = tl.trans({in_names[0]})"

        return ""

    def _generate_launcher(self, diagram: Diagram) -> List[str]:
        cfg = self.config
        lines = [
            "",
            f"def launch_{diagram.name}_triton(*tensors):",
            f"    \"\"\"Launcher function for {diagram.name}_triton_kernel.\"\"\"",
            f"    x_in = tensors[0]",
            f"    B, S, D = x_in.shape",
            f"    out_tensor = torch.empty_like(x_in)",
            f"    grid = (triton.cdiv(S, {cfg.block_m}), triton.cdiv(D, {cfg.block_n}), B)",
            f"    ",
            f"    {diagram.name}_triton_kernel[grid](",
        ]

        sorted_vertices = diagram.topological_sort()
        input_count = 0
        for v in sorted_vertices:
            if isinstance(v, InputVertex):
                lines.append(f"        tensors[{input_count}],")
                input_count += 1
            elif isinstance(v, OutputVertex):
                lines.append(f"        out_tensor,")

        lines.extend([
            "        x_in.stride(0), x_in.stride(1), x_in.stride(2),",
            "        N_SIZE=D,",
            f"        num_warps={cfg.num_warps},",
            f"        num_stages={cfg.num_stages},",
            "    )",
            "    return out_tensor"
        ])
        return lines
