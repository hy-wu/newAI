"""NVIDIA CUDA Tile IR Code Generator for FDIR Diagrams.

Generates tile-based CUDA/C++ code targeting the NVIDIA CUDA Tile IR specification
(https://docs.nvidia.com/cuda/tile-ir/). Abstracted tile operations include:
  - cuda::tile::load
  - cuda::tile::mma (Tensor Core matrix multiply-accumulate)
  - cuda::tile::apply (Pointwise activation mapping)
  - cuda::tile::add (Bypass addition)
  - cuda::tile::store
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
class TileConfig:
    """Hardware block-tiling parameters for CUDA Tile IR compilation."""
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32
    num_warps: int = 4
    num_stages: int = 3


class TileIRLowering:
    """Compiles FDIR Diagram AST down to NVIDIA CUDA Tile IR (cuda::tile) pseudocode."""

    def __init__(self, config: Optional[TileConfig] = None,
                 env: Optional[Dict[str, int]] = None):
        self.config = config or TileConfig()
        self.env = env or {"B": 1, "S": 128, "D": 768, "H": 12, "D_head": 64}

    def lower(self, diagram: Diagram) -> str:
        """Lower complete FDIR Diagram into a CUDA Tile IR kernel module string."""
        sorted_vertices = diagram.topological_sort()
        cfg = self.config

        code_lines = [
            f"// ====================================================================",
            f"// Auto-Generated NVIDIA CUDA Tile IR Code for FDIR: '{diagram.name}'",
            f"// Specification: https://docs.nvidia.com/cuda/tile-ir/",
            f"// Tile Sizing: TILE_M={cfg.tile_m}, TILE_N={cfg.tile_n}, TILE_K={cfg.tile_k}",
            f"// ====================================================================",
            f"#include <cuda/tile>",
            f"#include <cuda_fp16.h>",
            "",
            f"using namespace cuda::tile;",
            "",
            f"__global__ void __launch_bounds__({cfg.num_warps * 32})",
            f"{diagram.name}_cuda_tile_kernel(",
        ]

        # Function signatures: inputs + outputs
        sig_args = []
        for v in sorted_vertices:
            if isinstance(v, InputVertex):
                sig_args.append(f"    const half* __restrict__ ptr_{v.id}")
            elif isinstance(v, OutputVertex):
                sig_args.append(f"    half* __restrict__ ptr_{v.id}")

        code_lines.append(",\n".join(sig_args))
        code_lines.append(") {")
        code_lines.append("    // 1. Thread and Tile Block Indexing")
        code_lines.append("    int tile_idx_m = blockIdx.x;")
        code_lines.append("    int tile_idx_n = blockIdx.y;")
        code_lines.append("    int batch_idx  = blockIdx.z;")
        code_lines.append("")
        code_lines.append("    // 2. Tile Registers & Execution pipeline")

        # Topological lowering of vertices
        for v in sorted_vertices:
            in_props = diagram.get_input_propagators(v.id)
            in_names = [f"tile_{p.src_vertex_id}" for p in in_props]
            out_name = f"tile_{v.id}"

            vertex_code = self._lower_vertex(v, in_names, out_name, diagram)
            if vertex_code:
                code_lines.append(f"    // Vertex: {v.op_type} ({v.id})")
                code_lines.append(vertex_code)

        code_lines.append("}")
        return "\n".join(code_lines)

    def _lower_vertex(self, v: Vertex, in_names: List[str],
                      out_name: str, diagram: Diagram) -> str:
        cfg = self.config

        if isinstance(v, InputVertex):
            return (
                f"    tile<half, {cfg.tile_m}, {cfg.tile_n}> {out_name};\n"
                f"    load({out_name}, ptr_{v.id} + batch_idx * offset, layout::row_major());"
            )

        elif isinstance(v, OutputVertex):
            in_name = in_names[0] if in_names else "tile_prev"
            return f"    store(ptr_{v.id} + batch_idx * offset, {in_name}, layout::row_major());"

        elif isinstance(v, ContractionVertex):
            tb = v.attributes.get("transpose_b", False)
            tb_str = ", layout::col_major()" if tb else ", layout::row_major()"
            return (
                f"    tile<float, {cfg.tile_m}, {cfg.tile_n}> acc_{v.id} = 0.0f;\n"
                f"    #pragma unroll\n"
                f"    for (int k_tile = 0; k_tile < K_TOTAL; k_tile += {cfg.tile_k}) {{\n"
                f"        tile<half, {cfg.tile_m}, {cfg.tile_k}> tile_a = load({in_names[0]});\n"
                f"        tile<half, {cfg.tile_k}, {cfg.tile_n}> tile_b = load({in_names[1]}{tb_str});\n"
                f"        mma(acc_{v.id}, tile_a, tile_b, acc_{v.id}); // Tensor Core HW MMA\n"
                f"    }}\n"
                f"    tile<half, {cfg.tile_m}, {cfg.tile_n}> {out_name} = cast<half>(acc_{v.id});"
            )

        elif isinstance(v, AttentionVertex):
            return (
                f"    // Fused FlashAttention Tile Interaction Loop\n"
                f"    tile<float, {cfg.tile_m}, {cfg.tile_n}> acc_{v.id} = 0.0f;\n"
                f"    tile<float, {cfg.tile_m}, 1> max_score = -inf;\n"
                f"    tile<float, {cfg.tile_m}, 1> sum_exp = 0.0f;\n"
                f"    for (int s2 = 0; s2 < S_LEN; s2 += {cfg.tile_n}) {{\n"
                f"        tile<half, {cfg.tile_m}, {cfg.tile_k}> tile_q = load({in_names[0]});\n"
                f"        tile<half, {cfg.tile_k}, {cfg.tile_n}> tile_k = load({in_names[1]}, layout::col_major());\n"
                f"        tile<float, {cfg.tile_m}, {cfg.tile_n}> S_tile = 0.0f;\n"
                f"        mma(S_tile, tile_q, tile_k, S_tile);\n"
                f"        S_tile = scale(S_tile, 1.0f / sqrtf(64.0f));\n"
                f"        tile<float, {cfg.tile_m}, 1> new_max = reduce_max(S_tile, dim=1);\n"
                f"        tile<float, {cfg.tile_m}, {cfg.tile_n}> P_tile = exp(S_tile - new_max);\n"
                f"        tile<half, {cfg.tile_n}, {cfg.tile_k}> tile_v = load({in_names[2]});\n"
                f"        mma(acc_{v.id}, cast<half>(P_tile), tile_v, acc_{v.id});\n"
                f"    }}\n"
                f"    tile<half, {cfg.tile_m}, {cfg.tile_n}> {out_name} = cast<half>(acc_{v.id});"
            )

        elif isinstance(v, PointwiseVertex):
            sub_op = v.attributes.get("sub_op", "Add").lower()
            if len(in_names) >= 2:
                return f"    tile<half, {cfg.tile_m}, {cfg.tile_n}> {out_name} = apply(op::{sub_op}(), {in_names[0]}, {in_names[1]});"
            else:
                return f"    tile<half, {cfg.tile_m}, {cfg.tile_n}> {out_name} = apply(op::{sub_op}(), {in_names[0]});"

        elif isinstance(v, ResidualVertex):
            return f"    tile<half, {cfg.tile_m}, {cfg.tile_n}> {out_name} = add({in_names[0]}, {in_names[1]});"

        elif isinstance(v, NormVertex):
            norm_type = v.attributes.get("norm_type", "LayerNorm").lower()
            return f"    tile<half, {cfg.tile_m}, {cfg.tile_n}> {out_name} = tile_{norm_type}({in_names[0]});"

        elif isinstance(v, TransposeVertex):
            return f"    tile<half, {cfg.tile_m}, {cfg.tile_n}> {out_name} = transpose({in_names[0]});"

        return ""
