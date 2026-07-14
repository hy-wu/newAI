from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from .diagram import Diagram
from .nodes import (
    Vertex, ContractionVertex, AttentionVertex, PointwiseVertex,
    NormVertex, ResidualVertex, TransposeVertex, InputVertex, OutputVertex
)
from .types import Shape


@dataclass
class PerformanceReport:
    total_flops: int
    hbm_bytes: int
    kernel_launches: int
    peak_memory_bytes: int

    def __str__(self) -> str:
        gflops = self.total_flops / 1e9
        mb_traffic = self.hbm_bytes / (1024 * 1024)
        peak_mb = self.peak_memory_bytes / (1024 * 1024)
        return (
            f"=== Performance & Action Report ===\n"
            f"  FLOPs: {self.total_flops:,} ({gflops:.3f} GFLOPs)\n"
            f"  HBM Traffic: {self.hbm_bytes:,} Bytes ({mb_traffic:.2f} MB)\n"
            f"  Peak Activation Footprint: {peak_mb:.2f} MB\n"
            f"  Kernel Launch Overhead: {self.kernel_launches} ops\n"
        )


class CostModel:
    """Estimates computational action (FLOPs, memory footprint, bandwidth traffic) for FDIR diagrams.

    Peak memory is computed via liveness analysis over topological order:
    a tensor is "live" from the step it is produced until the last step that consumes it.
    """
    def __init__(self, env: Optional[Dict[str, int]] = None):
        self.env = env or {"B": 1, "S": 128, "D": 768, "D_out": 768, "H": 12, "D_head": 64}

    def _resolve_dim(self, d) -> int:
        """Resolve a dimension to int. Str dims are looked up in env."""
        if isinstance(d, int):
            return d
        return self.env.get(str(d), 1)

    def evaluate(self, diagram: Diagram) -> PerformanceReport:
        total_flops = 0
        hbm_bytes = 0
        kernel_launches = 0

        sorted_vertices = diagram.topological_sort()
        vertex_order = {v.id: i for i, v in enumerate(sorted_vertices)}

        # --- Liveness analysis for peak memory (D4) ---
        # For each vertex that produces a tensor, find the last consumer step.
        # The tensor is live from production step to last consumption step.
        tensor_sizes: Dict[str, int] = {}   # vertex_id -> output tensor size in bytes
        produce_step: Dict[str, int] = {}   # vertex_id -> step index where produced
        last_consume_step: Dict[str, int] = {}  # vertex_id -> last step index where consumed

        for i, v in enumerate(sorted_vertices):
            if isinstance(v, (InputVertex, OutputVertex)):
                # Inputs produce tensors at step i
                if isinstance(v, InputVertex):
                    t = v.attributes["tensor_type"]
                    tensor_sizes[v.id] = t.size_in_bytes(self.env)
                    produce_step[v.id] = i
                continue

            # Mark consumption of source vertices
            for p_id in v.inputs:
                if p_id in diagram.propagators:
                    src_id = diagram.propagators[p_id].src_vertex_id
                    last_consume_step[src_id] = i

            # Compute output tensor size
            in_props = diagram.get_input_propagators(v.id)
            out_props = diagram.get_output_propagators(v.id)

            # Estimate output size from first output propagator or first input
            if out_props:
                out_size = out_props[0].tensor_type.size_in_bytes(self.env)
            elif in_props:
                out_size = in_props[0].tensor_type.size_in_bytes(self.env)
            else:
                out_size = 0

            tensor_sizes[v.id] = out_size
            produce_step[v.id] = i

            kernel_launches += 1

            # Bytes read + written (HBM traffic)
            rw_bytes = sum(p.tensor_type.size_in_bytes(self.env) for p in in_props) + \
                       sum(p.tensor_type.size_in_bytes(self.env) for p in out_props)
            hbm_bytes += rw_bytes

            # FLOPs calculations per vertex type
            if isinstance(v, ContractionVertex):
                if len(in_props) >= 2:
                    t1_shape = in_props[0].tensor_type.shape
                    t2_shape = in_props[1].tensor_type.shape
                    # Matmul [M, K] x [K, N] -> 2 * M * N * K
                    m = self._resolve_dim(t1_shape.dims[-2]) if t1_shape.rank >= 2 else 1
                    k = self._resolve_dim(t1_shape.dims[-1])
                    n = self._resolve_dim(t2_shape.dims[-1])
                    batch = 1
                    for d in t1_shape.dims[:-2]:
                        batch *= self._resolve_dim(d)

                    flops = 2 * batch * m * n * k
                    total_flops += flops

            elif isinstance(v, AttentionVertex):
                # Standard SDPA attention FLOPs: 4 * B * H * S^2 * D_head
                b = self.env.get("B", 1)
                s = self.env.get("S", 128)
                h = v.attributes.get("num_heads", 12)
                d_head = v.attributes.get("head_dim", 64)
                flops = 4 * b * h * (s ** 2) * d_head
                total_flops += flops

            elif isinstance(v, (PointwiseVertex, NormVertex, ResidualVertex, TransposeVertex)):
                if in_props:
                    elem_cnt = in_props[0].tensor_type.shape.num_elements(self.env)
                    total_flops += elem_cnt

        # Compute peak memory: at each step, sum sizes of all live tensors
        peak_memory_bytes = 0
        num_steps = len(sorted_vertices)
        for step in range(num_steps):
            live_bytes = 0
            for v_id, prod in produce_step.items():
                last = last_consume_step.get(v_id, prod)  # If never consumed, only live at production
                if prod <= step <= last:
                    live_bytes += tensor_sizes.get(v_id, 0)
            peak_memory_bytes = max(peak_memory_bytes, live_bytes)

        return PerformanceReport(
            total_flops=total_flops,
            hbm_bytes=hbm_bytes,
            kernel_launches=kernel_launches,
            peak_memory_bytes=peak_memory_bytes
        )
