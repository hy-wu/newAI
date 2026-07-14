"""Dual Performance Evaluation Engine for FDIR Diagrams.

Provides two complementary evaluation perspectives:
- ModelPerformanceEvaluator: Theoretical model capacity, parameter footprint,
  expressiveness metrics (algorithm/architecture level).
- InfraPerformanceEvaluator: Hardware efficiency metrics — FLOPs utilization,
  HBM bandwidth efficiency, arithmetic intensity, SRAM footprint (infra level).

Together they feed the Design Agent closed-loop with dual feedback signals.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .diagram import Diagram
from .nodes import (
    Vertex, InputVertex, OutputVertex, ContractionVertex, AttentionVertex,
    PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex
)
from .cost import CostModel, PerformanceReport


# ============================================================
# Model Performance Evaluation (Architecture / Algorithm Level)
# ============================================================

@dataclass
class ModelPerformanceReport:
    """Architectural-level model capacity and complexity metrics."""
    total_parameters: int
    total_activations: int
    num_contraction_vertices: int
    num_attention_vertices: int
    num_pointwise_vertices: int
    num_residual_connections: int
    num_norm_layers: int
    depth: int                      # Longest path in the DAG
    theoretical_receptive_field: str  # "global" if Attention present, else "local"
    parameter_efficiency: float      # activations / parameters ratio
    sequence_scaling: str            # O(S^2) if Attention, O(S) otherwise

    def __str__(self) -> str:
        return (
            f"=== Model Performance Report ===\n"
            f"  Parameters:              {self.total_parameters:,}\n"
            f"  Activations:             {self.total_activations:,}\n"
            f"  Contraction Vertices:    {self.num_contraction_vertices}\n"
            f"  Attention Vertices:      {self.num_attention_vertices}\n"
            f"  Pointwise Vertices:      {self.num_pointwise_vertices}\n"
            f"  Residual Connections:     {self.num_residual_connections}\n"
            f"  Normalization Layers:     {self.num_norm_layers}\n"
            f"  DAG Depth:               {self.depth}\n"
            f"  Receptive Field:         {self.theoretical_receptive_field}\n"
            f"  Parameter Efficiency:    {self.parameter_efficiency:.4f}\n"
            f"  Sequence Scaling:        {self.sequence_scaling}\n"
        )


class ModelPerformanceEvaluator:
    """Evaluates architecture-level model capacity and theoretical properties.

    Answers: 'How expressive/efficient is this model architecture?'
    """

    def __init__(self, env: Optional[Dict[str, int]] = None):
        self.env = env or {}

    def evaluate(self, diagram: Diagram) -> ModelPerformanceReport:
        sorted_vertices = diagram.topological_sort()

        # Count vertex types
        n_contraction = sum(1 for v in sorted_vertices if isinstance(v, ContractionVertex))
        n_attention = sum(1 for v in sorted_vertices if isinstance(v, AttentionVertex))
        n_pointwise = sum(1 for v in sorted_vertices if isinstance(v, PointwiseVertex))
        n_residual = sum(1 for v in sorted_vertices if isinstance(v, ResidualVertex))
        n_norm = sum(1 for v in sorted_vertices if isinstance(v, NormVertex))

        # Parameter count: sum of weight tensor sizes (InputVertex tensors that
        # feed into ContractionVertex or AttentionVertex are likely weight matrices)
        total_params = 0
        total_activations = 0
        for v in sorted_vertices:
            if isinstance(v, InputVertex):
                t = v.attributes["tensor_type"]
                size = t.shape.num_elements(self.env)
                # Heuristic: if shape rank <= 2, it's a weight matrix (parameters)
                # If rank >= 3, it's an activation (e.g. (B, S, D))
                if t.shape.rank <= 2:
                    total_params += size
                else:
                    total_activations += size

        # DAG depth (longest path)
        depth = self._compute_dag_depth(diagram, sorted_vertices)

        # Receptive field
        has_attention = n_attention > 0
        receptive_field = "global (full sequence)" if has_attention else "local (per-element)"

        # Sequence scaling
        sequence_scaling = "O(S^2 * D)" if has_attention else "O(S * D)"

        # Parameter efficiency
        param_efficiency = (total_activations / total_params) if total_params > 0 else 0.0

        return ModelPerformanceReport(
            total_parameters=total_params,
            total_activations=total_activations,
            num_contraction_vertices=n_contraction,
            num_attention_vertices=n_attention,
            num_pointwise_vertices=n_pointwise,
            num_residual_connections=n_residual,
            num_norm_layers=n_norm,
            depth=depth,
            theoretical_receptive_field=receptive_field,
            parameter_efficiency=param_efficiency,
            sequence_scaling=sequence_scaling,
        )

    def _compute_dag_depth(self, diagram: Diagram, sorted_vertices: List[Vertex]) -> int:
        """Compute longest path in the DAG via dynamic programming."""
        depth: Dict[str, int] = {}
        for v in sorted_vertices:
            if not v.inputs:
                depth[v.id] = 0
            else:
                max_parent = 0
                for p_id in v.inputs:
                    if p_id in diagram.propagators:
                        src_id = diagram.propagators[p_id].src_vertex_id
                        max_parent = max(max_parent, depth.get(src_id, 0))
                depth[v.id] = max_parent + 1
        return max(depth.values()) if depth else 0


# ============================================================
# Infra Performance Evaluation (Hardware / Compute Level)
# ============================================================

@dataclass
class HardwareSpec:
    """Target GPU hardware specification for efficiency calculations."""
    name: str = "NVIDIA H100 SXM"
    peak_tflops_fp16: float = 989.4     # TF16 Tensor Core peak
    peak_tflops_fp32: float = 67.0      # FP32 CUDA Core peak
    hbm_bandwidth_tb_s: float = 3.35    # HBM3 bandwidth TB/s
    sram_per_sm_kb: float = 228.0       # Shared memory per SM
    num_sms: int = 132
    total_sram_kb: float = 228.0 * 132  # Total SRAM

    @property
    def peak_flops_per_sec(self) -> float:
        """Peak FP16 FLOPS."""
        return self.peak_tflops_fp16 * 1e12

    @property
    def hbm_bandwidth_bytes_per_sec(self) -> float:
        return self.hbm_bandwidth_tb_s * 1e12


@dataclass
class InfraPerformanceReport:
    """Hardware-level efficiency and utilization metrics."""
    total_flops: int
    hbm_bytes: int
    peak_memory_bytes: int
    kernel_launches: int
    arithmetic_intensity: float       # FLOPs / Byte
    estimated_time_compute_ms: float  # Time if compute-bound
    estimated_time_memory_ms: float   # Time if memory-bound
    estimated_time_ms: float          # max(compute, memory) — the bottleneck
    compute_utilization_pct: float    # % of peak FLOPs achieved
    memory_utilization_pct: float     # % of peak HBM BW achieved
    bottleneck: str                   # "compute-bound" or "memory-bound"
    tile_config_summary: str

    def __str__(self) -> str:
        return (
            f"=== Infra Performance Report ===\n"
            f"  Total FLOPs:             {self.total_flops:,} ({self.total_flops / 1e12:.4f} TFLOPs)\n"
            f"  HBM Traffic:             {self.hbm_bytes:,} Bytes ({self.hbm_bytes / 1e9:.3f} GB)\n"
            f"  Peak Activation Memory:  {self.peak_memory_bytes / 1e6:.2f} MB\n"
            f"  Kernel Launches:         {self.kernel_launches}\n"
            f"  Arithmetic Intensity:    {self.arithmetic_intensity:.2f} FLOPs/Byte\n"
            f"  Est. Time (Compute):     {self.estimated_time_compute_ms:.4f} ms\n"
            f"  Est. Time (Memory):      {self.estimated_time_memory_ms:.4f} ms\n"
            f"  Est. Total Time:         {self.estimated_time_ms:.4f} ms\n"
            f"  Compute Utilization:     {self.compute_utilization_pct:.2f}%\n"
            f"  Memory Utilization:      {self.memory_utilization_pct:.2f}%\n"
            f"  Bottleneck:              {self.bottleneck}\n"
            f"  Tile Config:             {self.tile_config_summary}\n"
        )


class InfraPerformanceEvaluator:
    """Evaluates hardware-level efficiency for FDIR diagrams on target GPUs.

    Answers: 'How efficiently does this computation use the hardware?'
    Computes arithmetic intensity, roofline-model utilization, and bottleneck analysis.
    """

    def __init__(self, env: Optional[Dict[str, int]] = None,
                 hardware: Optional[HardwareSpec] = None,
                 tile_m: int = 128, tile_n: int = 128, tile_k: int = 32):
        self.env = env or {}
        self.hardware = hardware or HardwareSpec()
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k

    def evaluate(self, diagram: Diagram) -> InfraPerformanceReport:
        # Use the existing CostModel for base metrics
        cost_model = CostModel(env=self.env)
        base_report = cost_model.evaluate(diagram)

        total_flops = base_report.total_flops
        hbm_bytes = base_report.hbm_bytes
        peak_mem = base_report.peak_memory_bytes
        kernel_launches = base_report.kernel_launches

        # Arithmetic intensity (AI)
        ai = total_flops / hbm_bytes if hbm_bytes > 0 else float('inf')

        # Roofline model timing estimates
        hw = self.hardware
        t_compute = (total_flops / hw.peak_flops_per_sec) * 1000  # ms
        t_memory = (hbm_bytes / hw.hbm_bandwidth_bytes_per_sec) * 1000  # ms
        t_total = max(t_compute, t_memory)

        # Utilization percentages (assuming ideal overlap)
        # Effective utilization = actual_time_doing_work / total_time
        compute_util = (t_compute / t_total * 100) if t_total > 0 else 0.0
        memory_util = (t_memory / t_total * 100) if t_total > 0 else 0.0

        bottleneck = "compute-bound" if t_compute >= t_memory else "memory-bound"

        tile_summary = f"TILE_M={self.tile_m}, TILE_N={self.tile_n}, TILE_K={self.tile_k}"

        return InfraPerformanceReport(
            total_flops=total_flops,
            hbm_bytes=hbm_bytes,
            peak_memory_bytes=peak_mem,
            kernel_launches=kernel_launches,
            arithmetic_intensity=ai,
            estimated_time_compute_ms=t_compute,
            estimated_time_memory_ms=t_memory,
            estimated_time_ms=t_total,
            compute_utilization_pct=compute_util,
            memory_utilization_pct=memory_util,
            bottleneck=bottleneck,
            tile_config_summary=tile_summary,
        )


# ============================================================
# Unified Dual Evaluator
# ============================================================

@dataclass
class DualPerformanceReport:
    """Combined model + infra performance report for Design Agent consumption."""
    model: ModelPerformanceReport
    infra: InfraPerformanceReport

    def __str__(self) -> str:
        return str(self.model) + "\n" + str(self.infra)

    def to_dict(self) -> Dict:
        """Serialize to dictionary for agent consumption."""
        return {
            "model": {
                "total_parameters": self.model.total_parameters,
                "total_activations": self.model.total_activations,
                "depth": self.model.depth,
                "num_contraction_vertices": self.model.num_contraction_vertices,
                "num_attention_vertices": self.model.num_attention_vertices,
                "num_residual_connections": self.model.num_residual_connections,
                "receptive_field": self.model.theoretical_receptive_field,
                "sequence_scaling": self.model.sequence_scaling,
                "parameter_efficiency": self.model.parameter_efficiency,
            },
            "infra": {
                "total_flops": self.infra.total_flops,
                "hbm_bytes": self.infra.hbm_bytes,
                "peak_memory_bytes": self.infra.peak_memory_bytes,
                "arithmetic_intensity": self.infra.arithmetic_intensity,
                "estimated_time_ms": self.infra.estimated_time_ms,
                "compute_utilization_pct": self.infra.compute_utilization_pct,
                "memory_utilization_pct": self.infra.memory_utilization_pct,
                "bottleneck": self.infra.bottleneck,
            },
        }


class DualEvaluator:
    """Unified evaluator producing both Model and Infra performance reports."""

    def __init__(self, env: Optional[Dict[str, int]] = None,
                 hardware: Optional[HardwareSpec] = None):
        self.env = env or {}
        self.model_eval = ModelPerformanceEvaluator(env=self.env)
        self.infra_eval = InfraPerformanceEvaluator(env=self.env, hardware=hardware)

    def evaluate(self, diagram: Diagram) -> DualPerformanceReport:
        model_report = self.model_eval.evaluate(diagram)
        infra_report = self.infra_eval.evaluate(diagram)
        return DualPerformanceReport(model=model_report, infra=infra_report)
