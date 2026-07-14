"""Autonomous Design Agent Environment for FDIR Architecture Search.

Provides a standardized environment for LLM/RL-based architecture search agents
to interact with the FDIR ecosystem. The agent observes dual performance feedback
(Model + Infra), proposes structured mutations on the FDIR diagram, and receives
updated metrics — forming a closed-loop optimization cycle.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import copy
import torch

from Feynman.fdir.diagram import Diagram
from Feynman.fdir.nodes import (
    Vertex, InputVertex, OutputVertex, ContractionVertex, AttentionVertex,
    PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex
)
from Feynman.fdir.types import TensorType, Shape, DType
from Feynman.fdir.evaluation import (
    DualEvaluator, DualPerformanceReport,
    ModelPerformanceEvaluator, InfraPerformanceEvaluator, HardwareSpec
)
from Feynman.fdir.rewriter import RewriteEngine
from Feynman.fdir.lowering import TorchLowering
from .profiler_runner import GPUProfiler


class MutationType(Enum):
    ADD_CONTRACTION = auto()
    ADD_ATTENTION = auto()
    ADD_RESIDUAL_BYPASS = auto()
    ADD_NORM_LAYER = auto()
    ADD_POINTWISE = auto()
    REMOVE_VERTEX = auto()
    FUSE_ATTENTION = auto()
    MODIFY_TILE_CONFIG = auto()
    SWAP_NORM_TYPE = auto()


@dataclass
class MutationAction:
    """A structured mutation proposal from the Design Agent."""
    mutation_type: MutationType
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Mutation({self.mutation_type.name}, params={self.params})"


@dataclass
class Observation:
    """Complete state observation for the Design Agent."""
    diagram_summary: Dict[str, Any]
    performance: DualPerformanceReport
    mutation_history: List[str]
    step: int
    physical_gpu_metrics: Optional[Dict[str, Any]] = None

    def to_prompt_context(self) -> str:
        """Serialize observation into a formatted prompt context for LLM agents."""
        lines = [
            f"=== Design Agent Observation (Step {self.step}) ===",
            f"Diagram Name: {self.diagram_summary['name']}",
            f"  Vertices Count:   {self.diagram_summary['num_vertices']}",
            f"  Propagators Count:{self.diagram_summary['num_propagators']}",
            f"  Vertex Types:     {self.diagram_summary['vertex_type_counts']}",
            "",
            str(self.performance),
        ]

        if self.physical_gpu_metrics:
            m = self.physical_gpu_metrics
            lines.append("=== Physical GPU Profile (Nsight/CUDA Profiler Telemetry) ===")
            if "error" in m:
                lines.append(f"  Profiling Error: {m['error']}")
            else:
                lines.append(f"  Device Name:            {m.get('cuda_device')}")
                lines.append(f"  Real Kernel Latency:    {m.get('latency_ms', 0.0):.4f} ms")
                lines.append(f"  Real Peak Memory usage: {m.get('peak_memory_mb', 0.0):.2f} MB")

        lines.extend([
            "",
            f"Mutation History ({len(self.mutation_history)} steps taken):",
        ])
        for i, m in enumerate(self.mutation_history[-5:], 1):
            lines.append(f"  {i}. {m}")
        return "\n".join(lines)


class DesignAgentInterface:
    """Closed-loop environment for autonomous architecture design agents.

    Exposes observe(), mutate(action), reset(), and get_available_mutations().
    """

    def __init__(self, diagram: Diagram,
                 env: Optional[Dict[str, int]] = None,
                 hardware: Optional[HardwareSpec] = None,
                 profiling_inputs: Optional[List[torch.Tensor]] = None):
        self.original_diagram = copy.deepcopy(diagram)
        self.diagram = copy.deepcopy(diagram)
        self.env = env or {}
        self.evaluator = DualEvaluator(env=self.env, hardware=hardware)
        self.rewriter = RewriteEngine()
        self.mutation_history: List[str] = []
        self.step = 0
        self.profiling_inputs = profiling_inputs

    def reset(self, diagram: Optional[Diagram] = None) -> Observation:
        if diagram is not None:
            self.original_diagram = copy.deepcopy(diagram)
        self.diagram = copy.deepcopy(self.original_diagram)
        self.mutation_history = []
        self.step = 0
        return self.observe()

    def observe(self) -> Observation:
        perf = self.evaluator.evaluate(self.diagram)
        summary = self._diagram_summary()

        # Gather physical hardware GPU profile if inputs are provided
        gpu_metrics = None
        if self.profiling_inputs is not None and GPUProfiler.is_gpu_available():
            try:
                module = TorchLowering().lower(self.diagram)
                gpu_metrics = GPUProfiler.profile_module(module, self.profiling_inputs)
            except Exception as e:
                gpu_metrics = {"error": f"Failed to run physical profile: {e}"}

        return Observation(
            diagram_summary=summary,
            performance=perf,
            mutation_history=list(self.mutation_history),
            step=self.step,
            physical_gpu_metrics=gpu_metrics,
        )

    def mutate(self, action: MutationAction) -> Observation:
        self.step += 1
        try:
            self._apply_mutation(action)
            self.mutation_history.append(f"[OK] {action}")
        except Exception as e:
            self.mutation_history.append(f"[FAIL] {action}: {e}")

        return self.observe()

    def get_available_mutations(self) -> List[MutationAction]:
        actions: List[MutationAction] = []
        removable = [v_id for v_id, v in self.diagram.vertices.items()
                     if not isinstance(v, (InputVertex, OutputVertex))]

        for v_id in removable[:3]:
            actions.append(MutationAction(
                MutationType.REMOVE_VERTEX, {"vertex_id": v_id}
            ))

        if self.diagram.inputs and self.diagram.outputs:
            actions.append(MutationAction(
                MutationType.ADD_RESIDUAL_BYPASS,
                {"from_vertex": self.diagram.inputs[0],
                 "to_before_vertex": self.diagram.outputs[0]}
            ))

        actions.append(MutationAction(MutationType.FUSE_ATTENTION))
        actions.append(MutationAction(MutationType.SWAP_NORM_TYPE))
        actions.append(MutationAction(
            MutationType.MODIFY_TILE_CONFIG,
            {"tile_m": 256, "tile_n": 128, "tile_k": 64}
        ))
        return actions

    def get_diagram(self) -> Diagram:
        return self.diagram

    def _apply_mutation(self, action: MutationAction) -> None:
        mt = action.mutation_type
        p = action.params

        if mt == MutationType.ADD_CONTRACTION:
            self._add_contraction(p)
        elif mt == MutationType.ADD_ATTENTION:
            self._add_attention(p)
        elif mt == MutationType.ADD_RESIDUAL_BYPASS:
            self._add_residual_bypass(p)
        elif mt == MutationType.ADD_NORM_LAYER:
            self._add_norm_layer(p)
        elif mt == MutationType.ADD_POINTWISE:
            self._add_pointwise(p)
        elif mt == MutationType.REMOVE_VERTEX:
            self._remove_vertex(p)
        elif mt == MutationType.FUSE_ATTENTION:
            self._fuse_attention()
        elif mt == MutationType.MODIFY_TILE_CONFIG:
            self._modify_tile_config(p)
        elif mt == MutationType.SWAP_NORM_TYPE:
            self._swap_norm_type(p)
        else:
            raise ValueError(f"Unknown mutation type: {mt}")

    def _add_contraction(self, params: Dict) -> None:
        vertex_id = params.get("id", f"contraction_{self.step}")
        after_vertex = params.get("after_vertex")
        if after_vertex and after_vertex in self.diagram.vertices:
            v = ContractionVertex(vertex_id)
            self.diagram.add_vertex(v)
            out_props = self.diagram.get_output_propagators(after_vertex)
            if out_props:
                t = out_props[0].tensor_type
                self.diagram.connect(after_vertex, vertex_id, t)

    def _add_attention(self, params: Dict) -> None:
        vertex_id = params.get("id", f"attention_{self.step}")
        num_heads = params.get("num_heads", 12)
        head_dim = params.get("head_dim", 64)
        v = AttentionVertex(vertex_id, num_heads=num_heads, head_dim=head_dim)
        self.diagram.add_vertex(v)

    def _add_residual_bypass(self, params: Dict) -> None:
        from_v = params.get("from_vertex")
        to_before = params.get("to_before_vertex")
        if not (from_v and to_before):
            raise ValueError("add_residual_bypass requires 'from_vertex' and 'to_before_vertex'")

        res_id = f"residual_{self.step}"
        res_v = ResidualVertex(res_id)
        self.diagram.add_vertex(res_v)

        out_props = self.diagram.get_output_propagators(from_v)
        if out_props:
            t = out_props[0].tensor_type
        else:
            if isinstance(self.diagram.vertices.get(from_v), InputVertex):
                t = self.diagram.vertices[from_v].attributes["tensor_type"]
            else:
                raise ValueError(f"Cannot determine tensor type from vertex '{from_v}'")

        self.diagram.connect(from_v, res_id, t, label="bypass")

    def _add_norm_layer(self, params: Dict) -> None:
        vertex_id = params.get("id", f"norm_{self.step}")
        norm_type = params.get("norm_type", "LayerNorm")
        after_vertex = params.get("after_vertex")
        v = NormVertex(vertex_id, norm_type=norm_type)
        self.diagram.add_vertex(v)
        if after_vertex and after_vertex in self.diagram.vertices:
            out_props = self.diagram.get_output_propagators(after_vertex)
            if out_props:
                self.diagram.connect(after_vertex, vertex_id, out_props[0].tensor_type)

    def _add_pointwise(self, params: Dict) -> None:
        vertex_id = params.get("id", f"pointwise_{self.step}")
        sub_op = params.get("sub_op", "GELU")
        v = PointwiseVertex(vertex_id, sub_op=sub_op)
        self.diagram.add_vertex(v)

    def _remove_vertex(self, params: Dict) -> None:
        vertex_id = params.get("vertex_id")
        if not vertex_id:
            raise ValueError("remove_vertex requires 'vertex_id'")
        if isinstance(self.diagram.vertices.get(vertex_id), (InputVertex, OutputVertex)):
            raise ValueError(f"Cannot remove Input/Output vertex '{vertex_id}'")

        in_props = self.diagram.get_input_propagators(vertex_id)
        out_props = self.diagram.get_output_propagators(vertex_id)

        predecessors = [(p.src_vertex_id, p.tensor_type) for p in in_props]
        successors = [(p.dst_vertex_id, p.tensor_type, p.label) for p in out_props]

        self.diagram.remove_vertex(vertex_id)

        if predecessors and successors:
            src_id, src_type = predecessors[0]
            for dst_id, dst_type, label in successors:
                if src_id in self.diagram.vertices and dst_id in self.diagram.vertices:
                    self.diagram.connect(src_id, dst_id, src_type, label=label)

    def _fuse_attention(self) -> None:
        self.diagram = self.rewriter.apply_all_passes(self.diagram)

    def _modify_tile_config(self, params: Dict) -> None:
        if "tile_m" in params:
            self.evaluator.infra_eval.tile_m = params["tile_m"]
        if "tile_n" in params:
            self.evaluator.infra_eval.tile_n = params["tile_n"]
        if "tile_k" in params:
            self.evaluator.infra_eval.tile_k = params["tile_k"]

    def _swap_norm_type(self, params: Dict = None) -> None:
        for v in self.diagram.vertices.values():
            if isinstance(v, NormVertex):
                current = v.attributes.get("norm_type", "LayerNorm")
                v.attributes["norm_type"] = "RMSNorm" if current == "LayerNorm" else "LayerNorm"

    def _diagram_summary(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        for v in self.diagram.vertices.values():
            t = v.__class__.__name__
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "name": self.diagram.name,
            "num_vertices": len(self.diagram.vertices),
            "num_propagators": len(self.diagram.propagators),
            "vertex_type_counts": type_counts,
            "input_ids": list(self.diagram.inputs),
            "output_ids": list(self.diagram.outputs),
        }
