from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diagram import Diagram
from .nodes import (
    Vertex, InputVertex, OutputVertex, ContractionVertex,
    AttentionVertex, PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex
)


class LoweredFDIRModule(nn.Module):
    """PyTorch module compiled directly from FDIR Diagram specification.

    Executes vertices in topological order, gathering inputs from
    source vertices via propagator edges.
    """
    def __init__(self, diagram: Diagram):
        super().__init__()
        self.diagram = diagram
        self.sorted_vertices = diagram.topological_sort()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        node_values: Dict[str, torch.Tensor] = {}

        # Bind Input Vertices to positional arguments
        input_idx = 0
        for v in self.sorted_vertices:
            if isinstance(v, InputVertex):
                if input_idx >= len(inputs):
                    raise RuntimeError(
                        f"Diagram expects at least {input_idx + 1} inputs, got {len(inputs)}"
                    )
                node_values[v.id] = inputs[input_idx]
                input_idx += 1

        # Execute vertices in topological order
        for v in self.sorted_vertices:
            if isinstance(v, InputVertex):
                continue
            if isinstance(v, OutputVertex):
                continue

            # Gather input tensors via propagators
            in_tensors: List[torch.Tensor] = []
            for p_id in v.inputs:
                prop = self.diagram.propagators[p_id]
                src_v_id = prop.src_vertex_id
                if src_v_id not in node_values:
                    raise RuntimeError(
                        f"Vertex {v.id}: source '{src_v_id}' has no computed value"
                    )
                in_tensors.append(node_values[src_v_id])

            res = self._execute_vertex(v, in_tensors)
            node_values[v.id] = res

        # Collect outputs: for each OutputVertex, return its source's value
        output_tensors: List[torch.Tensor] = []
        for out_id in self.diagram.outputs:
            out_v = self.diagram.vertices[out_id]
            if not out_v.inputs:
                raise RuntimeError(f"OutputVertex {out_id} has no input propagators")
            # Get the source vertex of the first input propagator
            prop = self.diagram.propagators[out_v.inputs[0]]
            src_val = node_values.get(prop.src_vertex_id)
            if src_val is None:
                raise RuntimeError(
                    f"OutputVertex {out_id}: source '{prop.src_vertex_id}' has no computed value"
                )
            output_tensors.append(src_val)

        # If no explicit outputs, return the last computed value
        if not output_tensors:
            last_v = self.sorted_vertices[-1]
            if last_v.id in node_values:
                return node_values[last_v.id]
            raise RuntimeError("No output vertices and no computed values found")

        return output_tensors[0] if len(output_tensors) == 1 else tuple(output_tensors)

    def _execute_vertex(self, v: Vertex, in_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Dispatch execution to the appropriate PyTorch op for a vertex type."""

        if isinstance(v, ContractionVertex):
            transpose_b = v.attributes.get("transpose_b", False)
            a, b = in_tensors[0], in_tensors[1]
            if transpose_b:
                b = b.transpose(-1, -2)
            return torch.matmul(a, b)

        elif isinstance(v, AttentionVertex):
            q, k, v_mat = in_tensors[0], in_tensors[1], in_tensors[2]
            return F.scaled_dot_product_attention(q, k, v_mat)

        elif isinstance(v, PointwiseVertex):
            sub_op = v.attributes.get("sub_op", "Add")
            if sub_op == "ReLU":
                return F.relu(in_tensors[0])
            elif sub_op == "GELU":
                return F.gelu(in_tensors[0])
            elif sub_op == "Softmax":
                return F.softmax(in_tensors[0], dim=-1)
            elif sub_op == "Sigmoid":
                return torch.sigmoid(in_tensors[0])
            elif sub_op == "Tanh":
                return torch.tanh(in_tensors[0])
            elif sub_op == "Add":
                return in_tensors[0] + in_tensors[1]
            elif sub_op == "Mul":
                return in_tensors[0] * in_tensors[1]
            elif sub_op == "Sub":
                return in_tensors[0] - in_tensors[1]
            else:
                return in_tensors[0]

        elif isinstance(v, ResidualVertex):
            return in_tensors[0] + in_tensors[1]

        elif isinstance(v, NormVertex):
            norm_type = v.attributes.get("norm_type", "LayerNorm")
            if norm_type == "RMSNorm":
                return self._rms_norm(in_tensors[0], v.attributes.get("eps", 1e-5))
            else:
                normalized_shape = in_tensors[0].shape[-1:]
                return F.layer_norm(in_tensors[0], normalized_shape)

        elif isinstance(v, TransposeVertex):
            perm = v.attributes["perm"]
            return in_tensors[0].permute(*perm)

        else:
            return in_tensors[0]

    @staticmethod
    def _rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """RMSNorm: x / sqrt(mean(x^2) + eps)"""
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return x / rms


class TorchLowering:
    """Compiles FDIR Diagram into executable PyTorch Module."""
    def __init__(self):
        pass

    def lower(self, diagram: Diagram) -> nn.Module:
        """Lowers FDIR Diagram to PyTorch nn.Module."""
        return LoweredFDIRModule(diagram)
