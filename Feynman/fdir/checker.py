from __future__ import annotations
from typing import Dict, List, Optional
from .diagram import Diagram
from .nodes import Vertex, ResidualVertex, ContractionVertex, AttentionVertex
from .types import TensorType, Shape


class ShapeMismatchError(Exception):
    """Raised when dimension or index conservation laws are violated in FDIR."""
    pass


class ShapeTypeChecker:
    """Validates dimension conservation laws across the entire FDIR diagram.

    Performs forward type inference through topological order, verifying:
    - Contraction axis matching (index conservation)
    - Residual bypass shape equality
    - Propagator declared types vs vertex-inferred types
    """
    def __init__(self, env: Optional[Dict[str, int]] = None):
        self.env = env or {}

    def check(self, diagram: Diagram) -> bool:
        sorted_vertices = diagram.topological_sort()
        # Maps vertex_id -> inferred output TensorType
        inferred_types: Dict[str, TensorType] = {}

        # Phase 1: Seed InputVertex output types
        for v in sorted_vertices:
            if v.op_type == "Input":
                inferred_types[v.id] = v.get_output_type([])

        # Phase 2: Forward inference and validation
        for v in sorted_vertices:
            if v.op_type == "Input":
                continue

            # Gather inferred types from source vertices of each input propagator
            input_types: List[TensorType] = []
            for p_id in v.inputs:
                if p_id not in diagram.propagators:
                    raise ShapeMismatchError(f"Vertex {v.id}: input propagator '{p_id}' not found in diagram")
                prop = diagram.propagators[p_id]
                src_id = prop.src_vertex_id
                if src_id not in inferred_types:
                    raise ShapeMismatchError(
                        f"Vertex {v.id}: source vertex '{src_id}' has no inferred output type "
                        f"(possible topological ordering issue)"
                    )
                input_types.append(inferred_types[src_id])

            # Specific conservation law checks
            if isinstance(v, ResidualVertex):
                if len(input_types) < 2:
                    raise ShapeMismatchError(
                        f"ResidualVertex {v.id} expects 2 inputs (bypass + main), got {len(input_types)}"
                    )
                t_bypass, t_main = input_types[0], input_types[1]
                if not t_bypass.shape.is_compatible(t_main.shape, self.env):
                    raise ShapeMismatchError(
                        f"Residual bypass shape conservation law violated at {v.id}: "
                        f"bypass shape {t_bypass.shape} != main branch shape {t_main.shape}"
                    )

            elif isinstance(v, ContractionVertex):
                if len(input_types) >= 2:
                    t1, t2 = input_types[0], input_types[1]
                    transpose_b = v.attributes.get("transpose_b", False)
                    k1 = t1.shape.dims[-1]
                    k2_idx = -1 if transpose_b else -2
                    k2 = t2.shape.dims[k2_idx] if abs(k2_idx) <= t2.shape.rank else None

                    if k1 is not None and k2 is not None:
                        if isinstance(k1, int) and isinstance(k2, int) and k1 != k2:
                            raise ShapeMismatchError(
                                f"Contraction index conservation law violated at {v.id}: "
                                f"contracted dimension {k1} != {k2}"
                            )
                        if isinstance(k1, str) and isinstance(k2, str) and k1 != k2:
                            # Check env resolution
                            r1 = self.env.get(k1)
                            r2 = self.env.get(k2)
                            if r1 is None or r2 is None or r1 != r2:
                                raise ShapeMismatchError(
                                    f"Contraction index conservation law violated at {v.id}: "
                                    f"contracted symbol '{k1}' != '{k2}'"
                                )

            # Infer output type via the vertex's own logic
            try:
                out_type = v.get_output_type(input_types)
            except (ValueError, NotImplementedError) as e:
                raise ShapeMismatchError(f"Type inference failed at vertex {v.id}: {e}") from e

            inferred_types[v.id] = out_type

            # Validate declared propagator types against inferred types (D1)
            for p_id in v.outputs:
                if p_id in diagram.propagators:
                    declared = diagram.propagators[p_id].tensor_type
                    if not out_type.shape.is_compatible(declared.shape, self.env):
                        raise ShapeMismatchError(
                            f"Propagator {p_id} from vertex {v.id}: "
                            f"declared shape {declared.shape} is incompatible with "
                            f"inferred output shape {out_type.shape}"
                        )

        return True
