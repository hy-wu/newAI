"""FDIR AST ↔ Python DSL Code Round-Trip Serialization.

Serializes an FDIR Diagram AST into executable Python DSL code,
and executes DSL code in a sandboxed namespace to reconstruct the Diagram AST.
"""

from __future__ import annotations
import inspect
from typing import Dict, Any
from .diagram import Diagram
from .types import TensorType, Shape, DType, Layout
from .nodes import (
    Vertex, InputVertex, OutputVertex, ContractionVertex, AttentionVertex,
    PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex
)


class FDIRCodeGen:
    """Round-trip serializer between FDIR Diagram AST and Python DSL code."""

    @staticmethod
    def ast_to_code(diagram: Diagram) -> str:
        """Serialize an FDIR Diagram AST into executable Python DSL code string."""
        lines = [
            "# Auto-generated Python DSL code from FDIR Diagram",
            "from Feynman.fdir import (",
            "    Diagram, TensorType, Shape, DType, Layout,",
            "    InputVertex, OutputVertex, ContractionVertex, AttentionVertex,",
            "    PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex",
            ")",
            "",
            "def build_diagram() -> Diagram:",
            f'    d = Diagram("{diagram.name}")',
            ""
        ]

        sorted_vertices = diagram.topological_sort()

        # Emit Vertex instantiations
        for v in sorted_vertices:
            if isinstance(v, InputVertex):
                t = v.attributes["tensor_type"]
                shape_str = f"Shape({t.shape.dims})"
                dtype_str = f"DType.{t.dtype.name}"
                lines.append(f'    t_{v.id} = TensorType(shape={shape_str}, dtype={dtype_str})')
                lines.append(f'    d.add_vertex(InputVertex("{v.id}", t_{v.id}))')

            elif isinstance(v, OutputVertex):
                lines.append(f'    d.add_vertex(OutputVertex("{v.id}"))')

            elif isinstance(v, ContractionVertex):
                tb = v.attributes.get("transpose_b", False)
                lines.append(f'    d.add_vertex(ContractionVertex("{v.id}", transpose_b={tb}))')

            elif isinstance(v, AttentionVertex):
                h = v.attributes.get("num_heads", 12)
                hd = v.attributes.get("head_dim", 64)
                lines.append(f'    d.add_vertex(AttentionVertex("{v.id}", num_heads={h}, head_dim={hd}))')

            elif isinstance(v, PointwiseVertex):
                sub_op = v.attributes.get("sub_op", "Add")
                lines.append(f'    d.add_vertex(PointwiseVertex("{v.id}", sub_op="{sub_op}"))')

            elif isinstance(v, ResidualVertex):
                lines.append(f'    d.add_vertex(ResidualVertex("{v.id}"))')

            elif isinstance(v, NormVertex):
                nt = v.attributes.get("norm_type", "LayerNorm")
                eps = v.attributes.get("eps", 1e-5)
                lines.append(f'    d.add_vertex(NormVertex("{v.id}", norm_type="{nt}", eps={eps}))')

            elif isinstance(v, TransposeVertex):
                perm = v.attributes.get("perm", ())
                lines.append(f'    d.add_vertex(TransposeVertex("{v.id}", perm={perm}))')

        lines.append("")
        lines.append("    # Propagators")

        # Emit Propagator connects
        for prop in diagram.propagators.values():
            t = prop.tensor_type
            shape_str = f"Shape({t.shape.dims})"
            dtype_str = f"DType.{t.dtype.name}"
            lbl_str = f', label="{prop.label}"' if prop.label else ""
            lines.append(
                f'    d.connect("{prop.src_vertex_id}", "{prop.dst_vertex_id}", '
                f'TensorType(shape={shape_str}, dtype={dtype_str}){lbl_str})'
            )

        lines.append("")
        lines.append("    return d")
        lines.append("")
        lines.append("diagram = build_diagram()")

        return "\n".join(lines)

    @staticmethod
    def code_to_ast(code_str: str) -> Diagram:
        """Execute Python DSL code string in a controlled namespace to produce an FDIR Diagram AST."""
        namespace: Dict[str, Any] = {
            "Diagram": Diagram,
            "TensorType": TensorType,
            "Shape": Shape,
            "DType": DType,
            "Layout": Layout,
            "InputVertex": InputVertex,
            "OutputVertex": OutputVertex,
            "ContractionVertex": ContractionVertex,
            "AttentionVertex": AttentionVertex,
            "PointwiseVertex": PointwiseVertex,
            "ResidualVertex": ResidualVertex,
            "NormVertex": NormVertex,
            "TransposeVertex": TransposeVertex,
        }

        exec(code_str, namespace)

        if "diagram" in namespace and isinstance(namespace["diagram"], Diagram):
            return namespace["diagram"]

        if "build_diagram" in namespace and callable(namespace["build_diagram"]):
            d = namespace["build_diagram"]()
            if isinstance(d, Diagram):
                return d

        raise ValueError("Executed code string did not assign a valid 'diagram' or 'build_diagram()' function.")
