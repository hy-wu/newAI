from __future__ import annotations
from collections import deque
from typing import Dict, List
from .nodes import Vertex, Propagator, InputVertex, OutputVertex
from .types import TensorType


class Diagram:
    """Graph container representing a Feynman Diagrammatic IR computational structure.

    Manages vertices (interaction nodes) and propagators (typed edges),
    supports topological sorting, and exports to Mermaid diagrams.
    """
    def __init__(self, name: str = "feynman_diagram"):
        self.name = name
        self.vertices: Dict[str, Vertex] = {}
        self.propagators: Dict[str, Propagator] = {}
        self.inputs: List[str] = []   # Input Vertex IDs (ordered)
        self.outputs: List[str] = []  # Output Vertex IDs (ordered)
        self._propagator_counter = 0

    def add_vertex(self, vertex: Vertex) -> Vertex:
        if vertex.id in self.vertices:
            raise ValueError(f"Vertex with ID '{vertex.id}' already exists in diagram.")
        self.vertices[vertex.id] = vertex
        if isinstance(vertex, InputVertex):
            self.inputs.append(vertex.id)
        elif isinstance(vertex, OutputVertex):
            self.outputs.append(vertex.id)
        return vertex

    def remove_vertex(self, vertex_id: str) -> None:
        """Remove a vertex and all its connected propagators."""
        if vertex_id not in self.vertices:
            raise KeyError(f"Vertex '{vertex_id}' not found.")
        v = self.vertices[vertex_id]

        # Collect propagators to remove
        props_to_remove = list(v.inputs) + list(v.outputs)
        for prop_id in props_to_remove:
            if prop_id in self.propagators:
                self.remove_propagator(prop_id)

        # Remove from input/output registries
        if vertex_id in self.inputs:
            self.inputs.remove(vertex_id)
        if vertex_id in self.outputs:
            self.outputs.remove(vertex_id)

        del self.vertices[vertex_id]

    def remove_propagator(self, prop_id: str) -> None:
        """Remove a propagator and unlink it from source/destination vertices."""
        if prop_id not in self.propagators:
            return
        prop = self.propagators[prop_id]
        src_v = self.vertices.get(prop.src_vertex_id)
        dst_v = self.vertices.get(prop.dst_vertex_id)
        if src_v and prop_id in src_v.outputs:
            src_v.outputs.remove(prop_id)
        if dst_v and prop_id in dst_v.inputs:
            dst_v.inputs.remove(prop_id)
        del self.propagators[prop_id]

    def connect(self, src_id: str, dst_id: str, tensor_type: TensorType, label: str = "") -> Propagator:
        if src_id not in self.vertices:
            raise KeyError(f"Source vertex '{src_id}' not found.")
        if dst_id not in self.vertices:
            raise KeyError(f"Destination vertex '{dst_id}' not found.")

        self._propagator_counter += 1
        prop_id = f"e_{src_id}_to_{dst_id}_{self._propagator_counter}"
        propagator = Propagator(
            id=prop_id,
            src_vertex_id=src_id,
            dst_vertex_id=dst_id,
            tensor_type=tensor_type,
            label=label or f"edge_{self._propagator_counter}"
        )
        self.propagators[prop_id] = propagator

        self.vertices[src_id].outputs.append(prop_id)
        self.vertices[dst_id].inputs.append(prop_id)
        return propagator

    def get_input_propagators(self, vertex_id: str) -> List[Propagator]:
        """Return all propagators feeding into a vertex, in order."""
        v = self.vertices[vertex_id]
        return [self.propagators[p_id] for p_id in v.inputs if p_id in self.propagators]

    def get_output_propagators(self, vertex_id: str) -> List[Propagator]:
        """Return all propagators leaving a vertex, in order."""
        v = self.vertices[vertex_id]
        return [self.propagators[p_id] for p_id in v.outputs if p_id in self.propagators]

    def topological_sort(self) -> List[Vertex]:
        """Kahn's algorithm with O(1) popleft via deque."""
        in_degree: Dict[str, int] = {v_id: 0 for v_id in self.vertices}
        for prop in self.propagators.values():
            in_degree[prop.dst_vertex_id] += 1

        queue = deque(v_id for v_id, deg in in_degree.items() if deg == 0)
        sorted_vertices: List[Vertex] = []

        while queue:
            curr_id = queue.popleft()
            sorted_vertices.append(self.vertices[curr_id])

            for prop_id in self.vertices[curr_id].outputs:
                if prop_id not in self.propagators:
                    continue
                prop = self.propagators[prop_id]
                in_degree[prop.dst_vertex_id] -= 1
                if in_degree[prop.dst_vertex_id] == 0:
                    queue.append(prop.dst_vertex_id)

        if len(sorted_vertices) != len(self.vertices):
            raise ValueError("Diagram contains cyclic dependency!")

        return sorted_vertices

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for prop in self.propagators.values():
            src = prop.src_vertex_id
            dst = prop.dst_vertex_id
            lbl = f"{prop.tensor_type.shape}"
            src_label = f"{self.vertices[src].op_type}:{src}"
            dst_label = f"{self.vertices[dst].op_type}:{dst}"
            lines.append(f'    {src}["{src_label}"] -->|"{lbl}"| {dst}["{dst_label}"]')
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Diagram '{self.name}': {len(self.vertices)} vertices, {len(self.propagators)} propagators>"
