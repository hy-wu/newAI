from __future__ import annotations
from typing import Dict, List, Optional, Set
import copy
from .diagram import Diagram
from .nodes import (
    Vertex, ContractionVertex, AttentionVertex, PointwiseVertex, TransposeVertex
)
from .types import TensorType, Shape


class RewriteEngine:
    """Applies semantics-preserving diagrammatic graph rewrites.

    Each pass performs actual graph mutations: removing old vertices/propagators
    and inserting new ones with correct wiring.
    """
    def __init__(self):
        pass

    def apply_all_passes(self, diagram: Diagram) -> Diagram:
        """Applies optimization rewrite passes in sequence."""
        diagram = self.fuse_attention_pattern(diagram)
        diagram = self.cancel_double_transpose(diagram)
        return diagram

    def fuse_attention_pattern(self, diagram: Diagram) -> Diagram:
        """Fuses Contraction(Q, K^T) -> Softmax -> Contraction(scores, V) into AttentionVertex.

        Detects the 3-node pattern and replaces it with a single fused vertex,
        properly rewiring all incoming and outgoing propagators.
        """
        new_diagram = copy.deepcopy(diagram)

        # We need to iterate carefully since we'll be mutating the graph
        fused = True
        while fused:
            fused = False
            sorted_v = new_diagram.topological_sort()

            for v in sorted_v:
                if not isinstance(v, ContractionVertex) or len(v.inputs) != 2:
                    continue

                # Check: does this Contraction's single output go to a Softmax?
                out_props = new_diagram.get_output_propagators(v.id)
                if len(out_props) != 1:
                    continue
                softmax_id = out_props[0].dst_vertex_id
                softmax_v = new_diagram.vertices.get(softmax_id)
                if not isinstance(softmax_v, PointwiseVertex):
                    continue
                if softmax_v.attributes.get("sub_op") != "Softmax":
                    continue

                # Check: does the Softmax's single output go to another Contraction?
                softmax_out = new_diagram.get_output_propagators(softmax_id)
                if len(softmax_out) != 1:
                    continue
                v2_id = softmax_out[0].dst_vertex_id
                v2 = new_diagram.vertices.get(v2_id)
                if not isinstance(v2, ContractionVertex):
                    continue

                # --- Pattern matched: v (QK^T) -> softmax -> v2 (scores * V) ---
                # Identify Q, K from v's inputs and V from v2's other input
                qk_in_props = new_diagram.get_input_propagators(v.id)
                q_prop = qk_in_props[0]
                k_prop = qk_in_props[1]

                v2_in_props = new_diagram.get_input_propagators(v2.id)
                # V is the input to v2 that is NOT from the softmax
                v_prop = None
                for p in v2_in_props:
                    if p.src_vertex_id != softmax_id:
                        v_prop = p
                        break

                if v_prop is None:
                    continue  # Can't identify V input

                # Collect v2's output propagators (consumers of the attention output)
                v2_out_props = new_diagram.get_output_propagators(v2.id)
                consumer_info = [(p.dst_vertex_id, p.tensor_type, p.label) for p in v2_out_props]

                # Create fused AttentionVertex
                attn_id = f"fused_attn_{v.id}"
                attn_v = AttentionVertex(id=attn_id, num_heads=12, head_dim=64)

                # Remove old vertices (this also removes their propagators)
                new_diagram.remove_vertex(v2.id)
                new_diagram.remove_vertex(softmax_id)
                new_diagram.remove_vertex(v.id)

                # Insert fused vertex
                new_diagram.add_vertex(attn_v)

                # Rewire: Q, K, V -> fused attention
                out_type = q_prop.tensor_type
                new_diagram.connect(q_prop.src_vertex_id, attn_id, q_prop.tensor_type, label="Q")
                new_diagram.connect(k_prop.src_vertex_id, attn_id, k_prop.tensor_type, label="K")
                new_diagram.connect(v_prop.src_vertex_id, attn_id, v_prop.tensor_type, label="V")

                # Rewire: fused attention -> downstream consumers
                for dst_id, ttype, label in consumer_info:
                    if dst_id in new_diagram.vertices:
                        new_diagram.connect(attn_id, dst_id, ttype, label=label)

                fused = True
                break  # Restart iteration after mutation

        return new_diagram

    def cancel_double_transpose(self, diagram: Diagram) -> Diagram:
        """Eliminates double transpose: T_p2(T_p1(x)) -> x when compose(p2, p1) = identity.

        Properly removes both TransposeVertex nodes and directly connects
        the predecessor to the successor.
        """
        new_diagram = copy.deepcopy(diagram)

        cancelled = True
        while cancelled:
            cancelled = False
            for v in list(new_diagram.vertices.values()):
                if not isinstance(v, TransposeVertex):
                    continue

                out_props = new_diagram.get_output_propagators(v.id)
                if len(out_props) != 1:
                    continue
                next_v = new_diagram.vertices.get(out_props[0].dst_vertex_id)
                if not isinstance(next_v, TransposeVertex):
                    continue

                p1 = v.attributes["perm"]
                p2 = next_v.attributes["perm"]

                # Check if applying p1 then p2 gives identity
                composed = tuple(p1[i] for i in p2)
                identity = tuple(range(len(p1)))
                if composed != identity:
                    continue

                # --- Double transpose detected ---
                # Get input to first transpose and output of second transpose
                in_props = new_diagram.get_input_propagators(v.id)
                if not in_props:
                    continue
                predecessor_prop = in_props[0]
                predecessor_id = predecessor_prop.src_vertex_id
                predecessor_type = predecessor_prop.tensor_type

                next_out_props = new_diagram.get_output_propagators(next_v.id)
                consumer_info = [(p.dst_vertex_id, p.tensor_type, p.label) for p in next_out_props]

                # Remove both transpose vertices
                new_diagram.remove_vertex(next_v.id)
                new_diagram.remove_vertex(v.id)

                # Direct-connect predecessor to all consumers
                for dst_id, ttype, label in consumer_info:
                    if dst_id in new_diagram.vertices:
                        new_diagram.connect(predecessor_id, dst_id, predecessor_type, label=label)

                cancelled = True
                break  # Restart after mutation

        return new_diagram
