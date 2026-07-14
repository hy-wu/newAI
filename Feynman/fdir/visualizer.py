"""Visual ML Feynman Diagram Renderer.

Generates visual representations of FDIR computational diagrams:
- to_tikz(): Publication-quality LaTeX tikz-feynman code (fermion lines, photon/wavy interaction lines, dashed bypasses).
- to_svg(): Standalone, color-coded SVG graphics.
- to_html(): Interactive HTML page wrapping SVG with CSS controls.
"""

from __future__ import annotations
import html
from typing import Dict, List, Tuple
from .diagram import Diagram
from .nodes import (
    Vertex, InputVertex, OutputVertex, ContractionVertex, AttentionVertex,
    PointwiseVertex, ResidualVertex, NormVertex, TransposeVertex
)


class FeynmanVisualizer:
    """Renderer for Visual Machine Learning Feynman Diagrams."""

    @staticmethod
    def to_tikz(diagram: Diagram) -> str:
        """Generate publication-ready LaTeX tikz-feynman document code."""
        sorted_vertices = diagram.topological_sort()
        lines = [
            "% Requires \\usepackage[compat=1.1.0]{tikz-feynman}",
            "\\begin{tikzpicture}",
            "  \\begin{feynman}"
        ]

        # Map vertex types to TikZ node styles
        # Input/Output -> plain, Vertices -> dot/blob with label
        for idx, v in enumerate(sorted_vertices):
            node_label = v.id.replace("_", "\\_")
            op_label = v.op_type
            if isinstance(v, InputVertex):
                lines.append(f"    \\vertex (v_{v.id}) {{\\small \\mathbf{{{node_label}}}}};")
            elif isinstance(v, OutputVertex):
                lines.append(f"    \\vertex (v_{v.id}) {{\\small \\mathbf{{{node_label}}}}};")
            else:
                lines.append(f"    \\node[dot, label=above:{{\\tiny {op_label}}}] (v_{v.id}) at ({idx*1.8}, 0) {{}};")

        lines.append("")
        lines.append("    % Diagram Propagator Lines (Feynman Rules)")

        # Draw propagators with style depending on line type
        for prop in diagram.propagators.values():
            src_id = prop.src_vertex_id
            dst_id = prop.dst_vertex_id
            src_v = diagram.vertices.get(src_id)

            style = "fermion"  # Standard state propagator
            if prop.label == "bypass" or "bypass" in prop.label.lower():
                style = "scalar, dashed"  # Residual bypass channel
            elif isinstance(src_v, AttentionVertex) or "attn" in src_id.lower():
                style = "boson"  # Interaction field line (wavy line)

            shape_str = str(prop.tensor_type.shape).replace("'", "")
            lines.append(f"    \\diagram* {{ (v_{src_id}) -- [{style}, edge label={{\\tiny ${shape_str}$}}] (v_{dst_id}) }};")

        lines.append("  \\end{feynman}")
        lines.append("\\end{tikzpicture}")
        return "\n".join(lines)

    @staticmethod
    def to_svg(diagram: Diagram, width: int = 900, height: int = 400) -> str:
        """Generate standalone SVG graphics representation of the FDIR Diagram."""
        sorted_vertices = diagram.topological_sort()
        num_v = len(sorted_vertices)

        # Color palette per vertex type
        color_map = {
            "Input": "#2563eb",        # Blue
            "Output": "#0f172a",       # Dark slate
            "Contraction": "#dc2626",   # Red
            "Attention": "#9333ea",     # Purple
            "Pointwise": "#16a34a",     # Green
            "ResidualAdd": "#ea580c",   # Orange
            "Norm": "#475569",          # Gray
            "Transpose": "#0284c7",     # Sky blue
        }

        # Positioning layout (left-to-right flow with vertical distribution)
        padding_x = 80
        step_x = (width - 2 * padding_x) / max(num_v - 1, 1)
        center_y = height / 2.0

        positions: Dict[str, Tuple[float, float]] = {}

        # Layout vertices: offset inputs/outputs and bypasses
        input_count = sum(1 for v in sorted_vertices if isinstance(v, InputVertex))
        inp_idx = 0

        for idx, v in enumerate(sorted_vertices):
            if isinstance(v, InputVertex):
                y_pos = center_y + (inp_idx - (input_count - 1) / 2.0) * 80.0
                positions[v.id] = (padding_x, y_pos)
                inp_idx += 1
            elif isinstance(v, OutputVertex):
                positions[v.id] = (width - padding_x, center_y)
            else:
                x_pos = padding_x + idx * step_x
                # Add slight vertical offset for branching vertices
                offset_y = 0.0
                if "proj_K" in v.id: offset_y = -60.0
                if "proj_V" in v.id: offset_y = 60.0
                positions[v.id] = (x_pos, center_y + offset_y)

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}px" style="background:#0f172a; font-family: Inter, sans-serif;">',
            '  <defs>',
            '    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
            '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/>',
            '    </marker>',
            '    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
            '      <feGaussianBlur stdDeviation="3" result="blur"/>',
            '      <feComposite in="SourceGraphic" in2="blur" operator="over"/>',
            '    </filter>',
            '  </defs>',
            '  <style>',
            '    .prop-line { stroke: #64748b; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }',
            '    .prop-bypass { stroke: #f97316; stroke-width: 2.5; stroke-dasharray: 6,4; fill: none; marker-end: url(#arrow); }',
            '    .prop-attn { stroke: #a855f7; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }',
            '    .node-title { fill: #f8fafc; font-size: 13px; font-weight: 600; text-anchor: middle; }',
            '    .node-sub { fill: #94a3b8; font-size: 10px; text-anchor: middle; }',
            '    .edge-label { fill: #cbd5e1; font-size: 10px; text-anchor: middle; background: #1e293b; }',
            '  </style>',
            '  <!-- Title Banner -->',
            f'  <text x="{width/2}" y="30" fill="#e2e8f0" font-size="16" font-weight="700" text-anchor="middle">Visual Machine Learning Feynman Diagram: {diagram.name}</text>',
            '  <!-- Propagator Edges -->'
        ]

        # Draw propagators
        for prop in diagram.propagators.values():
            src_x, src_y = positions.get(prop.src_vertex_id, (padding_x, center_y))
            dst_x, dst_y = positions.get(prop.dst_vertex_id, (width - padding_x, center_y))

            cls_name = "prop-line"
            if prop.label == "bypass" or "bypass" in prop.label.lower():
                cls_name = "prop-bypass"
            elif "attn" in prop.src_vertex_id.lower():
                cls_name = "prop-attn"

            # Curved control points if offset
            if abs(src_y - dst_y) > 10:
                path_d = f"M {src_x} {src_y} C {src_x + (dst_x - src_x)/2} {src_y}, {src_x + (dst_x - src_x)/2} {dst_y}, {dst_x} {dst_y}"
            else:
                path_d = f"M {src_x} {src_y} L {dst_x} {dst_y}"

            mid_x = (src_x + dst_x) / 2.0
            mid_y = (src_y + dst_y) / 2.0 - 8.0
            shape_text = html.escape(str(prop.tensor_type.shape))

            svg_parts.append(f'  <path d="{path_d}" class="{cls_name}"/>')
            svg_parts.append(f'  <text x="{mid_x}" y="{mid_y}" class="edge-label">{shape_text}</text>')

        # Draw Vertex Nodes
        svg_parts.append('  <!-- Vertex Interaction Nodes -->')
        for v in sorted_vertices:
            x, y = positions.get(v.id, (padding_x, center_y))
            color = color_map.get(v.op_type, "#38bdf8")

            if isinstance(v, (InputVertex, OutputVertex)):
                svg_parts.append(f'  <g transform="translate({x},{y})">')
                svg_parts.append(f'    <rect x="-35" y="-18" width="70" height="36" rx="8" fill="{color}" filter="url(#glow)"/>')
                svg_parts.append(f'    <text y="4" class="node-title" font-size="11">{html.escape(v.id)}</text>')
                svg_parts.append('  </g>')
            else:
                svg_parts.append(f'  <g transform="translate({x},{y})">')
                svg_parts.append(f'    <circle r="22" fill="{color}" filter="url(#glow)"/>')
                svg_parts.append(f'    <text y="-28" class="node-title">{html.escape(v.id)}</text>')
                svg_parts.append(f'    <text y="4" class="node-title" font-size="10">{html.escape(v.op_type[:6])}</text>')
                svg_parts.append('  </g>')

        svg_parts.append('</svg>')
        return "\n".join(svg_parts)

    @classmethod
    def to_html(cls, diagram: Diagram) -> str:
        """Wrap the SVG output in a self-contained HTML page."""
        svg_code = cls.to_svg(diagram)
        tikz_code = html.escape(cls.to_tikz(diagram))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FDIR Feynman Diagram: {diagram.name}</title>
  <style>
    body {{ background: #020617; color: #f8fafc; font-family: system-ui, sans-serif; padding: 2rem; }}
    .card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }}
    pre {{ background: #1e293b; padding: 1rem; border-radius: 8px; overflow-x: auto; color: #38bdf8; }}
  </style>
</head>
<body>
  <h1>Visual Machine Learning Feynman Diagram</h1>
  <div class="card">
    <h2>Interactive Visual Computation Diagram</h2>
    {svg_code}
  </div>
  <div class="card">
    <h2>LaTeX tikz-feynman Source Code</h2>
    <pre><code>{tikz_code}</code></pre>
  </div>
</body>
</html>"""
