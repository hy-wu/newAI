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
    def _compute_2d_layout(diagram: Diagram) -> Dict[str, Tuple[float, float]]:
        """Compute topological level-based 2D coordinates (level_x, rank_y) for vertices."""
        sorted_vertices = diagram.topological_sort()
        levels: Dict[str, int] = {}

        # 1. Compute horizontal level (depth)
        for v in sorted_vertices:
            if isinstance(v, InputVertex):
                levels[v.id] = 0
            else:
                in_props = diagram.get_input_propagators(v.id)
                if in_props:
                    max_parent_level = max(
                        levels.get(p.src_vertex_id, 0) for p in in_props
                    )
                    levels[v.id] = max_parent_level + 1
                else:
                    levels[v.id] = 0

        # Group vertices by level
        level_groups: Dict[int, List[Vertex]] = {}
        for v in sorted_vertices:
            lvl = levels[v.id]
            level_groups.setdefault(lvl, []).append(v)

        # 2. Compute 2D grid coordinates (X, Y)
        positions: Dict[str, Tuple[float, float]] = {}
        for lvl, group in level_groups.items():
            num_in_group = len(group)
            for idx, v in enumerate(group):
                # Spread vertically around y=0
                y_coord = (idx - (num_in_group - 1) / 2.0) * 1.5
                positions[v.id] = (float(lvl * 2.5), y_coord)

        return positions

    @classmethod
    def to_tikz(cls, diagram: Diagram) -> str:
        """Generate publication-ready LaTeX tikz-feynman document code with multi-level 2D layout."""
        sorted_vertices = diagram.topological_sort()
        positions = cls._compute_2d_layout(diagram)

        lines = [
            "% Requires \\usepackage[compat=1.1.0]{tikz-feynman}",
            "\\begin{tikzpicture}",
            "  \\begin{feynman}"
        ]

        # Place vertices using 2D (x, y) grid coordinates
        for v in sorted_vertices:
            node_label = v.id.replace("_", "\\_")
            op_label = v.op_type
            x, y = positions.get(v.id, (0.0, 0.0))

            if isinstance(v, (InputVertex, OutputVertex)):
                lines.append(f"    \\vertex (v_{v.id}) at ({x:.2f}, {y:.2f}) {{\\small \\mathbf{{{node_label}}}}};")
            else:
                lines.append(f"    \\node[dot, label=above:{{\\tiny {node_label} ({op_label})}}] (v_{v.id}) at ({x:.2f}, {y:.2f}) {{}};")

        lines.append("")
        lines.append("    % Diagram Propagator Lines (Feynman Rules)")

        # Draw propagators with line styles
        for prop in diagram.propagators.values():
            src_id = prop.src_vertex_id
            dst_id = prop.dst_vertex_id
            src_v = diagram.vertices.get(src_id)

            style = "fermion"  # Standard state propagator
            if prop.label == "bypass" or "bypass" in prop.label.lower():
                style = "scalar, dashed, bend left=45"  # Residual bypass channel (curved)
            elif isinstance(src_v, AttentionVertex) or "attn" in src_id.lower():
                style = "boson"  # Interaction field line (wavy line)

            shape_str = str(prop.tensor_type.shape).replace("'", "")
            lines.append(f"    \\diagram* {{ (v_{src_id}) -- [{style}, edge label={{\\tiny ${shape_str}$}}] (v_{dst_id}) }};")

        lines.append("  \\end{feynman}")
        lines.append("\\end{tikzpicture}")
        return "\n".join(lines)

    @classmethod
    def to_svg(cls, diagram: Diagram, width: int = 960, height: int = 480) -> str:
        """Generate standalone SVG graphics with multi-level 2D layout."""
        sorted_vertices = diagram.topological_sort()
        grid_pos = cls._compute_2d_layout(diagram)

        # Scale 2D grid coordinates to SVG pixel space
        max_x = max(pos[0] for pos in grid_pos.values()) if grid_pos else 1.0
        padding_x = 90
        padding_y = 60
        scale_x = (width - 2 * padding_x) / max(max_x, 1.0)
        center_y = height / 2.0
        scale_y = 90.0

        positions: Dict[str, Tuple[float, float]] = {}
        for v_id, (gx, gy) in grid_pos.items():
            positions[v_id] = (padding_x + gx * scale_x, center_y - gy * scale_y)

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
            '    .edge-label { fill: #cbd5e1; font-size: 10px; text-anchor: middle; }',
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
            is_bypass = prop.label == "bypass" or "bypass" in prop.label.lower()
            if is_bypass:
                cls_name = "prop-bypass"
            elif "attn" in prop.src_vertex_id.lower():
                cls_name = "prop-attn"

            # Curved control points for residual bypass or multi-level offsets
            if is_bypass:
                # Arc overhead
                arc_height = min(src_y, dst_y) - 90.0
                path_d = f"M {src_x} {src_y} C {src_x + 40} {arc_height}, {dst_x - 40} {arc_height}, {dst_x} {dst_y}"
                mid_x = (src_x + dst_x) / 2.0
                mid_y = arc_height + 14.0
            elif abs(src_y - dst_y) > 10:
                path_d = f"M {src_x} {src_y} C {src_x + (dst_x - src_x)/2} {src_y}, {src_x + (dst_x - src_x)/2} {dst_y}, {dst_x} {dst_y}"
                mid_x = (src_x + dst_x) / 2.0
                mid_y = (src_y + dst_y) / 2.0 - 8.0
            else:
                path_d = f"M {src_x} {src_y} L {dst_x} {dst_y}"
                mid_x = (src_x + dst_x) / 2.0
                mid_y = src_y - 8.0

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
