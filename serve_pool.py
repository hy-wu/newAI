r"""
Local paper pool server — browse PDFs, extracted figures, and TeX source in browser.

Responsive layout:
  - Wide screen (>=1200px): 3-column grid  [Paper List | PDF | Figures]
  - Narrow screen: accordion expand/collapse

Usage:
    python serve_pool.py                         # start on http://localhost:8899
    python serve_pool.py --port 8000             # custom port
    python serve_pool.py --pool D:\my_papers     # custom pool path
"""

import argparse
import html
import json
import sqlite3
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DB = PROJECT_ROOT / ".snowcite" / "papers.db"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Pool</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f5; color: #333; height: 100vh; overflow: hidden; }
  a { color: #1a73e8; text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* ========== Narrow layout (default, <1200px) ========== */
  .narrow-body { height: 100vh; display: flex; flex-direction: column; }
  .narrow-body #search { display: block; width: 100%; padding: 10px 14px; font-size: 1em;
    border: none; border-bottom: 1px solid #ddd; outline: none; flex-shrink: 0; }
  .narrow-body #search:focus { border-color: #1a73e8; }
  .narrow-body #papers { flex: 1; overflow-y: auto; padding: 12px; }
  .narrow-body .subtitle { text-align: center; padding: 8px; color: #888; font-size: 0.85em;
    border-top: 1px solid #eee; flex-shrink: 0; }

  .paper { background: #fff; border-radius: 8px; margin-bottom: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }
  .paper-header { padding: 12px 16px; cursor: pointer; display: flex; align-items: center;
                   gap: 10px; user-select: none; }
  .paper-header:hover { background: #f0f7ff; }
  .paper-id { color: #1a73e8; font-weight: bold; min-width: 36px; font-size: 0.9em; }
  .paper-title { flex: 1; font-weight: 500; font-size: 0.95em; }
  .paper-year { color: #888; font-size: 0.85em; min-width: 44px; }
  .toggle-icon { color: #888; transition: transform .2s; font-size: 11px; }
  .paper.open .toggle-icon { transform: rotate(90deg); }
  .paper-body { display: none; padding: 0 16px 16px; border-top: 1px solid #eee; }
  .paper.open .paper-body { display: block; }
  .paper-meta { display: flex; gap: 16px; margin: 10px 0; font-size: 0.85em; color: #555; }
  .paper-links { margin: 8px 0; display: flex; gap: 8px; flex-wrap: wrap; }
  .btn { display: inline-block; padding: 5px 12px; background: #1a73e8; color: #fff !important;
          text-decoration: none !important; border-radius: 4px; font-size: 0.85em; }
  .btn:hover { background: #1557b0; }
  .btn-search { background: #0078d4; cursor: pointer; }
  .btn-search:hover { background: #005a9e; }
  .pdf-viewer { margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; background: #eee; }
  .pdf-viewer embed { display: block; width: 100%; height: 500px; }
  .figures { margin: 10px 0; }
  .figures h4, .tex-section h4 { margin-bottom: 6px; color: #555; font-size: 0.9em; }
  .figure-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px; }
  .figure-item { border: 1px solid #ddd; border-radius: 4px; padding: 6px; background: #fafafa; text-align: center; }
  .figure-item img, .figure-item embed { max-width: 100%; max-height: 250px; }
  .figure-item .fig-caption { font-size: 0.8em; color: #666; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .no-figures { color: #999; font-style: italic; font-size: 0.9em; }
  pre { background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 10px;
        overflow-x: auto; font-size: 0.8em; max-height: 400px; white-space: pre-wrap; word-break: break-all; }
  .tex-section { margin: 10px 0; }
  .tex-toggle { color: #1a73e8; cursor: pointer; font-size: 0.85em; }
  .tex-toggle:hover { text-decoration: underline; }

  /* ========== Wide layout (>=1200px, 3-column) ========== */
  .wide-body { height: 100vh; display: flex; flex-direction: column; }
  .wide-body .top-bar { display: flex; align-items: center; gap: 12px; padding: 8px 16px;
    border-bottom: 1px solid #ddd; background: #fff; flex-shrink: 0; }
  .wide-body .top-bar input { flex: 1; max-width: 500px; padding: 8px 12px; font-size: 0.95em;
    border: 1px solid #ddd; border-radius: 6px; outline: none; }
  .wide-body .top-bar input:focus { border-color: #1a73e8; }
  .wide-body .top-bar .info { color: #888; font-size: 0.85em; margin-left: auto; }
  .wide-grid { display: flex; flex: 1; overflow: hidden; }
  .divider { width: 4px; cursor: col-resize; background: transparent; flex-shrink: 0; position: relative; z-index: 10; }
  .divider:hover, .divider.active { background: #1a73e8; }
  @media (prefers-color-scheme: dark) {
    .divider:hover, .divider.active { background: #4a9eff; }
  }

  .wide-list { overflow-y: auto; border-right: none; background: #fff; width: 320px; min-width: 200px; max-width: 500px; flex-shrink: 0; }
  .wide-list .list-item { padding: 10px 14px; cursor: pointer; border-bottom: 1px solid #f0f0f0;
    display: flex; gap: 8px; align-items: baseline; }
  .wide-list .list-item:hover { background: #f0f7ff; }
  .wide-list .list-item.active { background: #e8f0fe; border-left: 3px solid #1a73e8; }
  .wide-list .list-item .li-id { color: #1a73e8; font-weight: bold; font-size: 0.85em; min-width: 30px; }
  .wide-list .list-item .li-title { font-size: 0.9em; }
  .wide-list .list-item .li-year { color: #888; font-size: 0.8em; margin-left: auto; }
  .wide-list .empty-list { padding: 40px 20px; text-align: center; color: #999; }

  .wide-center { overflow-y: auto; padding: 16px; display: flex; flex-direction: column; background: #fafafa; flex: 1; min-width: 300px; }
  .wide-center .placeholder { margin: auto; text-align: center; color: #999; font-size: 1.1em; }
  .wide-center .pdf-toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
  .wide-center .pdf-toolbar .pt-title { font-weight: 600; font-size: 0.95em; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .wide-center .pdf-viewer { flex: 1; min-height: 0; border: 1px solid #ddd; border-radius: 4px; background: #eee; }
  .wide-center .pdf-viewer embed { width: 100%; height: 100%; min-height: 400px; }

  .wide-right { overflow-y: auto; padding: 16px; border-left: none; background: #fff; width: 340px; min-width: 200px; max-width: 600px; flex-shrink: 0; }
  .wide-right .placeholder { margin-top: 40px; text-align: center; color: #999; }
  .wide-right .figures { margin: 0; }
  .wide-right .figure-grid { grid-template-columns: 1fr; }
  .wide-right .tex-section pre { max-height: 300px; }

  /* ========== Dark mode ========== */
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a2e; color: #e0e0e0; }
    .paper { background: #16213e; }
    .paper-header:hover { background: #1a2744; }
    .paper-body { border-color: #2a2a4a; }
    .pdf-viewer { background: #111; border-color: #333; }
    .figure-item { background: #1e1e3a; border-color: #333; }
    pre { background: #111; border-color: #333; color: #ccc; }
    #search, .wide-body .top-bar { background: #16213e; color: #e0e0e0; border-color: #333; }
    .wide-body .top-bar input { background: #1a1a2e; color: #e0e0e0; border-color: #333; }
    .wide-list { background: #16213e; }
    .wide-list .list-item { border-color: #2a2a4a; }
    .wide-list .list-item:hover { background: #1a2744; }
    .wide-list .list-item.active { background: #1a2a4e; border-left-color: #4a9eff; }
    .wide-center { background: #16162a; }
    .wide-right { background: #16213e; }
    .wide-body .top-bar { background: #16213e; border-color: #2a2a4a; }
    .paper-meta { color: #aaa; }
    .paper-year { color: #888; }
  }
</style>
</head>
<body>
<div id="app"></div>
<script>
const PAPERS = __PAPERS_JSON__;
const POOL_PATH = "__POOL_PATH__";
const WIDE_BP = 1200;

// ── helpers ──────────────────────────────────────────────────────────
function escapeHtml(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function figHtml(fig) {
  var p = escapeHtml(fig[0]), ext = fig[1], name = p.split('/').pop();
  if (ext === '.pdf')
    return '<div class="figure-item"><embed src="'+p+'" type="application/pdf" width="100%" height="250px"><div class="fig-caption">'+name+'</div></div>';
  return '<div class="figure-item"><img src="'+p+'" alt="'+name+'" loading="lazy"><div class="fig-caption">'+name+'</div></div>';
}

// ── build narrow (accordion) ─────────────────────────────────────────
function renderNarrow(filter) {
  var out = '<div class="narrow-body">';
  out += '<input id="search" type="text" placeholder="Search papers..." value="'+escapeHtml(filter||'')+'" oninput="onSearch(this.value)">';
  out += '<div id="papers">';
  var shown = 0;
  PAPERS.forEach(function(p) {
    if (filter && p.t.indexOf(filter) < 0 && (filter.length <= 3 || (p.t + ' ' + p.y).indexOf(filter) < 0)) return;
    shown++;
    out += '<div class="paper" data-id="'+p.id+'">';
    out += '<div class="paper-header" onclick="togglePaperNarrow(this)">';
    out += '<span class="paper-id">['+p.id+']</span>';
    out += '<span class="paper-title">'+escapeHtml(p.t)+'</span>';
    out += '<span class="paper-year">('+p.y+')</span>';
    out += '<span class="toggle-icon">&#9654;</span></div>';
    out += '<div class="paper-body">';
    out += '<div class="paper-meta"><span>Status: <b>'+escapeHtml(p.st)+'</b></span><span>arXiv:'+escapeHtml(p.a)+'</span></div>';
    if (p.pdf) {
      out += '<div class="paper-links"><a href="'+p.pdf+'" target="_blank" class="btn">New tab</a> <a class="btn btn-search" data-pid="'+p.id+'" onclick="searchBing(this.dataset.pid)">Bing</a></div>';
      out += '<div class="pdf-viewer"><embed src="'+p.pdf+'#view=FitH" type="application/pdf"></div>';
    } else {
      out += '<div class="paper-links"><span style="color:#999">PDF not available</span> <a class="btn btn-search" data-pid="'+p.id+'" onclick="searchBing(this.dataset.pid)">Bing</a></div>';
    }
    if (p.figs && p.figs.length) {
      out += '<div class="figures"><h4>Extracted Figures</h4><div class="figure-grid">';
      p.figs.forEach(function(f) { out += figHtml(f); });
      out += '</div></div>';
    }
    out += '<div class="tex-section">';
    out += '<h4>TeX Source <span class="tex-toggle" onclick="toggleTexNarrow(this)">(show/hide)</span></h4>';
    out += '<pre style="display:none">'+(p.tc || 'No TeX source available')+'</pre>';
    out += '</div></div></div>';
  });
  out += '</div>';
  out += '<div class="subtitle">'+shown+'/'+PAPERS.length+' papers &middot; Pool: '+POOL_PATH+'</div>';
  out += '</div>';
  document.getElementById('app').innerHTML = out;
}

function togglePaperNarrow(header) {
  header.parentElement.classList.toggle('open');
}
function toggleTexNarrow(el) {
  var pre = el.parentElement.nextElementSibling;
  pre.style.display = pre.style.display === 'none' ? 'block' : 'none';
}

// ── build wide (3-column) ────────────────────────────────────────────
var selectedId = null;

function renderWide(filter) {
  var out = '<div class="wide-body">';
  out += '<div class="top-bar">';
  out += '<b>Paper Pool</b>';
  out += '<input id="search-wide" type="text" placeholder="Search papers..." value="'+escapeHtml(filter||'')+'" oninput="onSearch(this.value)">';
  out += '<span class="info">'+PAPERS.length+' papers &middot; '+POOL_PATH+'</span>';
  out += '</div>';
  out += '<div class="wide-grid">';

  // ── Left: list ──
  out += '<div class="wide-list" id="wide-list">';
  var shown = 0;
  PAPERS.forEach(function(p) {
    if (filter && p.t.indexOf(filter) < 0 && (filter.length <= 3 || (p.t + ' ' + p.y).indexOf(filter) < 0)) return;
    shown++;
    var active = (p.id === selectedId) ? ' active' : '';
    out += '<div class="list-item'+active+'" data-id="'+p.id+'" onclick="selectPaper('+p.id+')">';
    out += '<span class="li-id">['+p.id+']</span>';
    out += '<span class="li-title">'+escapeHtml(p.t)+'</span>';
    out += '<span class="li-year">('+p.y+')</span></div>';
  });
  if (!shown) out += '<div class="empty-list">No matching papers</div>';
  out += '</div>';

  // ── Divider 1 ──
  out += '<div class="divider" data-panel="left" onmousedown="startResize(event, \\'left\\')"></div>';

  // ── Center: PDF ──
  out += '<div class="wide-center" id="wide-center">';
  var sel = getPaperById(selectedId);
  if (sel) {
    var searchBtnWide = '<a class="btn btn-search" data-pid="'+sel.id+'" onclick="searchBing(this.dataset.pid)">Bing</a>';
    if (sel.pdf) {
      out += '<div class="pdf-toolbar"><span class="pt-title">['+sel.id+'] '+escapeHtml(sel.t)+'</span>';
      out += '<a href="'+sel.pdf+'" target="_blank" class="btn">New Tab</a> '+searchBtnWide+'</div>';
      out += '<div class="pdf-viewer"><embed src="'+sel.pdf+'#view=FitH" type="application/pdf"></div>';
    } else {
      out += '<div style="text-align:center;padding:40px 16px">';
      out += '<div class="placeholder" style="margin-bottom:16px">PDF not available for this paper</div>';
      out += searchBtnWide+'</div>';
    }
  } else {
    out += '<div class="placeholder">Select a paper from the list</div>';
  }
  out += '</div>';

  // ── Right: Figures + TeX ──
  out += '<div class="divider" data-panel="right" onmousedown="startResize(event, \\'right\\')"></div>';
  out += '<div class="wide-right" id="wide-right">';
  if (sel) {
    if (sel.figs && sel.figs.length) {
      out += '<div class="figures"><h4>Extracted Figures</h4><div class="figure-grid">';
      sel.figs.forEach(function(f) { out += figHtml(f); });
      out += '</div></div>';
    } else {
      out += '<p class="no-figures">No extracted figures</p>';
    }
    out += '<div class="tex-section">';
    out += '<h4>TeX Source <span class="tex-toggle" onclick="toggleTexWide()">(show/hide)</span></h4>';
    out += '<pre id="tex-wide" style="display:none">'+(sel.tc || 'No TeX source available')+'</pre>';
    out += '</div>';
  } else {
    out += '<div class="placeholder">Select a paper to view figures</div>';
  }
  out += '</div>';

  out += '</div></div>';
  document.getElementById('app').innerHTML = out;
  loadPanelWidths();
}

function getPaperById(id) {
  for (var i = 0; i < PAPERS.length; i++) { if (PAPERS[i].id === id) return PAPERS[i]; }
  return null;
}

function selectPaper(id) {
  selectedId = id;
  var f = getFilter();
  var scrollTop = 0;
  var listEl = document.getElementById('wide-list');
  if (listEl) scrollTop = listEl.scrollTop;
  renderWide(f);
  setTimeout(function() {
    var newList = document.getElementById('wide-list');
    if (newList && scrollTop > 0) newList.scrollTop = scrollTop;
  }, 0);
}

function toggleTexWide() {
  var el = document.getElementById('tex-wide');
  if (el) el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

// ── search ───────────────────────────────────────────────────────────
function getFilter() {
  var el = document.getElementById('search') || document.getElementById('search-wide');
  return el ? el.value.toLowerCase() : '';
}

function onSearch(val) {
  var f = val.toLowerCase();
  if (window.innerWidth >= WIDE_BP) renderWide(f);
  else renderNarrow(f);
}

// ── layout switch ────────────────────────────────────────────────────
var resizeTimer = null;
function onResize() {
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(function() {
    var f = getFilter();
    if (window.innerWidth >= WIDE_BP) renderWide(f);
    else renderNarrow(f);
  }, 200);
}

// ── keyboard nav ─────────────────────────────────────────────────────
function onKeyNav(e) {
  if (window.innerWidth < WIDE_BP) return;
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    var items = document.querySelectorAll('.list-item');
    var idx = -1;
    for (var i = 0; i < items.length; i++) {
      if (items[i].classList.contains('active')) { idx = i; break; }
    }
    if (e.key === 'ArrowDown') idx = Math.min(idx + 1, items.length - 1);
    else idx = Math.max(idx - 1, 0);
    if (idx >= 0 && items[idx]) {
      var id = parseInt(items[idx].getAttribute('data-id'));
      selectPaper(id);
    }
    e.preventDefault();
  }
}

// ── panel resize ─────────────────────────────────────────────────────
var resizeState = null;

function startResize(e, side) {
  e.preventDefault();
  resizeState = { side: side, startX: e.clientX };
  var div = e.target;
  div.classList.add('active');
  document.addEventListener('mousemove', doResize);
  document.addEventListener('mouseup', stopResize);
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
}

function doResize(e) {
  if (!resizeState) return;
  var dx = e.clientX - resizeState.startX;
  var list = document.getElementById('wide-list');
  var right = document.getElementById('wide-right');
  if (resizeState.side === 'left' && list) {
    var w = parseInt(list.offsetWidth) + dx;
    w = Math.max(200, Math.min(500, w));
    list.style.width = w + 'px';
  } else if (resizeState.side === 'right' && right) {
    var w = parseInt(right.offsetWidth) - dx;
    w = Math.max(200, Math.min(600, w));
    right.style.width = w + 'px';
  }
  resizeState.startX = e.clientX;
}

function stopResize(e) {
  if (!resizeState) return;
  document.querySelectorAll('.divider.active').forEach(function(d) { d.classList.remove('active'); });
  document.removeEventListener('mousemove', doResize);
  document.removeEventListener('mouseup', stopResize);
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
  // save widths
  var list = document.getElementById('wide-list');
  var right = document.getElementById('wide-right');
  var vals = {};
  if (list) vals.leftW = list.style.width;
  if (right) vals.rightW = right.style.width;
  try { localStorage.setItem('pool_panel_widths', JSON.stringify(vals)); } catch(e) {}
  resizeState = null;
}

function loadPanelWidths() {
  try {
    var vals = JSON.parse(localStorage.getItem('pool_panel_widths'));
    if (vals && vals.leftW) document.getElementById('wide-list').style.width = vals.leftW;
    if (vals && vals.rightW) document.getElementById('wide-right').style.width = vals.rightW;
  } catch(e) {}
  // apply after render
  setTimeout(function() {
    try {
      var vals = JSON.parse(localStorage.getItem('pool_panel_widths'));
      if (vals && vals.leftW) document.getElementById('wide-list').style.width = vals.leftW;
      if (vals && vals.rightW) document.getElementById('wide-right').style.width = vals.rightW;
    } catch(e) {}
  }, 50);
}

// ── init ─────────────────────────────────────────────────────────────
document.addEventListener('keydown', onKeyNav);
window.addEventListener('resize', onResize);

function searchBing(id) {
  var p = getPaperById(parseInt(id, 10));
  if (!p) return;
  var q = encodeURIComponent(p.t);
  window.open('https://www.bing.com/search?q=' + q, '_blank');
}

// clean and index paper data for search
PAPERS.forEach(function(p) {
  p.t = p.title || '';
  p.y = p.year || '?';
  p.st = p.status || 'unreviewed';
  p.a = p.arxiv_id || '';
});

(function init() {
  if (window.innerWidth >= WIDE_BP) renderWide('');
  else renderNarrow('');
})();
</script>
</body>
</html>"""


def get_papers_from_db():
    """Fetch papers from snowcite DB."""
    if not DB.exists():
        return []
    conn = sqlite3.connect(str(DB))
    cur = conn.execute("""
        SELECT p.id, p.source, p.source_id, p.title, p.year, p.authors_json,
               COALESCE(r.status, 'unreviewed') as status
        FROM papers p
        LEFT JOIN reviews r ON r.paper_id = p.id
        WHERE p.source = 'arxiv' AND p.source_id IS NOT NULL AND p.source_id != ''
        ORDER BY p.id
    """)
    rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
    conn.close()
    return rows


def scan_pool(pool_dir):
    """Read what's on disk in the pool directory."""
    papers = []
    arxiv_dir = pool_dir / "arxiv"
    if not arxiv_dir.exists():
        return papers
    for folder in sorted(arxiv_dir.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split("_", 1)
        arxiv_id = parts[0]
        pdfs = list(folder.glob("*.pdf"))
        source_dir = folder / "source"
        figures, tex_files = [], []
        if source_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.pdf"):
                figures.extend(source_dir.rglob(ext))
            tex_files = sorted(source_dir.rglob("*.tex"))
        figures = sorted(figures)
        papers.append({
            "arxiv_id": arxiv_id,
            "folder": folder.name,
            "pdf_path": str(pdfs[0].relative_to(pool_dir)) if pdfs else None,
            "figures": [(str(f.relative_to(pool_dir)), f.suffix.lower()) for f in figures],
            "tex_files": [str(f.relative_to(pool_dir)) for f in tex_files],
        })
    return papers


def read_tex_content(pool_dir, rel_path, max_chars=2000):
    """Read first portion of a .tex file for preview."""
    full_path = pool_dir / rel_path
    if not full_path.exists():
        return ""
    try:
        text = full_path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... (truncated)"
        return html.escape(text)
    except Exception:
        return ""


def build_papers_json(pool_dir, db_papers, disk_info):
    """Build JSON-serializable list of paper data for the client."""
    papers_data = []
    disk_map = {d["arxiv_id"]: d for d in disk_info}

    for p in db_papers:
        source_id = p.get("source_id", "") or ""
        arxiv_id = source_id.split("v")[0] if "v" in source_id else source_id
        title = p["title"]

        disk = disk_map.get(arxiv_id)
        figs = disk["figures"] if disk and disk["figures"] else []
        pdf_path = disk["pdf_path"] if disk else None
        tex_content = ""
        if disk and disk["tex_files"]:
            tex_content = read_tex_content(pool_dir, disk["tex_files"][0])

        papers_data.append({
            "id": p["id"],
            "title": title,
            "year": p.get("year", "?"),
            "status": p.get("status", "unreviewed"),
            "arxiv_id": arxiv_id,
            "pdf": pdf_path,
            "figs": figs,
            "tc": tex_content,
        })

    return papers_data


class PoolHandler(SimpleHTTPRequestHandler):
    """Custom handler serving pool directory + generated index."""

    def __init__(self, *args, **kwargs):
        self.pool_dir = kwargs.pop("pool_dir")
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path in ("/", ""):
            return str(self.pool_dir)
        return str(self.pool_dir / path.lstrip("/"))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_generated_index()
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            super().do_GET()

    def send_generated_index(self):
        db_papers = get_papers_from_db()
        disk_info = scan_pool(self.pool_dir)
        papers_data = build_papers_json(self.pool_dir, db_papers, disk_info)

        page = HTML_TEMPLATE.replace("__PAPERS_JSON__", json.dumps(papers_data, ensure_ascii=False))
        page = page.replace("__POOL_PATH__", html.escape(str(self.pool_dir)))

        self.send_response(200)
        body = page.encode("utf-8")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"  [{self.address_string()}] {args[0]} {args[1]} {args[2]}")


def main():
    parser = argparse.ArgumentParser(description="Paper pool browser server")
    parser.add_argument("--port", type=int, default=8899, help="Port (default: 8899)")
    parser.add_argument("--pool", default=r"F:\文献池", help="Pool directory")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    pool_dir = Path(args.pool)
    if not pool_dir.exists():
        print(f"Error: pool directory not found: {pool_dir}")
        print("Run fetch_papers.py first to populate the pool.")
        sys.exit(1)

    def make_handler(*handler_args):
        return PoolHandler(*handler_args, pool_dir=pool_dir)

    server = HTTPServer(("127.0.0.1", args.port), make_handler)
    url = f"http://127.0.0.1:{args.port}"

    print(f"\n  Paper Pool server started!")
    print(f"  Open in browser: {url}")
    print(f"  Pool directory:  {pool_dir}")
    print(f"  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
