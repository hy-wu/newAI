r"""
Download arXiv paper PDFs + TeX source + BibTeX to a local pool directory.

Usage:
    python fetch_papers.py                              # download all unreviewed/approved papers
    python fetch_papers.py --id 24 26 27                # download specific IDs
    python fetch_papers.py --status approved            # only approved papers
    python fetch_papers.py --pool D:\my_papers          # custom pool dir (default: F:\文献池)
    python fetch_papers.py --bib-only                   # generate .bib only, no downloads
    python fetch_papers.py --tex-only 24                # TeX source only, no PDF
"""

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import io
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DB = PROJECT_ROOT / ".snowcite" / "papers.db"

# arXiv URLs 格式
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_EPRINT = "https://arxiv.org/e-print/{arxiv_id}"


def get_papers(paper_ids=None, status=None, non_arxiv_only=False):
    """Query snowcite DB for papers."""
    conn = sqlite3.connect(str(DB))
    query = """
        SELECT p.id, p.source, p.source_id, p.title, p.year, p.authors_json, p.doi, p.venue,
               COALESCE(r.status, 'unreviewed') as status
        FROM papers p
        LEFT JOIN reviews r ON r.paper_id = p.id
        WHERE 1=1
    """
    params = []
    if paper_ids:
        placeholders = ",".join("?" * len(paper_ids))
        query += f" AND p.id IN ({placeholders})"
        params.extend(paper_ids)
    if status:
        query += " AND r.status = ?"
        params.append(status)
    if non_arxiv_only:
        query += " AND (p.source != 'arxiv' OR p.source_id IS NULL OR p.source_id = '')"
    query += " ORDER BY p.id"

    cur = conn.execute(query, params)
    rows = [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
    conn.close()
    return rows


def resolve_arxiv_via_semantic_scholar(doi, title):
    """Query Semantic Scholar by DOI or title to find the arXiv ID."""
    import urllib.parse
    import time

    # Try by DOI first (more precise)
    if doi:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,externalIds"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'snowcite/0.4'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                ext = data.get('externalIds', {})
                arxiv_id = ext.get('ArXiv')
                if arxiv_id:
                    return arxiv_id
        except Exception:
            pass

    # Fallback: search by title
    if title:
        clean = title.strip().replace('\n', ' ').replace('"', ' ')
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(clean)}&limit=5&fields=title,externalIds"
        try:
            time.sleep(0.3)  # rate limit courtesy
            req = urllib.request.Request(url, headers={'User-Agent': 'snowcite/0.4'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                for paper in data.get('data', []):
                    ext = paper.get('externalIds', {})
                    arxiv_id = ext.get('ArXiv')
                    if arxiv_id:
                        return arxiv_id
        except Exception:
            pass

    return None


def update_paper_arxiv(paper_id, arxiv_id):
    """Update a paper's source and source_id to arxiv."""
    conn = sqlite3.connect(str(DB))
    conn.execute("UPDATE papers SET source='arxiv', source_id=? WHERE id=?", (arxiv_id, paper_id))
    conn.commit()
    conn.close()


def sanitize_filename(title):
    """Clean title for use as folder/filename."""
    keep = title.replace(" ", "_").replace(":", "").replace("/", "_")
    keep = "".join(c for c in keep if c.isalnum() or c in "_-.")
    return keep[:80]


def download_file(url, dest, timeout=30):
    """Download a file with progress indicator."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "snowcite/0.3"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        return True, len(data)
    except Exception as e:
        return False, str(e)


async def fetch_paper(args, pool_dir, paper):
    """Download PDF and/or TeX for one paper."""
    pid = paper["id"]
    source = paper["source"]
    source_id = paper["source_id"] or ""
    title = paper["title"]
    year = paper["year"] or 0

    # Skip non-arXiv papers (no PDF/TeX available)
    if source not in ("arxiv",) or not source_id:
        return {"id": pid, "status": "skipped", "reason": f"not from arXiv (source={source})"}

    # Extract arXiv ID (strip version suffix for folder naming)
    arxiv_id = source_id.split("v")[0] if "v" in source_id else source_id

    # Paper folder: F:\文献池\arxiv\2405.04620_TitleWords\
    folder_name = f"{arxiv_id}_{sanitize_filename(title)[:40]}"
    paper_dir = pool_dir / "arxiv" / folder_name
    paper_dir.mkdir(parents=True, exist_ok=True)

    result = {"id": pid, "arxiv_id": arxiv_id, "dir": str(paper_dir), "files": []}

    # 1. Download PDF
    if not args.tex_only:
        pdf_path = paper_dir / f"{arxiv_id}.pdf"
        if pdf_path.exists():
            result["files"].append({"type": "pdf", "status": "exists", "path": str(pdf_path)})
        else:
            pdf_url = ARXIV_PDF.format(arxiv_id=arxiv_id)
            ok, info = download_file(pdf_url, pdf_path)
            if ok:
                result["files"].append({"type": "pdf", "status": "downloaded", "size": info, "path": str(pdf_path)})
            else:
                result["files"].append({"type": "pdf", "status": "failed", "error": info})

    # 2. Download TeX source (arXiv e-print = tar.gz with .tex files)
    if not args.no_tex:
        tex_path = paper_dir / f"{arxiv_id}.tar.gz"
        if tex_path.exists():
            result["files"].append({"type": "tex_source", "status": "exists", "path": str(tex_path)})
        else:
            eprint_url = ARXIV_EPRINT.format(arxiv_id=arxiv_id)
            ok, info = download_file(eprint_url, tex_path)
            if ok:
                result["files"].append({"type": "tex_source", "status": "downloaded", "size": info, "path": str(tex_path)})
                # Extract to see what's inside
                extract_dir = paper_dir / "source"
                extract_dir.mkdir(exist_ok=True)
                try:
                    with tarfile.open(tex_path, "r:gz") as tar:
                        tar.extractall(path=extract_dir, filter='data')
                    tex_files = list(extract_dir.rglob("*.tex"))
                    fig_files = list(extract_dir.rglob("*.pdf")) + list(extract_dir.rglob("*.png")) + list(extract_dir.rglob("*.jpg"))
                    result["files"].append({
                        "type": "tex_extracted",
                        "tex_count": len(tex_files),
                        "fig_count": len(fig_files),
                        "extract_dir": str(extract_dir),
                    })
                except Exception as e:
                    result["files"].append({"type": "tex_extract_error", "error": str(e)})
            else:
                result["files"].append({"type": "tex_source", "status": "failed", "error": info})

    # 3. Save BibTeX entry
    authors = json.loads(paper.get("authors_json", "[]") or "[]")
    author_str = " and ".join(authors) if authors else "{Anonymous}"
    bib = f"@article{{{arxiv_id},\n"
    bib += f"  author = {{{author_str}}},\n"
    bib += f"  title = {{{title}}},\n"
    bib += f"  year = {{{year}}},\n"
    if paper.get("doi"):
        bib += f"  doi = {{{paper['doi']}}},\n"
    if paper.get("venue"):
        bib += f"  journal = {{{paper['venue']}}},\n"
    bib += f"  archivePrefix = {{arXiv}},\n"
    bib += f"  eprint = {{{arxiv_id}}}\n"
    bib += "}\n"

    bib_path = paper_dir / f"{arxiv_id}.bib"
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(bib)
    result["files"].append({"type": "bib", "path": str(bib_path)})

    return result


def generate_collection_bib(pool_dir, papers):
    """Generate one big BibTeX file for all papers."""
    entries = []
    for p in papers:
        source_id = p.get("source_id", "") or ""
        arxiv_id = source_id.split("v")[0] if "v" in source_id else source_id
        if not arxiv_id:
            continue
        authors = json.loads(p.get("authors_json", "[]") or "[]")
        author_str = " and ".join(authors) if authors else "{Anonymous}"
        bib = f"@article{{{arxiv_id},\n"
        bib += f"  author = {{{author_str}}},\n"
        bib += f"  title = {{{p['title']}}},\n"
        bib += f"  year = {{{p['year']}}},\n"
        if p.get("doi"):
            bib += f"  doi = {{{p['doi']}}},\n"
        if p.get("venue"):
            bib += f"  journal = {{{p['venue']}}},\n"
        bib += f"  archivePrefix = {{arXiv}},\n"
        bib += f"  eprint = {{{arxiv_id}}}\n"
        bib += "}\n"
        entries.append(bib)

    bib_path = pool_dir / "all_references.bib"
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))
    return bib_path


async def main():
    parser = argparse.ArgumentParser(description="Download arXiv papers to local pool")
    parser.add_argument("--id", nargs="+", type=int, help="Paper IDs to fetch")
    parser.add_argument("--status", help="Filter by review status")
    parser.add_argument("--pool", default=r"F:\文献池", help="Pool directory (default: F:\\文献池)")
    parser.add_argument("--bib-only", action="store_true", help="Only generate BibTeX, skip downloads")
    parser.add_argument("--tex-only", action="store_true", help="Only download TeX source, not PDF")
    parser.add_argument("--no-tex", action="store_true", help="Skip TeX source download")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--resolve-arxiv", action="store_true",
        help="Look up arXiv IDs for non-arxiv papers via Semantic Scholar, then download")
    args = parser.parse_args()

    pool_dir = Path(args.pool)
    if not args.dry_run:
        pool_dir.mkdir(parents=True, exist_ok=True)

    # Get papers
    if args.resolve_arxiv:
        papers = get_papers(paper_ids=args.id, status=args.status, non_arxiv_only=True)
        if not papers:
            print("All papers already have arXiv IDs. Nothing to resolve.")
            return

        print(f"Found {len(papers)} non-arxiv papers to resolve\n")
        resolved = 0
        for p in papers:
            print(f"  [{p['id']}] {p['title'][:60]}...", end=" ")
            sys.stdout.flush()
            arxiv_id = resolve_arxiv_via_semantic_scholar(p.get('doi'), p['title'])
            if arxiv_id:
                update_paper_arxiv(p['id'], arxiv_id)
                print(f"-> arXiv:{arxiv_id}")
                resolved += 1
            else:
                print("not found")
        print(f"\nResolved {resolved}/{len(papers)} papers to arXiv. Re-run without --resolve-arxiv to download.")

        # Re-fetch for download if we also want to download
        if not args.dry_run and resolved > 0:
            papers = get_papers(paper_ids=args.id, status=args.status)
        else:
            return
    else:
        papers = get_papers(paper_ids=args.id, status=args.status)

    if not papers:
        print("No papers found matching criteria.")
        return

    print(f"\nFound {len(papers)} papers to fetch\n")
    print(f"{'ID':>4}  {'Status':>10}  Title")
    print("-" * 70)
    for p in papers:
        print(f"  {p['id']:>3}  {p['status']:>10}  {p['title'][:55]}")
    print()

    if args.dry_run:
        return

    if args.bib_only:
        bib = generate_collection_bib(pool_dir, papers)
        print(f"BibTeX written to: {bib}")
        return

    # Download each paper
    for paper in papers:
        print(f"\n{'='*60}")
        print(f"  Paper [{paper['id']}]: {paper['title'][:50]}")
        print(f"{'='*60}")

        result = await fetch_paper(args, pool_dir, paper)

        for f in result.get("files", []):
            t = f["type"]
            if t == "pdf":
                if f["status"] == "downloaded":
                    print(f"  [PDF]   Downloaded ({f['size']/1024:.0f} KB): {f['path']}")
                elif f["status"] == "exists":
                    print(f"  [PDF]   Already exists: {f['path']}")
                elif f["status"] == "failed":
                    print(f"  [PDF]   Failed: {f.get('error', '')}")
            elif t == "tex_source":
                if f["status"] == "downloaded":
                    print(f"  [TEX]   Downloaded ({f['size']/1024:.0f} KB): {f['path']}")
                elif f["status"] == "exists":
                    print(f"  [TEX]   Already exists: {f['path']}")
                elif f["status"] == "failed":
                    print(f"  [TEX]   Failed: {f.get('error', '')}")
            elif t == "tex_extracted":
                print(f"  [TEX]   Extracted: {f['tex_count']} .tex files, {f['fig_count']} figures")
            elif t == "bib":
                print(f"  [BIB]   Saved: {f['path']}")

    # Generate collection BibTeX
    bib = generate_collection_bib(pool_dir, papers)
    print(f"\n  [BIB]   Collection BibTeX: {bib}")

    # Generate index.html for easy browsing
    index_path = pool_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>文献池</title></head><body>\n")
        f.write(f"<h1>文献池 ({len(papers)} papers)</h1>\n<ul>\n")
        for p in papers:
            source_id = p.get("source_id", "") or ""
            arxiv_id = source_id.split("v")[0] if "v" in source_id else source_id
            authors = json.loads(p.get("authors_json", "[]") or "[]")
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            title_esc = p["title"].replace("<", "&lt;").replace(">", "&gt;")

            # Build links
            pdf_link = f"arxiv/{arxiv_id}_{sanitize_filename(p['title'])[:40]}/{arxiv_id}.pdf"
            tex_link = f"https://arxiv.org/e-print/{arxiv_id}"
            abs_link = f"https://arxiv.org/abs/{arxiv_id}"

            f.write(f"  <li>\n")
            f.write(f"    <b>{title_esc}</b><br>\n")
            f.write(f"    {author_str} ({p['year']})<br>\n")
            f.write(f"    <a href='{pdf_link}'>PDF</a> | ")
            f.write(f"    <a href='{tex_link}'>TeX Source</a> | ")
            f.write(f"    <a href='{abs_link}'>arXiv</a>\n")
            f.write(f"  </li>\n\n")
        f.write("</ul></body></html>\n")
    print(f"  [HTML]  Browser index: {index_path}")

    print(f"\nDone! Pool directory: {pool_dir}")


if __name__ == "__main__":
    # Handle Windows event loop policy for asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
