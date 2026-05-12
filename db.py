"""Snowcite 資料庫查詢工具。

用法：
    python db.py                     列出所有論文
    python db.py --status approved   按狀態篩選
    python db.py --id 24             查看單篇詳情
    python db.py --count             只看統計
    python db.py --sql "SELECT * FROM papers WHERE year >= 2025"
"""

import sqlite3
import argparse
import json
import os
import textwrap

DB = os.path.join(os.path.dirname(__file__), ".snowcite", "papers.db")

# Use ASCII-safe symbols for Windows GBK terminals
COLORS = {
    "unreviewed": "[ ]",
    "approved": "[OK]",
    "rejected": "[NO]",
    "maybe": "[?]",
}


def list_papers(status=None, limit=100):
    conn = sqlite3.connect(DB)
    query = """
        SELECT p.id, p.year, p.title, COALESCE(r.status, 'unreviewed') as status, p.venue, p.source
        FROM papers p
        LEFT JOIN reviews r ON r.paper_id = p.id
    """
    params = []
    if status:
        query += " WHERE r.status = ?"
        params.append(status)
    query += " ORDER BY p.id LIMIT ?"
    params.append(limit)

    cur = conn.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    print(f"Total: {len(rows)} papers\n")
    print(f"{'':>3} {'Yr':>4} {'Status':>10}  Title")
    print("-" * 80)
    for r in rows:
        icon = COLORS.get(r[3], "?")
        t = r[2].encode("utf-8", errors="replace").decode("utf-8")
        print(f"{icon} {r[0]:>3} ({r[1]}) {r[3]:>10}  {t[:60]}")


def show_paper(paper_id):
    conn = sqlite3.connect(DB)
    cur = conn.execute(
        """SELECT p.*, COALESCE(r.status, 'unreviewed') as status, r.reason, r.note
           FROM papers p
           LEFT JOIN reviews r ON r.paper_id = p.id
           WHERE p.id = ?""",
        (paper_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        print(f"Paper ID {paper_id} not found")
        return

    cols = [d[0] for d in cur.description]
    data = dict(zip(cols, row))

    print(f"\n{'='*60}")
    print(f"  [{data['id']}] {data['title']}")
    print(f"{'='*60}")
    print(f"  Year:   {data['year']}")
    print(f"  Source: {data['source']} ({data.get('source_id', '')})")
    print(f"  DOI:    {data.get('doi', '') or 'N/A'}")
    print(f"  Venue:  {data.get('venue', '') or 'N/A'}")
    print(f"  Status: {data.get('status', 'unreviewed')}")
    if data.get("reason"):
        print(f"  Reason: {data['reason']}")
    authors = json.loads(data.get("authors_json", "[]") or "[]")
    if authors:
        print(f"   Authors: {', '.join(authors)}")
    abstract = data.get("abstract", "")
    if abstract:
        wrapped = textwrap.fill(abstract, width=70)
        print(f"\n   Abstract: {abstract}")
    print()


def count():
    conn = sqlite3.connect(DB)
    cur = conn.execute("SELECT COUNT(*) FROM papers")
    total = cur.fetchone()[0]
    cur = conn.execute("SELECT status, COUNT(*) FROM reviews GROUP BY status")
    counts = dict(cur.fetchall())
    conn.close()

    print(f"Total papers: {total}")
    for s in ["approved", "maybe", "rejected", "unreviewed"]:
        n = counts.get(s, 0)
        icon = COLORS.get(s, "?")
        print(f"  {icon} {s}: {n}")


def raw_sql(sql):
    conn = sqlite3.connect(DB)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        print(f"  {'  '.join(cols)}")
        print("-" * 60)
        for r in rows[:50]:
            print(f"  {'  '.join(str(v)[:40] for v in r)}")
        print(f"\nTotal: {len(rows)} rows" if len(rows) > 50 else "")
    except Exception as e:
        print(f"SQL error: {e}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snowcite paper query tool")
    parser.add_argument("--status", help="Filter by status (approved/rejected/unreviewed/maybe)")
    parser.add_argument("--id", type=int, help="Show single paper details")
    parser.add_argument("--count", action="store_true", help="Show counts only")
    parser.add_argument("--sql", help="Run raw SQL query")
    parser.add_argument("--limit", type=int, default=100, help="Max rows to show")
    args = parser.parse_args()

    if args.count:
        count()
    elif args.id:
        show_paper(args.id)
    elif args.sql:
        raw_sql(args.sql)
    elif args.status:
        list_papers(status=args.status, limit=args.limit)
    else:
        list_papers(limit=args.limit)
