#!/usr/bin/env python3
import argparse, csv, json, re
from pathlib import Path
from collections import defaultdict
from typing import Optional

GID_RE = re.compile(r"^G\d{4}$", re.I)

def is_group_entry(obj) -> Optional[str]:
    # prefer G#### in label_id; else extract from /groups/G#### in label_link
    lid = str(obj.get("label_id", "")).strip()
    if GID_RE.match(lid): return lid.upper()
    link = str(obj.get("label_link", "")).lower()
    m = re.search(r"/groups/(G\d{4})", link, re.I)
    return m.group(1).upper() if m else None

def load_linking_groups(link_path: Path) -> dict[str, set[str]]:
    """Return doc_id -> set(G####) for a linking/*.jsonl split."""
    doc2g = defaultdict(set)
    if not link_path.exists(): return doc2g
    with link_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            gid = is_group_entry(obj)
            if not gid: 
                continue
            doc_id = (obj.get("document") or obj.get("doc_id") or obj.get("id") or "").strip()
            if doc_id:
                doc2g[doc_id].add(gid)
    return doc2g

def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source","url","doc_id","label","text"])
        for r in rows:
            w.writerow(r)
    print(f"[info] wrote {out_csv} ({len(rows)} rows)")

def main():
    ap = argparse.ArgumentParser(description="Export AnnoCTR to CTI CSV (source,url,doc_id,label,text).")
    ap.add_argument("--root", required=True, help="Path that contains AnnoCTR/ (local clone root)")
    ap.add_argument("--out_csv", default="annoctr_bm25_dataset.csv")
    ap.add_argument("--multi", action="store_true", help="Emit one row per (doc,group); default picks first")
    args = ap.parse_args()

    base = Path(args.root) / "AnnoCTR"
    rows = []
    for split in ("train","dev","test"):
        text_dir = base / "text" / split
        link_path = base / "linking" / f"{split}.jsonl"
        doc2g = load_linking_groups(link_path)

        for p in sorted(text_dir.glob("*.txt")):
            doc_id = p.stem
            if doc_id not in doc2g or not doc2g[doc_id]:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            text = re.sub(r"\s+", " ", text).strip()
            gids = sorted(doc2g[doc_id])
            if args.multi:
                for gid in gids:
                    rows.append(("annoctr", str(p), doc_id, gid, text))
            else:
                # single-label baseline: pick deterministic first
                rows.append(("annoctr", str(p), doc_id, gids[0], text))

    if not rows:
        raise SystemExit("[error] No (doc, group) pairs found. Check your AnnoCTR path.")
    write_csv(rows, Path(args.out_csv))

if __name__ == "__main__":
    main()
