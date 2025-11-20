#!/usr/bin/env python3
"""
Sanitize CTI CSVs by redacting actor aliases from text (and optionally from url/doc_id).

Input CSV schema (from build_cti_dataset.py):
    source, url, doc_id, label, text

Features:
- Accepts a folder of CSVs and/or a list of CSV files
- Loads alias map (G#### -> [aliases]) from g2aliases.json
- Optional alias stoplist to ignore ambiguous aliases (e.g., "play")
- Redacts aliases with whole-word, case-insensitive regexes
- Preserves original columns; can optionally blank the 'label' column
- Handles huge cells, weird CSVs with NUL bytes, and .csv.gz

Usage examples:
  python sanitize_csv_aliases.py --in_dir ./cti_datasets --aliases g2aliases.json --out_dir ./sanitized_csvs

******************************
* Redact actor from text for APT Report
******************************
python sanitize_csv_aliases.py --in_csvs ./cti_bm25_test_dataset_APT_Reports.csv --aliases g2aliases.json --out_dir ./sanitized_csvs

"""

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

# ---- allow very large CSV cells (long 'text' columns) ----
_max = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max)
        break
    except OverflowError:
        _max //= 2  # macOS sometimes needs this backoff

RE_WHITESPACE = re.compile(r"\s+")

def open_csv_any(path: Path, mode: str = "r"):
    """Open .csv or .csv.gz in text mode with UTF-8."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt" if "r" in mode else "wt", encoding="utf-8", newline="")
    return path.open("r" if "r" in mode else "w", encoding="utf-8", newline="")

def sanitize_nul_lines(fp: Iterable[str]):
    """Yield lines with NUL bytes removed (csv module chokes on NULs)."""
    for line in fp:
        yield line.replace("\x00", "")

def discover_csvs(in_dir: str = None, in_csvs: Sequence[str] = None) -> List[Path]:
    """Collect CSV paths from directory (recursive) and/or explicit list."""
    out, seen = [], set()
    if in_dir:
        for p in Path(in_dir).rglob("*.csv*"):  # matches .csv and .csv.gz
            p = p.resolve()
            if p.exists() and str(p) not in seen:
                out.append(p); seen.add(str(p))
    if in_csvs:
        for c in in_csvs:
            p = Path(c).resolve()
            if p.exists() and p.suffix in {".csv", ".gz"} and str(p) not in seen:
                out.append(p); seen.add(str(p))
    return out

def load_alias_map(path: str) -> dict:
    """Load G-code -> [aliases] (lowercased) from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for gid, aliases in data.items():
        low = []
        for a in aliases or []:
            a = (a or "").strip().lower()
            if a:
                low.append(a)
        if low:
            out[gid.upper()] = sorted(set(low))
    return out

def load_stoplist(path: str) -> set:
    """Load optional alias stoplist (one lowercase alias per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip().lower() for ln in f if ln.strip()}

def compile_redaction_patterns(g2aliases: dict, stoplist: set = None, alias_min_len: int = 3):
    """
    Build list of (regex, alias) for redaction.
    - Skips stoplisted aliases
    - Skips ultra-short single words (unless multi-word or contains digits)
    - Sorts longer aliases first to avoid partial-overlap issues
    """
    stoplist = stoplist or set()
    pairs = []
    for aliases in g2aliases.values():
        for a in aliases:
            if a in stoplist:
                continue
            keep = (" " in a) or any(ch.isdigit() for ch in a) or len(a) >= alias_min_len
            if not keep:
                continue
            pat = re.compile(rf"(?<!\w){re.escape(a)}(?!\w)", re.IGNORECASE)
            pairs.append((len(a), pat, a))
    pairs.sort(key=lambda t: t[0], reverse=True)
    return [(pat, a) for _, pat, a in pairs]

def redact_text(text: str, patterns, replacement: str) -> str:
    """Apply all alias redactions to text (longest aliases first)."""
    t = text
    for pat, _ in patterns:
        t = pat.sub(replacement, t)
    return t

def process_csv(in_path: Path, out_path: Path, patterns, redact_fields=("text",), strip_label=False, replacement="[REDACTED_ACTOR]"):
    """
    Read one CSV and write a sanitized copy.
    - Redacts in specified fields (default: text)
    - Optionally blanks 'label' for blind prediction
    - Preserves other columns as-is
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_csv_any(in_path, "r") as fin, open_csv_any(out_path, "w") as fout:
        reader = csv.DictReader(sanitize_nul_lines(fin))
        if not reader.fieldnames:
            raise ValueError(f"{in_path} has no header row.")
        # Ensure standard columns exist; if not, we still pass through gracefully
        fieldnames = list(reader.fieldnames)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        n_rows, n_kept = 0, 0
        for row in reader:
            n_rows += 1
            # Redact in chosen fields
            for col in redact_fields:
                if col in row and row[col]:
                    row[col] = redact_text(row[col], patterns, replacement)
            if strip_label and "label" in row:
                row["label"] = ""
            writer.writerow(row); n_kept += 1

    print(f"[info] {in_path.name}: wrote {n_kept}/{n_rows} → {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Redact actor aliases in CSVs produced by build_cti_dataset.py.")
    ap.add_argument("--in_dir", help="Folder containing CSVs (recursively scanned)")
    ap.add_argument("--in_csvs", nargs="*", help="One or more CSV paths")
    ap.add_argument("--aliases", required=True, help="Path to g2aliases.json")
    ap.add_argument("--out_dir", required=True, help="Output folder for sanitized CSVs")
    ap.add_argument("--alias_stoplist", help="Optional alias stoplist file (lowercase, one per line)")
    ap.add_argument("--alias_min_len", type=int, default=3, help="Min length for single-word aliases (unless digits/space)")
    ap.add_argument("--redact_doc_id", action="store_true", help="Also redact in 'doc_id' (if file names leak actors)")
    ap.add_argument("--redact_url", action="store_true", help="Also redact in 'url'")
    ap.add_argument("--strip_label", action="store_true", help="Blank the 'label' column for blind prediction")
    ap.add_argument("--replacement", default="[REDACTED_ACTOR]", help="Replacement token")
    args = ap.parse_args()

    csvs = discover_csvs(args.in_dir, args.in_csvs)
    if not csvs:
        raise SystemExit("No input CSVs found. Provide --in_dir and/or --in_csvs.")

    g2 = load_alias_map(args.aliases)
    stop = load_stoplist(args.alias_stoplist) if args.alias_stoplist else set()
    patterns = compile_redaction_patterns(g2, stoplist=stop, alias_min_len=args.alias_min_len)

    fields = ["text"]
    if args.redact_doc_id:
        fields.append("doc_id")
    if args.redact_url:
        fields.append("url")

    out_dir = Path(args.out_dir)
    for p in csvs:
        rel = p.name  # keep just filename; you could mirror folders if you prefer
        out_path = out_dir / rel
        process_csv(p, out_path, patterns, redact_fields=fields, strip_label=args.strip_label, replacement=args.replacement)

if __name__ == "__main__":
    main()
