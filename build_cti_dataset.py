#!/usr/bin/env python3

"""
Sample Useage

python build_cti_dataset.py \
  --aliases g2aliases.json \
  --pdf_root APT_CyberCriminal_Campagin_Collections \
  --out_csv cti_bm25_dataset_apt_cybecriminal_campaigns.csv \
  --min_chars 600 --min_hits 1

python build_cti_dataset.py --aliases g2aliases.json --pdf_root mitre_apt_ref_reports --out_csv cti_bm25_dataset_mitre_refs.csv --min_chars 600 --min_hits 1

********************************
Build Dataset CTI-HAL Reports
********************************
python build_cti_dataset.py --aliases g2aliases.json --pdf_root ./reports --out_csv cti_bm25_dataset_cti-hal.csv --min_chars 600 --min_hits 1

********************************
Build Dataset from RSS Feeds
********************************
python build_cti_dataset.py --aliases g2aliases.json --out_csv cti_bm25_dataset_feeds.csv --min_chars 600 --min_hits 1

********************************
Build Test Dataset from APT Reports
********************************
python build_cti_dataset.py --aliases g2aliases.json --pdf_root ./APT\ Reports --out_csv cti_bm25_test_dataset_APT_Reports.csv --min_chars 600 --min_hits 1

"""


import argparse, csv, html, json, re, time, hashlib, sys
import feedparser, requests
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text
import time, math, requests, feedparser

HEADERS = {"User-Agent": "cti-curator/0.2 (+research)"}

def get_feed_bytes(url: str, timeout: int = 20, retries: int = 3, backoff: float = 0.6) -> bytes:
    """
    Fetch RSS/Atom with requests so we control timeout/retries.
    Returns raw bytes suitable for feedparser.parse().
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
        except (requests.Timeout, requests.ConnectionError):
            pass
        # exponential backoff
        sleep_s = backoff * (2 ** attempt)
        time.sleep(sleep_s)
    return b""


def load_g2aliases(path: str) -> Dict[str, List[str]]:
    g2 = json.loads(Path(path).read_text(encoding="utf-8"))
    return {gid.upper(): sorted({a.strip().lower() for a in aliases if a.strip()})
            for gid, aliases in g2.items()}

def compile_alias_patterns(g2aliases: Dict[str, List[str]]):
    """
    Compile regex patterns for each alias in the G-code alias map.

    Each alias is converted into a case-insensitive regex that matches
    whole words (using word-boundary lookarounds) to avoid partial matches.
    Longer aliases are sorted first so more specific patterns are checked
    before shorter, overlapping ones.

    Returns:
        List of (compiled_pattern, alias, gid) tuples.
    """
    pairs = []
    for gid, aliases in g2aliases.items():
        for a in aliases:
            pat = re.compile(rf"(?<!\w){re.escape(a)}(?!\w)", re.IGNORECASE)
            pairs.append((len(a), pat, a, gid))
    # longer aliases first
    pairs.sort(key=lambda t: t[0], reverse=True)
    return [(pat, a, gid) for _, pat, a, gid in pairs]

def fetch(url: str, timeout=25) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code != 200: return None
        return r.text
    except Exception:
        return None

def extract_text(html_str: str) -> str:
    soup = BeautifulSoup(html_str, "html.parser")
    for tag in soup(["script","style","nav","header","footer","aside"]): tag.decompose()
    node = soup.find("article") or soup.body or soup
    txt = node.get_text(separator=" ", strip=True)
    txt = html.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def _resolve_entry_url(entry) -> Optional[str]:
    # Prefer the common 'link' field
    url = entry.get("link")
    if not url:
        # Fall back to other typical fields seen in some feeds
        url = entry.get("id") or entry.get("href")
        if not url and isinstance(entry.get("links"), list) and entry["links"]:
            url = entry["links"][0].get("href")
    return url


from urllib.parse import urlparse

def harvest_feeds(feeds, min_chars: int = 400, delay: float = 0.4,
                  feed_timeout: int = 20, feed_retries: int = 3) -> list[tuple[str,str,str]]:
    """
    Parse a list of RSS/Atom feeds and return (source, url, text) tuples.
    Uses requests for fetching (with timeout/retries) to avoid urllib timeouts.
    """
    rows = []
    for feed in feeds or []:
        raw = get_feed_bytes(feed, timeout=feed_timeout, retries=feed_retries)
        if not raw:
            print(f"[warn] feed timeout/empty: {feed}")
            continue
        fd = feedparser.parse(raw)
        for e in fd.entries:
            # resolve URL (handles list-of-dicts 'link' cases too)
            url = None
            link = e.get("link")
            if isinstance(link, str):
                url = link
            elif isinstance(link, list) and link:
                url = link[0].get("href")
            if not url:
                # try other common fields
                url = e.get("id") or (e.get("links", [{}])[0].get("href") if e.get("links") else None)
            if not url:
                continue

            html_str = fetch(url)  # your existing requests-based page fetcher
            if not html_str:
                continue
            txt = extract_text(html_str)
            if len(txt) < min_chars:
                continue
            src = urlparse(url).netloc or "feed"
            rows.append((src, url, txt))
            time.sleep(delay)
    return rows


def harvest_local(folder: str, min_chars=400) -> List[Tuple[str,str,str]]:
    rows = []
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() not in {".txt",".md"}: continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) < min_chars: continue
        rows.append(("local", str(p), txt))
    return rows

def harvest_pdfs(root: str, min_chars: int = 400) -> list[tuple[str, str, str]]:
    """
    Walk a root directory recursively and extract text from all PDFs.

    Expected layout (any depth is fine):
        root/
          source_a/
            file1.pdf
            file2.pdf
          source_b/
            subdir/
              file3.pdf
          ...

    For each PDF:
      - extract text with pdfminer.six,
      - whitespace-normalize,
      - keep only documents with at least `min_chars` characters,
      - set `source` to the top-level subfolder name under `root`
        (falls back to 'local-pdf' if not applicable).

    Returns:
      List of (source, path, text) tuples.
    """
    rows: list[tuple[str, str, str]] = []
    root_path = Path(root).resolve()
    for pdf_path in root_path.rglob("*.pdf"):
        try:
            text = pdf_extract_text(str(pdf_path)) or ""
        except Exception:
            continue  # skip unreadable PDFs

        # normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < min_chars:
            continue

        # pick source as the first directory under root (folder-of-folders)
        try:
            rel = pdf_path.relative_to(root_path)
            source = rel.parts[0] if len(rel.parts) >= 2 else "local-pdf"
        except Exception:
            source = "local-pdf"

        rows.append((source, str(pdf_path), text))
    return rows


def label_with_aliases(text: str, patterns, min_hits: int = 1, dominance_ratio: float = 1.5):
    """
    Assign the most likely actor label (G-code) to a document using alias matches.

    This function scans a text for known actor aliases (precompiled regex patterns)
    and counts how many times each actor (G-code) appears. The actor with the
    highest count is selected as the document label if it meets the following criteria:

      1. The actor has at least `min_hits` total alias matches.
      2. The actor's count is `dominance_ratio` times greater than the next highest
         actor count (to filter out ambiguous documents mentioning multiple actors).

    The dominance ratio is a heuristic confidence threshold that improves label
    precision at the cost of recall: higher ratios produce fewer but cleaner labels.

    Args:
        text: The full text of the document to be labeled.
        patterns: A list of tuples (regex_pattern, alias, gid), as built by
                  `compile_alias_patterns()`.
        min_hits: Minimum number of alias matches required for the top actor
                  to qualify as a label (default = 1).
        dominance_ratio: Required ratio between the top actor's count and the
                         next highest count to accept the label (default = 1.5).

    Returns:
        tuple:
            - label (str | None): The dominant actor's G-code if a confident label
              is found, otherwise None.
            - counts (collections.Counter): A counter of all alias hits per actor
              within the document.

    Example:
        >>> text = "APT29, also known as Cozy Bear, targeted several agencies."
        >>> gid, counts = label_with_aliases(text, patterns)
        >>> gid
        'G0016'
        >>> counts
        Counter({'G0016': 2})
    """
    t = text.lower()
    from collections import Counter, defaultdict
    counts = Counter()
    longest = defaultdict(int)

    for pat, alias, gid in patterns:
        for _ in pat.finditer(t):
            counts[gid] += 1
            longest[gid] = max(longest[gid], len(alias))

    if not counts:
        return None, counts

    best_gid, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count < min_hits:
        return None, counts

    others = [c for g, c in counts.items() if g != best_gid]
    if others:
        ratio = best_count / max(others)
        if ratio < dominance_ratio:
            return None, counts  # too ambiguous

    return best_gid, counts


def write_csv(rows: List[Tuple[str,str,str,str]], out_csv: str):
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["source","url","doc_id","label","text"])
        for src, url, gid, txt in rows:
            doc_id = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:12]
            w.writerow([src, url, doc_id, gid, txt])
    print(f"[info] wrote {out_csv} ({len(rows)} rows)")

def main():
    ap = argparse.ArgumentParser(description="Build weak-labeled CTI dataset using alias map.")
    ap.add_argument("--aliases", default="g2aliases.json", help="g2aliases.json (G#### -> [aliases])")
    ap.add_argument("--out_csv", default="cti_bm25_dataset.csv")
    ap.add_argument("--feeds", nargs="*", default=[

# "https://www.mandiant.com/resources/blog/rss.xml",
# "https://www.microsoft.com/en-us/security/blog/feed/",
# "https://www.crowdstrike.com/blog/feed/",
# "https://unit42.paloaltonetworks.com/feed/",
# "https://blog.talosintelligence.com/feed/",
# "https://securelist.com/feed/",
# "https://www.sentinelone.com/blog/feed/",
# "https://www.welivesecurity.com/feed/",
# "https://nakedsecurity.sophos.com/feed/",
# "https://research.checkpoint.com/feed/",
# "https://www.mcafee.com/blogs/feed/",
# "https://blog.malwarebytes.com/feed/",
# "https://therecord.media/feed/",
# "https://www.rapid7.com/blog/rss/",
                                                    ], help="RSS/Atom feed URLs")
    ap.add_argument("--feed_timeout", type=int, default=20)
    ap.add_argument("--feed_retries", type=int, default=3)
    ap.add_argument("--local_dir", help="Folder of .txt/.md to include at CTI docs")
    ap.add_argument("--pdf_root", type=str, help="Root folder containing subfolders of PDFs to ingest")
    ap.add_argument("--min_chars", type=int, default=400)
    ap.add_argument("--min_hits", type=int, default=1, help="Min hits of the top gid to accept")
    ap.add_argument("--dominance_ratio", type=float, default=3, help="Require top actor hits ≥ ratio × next best")

    args = ap.parse_args()

    g2aliases = load_g2aliases(args.aliases)
    patterns = compile_alias_patterns(g2aliases)
    print(f"[info] groups={len(g2aliases)} aliases={sum(len(v) for v in g2aliases.values())}")

    rows = []
    if args.feeds:
        rows += harvest_feeds(args.feeds, min_chars=args.min_chars)
    if args.local_dir:
        rows += harvest_local(args.local_dir, min_chars=args.min_chars)
    print(f"[info] collected={len(rows)}")

    if args.pdf_root:
        print(f"[info] harvesting PDFs from: {args.pdf_root}")
        rows += harvest_pdfs(args.pdf_root, min_chars=args.min_chars)


    labeled = []
    for src, url, txt in rows:
        gid, counts = label_with_aliases(txt, patterns, min_hits= args.min_hits, dominance_ratio=args.dominance_ratio)
        if gid is None or counts.get(gid, 0) < args.min_hits: 
            continue
        labeled.append((src, url, gid, txt))
    print(f"[info] labeled={len(labeled)}")

    if not labeled:
        sys.exit("[error] No labeled docs—add feeds/local_dir or lower thresholds.")
    write_csv(labeled, args.out_csv)

if __name__ == "__main__":
    main()


