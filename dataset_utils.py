#!/usr/bin/env python3
"""
Dataset utilities for CTI CSVs (source,url,doc_id,label,text).
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import sys, csv, gzip

# --- allow very large CSV cells (long 'text' columns) ---
_max = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max)
        break
    except OverflowError:
        _max //= 2  # macOS sometimes needs this backoff

def open_csv_any(path: Path):
    """
    Open a CSV or gzipped CSV safely with UTF-8 encoding.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")

def sanitize_nul_lines(fp):
    """
    Yield lines from a file-like, removing NUL bytes that break csv.reader.
    """
    for line in fp:
        if "\x00" in line:
            line = line.replace("\x00", "")
        yield line


def discover_csvs(train_dir: Optional[str], csvs: Optional[Sequence[str]]) -> List[Path]:
    """
    Collect CSV paths from a directory (recursive) and/or an explicit list.
    Deduplicates and keeps only existing files.
    """
    paths: List[Path] = []
    if train_dir:
        for p in Path(train_dir).rglob("*.csv"):
            paths.append(p.resolve())
    if csvs:
        for c in csvs:
            p = Path(c).resolve()
            if p.suffix.lower() == ".csv":
                paths.append(p)
    out, seen = [], set()
    for p in paths:
        if p.exists() and str(p) not in seen:
            out.append(p); seen.add(str(p))
    return out


def load_labeled_rows(csv_paths: Sequence[Path]) -> Tuple[List[str], List[str], List[str]]:
    """
    Read texts, labels, and doc_ids from multiple CSVs with schema:
    source,url,doc_id,label,text
    """
    texts, labels, ids = [], [], []
    required = {"source", "url", "doc_id", "label", "text"}
    for path in csv_paths:
        with open_csv_any(path) as f:
            r = csv.DictReader(sanitize_nul_lines(f))
            if not r.fieldnames or not required.issubset(r.fieldnames):
                missing = required - set(r.fieldnames or [])
                raise ValueError(f"{path} missing columns: {missing}")
            for row in r:
                lab = (row.get("label") or "").strip()
                txt = (row.get("text") or "").strip()
                did = (row.get("doc_id") or "").strip()
                if not lab or not txt:
                    continue
                labels.append(lab.upper())
                texts.append(txt)
                ids.append(did)
    return texts, labels, ids


def drop_rare_classes(texts: List[str], labels: List[str], min_count: int) -> Tuple[List[str], List[str]]:
    """
    Remove samples whose class frequency is less than `min_count`.
    """
    if min_count <= 1:
        return texts, labels
    cnt = Counter(labels)
    keep = [i for i, y in enumerate(labels) if cnt[y] >= min_count]
    return [texts[i] for i in keep], [labels[i] for i in keep]


def stratified_guard_split(
    X: List[str], y: List[str], test_frac: float = 0.25, seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Per-class split that guarantees at least one training sample per class.
    Classes with only 1 item are kept entirely in train.
    """
    rng = np.random.default_rng(seed)
    by_cls = defaultdict(list)
    for i, lab in enumerate(y):
        by_cls[lab].append(i)

    train_idx, test_idx = [], []
    for _, idxs in by_cls.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        if len(idxs) == 1:
            train_idx.extend(idxs.tolist())
        else:
            n_test = max(1, int(round(len(idxs) * test_frac)))
            n_test = min(n_test, len(idxs) - 1)
            test_idx.extend(idxs[:n_test].tolist())
            train_idx.extend(idxs[n_test:].tolist())

    Xtr = [X[i] for i in train_idx]; ytr = [y[i] for i in train_idx]
    Xte = [X[i] for i in test_idx];  yte = [y[i] for i in test_idx]
    return Xtr, Xte, ytr, yte

def count_labels(csv_paths: Sequence[Path]) -> Counter:
    """
    Count how many rows belong to each label (G-code) across CSVs.

    Args:
        csv_paths: Iterable of CSV files with schema source,url,doc_id,label,text.

    Returns:
        Counter mapping {label -> count}.
    """
    c = Counter()
    for path in csv_paths:
        with open_csv_any(path) as f:
            r = csv.DictReader(sanitize_nul_lines(f))
            for row in r:
                lab = (row.get("label") or "").strip().upper()
                txt = (row.get("text") or "").strip()
                if lab and txt:
                    c[lab] += 1
    return c