#!/usr/bin/env python3
import os, re, json, glob, argparse
from collections import defaultdict, Counter
from typing import List, Tuple
import numpy as np

# -------- tokenization tuned for CTI --------
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:\\-]+")

def tok(s: str) -> List[str]:
    """
    Tokenize CTI text into case-folded alphanumeric-ish tokens.

    Uses a permissive regex that keeps terms like APT29, G0032, file paths,
    and indicators-ish strings intact. Lowercases before matching.

    Args:
        s: Raw text.

    Returns:
        List of tokens (strings).
    """
    return TOKEN_RE.findall(s.lower())

# -------- find ATT&CK groups in linking --------
GROUP_ID_RE = re.compile(r"^G\d{4}$", re.I)

def is_group_rec(obj: dict) -> Tuple[bool, str]:
    """
    Decide whether a linking.jsonl row refers to an ATT&CK Group and normalize its label.

    Heuristics:
      - Prefer `label_id` if it's a G####.
      - Else, if `label_link` contains `/groups/G####`, extract that.
      - Else, fall back to `label_title`/`label` (rare) or 'UNKNOWN_GROUP'.

    Args:
        obj: A single JSON object from AnnoCTR linking/*.jsonl.

    Returns:
        (is_group, canonical_label): bool and chosen label string (usually G####).
    """
    lid = str(obj.get("label_id", "")).strip()
    link = str(obj.get("label_link", "")).strip().lower()
    title = str(obj.get("label_title") or obj.get("label") or "").strip()
    if GROUP_ID_RE.match(lid):
        return True, lid.upper()
    if "/groups/" in link:
        m = re.search(r"/groups/(G\d{4})", link, re.I)
        if m:
            return True, m.group(1).upper()
        # fallback to title if no ID in URL
        return True, title if title else "UNKNOWN_GROUP"
    return False, ""

def load_texts_and_group_labels(root: str):
    """
    Load AnnoCTR texts and per-document Group labels.

    Walks AnnoCTR/text/{train,dev,test} and aligns with linking/{split}.jsonl.
    If a doc maps to multiple groups, picks a deterministic single label
    (alphabetical first) to keep the baseline single-label.

    Args:
        root: Path whose child is 'AnnoCTR/'.

    Returns:
        texts: List[str] document bodies.
        labels: List[str] normalized labels (usually G####).
    """
    base = os.path.join(root, "AnnoCTR")
    texts, labels = [], []
    for split in ("train", "dev", "test"):
        text_dir = os.path.join(base, "text", split)
        link_path = os.path.join(base, "linking", f"{split}.jsonl")

        doc2groups = defaultdict(set)
        if os.path.exists(link_path):
            with open(link_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    doc_id = str(obj.get("document") or obj.get("doc_id") or obj.get("id") or "").strip()
                    ok, lab = is_group_rec(obj)
                    if doc_id and ok and lab:
                        doc2groups[doc_id].add(lab.upper())

        for p in glob.glob(os.path.join(text_dir, "*.txt")):
            doc_id = os.path.splitext(os.path.basename(p))[0]
            if doc_id not in doc2groups or len(doc2groups[doc_id]) == 0:
                continue
            with open(p, "r", encoding="utf-8") as fh:
                t = fh.read()
            # choose one label deterministically for single-label baseline
            lab = sorted(doc2groups[doc_id])[0]
            texts.append(t)
            labels.append(lab)
    return texts, labels

# -------- safe split to avoid unseen-in-train classes in test --------
def safe_label_aware_split(X, y, test_frac=0.3, seed=42):
    """
    Stratified-ish split by class with a guard so every class has train coverage.

    For each class, hold out up to round(len * test_frac) examples for test,
    but never all of them (i.e., keep at least one train example per class).
    Classes with a single item are forced into train.

    Args:
        X: Sequence of inputs (texts).
        y: Sequence of labels.
        test_frac: Desired per-class test fraction.
        seed: RNG seed for reproducibility.

    Returns:
        Xtr, Xte, ytr, yte: Train/test partitions.
    """
    rng = np.random.default_rng(seed)
    by_cls = defaultdict(list)
    for i, lab in enumerate(y):
        by_cls[lab].append(i)
    train_idx, test_idx = [], []
    for _, idxs in by_cls.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        if len(idxs) == 1:
            train_idx.extend(idxs.tolist())
        else:
            n_test = max(1, int(round(len(idxs)*test_frac)))
            n_test = min(n_test, len(idxs)-1)
            test_idx.extend(idxs[:n_test].tolist())
            train_idx.extend(idxs[n_test:].tolist())
    Xtr = [X[i] for i in train_idx]; ytr = [y[i] for i in train_idx]
    Xte = [X[i] for i in test_idx];  yte = [y[i] for i in test_idx]
    return Xtr, Xte, ytr, yte

# -------- BM25 classifiers --------
def bm25_knn_predict(train_texts, train_labels, test_texts, k=3):
    """
    BM25 k-NN classifier: vote among the top-k most similar train docs.

    Pipeline:
      - Tokenize all train docs, build BM25Okapi on the train corpus.
      - For each test doc, score against train; collect top-k labels.
      - Majority vote; if tie, choose label with largest sum of BM25 scores.

    Args:
        train_texts: List of training documents.
        train_labels: List of training labels (aligned with train_texts).
        test_texts: List of documents to classify.
        k: Number of neighbors to vote.

    Returns:
        List[str]: Predicted labels for each test document.
    """
    from rank_bm25 import BM25Okapi
    corpus_tok = [tok(t) for t in train_texts]
    bm25 = BM25Okapi(corpus_tok)
    preds = []
    for t in test_texts:
        q = tok(t)
        scores = bm25.get_scores(q)
        idx = np.argsort(scores)[::-1][:k]
        top_labs = [train_labels[i] for i in idx]
        # majority vote; tiebreak by sum of scores
        counts = Counter(top_labs)
        best = max(counts.values())
        tied = [lab for lab,c in counts.items() if c==best]
        if len(tied)==1:
            preds.append(tied[0])
        else:
            sums = {lab: float(np.sum([scores[i] for i in idx if train_labels[i]==lab])) for lab in tied}
            preds.append(max(sums.items(), key=lambda kv: kv[1])[0])
    return preds

def bm25_classsum_predict(train_texts, train_labels, test_texts):
    """
    BM25 class-sum classifier: sum BM25 scores over all train docs per class.

    For each test doc:
      - Score similarity to every train doc (BM25Okapi).
      - Aggregate by class label (sum of scores).
      - Predict the label with the largest aggregate score.

    Args:
        train_texts: List of training documents.
        train_labels: List of training labels (aligned).
        test_texts: List of documents to classify.

    Returns:
        List[str]: Predicted labels.
    """
    from rank_bm25 import BM25Okapi
    corpus_tok = [tok(t) for t in train_texts]
    bm25 = BM25Okapi(corpus_tok)
    idx_by_lab = defaultdict(list)
    for i, lab in enumerate(train_labels):
        idx_by_lab[lab].append(i)
    labs = sorted(idx_by_lab)
    preds = []
    for t in test_texts:
        q = tok(t)
        scores = bm25.get_scores(q)
        best_lab, best_score = None, -1.0
        for lab in labs:
            s = float(np.sum(scores[idx_by_lab[lab]]))
            if s > best_score:
                best_lab, best_score = lab, s
        preds.append(best_lab)
    return preds

# -------- metrics --------
from sklearn.metrics import classification_report, f1_score

def evaluate(y_true, y_pred, title):
    """
    Print a compact per-class report and macro-F1.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.
        title: Header for the block.
    """
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))

# -------- main --------
def main():
    """
    Train & evaluate BM25 baselines on AnnoCTR:
      - Loads texts and labels (ATT&CK groups) from local AnnoCTR clone.
      - Drops ultra-rare classes if --min_count > 1.
      - Makes a safe per-class split.
      - Evaluates BM25 k-NN and class-sum variants.

    CLI:
      --root       Path containing AnnoCTR/
      --min_count  Minimum class frequency to keep (default 1)
      --test_frac  Per-class test fraction (default 0.3)
      --k          k for k-NN (default 3)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./annoctr", help="Path that contains AnnoCTR/...")
    ap.add_argument("--min_count", type=int, default=1, help="Drop classes with < min_count docs")
    ap.add_argument("--test_frac", type=float, default=0.3, help="Fraction for test split per class")
    ap.add_argument("--k", type=int, default=3, help="k for k-NN BM25")
    args = ap.parse_args()

    texts, labels = load_texts_and_group_labels(args.root)
    if not texts:
        raise SystemExit("No labeled (group) docs found. Check linking/*.jsonl for G#### or /groups/.")

    # normalize labels
    labels = [lab.upper() for lab in labels]

    # optional: drop ultra-rare classes (stabilizes tiny-data)
    if args.min_count > 1:
        cnt = Counter(labels)
        keep = [i for i,lab in enumerate(labels) if cnt[lab] >= args.min_count]
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]

    print(f"Total docs: {len(texts)} | Classes: {len(set(labels))} | Min count kept: {args.min_count}")
    print("Top 10 class counts:", Counter(labels).most_common(10))

    Xtr, Xte, ytr, yte = safe_label_aware_split(texts, labels, test_frac=args.test_frac)
    print("Train size:", len(ytr), "| Test size:", len(yte))

    y_pred_knn = bm25_knn_predict(Xtr, ytr, Xte, k=args.k)
    evaluate(yte, y_pred_knn, f"BM25 k-NN (k={args.k})")

    y_pred_sum = bm25_classsum_predict(Xtr, ytr, Xte)
    evaluate(yte, y_pred_sum, "BM25 Class-Sum")

if __name__ == "__main__":
    main()
