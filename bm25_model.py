#!/usr/bin/env python3
"""
Reusable BM25 text classifier.

- Pure model code: fit / predict / save / load
"""

import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

# Permissive CTI-friendly tokenizer (keeps APT29, G0032, paths, URLs)
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:\\-]+")

def default_tokenize(text: str) -> List[str]:
    """Lowercase and split text into analysis-friendly tokens."""
    return TOKEN_RE.findall(text.lower())


class BM25Classifier:
    """
    Minimal BM25-based classifier with two inference modes and persistence.

    Modes:
      - 'classsum': sum BM25 scores across all training docs per class
      - 'knn'     : majority vote among top-k most similar training docs

    Notes:
      - Tokenizer is injectable for customization/testing.
      - Model can be saved/loaded with pickle after fitting.
    """

    def __init__(self, mode: str = "classsum", k: int = 3, tokenizer=default_tokenize):
        if mode not in {"classsum", "knn"}:
            raise ValueError("mode must be 'classsum' or 'knn'")
        self.mode = mode
        self.k = k
        self.tokenizer = tokenizer

        # Learned state after fit()
        self._bm25: Optional[BM25Okapi] = None
        self._labels: Optional[np.ndarray] = None
        self._idx_by_lab: Optional[Dict[str, np.ndarray]] = None

    def fit(self, train_texts: List[str], train_labels: List[str]) -> "BM25Classifier":
        """
        Build BM25 index over the tokenized training corpus and store labels.

        Args:
          train_texts: Raw training documents (strings).
          train_labels: One label per doc (e.g., G-codes).
        """
        corpus_tok = [self.tokenizer(t) for t in train_texts]
        self._bm25 = BM25Okapi(corpus_tok)
        self._labels = np.array(train_labels)

        if self.mode == "classsum":
            idx_by_lab = defaultdict(list)
            for i, lab in enumerate(train_labels):
                idx_by_lab[lab].append(i)
            self._idx_by_lab = {lab: np.array(ix, dtype=int) for lab, ix in idx_by_lab.items()}
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for new texts using the configured mode.
        """
        if self._bm25 is None or self._labels is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        preds: List[str] = []
        for t in texts:
            q = self.tokenizer(t)
            scores = self._bm25.get_scores(q)

            if self.mode == "knn":
                idx = np.argsort(scores)[::-1][: self.k]
                top_labels = self._labels[idx]
                counts = Counter(top_labels)
                best = max(counts.values())
                tied = [lab for lab, c in counts.items() if c == best]
                if len(tied) == 1:
                    preds.append(tied[0])
                else:
                    # tie-break by highest summed scores among tied labels
                    sums = {lab: float(np.sum([scores[i] for i in idx if self._labels[i] == lab])) for lab in tied}
                    preds.append(max(sums.items(), key=lambda kv: kv[1])[0])
            else:
                # classsum
                best_lab, best_score = None, -1.0
                for lab, idxs in self._idx_by_lab.items():
                    s = float(np.sum(scores[idxs]))
                    if s > best_score:
                        best_lab, best_score = lab, s
                preds.append(best_lab)

        return preds

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """
        Persist the trained model (BM25 index, labels, indices, settings).
        """
        if self._bm25 is None or self._labels is None:
            raise RuntimeError("Nothing to save. Train the model first.")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "mode": self.mode,
                    "k": self.k,
                    "labels": self._labels,
                    "bm25": self._bm25,
                    "idx_by_lab": self._idx_by_lab,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str) -> "BM25Classifier":
        """
        Load a trained model from disk.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        m = cls(mode=obj["mode"], k=obj["k"])
        m._labels = obj["labels"]
        m._bm25 = obj["bm25"]
        m._idx_by_lab = obj["idx_by_lab"]
        return m