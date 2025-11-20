#!/usr/bin/env python3
"""
BM25 CLI:
- Train from CSVs/folder of CSVs and save the model
- Predict on new CSVs using a saved model

Usage examples:
*******************************
* Train BM25 model on CTI datasets
*******************************
python bm25_cli.py train \
  --train_dir ./cti_datasets \
  --min_count 25 \
  --mode knn \
  --k 3 \
  --test_frac 0.2 \
  --save_model models/bm25_cti.pkl

*******************************
* Predict on sanitized APT Reports
*******************************  
python bm25_cli.py predict \
  --model models/bm25_cti.pkl \
  --predict_dir ./sanitized_csvs \
  --out_csv ./predictions/bm25_predictions_APT_Reports.csv

*******************************
* Summarize an entire folder of CSVs (recursively)
*******************************
python bm25_cli.py summarize --dir ./datasets/cti_csvs --out label_counts.csv

"""

import argparse
import csv
from pathlib import Path
from typing import List

from sklearn.metrics import classification_report, f1_score, accuracy_score
from dataset_utils import discover_csvs, load_labeled_rows, drop_rare_classes, stratified_guard_split, count_labels


from bm25_model import BM25Classifier
from dataset_utils import (
    discover_csvs,
    drop_rare_classes,
    load_labeled_rows,
    stratified_guard_split,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BM25 trainer/predictor/summarizer for CTI CSV datasets.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Train
    tr = sub.add_parser("train", help="Train a BM25 model on CSVs")
    tr.add_argument("--train_dir", type=str, help="Folder of CSVs (recursively scanned)")
    tr.add_argument("--train_csvs", nargs="*", help="One or more CSV files")
    tr.add_argument("--min_count", type=int, default=10, help="Drop classes with < min_count docs")
    tr.add_argument("--mode", choices=["classsum", "knn"], default="knn")
    tr.add_argument("--k", type=int, default=3, help="k for knn mode")
    tr.add_argument("--test_frac", type=float, default=0.15, help="Per-class test fraction")
    tr.add_argument("--save_model", type=str, required=True, help="Path to save model (.pkl)")

    # Predict
    pr = sub.add_parser("predict", help="Load a saved model and predict CSVs")
    pr.add_argument("--model", type=str, required=True, help="Path to saved model (.pkl)")
    pr.add_argument("--predict_dir", type=str, help="Folder of CSVs to predict")
    pr.add_argument("--predict_csvs", nargs="*", help="One or more CSVs to predict")
    pr.add_argument("--out_csv", type=str, default="bm25_predictions.csv")

    # Summarize
    su = sub.add_parser("summarize", help="Summarize label counts across CSVs")
    su.add_argument("--dir", dest="sum_dir", type=str, help="Folder of CSVs (recursively scanned)")
    su.add_argument("--csvs", dest="sum_csvs", nargs="*", help="One or more CSV files")
    su.add_argument("--out", dest="sum_out", type=str, help="Optional path to write counts (CSV)")
    su.add_argument("--top", dest="sum_top", type=int, default=50, help="How many labels to print")


    return ap

def print_metrics(y_true, y_pred, title: str = "Evaluation"):
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Macro F1 :", f1_score(y_true, y_pred, average="macro", zero_division=0))
    print("Weighted F1:", f1_score(y_true, y_pred, average="weighted", zero_division=0))

def do_train(args: argparse.Namespace) -> None:
    csvs = discover_csvs(args.train_dir, args.train_csvs)
    if not csvs:
        raise SystemExit("No training CSVs found. Provide --train_dir and/or --train_csvs.")
    print(f"[info] training on {len(csvs)} CSVs")

    texts, labels, _ = load_labeled_rows(csvs)
    if not texts:
        raise SystemExit("No rows loaded from CSVs.")

    texts, labels = drop_rare_classes(texts, labels, args.min_count)
    print(f"[info] kept {len(texts)} docs after min_count={args.min_count}")

    Xtr, Xte, ytr, yte = stratified_guard_split(texts, labels, test_frac=args.test_frac)
    clf = BM25Classifier(mode=args.mode, k=args.k).fit(Xtr, ytr)

    # # quick internal evaluation
    # preds = clf.predict(Xte)
    # print("\n=== Holdout Evaluation ===")
    # print(classification_report(yte, preds, digits=3, zero_division=0))
    # print("Macro F1:", f1_score(yte, preds, average="macro", zero_division=0))

    preds = clf.predict(Xte, )
    print_metrics(yte, preds, title="Holdout Evaluation")


    # re-fit on all data, then save (so the saved model is full-data)
    clf = BM25Classifier(mode=args.mode, k=args.k).fit(texts, labels)
    Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.save_model)
    print(f"[info] saved model → {args.save_model}")


def write_predictions(out_path: str, doc_ids: List[str], labels: List[str]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "predicted_label"])
        for i, lab in zip(doc_ids, labels):
            w.writerow([i, lab])
    print(f"[info] wrote predictions → {out_path}")


def do_predict(args: argparse.Namespace) -> None:
    clf = BM25Classifier.load(args.model)
    csvs = discover_csvs(args.predict_dir, args.predict_csvs)
    if not csvs:
        raise SystemExit("No CSVs found to predict. Provide --predict_dir and/or --predict_csvs.")
    print(f"[info] predicting {len(csvs)} CSVs")

    texts, labels, doc_ids = load_labeled_rows(csvs)
    preds = clf.predict(texts)
    write_predictions(args.out_csv, doc_ids, preds)

    # If labels are present, print evaluation metrics
    if labels is not None and len(labels) == len(preds):
        # Optional: guard against placeholder labels if you use them
        # Here we just compute metrics directly.
        print_metrics(labels, preds, title="Prediction Evaluation")
    else:
        print("[info] no labels available for evaluation; only wrote predictions.")


def do_summarize(args: argparse.Namespace) -> None:
    """
    Summarize labels across many CSVs and optionally write a counts CSV.
    """
    csvs = discover_csvs(args.sum_dir, args.sum_csvs)
    if not csvs:
        raise SystemExit("No CSVs found. Provide --dir and/or --csvs.")
    print(f"[info] summarizing {len(csvs)} CSVs")

    counts = count_labels(csvs)
    total = sum(counts.values())
    uniq = len(counts)
    print(f"[info] total docs: {total} | unique labels: {uniq}")

    # Print top-K
    for lab, n in counts.most_common(args.sum_top):
        pct = (n / total * 100) if total else 0.0
        print(f"{lab:>6}  {n:5d}  ({pct:5.1f}%)")

    # Optional: write counts to CSV
    if args.sum_out:
        with open(args.sum_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "count"])
            for lab, n in counts.most_common():
                w.writerow([lab, n])
        print(f"[info] wrote counts → {args.sum_out}")

def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.cmd == "train":
        do_train(args)
    elif args.cmd == "predict":
        do_predict(args)
    elif args.cmd == "summarize":
        do_summarize(args)



if __name__ == "__main__":
    main()



