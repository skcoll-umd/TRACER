#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_attribution.py

Evaluate LLM malware-attribution outputs against ground truth (GT).

Features
--------
- Pairs GT and Pred files by basename (handles *_output.json).
- Identity fields (Threat_actor, Group_Name, Group_ID, malware_family, Campaign_Name):
  * exact match (normalized)
  * alias-aware match (via STIX + optional alias CSV)
  * fuzzy similarity (RapidFuzz if installed, else token Jaccard)
  * scores: 1.0 (exact/alias/high-fuzzy), 0.5 (mid-fuzzy), 0.0 otherwise
- TTP coverage: precision / recall / F1 over technique IDs (e.g., T1059, T1059.001).
- Technique→Tactic accuracy: correct if any predicted TAxxxx matches any GT TAxxxx for that technique.
- Composite score with publication-friendly default weights (configurable).
- Outputs:
  * per-report JSON (diagnostics)
  * summary CSV
  * optional plots (overall scores bar; distributions; scatter)

CLI
---
pip install rapidfuzz matplotlib

python evaluate_attribution.py \
  --gt-folder GT --pred-folder Preds --out-folder results \
  --stix-bundle enterprise-attack.json \
  --alias-csv aliases.csv \
  --make-plots
"""
import os
import re
import json
import csv
import argparse
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

# Optional fuzzy matcher (recommended): pip install rapidfuzz
try:
    from rapidfuzz.fuzz import ratio as fuzz_ratio
except Exception:
    fuzz_ratio = None

# Optional plotting (recommended): pip install matplotlib
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ------------------------- Config & Defaults -------------------------
IDENTITY_FIELDS = ["Threat_actor", "Group_Name", "Group_ID", "malware_family", "Campaign_Name"]

DEFAULT_WEIGHTS = {
    "Threat_actor":     0.20,
    "Group_Name":       0.10,
    "Group_ID":         0.10,
    "malware_family":   0.10,
    "Campaign_Name":    0.10,
    "ttp_f1":           0.25,
    "tactic_accuracy":  0.15,
}

FUZZY_FULL    = 0.95   # similarity >= this → 1.0 (if not exact/alias)
FUZZY_PARTIAL = 0.85   # similarity >= this → 0.5

NOISE_TOKENS = {
    "group","team","apt","threat","actor","the","operation","campaign","unit",
    "co","ltd","inc","llc","aka"
}


# ------------------------- Utility Functions -------------------------
def norm_text(s: str) -> str:
    """Normalize text for robust comparison (case/space/punct/stopword)."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower().strip()
    s = re.sub(r"[\s/_\-–—]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    tokens = [t for t in s.split() if t not in NOISE_TOKENS]
    return " ".join(tokens).strip()

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb: return 1.0
    if not sa or not sb:  return 0.0
    return len(sa & sb) / len(sa | sb)

def fuzzy_sim(a: str, b: str) -> float:
    """Return [0..1] similarity using RapidFuzz if available; else Jaccard."""
    na, nb = norm_text(a), norm_text(b)
    if not na and not nb: return 1.0
    if not na or not nb:  return 0.0
    if fuzz_ratio:
        return fuzz_ratio(na, nb) / 100.0
    return jaccard(na, nb)

def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def key_any(d: dict, *candidates):
    for c in candidates:
        if c in d: return d[c]
    return None

def ensure_dict(x):
    if isinstance(x, dict): return x
    if isinstance(x, (list, str)):
        try:
            j = json.loads(x)
            if isinstance(j, dict): return j
        except Exception: pass
    return {}

def ensure_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str):  return [x]
    if x is None:           return []
    return list(x)


# ------------------------- Alias DB -------------------------
class AliasDB:
    """
    canonical_name -> aliases (including canonical)
    alias(normalized) -> canonical_name
    canonical_name -> group_id (Gxxxx) optional
    """
    def __init__(self):
        self.canonical_from_alias: Dict[str, str] = {}
        self.aliases_for_canonical: Dict[str, Set[str]] = defaultdict(set)
        self.group_id_for_canonical: Dict[str, str] = {}

    def add(self, canonical: str, group_id: Optional[str], aliases: List[str]):
        names = {canonical} | {a for a in aliases if a}
        for n in names:
            self.canonical_from_alias[norm_text(n)] = canonical
        self.aliases_for_canonical[canonical].update(names)
        if group_id:
            self.group_id_for_canonical[canonical] = group_id

    def canonical(self, name: str) -> Optional[str]:
        return self.canonical_from_alias.get(norm_text(name))

    def aliases(self, canonical: str) -> Set[str]:
        return self.aliases_for_canonical.get(canonical, set())

    def gid(self, canonical: str) -> Optional[str]:
        return self.group_id_for_canonical.get(canonical)


def load_aliases_from_csv(path: str) -> AliasDB:
    """
    CSV format: canonical,group_id,aliases
    aliases is semicolon-separated, e.g. "Sandworm;Voodoo Bear;Unit 74455"
    """
    db = AliasDB()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = (row.get("canonical") or "").strip()
            gid = (row.get("group_id") or "").strip() or None
            aliases = [a.strip() for a in (row.get("aliases") or "").split(";") if a.strip()]
            if canonical:
                db.add(canonical, gid, aliases)
    return db

def load_aliases_from_stix_bundle(path: str) -> AliasDB:
    """
    Parse MITRE ATT&CK STIX 2.1 bundle JSON (enterprise-attack).
    For each intrusion-set:
      - name (canonical)
      - aliases
      - external_references: source_name 'mitre-attack', external_id Gxxxx
    """
    db = AliasDB()
    bundle = load_json(path)
    if not bundle or "objects" not in bundle:
        return db
    for obj in bundle["objects"]:
        if not isinstance(obj, dict): continue
        if obj.get("type") != "intrusion-set": continue
        canonical = obj.get("name")
        aliases = ensure_list(obj.get("aliases"))
        gid = None
        for ref in ensure_list(obj.get("external_references")):
            if isinstance(ref, dict) and ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id")
                if ext_id and re.match(r"^G\d{4}$", ext_id):
                    gid = ext_id
                    break
        if canonical:
            db.add(canonical, gid, aliases)
    return db


# ------------------------- Identity Matching -------------------------
def match_identity(pred: str, gold: str, field: str, alias_db: AliasDB) -> Tuple[float, str, str]:
    """
    Returns (score, match_type, note):
      score: 1.0, 0.5, 0.0
      match_type: 'exact' | 'alias' | 'fuzzy-high' | 'fuzzy-mid' | 'mismatch' | 'missing'
      note: extra info (e.g., similarity, canonical resolution)
    """
    if gold is None and pred is None:
        return 1.0, "exact", "both missing"
    if gold is None:
        return 0.0, "missing", "gold missing"
    if pred is None or str(pred).strip() in ("", "unknown"):
        return 0.0, "missing", "prediction missing/unknown"

    p_raw, g_raw = str(pred).strip(), str(gold).strip()
    if norm_text(p_raw) == norm_text(g_raw):
        return 1.0, "exact", ""

    # Alias logic for actors/groups
    if field in ("Threat_actor", "Group_Name"):
        p_can = alias_db.canonical(p_raw) or p_raw
        g_can = alias_db.canonical(g_raw) or g_raw
        if norm_text(p_can) == norm_text(g_can):
            return 1.0, "alias", f"canonical({p_raw})={p_can} ; canonical({g_raw})={g_can}"
        # group-id equality if both have known gid
        p_gid = alias_db.gid(p_can) if alias_db.canonical(p_raw) else None
        g_gid = alias_db.gid(g_can) if alias_db.canonical(g_raw) else None
        if p_gid and g_gid and p_gid == g_gid:
            return 1.0, "alias", f"shared_gid={p_gid}"

    # Group_ID: normalize e.g. g0034 -> G0034
    if field == "Group_ID":
        p_gid = re.sub(r"\s", "", p_raw.upper())
        g_gid = re.sub(r"\s", "", g_raw.upper())
        if p_gid == g_gid:
            return 1.0, "exact", ""

    # Fuzzy fallback
    sim = fuzzy_sim(p_raw, g_raw)
    if sim >= FUZZY_FULL:
        return 1.0, "fuzzy-high", f"sim={sim:.2f}"
    if sim >= FUZZY_PARTIAL:
        return 0.5, "fuzzy-mid", f"sim={sim:.2f}"
    return 0.0, "mismatch", f"sim={sim:.2f}"


# ------------------------- TTP & Tactic Scoring -------------------------
def collect_techniques(obj: dict) -> Set[str]:
    """
    Techniques are taken from:
      - keys of Attack_to_TTP_map / attack_ttp_map
      - attack_ttp_ids (if present)
    """
    techniques = set()
    amap = key_any(obj, "Attack_to_TTP_map", "attack_ttp_map") or {}
    amap = ensure_dict(amap)
    techniques |= set(amap.keys())
    ids = key_any(obj, "attack_ttp_ids", "technique_ids")
    if ids:
        techniques |= set(ensure_list(ids))
    return {t.strip() for t in techniques if isinstance(t, str) and t.strip()}

def collect_tactic_map(obj: dict) -> Dict[str, List[str]]:
    """
    Normalize to: { technique_id: [TAxxxx, ...] }
    Accepts:
      - Techniques_to_Tactic (dict: T->TA or T->list[TA])
      - technique_to_tactics (dict: T->list[TA])
    """
    raw = key_any(obj, "Techniques_to_Tactic", "technique_to_tactics") or {}
    raw = ensure_dict(raw)
    out = {}
    for k, v in raw.items():
        lst = ensure_list(v)
        norm = []
        for x in lst:
            if isinstance(x, str):
                x = x.strip().upper()
                if re.match(r"^TA\d{4}$", x):
                    norm.append(x)
        if norm:
            out[k] = norm
    return out

def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1

def score_ttp_sets(gt: Set[str], pr: Set[str]) -> Dict[str, float]:
    tp = len(gt & pr)
    fp = len(pr - gt)
    fn = len(gt - pr)
    p, r, f = prf1(tp, fp, fn)
    return {
        "ttp_tp": float(tp),
        "ttp_fp": float(fp),
        "ttp_fn": float(fn),
        "ttp_precision": round(p, 4),
        "ttp_recall": round(r, 4),
        "ttp_f1": round(f, 4),
    }

def score_tactics(gt_map: Dict[str, List[str]], pr_map: Dict[str, List[str]]) -> Dict[str, float]:
    total, correct = 0, 0
    for ttp, gt_tactics in gt_map.items():
        if not gt_tactics:
            continue
        total += 1
        pr_tactics = pr_map.get(ttp, [])
        # correct if ANY predicted TA matches ANY GT TA for this TTP
        ok = any(pt in set(gt_tactics) for pt in pr_tactics)
        if ok:
            correct += 1
    acc = correct / total if total else 0.0
    return {
        "tactic_total": float(total),
        "tactic_correct": float(correct),
        "tactic_accuracy": round(acc, 4),
    }


# ------------------------- Pairing & Evaluation -------------------------
def pair_files(gt_folder: str, pred_folder: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Return list of (base, gt_path, pred_path).
    Matches *_output.json to the GT basename as well.
    """
    gt_files = [f for f in os.listdir(gt_folder) if f.lower().endswith(".json")]
    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith(".json")]

    pred_index = {}
    for f in pred_files:
        b = os.path.splitext(f)[0]
        pred_index[b] = f
        if b.endswith("_output"):
            pred_index[b[:-7]] = f

    pairs = []
    for g in gt_files:
        base = os.path.splitext(g)[0]
        p = pred_index.get(base)
        pairs.append((base, os.path.join(gt_folder, g), os.path.join(pred_folder, p) if p else None))
    return pairs

def evaluate_pair(gt_obj: dict, pr_obj: dict, alias_db: AliasDB, weights: Dict[str, float]) -> dict:
    # Identity field values (accept common variants)
    vals_gt = {
        "Threat_actor":  key_any(gt_obj, "Threat_actor", "threat_actor", "actor", "group", "group_name") or "",
        "Group_Name":    key_any(gt_obj, "Group_Name", "group_name") or "",
        "Group_ID":      key_any(gt_obj, "Group_ID", "group_id") or "",
        "malware_family":key_any(gt_obj, "malware_family", "family") or "",
        "Campaign_Name": key_any(gt_obj, "Campaign_Name", "campaign_name") or "",
    }
    vals_pr = {
        "Threat_actor":  key_any(pr_obj, "Threat_actor", "threat_actor", "Group_Name", "group_name") or "",
        "Group_Name":    key_any(pr_obj, "Group_Name", "group_name") or "",
        "Group_ID":      key_any(pr_obj, "Group_ID", "group_id") or "",
        "malware_family":key_any(pr_obj, "malware_family") or "",
        "Campaign_Name": key_any(pr_obj, "Campaign_Name", "campaign_name") or "",
    }

    # Identity scoring
    id_scores, id_meta = {}, {}
    for field in IDENTITY_FIELDS:
        s, t, note = match_identity(vals_pr.get(field, ""), vals_gt.get(field, ""), field, alias_db)
        id_scores[field] = s
        id_meta[field] = {"prediction": vals_pr.get(field, ""), "gold": vals_gt.get(field, ""), "type": t, "note": note}

    # If Group_ID differs but canonical actor GID aligns, grant full credit
    try:
        gt_actor_can = alias_db.canonical(vals_gt["Threat_actor"]) or vals_gt["Threat_actor"]
        pr_actor_can = alias_db.canonical(vals_pr["Threat_actor"]) or vals_pr["Threat_actor"]
        gt_gid_known = alias_db.gid(gt_actor_can)
        pr_gid_known = alias_db.gid(pr_actor_can)
        gold_gid = (vals_gt["Group_ID"] or gt_gid_known or "").upper()
        if gold_gid and pr_gid_known and pr_gid_known.upper() == gold_gid and id_scores["Group_ID"] < 1.0:
            id_scores["Group_ID"] = 1.0
            id_meta["Group_ID"]["type"] = "alias"
            id_meta["Group_ID"]["note"] = f"matched via canonical gid {pr_gid_known}"
    except Exception:
        pass

    # TTP coverage & Tactic mapping
    gt_ttps = collect_techniques(gt_obj)
    pr_ttps = collect_techniques(pr_obj)
    ttp_metrics = score_ttp_sets(gt_ttps, pr_ttps)

    gt_tactics = collect_tactic_map(gt_obj)
    pr_tactics = collect_tactic_map(pr_obj)
    tactic_metrics = score_tactics(gt_tactics, pr_tactics)

    # Composite score
    composite = 0.0
    composite += weights.get("Threat_actor", 0)    * id_scores["Threat_actor"]
    composite += weights.get("Group_Name", 0)      * id_scores["Group_Name"]
    composite += weights.get("Group_ID", 0)        * id_scores["Group_ID"]
    composite += weights.get("malware_family", 0)  * id_scores["malware_family"]
    composite += weights.get("Campaign_Name", 0)   * id_scores["Campaign_Name"]
    composite += weights.get("ttp_f1", 0)          * ttp_metrics["ttp_f1"]
    composite += weights.get("tactic_accuracy", 0) * tactic_metrics["tactic_accuracy"]

    return {
        "identity_scores": id_scores,
        "identity_details": id_meta,
        "ttp_metrics": ttp_metrics,
        "tactic_metrics": tactic_metrics,
        "overall_score": round(composite, 4),
        "counts": {"gt_ttps": len(gt_ttps), "pr_ttps": len(pr_ttps)},
    }


# ------------------------- Plots -------------------------
def make_plots(rows: List[dict], out_folder: str):
    if not rows:
        return
    if not _HAS_MPL:
        print("[WARN] matplotlib not installed; skipping plots.")
        return

    plots_dir = os.path.join(out_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Bar chart: overall score per report
    reports = [r["report"] for r in rows]
    scores  = [float(r["overall_score"]) for r in rows]

    plt.figure(figsize=(max(6, 0.45*len(rows)), 4.5))
    plt.bar(range(len(scores)), scores)
    plt.xticks(range(len(scores)), reports, rotation=75, ha="right", fontsize=8)
    plt.ylabel("Overall score")
    plt.ylim(0, 1.0)
    plt.title("Overall score per report")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "overall_scores_bar.png"), dpi=160)
    plt.close()

    # Histograms: TTP F1 and Tactic Accuracy
    ttp_f1 = [float(r["ttp_f1"]) for r in rows]
    tac_acc = [float(r["tactic_accuracy"]) for r in rows]

    plt.figure(figsize=(6,4))
    plt.hist(ttp_f1, bins=10)
    plt.xlabel("TTP F1")
    plt.ylabel("Count of reports")
    plt.title("Distribution of TTP F1")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ttp_f1_distribution.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(tac_acc, bins=10)
    plt.xlabel("Tactic accuracy")
    plt.ylabel("Count of reports")
    plt.title("Distribution of technique→tactic accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tactic_accuracy_distribution.png"), dpi=160)
    plt.close()

    # Scatter: TTP F1 vs Tactic accuracy
    plt.figure(figsize=(6,5))
    plt.scatter(ttp_f1, tac_acc)
    plt.xlabel("TTP F1")
    plt.ylabel("Tactic accuracy")
    plt.title("TTP F1 vs Tactic accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ttpF1_vs_tacticAcc_scatter.png"), dpi=160)
    plt.close()

    print(f"[PLOTS] wrote: {plots_dir}")


# ------------------------- CLI -------------------------
def main():
    global FUZZY_FULL, FUZZY_PARTIAL

    ap = argparse.ArgumentParser(description="Evaluate LLM malware attribution against ground truth.")
    ap.add_argument("--gt-folder",    required=True, help="Folder with ground truth JSON files")
    ap.add_argument("--pred-folder",  required=True, help="Folder with predicted JSON files")
    ap.add_argument("--out-folder",   default="attribution_results", help="Output folder for per-report JSON + summary CSV")
    ap.add_argument("--stix-bundle",  help="Optional MITRE ATT&CK STIX 2.1 enterprise bundle JSON for alias DB")
    ap.add_argument("--alias-csv",    help="Optional alias CSV: canonical,group_id,aliases (semicolon-separated)")
    ap.add_argument("--weights-json", help="Optional JSON to override metric weights")
    ap.add_argument("--fuzzy-full",   type=float, default=FUZZY_FULL,    help="Full-credit fuzzy threshold (default 0.95)")
    ap.add_argument("--fuzzy-partial",type=float, default=FUZZY_PARTIAL, help="Partial-credit fuzzy threshold (default 0.85)")
    ap.add_argument("--make-plots",   action="store_true", help="Save charts to attribution_results/plots")
    args = ap.parse_args()

    # update globals from CLI
    FUZZY_FULL    = args.fuzzy_full
    FUZZY_PARTIAL = args.fuzzy_partial

    os.makedirs(args.out_folder, exist_ok=True)
    per_report_dir = os.path.join(args.out_folder, "per_report")
    os.makedirs(per_report_dir, exist_ok=True)

    # Load weights
    weights = DEFAULT_WEIGHTS.copy()
    if args.weights_json and os.path.exists(args.weights_json):
        try:
            with open(args.weights_json, "r", encoding="utf-8") as f:
                w = json.load(f)
            for k, v in w.items():
                if k in weights:
                    weights[k] = float(v)
        except Exception as e:
            print(f"[WARN] Could not load weights JSON: {e}")

    # Build alias DB: STIX then CSV overrides/extends
    alias_db = AliasDB()
    if args.stix_bundle and os.path.exists(args.stix_bundle):
        alias_db = load_aliases_from_stix_bundle(args.stix_bundle)
        print(f"[INFO] Loaded STIX alias DB with {len(alias_db.aliases_for_canonical)} intrusion sets.")
    if args.alias_csv and os.path.exists(args.alias_csv):
        custom_db = load_aliases_from_csv(args.alias_csv)
        for can, aliases in custom_db.aliases_for_canonical.items():
            alias_db.add(can, custom_db.gid(can), list(aliases))
        print(f"[INFO] Merged aliases from CSV.")

    # Pair files
    pairs = pair_files(args.gt_folder, args.pred_folder)
    rows = []

    for base, gt_path, pr_path in pairs:
        if not pr_path or not os.path.exists(pr_path):
            print(f"[MISS] No prediction file for {base}")
            continue
        gt_obj = load_json(gt_path)
        pr_obj = load_json(pr_path)
        if gt_obj is None or pr_obj is None:
            print(f"[ERR] Failed to load JSON for {base}")
            continue

        res = evaluate_pair(gt_obj, pr_obj, alias_db, weights)

        # Write per-report JSON
        out_json = os.path.join(per_report_dir, f"{base}.eval.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        # For summary CSV
        row = {
            "report": base,
            "overall_score": res["overall_score"],
            "Threat_actor": res["identity_scores"]["Threat_actor"],
            "Group_Name": res["identity_scores"]["Group_Name"],
            "Group_ID": res["identity_scores"]["Group_ID"],
            "malware_family": res["identity_scores"]["malware_family"],
            "Campaign_Name": res["identity_scores"]["Campaign_Name"],
            "ttp_precision": res["ttp_metrics"]["ttp_precision"],
            "ttp_recall": res["ttp_metrics"]["ttp_recall"],
            "ttp_f1": res["ttp_metrics"]["ttp_f1"],
            "tactic_accuracy": res["tactic_metrics"]["tactic_accuracy"],
            "gt_ttps": res["counts"]["gt_ttps"],
            "pr_ttps": res["counts"]["pr_ttps"],
        }
        rows.append(row)
        print(f"[OK] {base}: overall={row['overall_score']:.3f}  ttp_f1={row['ttp_f1']:.3f}  tactic_acc={row['tactic_accuracy']:.3f}")

    # Write summary CSV
    if rows:
        out_csv = os.path.join(args.out_folder, "summary.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[DONE] Wrote summary → {out_csv}")
        print(f"[DONE] Per-report JSON → {per_report_dir}")

        if args.make_plots:
            make_plots(rows, args.out_folder)
    else:
        print("[INFO] No paired files evaluated.")


if __name__ == "__main__":
    main()
