import json
import argparse
import os
import csv
from typing import List, Dict, Any, Set


# ---------- Helpers ----------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    return name.strip().lower()


def overlap_ratio(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


# ---------- Parse MITRE Enterprise ATT&CK ----------
def load_mitre_actors_from_enterprise(path: str) -> List[Dict]:
    data = load_json(path)
    objects = data.get("objects", [])
    id_index = {obj.get("id"): obj for obj in objects}

    intrusion_sets: Dict[str, Dict[str, Any]] = {}

    # Collect intrusion set info
    for obj in objects:
        if obj.get("type") == "intrusion-set":
            stix_id = obj.get("id")
            name = obj.get("name", "")
            aliases = obj.get("aliases", [])

            group_id = None
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack" and \
                   ref.get("external_id", "").startswith("G0"):
                    group_id = ref.get("external_id")
                    break

            intrusion_sets[stix_id] = {
                "stix_id": stix_id,
                "group_id": group_id,
                "name": name,
                "aliases": aliases,
                "techniques": set(),
                "malware": set(),
            }

    # Collect relationships
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "uses":
            continue

        source_ref = obj.get("source_ref")
        target_ref = obj.get("target_ref")

        if source_ref not in intrusion_sets:
            continue

        # Techniques
        if target_ref.startswith("attack-pattern--"):
            tech_obj = id_index.get(target_ref, {})
            technique_id = None
            for ref in tech_obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack" and \
                   ref.get("external_id", "").startswith("T"):
                    technique_id = ref.get("external_id")
                    break
            if technique_id:
                intrusion_sets[source_ref]["techniques"].add(technique_id)

        # Malware
        elif target_ref.startswith("malware--"):
            mal_obj = id_index.get(target_ref, {})
            malware_name = mal_obj.get("name")
            if malware_name:
                intrusion_sets[source_ref]["malware"].add(malware_name)

    # Convert to list
    mitre_actors = []
    for intr in intrusion_sets.values():
        mitre_actors.append({
            "group_id": intr["group_id"],
            "name": intr["name"],
            "aliases": intr["aliases"],
            "techniques": sorted(list(intr["techniques"])),
            "malware": sorted(list(intr["malware"])),
        })

    return mitre_actors


# ---------- Deterministic Matching ----------
def build_alias_index(mitre_actors: List[Dict]) -> Dict[str, Dict]:
    idx = {}
    for actor in mitre_actors:
        names = [actor.get("name", "")] + actor.get("aliases", [])
        for n in names:
            norm = normalize_name(n)
            if norm:
                idx[norm] = actor
    return idx


def find_candidate_actors(
    llm_record: Dict,
    mitre_actors: List[Dict],
    technique_threshold: float = 0.2
):
    alias_index = build_alias_index(mitre_actors)

    llm_name = llm_record.get("Threat_actor") or llm_record.get("Group_Name")
    llm_norm_name = normalize_name(llm_name) if llm_name else ""

    llm_techniques = set(llm_record.get("Attack_to_TTP_map", {}).keys())

    llm_malware = set()
    mf = llm_record.get("malware_family")
    if isinstance(mf, str):
        llm_malware.add(mf)
    elif isinstance(mf, list):
        llm_malware.update(mf)

    candidates = []

    for actor in mitre_actors:
        actor_name = actor.get("name", "")
        actor_aliases = actor.get("aliases", [])
        actor_techniques = set(actor.get("techniques", []))
        actor_malware = set(actor.get("malware", []))

        # Deterministic signals
        name_match = normalize_name(actor_name) == llm_norm_name or \
                     llm_norm_name in [normalize_name(a) for a in actor_aliases]

        technique_overlap = overlap_ratio(llm_techniques, actor_techniques)
        malware_overlap = overlap_ratio(llm_malware, actor_malware)

        score = 0.0
        if name_match:
            score += 1.0
        score += technique_overlap
        score += 0.5 * malware_overlap

        if score == 0:
            continue

        if not name_match and technique_overlap < technique_threshold:
            continue

        candidates.append({
            "group_id": actor["group_id"],
            "name": actor_name,
            "score": round(score, 3),
            "name_match": name_match,
            "technique_overlap": round(technique_overlap, 3),
            "malware_overlap": round(malware_overlap, 3),
            "shared_techniques": list(llm_techniques & actor_techniques),
            "shared_malware": list(llm_malware & actor_malware),
        })

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


# ---------- Folder processing + CSV saving ----------
def process_folder(
    llm_folder: str,
    gt_folder: str,
    mitre_actors: List[Dict],
    csv_output: str
):
    rows = []

    for filename in os.listdir(llm_folder):
        if not filename.endswith(".json"):
            continue

        pred_path = os.path.join(llm_folder, filename)
        llm_record = load_json(pred_path)

        # --- load ground truth file with the same name, if present ---
        gt_actor = None
        if gt_folder:
            gt_path = os.path.join(gt_folder, filename)
            if os.path.exists(gt_path):
                gt_record = load_json(gt_path)
                # handle both "threat_actor" and "Threat_actor"
                gt_actor = gt_record.get("threat_actor") or gt_record.get("Threat_actor")

        candidates = find_candidate_actors(llm_record, mitre_actors)

        for c in candidates:
            rows.append({
                "file": filename,

                # ground truth column
                "ground_truth_actor": gt_actor,

                # predicted actor from LLM JSON
                "predicted_actor": llm_record.get("Threat_actor"),

                # candidate from deterministic model
                "candidate_group_id": c["group_id"],
                "candidate_name": c["name"],
                "score": c["score"],
                "name_match": c["name_match"],
                "technique_overlap": c["technique_overlap"],
                "malware_overlap": c["malware_overlap"],
                "shared_techniques": ";".join(c["shared_techniques"]),
                "shared_malware": ";".join(c["shared_malware"]),
            })

    if not rows:
        print("[!] No rows generated – check your folders / JSON format.")
        return

    # Write to CSV
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Saved results to {csv_output}")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enterprise", required=True, help="enterprise-attack.json")
    parser.add_argument("--llm-folder", required=True, help="folder with predicted JSONs")
    parser.add_argument("--gt-folder", required=True, help="folder with ground-truth JSONs")
    parser.add_argument("--output", default="attribution_results.csv")
    args = parser.parse_args()

    print("[+] Loading MITRE ATT&CK...")
    mitre_actors = load_mitre_actors_from_enterprise(args.enterprise)

    print(f"[+] Processing folder: {args.llm_folder}")
    process_folder(args.llm_folder, args.gt_folder, mitre_actors, args.output)


if __name__ == "__main__":
    main()
