#!/usr/bin/env python3
import os, sys, json, argparse, requests, re

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-3-7-sonnet-latest"  # change if your org uses a pinned model

SYSTEM_INSTRUCTION = (
    "You are a senior security analyst and threat report writer with deep knowledge of "
    "ATT&CK IDs, threat actor, group ids,  malware families, campaign naming conventions, and common infrastructure hints. "
    "You will be provided with a single JSON containing structured fields extracted from a malware report. "
    "Your task is to analyze the provided data and produce a single JSON object (and ONLY JSON, no prose, "
    "no explanation) that contains the following keys exactly:\n\n"
    "  - Threat_actor: string  (best-guess threat actor alias based on all the hints provided)\n"
    "  - Group_Name: string ('identify the correct group name')\n"
    "  - Group_ID: string (MITRE ATT&CK Gxxxx group ID if identifiable;)\n"
    "  - malware_family: string (best guess;)\n"
    "  - Campaign_Name: string (best-guess campaign identifier)\n"
    "  - Attack_to_TTP_map: object mapping each TTP ID (e.g. 'T1059') to a short attack/technique name or 'unknown'\n"
    "  - Techniques_to_Tactic: object mapping each TTP ID to its ATT&CK tactic ID (e.g. 'TA0001') or 'unknown'\n\n"
    "Rules & constraints:\n"
    "  1) Output MUST be valid JSON parseable by standard json.loads().\n"
    "  2) Do NOT include any extra keys beyond those specified. If you cannot determine a value, use the string 'unknown'.\n"
    "  3) Prefer canonical ATT&CK technique names.\n"
    "  4) Use the input's tactic IDs when provided; otherwise infer.\n"
    "  5) Keep all values as strings except the two mapping objects.\n"
    "  6) Do NOT wrap the JSON in markdown or code fences.\n"
)

def build_user_prompt(report_json: dict) -> str:
    input_json_str = json.dumps(report_json, indent=2, ensure_ascii=False)
    return (
        "The following JSON (exactly as given) contains extracted report information. "
        "Analyze it throughly like an expert malware analyst for threat actor name, group id, group name, campaign name, malware family, attack to ttp map, techniques to tactic map and produce the required output JSON.\n\n"
        f"INPUT_JSON:\n{input_json_str}\n\n"
        "Produce the output JSON now, and ONLY the JSON (no commentary, no notes)."
    )

def call_anthropic(system_prompt: str, user_prompt: str, api_key: str,
                   model: str, max_tokens: int = 1500, temperature: float = 0.5) -> str:
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=90)
    try:
        resp.raise_for_status()
    except Exception:
        raise RuntimeError(f"Anthropic error {resp.status_code}: {resp.text}")

    data = resp.json()
    chunks = [b.get("text","") for b in data.get("content", []) if b.get("type")=="text"]
    return "\n".join(chunks).strip()

def extract_json_from_text(text: str) -> str | None:
    s = text.strip()
    try:
        json.loads(s); return s
    except: pass
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last != -1 and last > first:
        cand = s[first:last+1]
        cand = re.sub(r",\s*}", "}", cand)
        cand = re.sub(r",\s*]", "]", cand)
        try:
            json.loads(cand); return cand
        except: pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Batch Malware Attribution with Claude")
    parser.add_argument("input_folder", help="Folder containing input JSON files")
    parser.add_argument("output_folder", help="Folder where output JSON files will be saved")
    parser.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: API key missing. Set ANTHROPIC_API_KEY or use --api-key.", file=sys.stderr)
        sys.exit(1)

    input_folder = args.input_folder
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # get list of .json files
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]

    if not files:
        print("No JSON files found in input folder!")
        sys.exit(0)

    print(f"Found {len(files)} files. Processing...\n")

    for filename in files:
        input_path = os.path.join(input_folder, filename)

        # load input JSON
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                report_json = json.load(f)
        except Exception as e:
            print(f"[ERROR] Cannot read {filename}: {e}")
            continue

        # build prompts
        system_prompt = SYSTEM_INSTRUCTION
        user_prompt = build_user_prompt(report_json)

        # call the model
        try:
            raw_output = call_anthropic(
                api_key=args.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
        except Exception as e:
            print(f"[ERROR] Model call failed for {filename}: {e}")
            continue

        json_str = extract_json_from_text(raw_output)

        if not json_str:
            print(f"[ERROR] Model output for {filename} was not valid JSON.")
            continue

        # save file
        output_path = os.path.join(
            output_folder,
            filename.replace(".json", "_output.json")
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        except Exception as e:
            print(f"[ERROR] Could not save output for {filename}: {e}")
            continue

        print(f"[OK] Saved output → {output_path}")

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
