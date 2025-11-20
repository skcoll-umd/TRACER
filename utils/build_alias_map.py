import json, requests, re
from collections import defaultdict

URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
GID_RE = re.compile(r"^G\d{4}$", re.I)

data = requests.get(URL, timeout=30).json()
g2aliases = defaultdict(set)

for obj in data["objects"]:
    if obj.get("type") != "intrusion-set":
        continue
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            gid = ref.get("external_id")
            if gid and GID_RE.match(gid):
                gid = gid.upper()
                g2aliases[gid].add(obj.get("name", ""))
                for a in obj.get("aliases", []) or []:
                    g2aliases[gid].add(a)

final = {gid: sorted({a.lower() for a in als if len(a) > 2})
         for gid, als in g2aliases.items()}

with open("g2aliases.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2)
    print(f"Wrote g2aliases.json with {len(final)} groups")
