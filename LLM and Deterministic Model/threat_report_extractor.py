import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

import argparse, json, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ----- deps -----
try:
    import pdfplumber
    import spacy
except Exception:
    raise SystemExit("Install deps:\n  pip install pdfplumber spacy requests\n  python -m spacy download en_core_web_sm")

import requests

# Optional OCR deps only if --ocr used
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    pass

# ==================== REGEX / CONSTANTS ====================
RE_MD5 = re.compile(r'\b[a-fA-F0-9]{32}\b')
RE_SHA1 = re.compile(r'\b[a-fA-F0-9]{40}\b')
RE_SHA256 = re.compile(r'\b[a-fA-F0-9]{64}\b')
RE_ATTACK_ID = re.compile(r'\bT\d{4}(?:\.\d{3})?\b', re.IGNORECASE)

# Distinguish ATT&CK tactics (TA000x) from vendor TA groups (TA505)
RE_TACTIC_TA000X = re.compile(r'\bTA0\d{3}\b', re.IGNORECASE)     # NOT actors
RE_VENDOR_TA = re.compile(r'\bTA\d{3,5}\b', re.IGNORECASE)         # vendor actor naming (3–5 digits, e.g. TA505)

RE_URL = re.compile(r'https?://[^\s,;\'")]+', re.IGNORECASE)
RE_IPV4 = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
RE_DOMAIN = re.compile(r'\b((?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,63})\b', re.IGNORECASE)

# Mutex
RE_MUTEX_KEYED = re.compile(r'\b(?:Mutex|Mutant|CreateMutex(?:A|W)?|Named\s+Mutex)[:\s]*([A-Za-z0-9_:\-\\/{\}]+)\b', re.IGNORECASE)
RE_MUTEX_GLOBAL = re.compile(r'\b(?:Global\\|Local\\|BaseNamedObjects\\)[A-Za-z0-9_:\-\\/{\}.]{4,}\b')

# Narrative noise
HEADER_WORDS = {"conclusion","recommendations","executive summary","summary","indicators of compromise","iocs","mitigations","appendix","table of contents"}

# Heuristics
MALWARE_SUFFIX = r'(?:RAT|Trojan|Backdoor|Loader|Stealer|Bot|Worm|Ransomware)'
VERSION_TOK = r'(?:v\d+(?:\.\d+)?|V\d+(?:\.\d+)?|\d+\.\d+)'

# Vendor taxonomy shapes
ACTOR_TAXONOMY = [
    re.compile(r'\bAPT[\s\-]?\d{1,3}\b', re.I),
    re.compile(r'\bUNC[\s\-]?\d{1,5}\b', re.I),
    re.compile(r'\bDEV[\s\-]?\d{3,5}\b', re.I),
    re.compile(r'\bStorm[\s\-]?\d{3,5}\b', re.I),
    re.compile(r'\bTG[\s\-]?\d{3,5}\b', re.I),
    RE_VENDOR_TA,  # TA### (vendor group naming)
]

# --- Optional Microsoft alias map ---
try:
    from actor_aliases import ACTOR_ALIASES  # dict: alias -> Microsoft canonical
    _ALIASES_CI = {k.strip().lower(): v for k, v in ACTOR_ALIASES.items()}
    _MS_CANON   = {v.strip().lower() for v in ACTOR_ALIASES.values()}
except Exception:
    _ALIASES_CI = {}
    _MS_CANON   = set()

COMMON_TLDS = {
    "com","net","org","io","co","ru","cn","uk","de","jp","fr","au","in","ca","us","es","it","nl","pl","br","se","no","fi","ch","cz","ua","za","tr","vn","kr","tw","sg","hk","be","dk","at","gr","pt","ro","hu","sk","ie","nz","mx","il","sa","ae","ir","cl","ar","pk","id"
}
BANNED_TLDS = {"exe","dll","dat","tmp","txt","zip","rar","7z","gz","pdf","doc","docx","xls","xlsx","ppt","pptx","bin"}
WINDOWS_PROCESSES = {"taskmgr.exe","processhacker.exe","procexp.exe","svchost.exe","winlogon.exe","lsass.exe","explorer.exe"}

DELIVERY = [
    "spearphish","phishing","smishing","vishing","malspam","email",
    "drive-by","drive by","exploit kit","malvertising","supply chain",
    "software update","third-party update","third party update",
    "container image","docker image","npm package","pip package","iso",
    "firmware update","usb","watering hole","watering-hole","rdp",
    "credential stuffing","brute force","malicious attachment","macro",
    "website","social media","attachment","document","waterin g hole"
]

SECTORS = [
    "energy","financial","finance","government","public sector","healthcare",
    "manufacturing","telecom","telecommunications","education","retail",
    "transport","defense","insurance","utilities","legal","media","hospitality",
    "pharmaceuticals","chemical","technology","real estate","mining","agriculture","aerospace"
]

INFRA_KEYWORDS = {
  "services": [
    "c2",
    "command and control",
    "beacon",
    "rdp",
    "vnc",
    "winrm",
    "ssh",
    "telnet",
    "smb",
    "nfs",
    "rpc",
    "http",
    "https",
    "web protocols",
    "websocket",
    "dns",
    "dns-tunneling",
    "smtp",
    "imap",
    "pop3",
    "ftp",
    "sftp",
    "scp",
    "tcp",
    "udp",
    "icmp",
    "tls",
    "ssl",
    "vpn",
    "proxy",
    "reverse proxy",
    "domain fronting",
    "tor",
    "onion",
    "cloud storage",
    "s3",
    "sharepoint",
    "googledrive",
    "dropbox",
    "webhook",
    "api",
    "mysql",
    "mssql",
    "postgresql",
    "redis",
    "memcached",
    "mongodb",
    "ldap",
    "kerberos",
    "ntp",
    "dhcp",
    "coap",
    "mqtt",
    "icp/udp",
    "direct cloud vm",
    "file transfer protocol",
    "publish/subscribe",
    "custom protocol"
  ],
  "tools": {
    "c2_frameworks": [
      "cobalt strike",
      "metasploit",
      "meterpreter",
      "sliver",
      "brute ratel",
      "mythic",
      "havoc",
      "sliver",
      "covenant",
      "powercat",
      "empire",
      "powerless (powershell empire variants)"
    ],
    "credential_and_privilege_tools": [
      "mimikatz",
      "secretsdump/impacket",
      "creds_dumpers",
      "lsass dumper",
      "procdump",
      "secretsdump.py",
      "gsecdump"
    ],
    "post_exploitation_and_lateral": [
      "psexec",
      "wmiexec",
      "smbexec",
      "impacket-scripts",
      "bloodhound",
      "ridenum",
      "adfind",
      "adldap tools"
    ],
    "file_transfer_and_sync": [
      "rclone",
      "curl",
      "wget",
      "scp",
      "sftp",
      "ftp",
      "powershell (Invoke-WebRequest/Invoke-RestMethod)",
      "bitsadmin",
      "certutil",
      "ftp clients",
      "cloudcli (awscli, az cli, gcloud)"
    ],
    "tunneling_and_proxy": [
      "netcat",
      "ncat",
      "socat",
      "ssh -R / -L",
      "htran",
      "stunnel",
      "chisel",
      "ngrok",
      "frp",
      "reverse proxy tools",
      "domain fronting libs"
    ],
    "webshells_and_uploaders": [
      "china chopper",
      "aspx webshell",
      "php webshell",
      "weevely",
      "webshells (generic)",
      "uploaders/backdoors"
    ],
    "living_off_the_land (LOLBAS/LOLBIN)": [
      "powershell",
      "mshta",
      "rundll32",
      "regsvr32",
      "bitsadmin",
      "certutil",
      "wmic",
      "schtasks",
      "tasklist",
      "taskkill",
      "sc.exe",
      "net.exe",
      "netsh",
      "netstat",
      "schtasks",
      "schtasks.exe"
    ],
    "exfiltration_and_sync_tools": [
      "rclone",
      "resilio",
      "ftp clients",
      "cloud storage cli (awscli, gcloud, az cli)",
      "webhooks",
      "smtp exfil"
    ],
    "credential_access_frameworks_and_helpers": [
      "powersploit",
      "nishang",
      "impacket",
      "powerup",
      "seatbelt"
    ],
    "recon_and_scanning": [
      "nmap",
      "masscan",
      "zmap",
      "nbtscan",
      "dnsrecon",
      "amass",
      "shodan clients"
    ],
    "utilities_and_helpers": [
      "python",
      "perl",
      "ruby",
      "bash",
      "cmd",
      "powershell",
      "powershell core",
      "docker",
      "kubectl",
      "git",
      "npm",
      "pip"
    ]
  }
}


# Benign domain allowlist (helps suppress vendor links in IOCs)
DEFAULT_DOMAIN_ALLOWLIST = {
    "microsoft.com","learn.microsoft.com","github.com","raw.githubusercontent.com",
    "mitre.org","attack.mitre.org","nist.gov","cloudflare.com","google.com","googleusercontent.com",
    "docs.google.com","drive.google.com","dropbox.com","box.com","wikipedia.org","medium.com",
    "twitter.com","x.com","linkedin.com","facebook.com","youtube.com","youtu.be",
    "socradar.io","socradar.com","unit42.paloaltonetworks.com","blog.google","security.microsoft.com",
    "virustotal.com","virustotal.org","any.run","malwarebytes.com","kaspersky.com","symantec.com",
    "trellix.com","synacktiv.com","cyble.com","pentestlab.blog", "cloud.google.com", "archive.org", 
}

# ==================== HELPERS ====================
def dedupe(seq):
    seen=set(); out=[]
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def is_name_shape(s: str, max_tokens=5) -> bool:
    s = s.replace("\n"," ").strip()
    toks = [t for t in s.split() if t]
    if not (1 <= len(toks) <= max_tokens): return False
    bad = {"the","a","an","and","but","or","if","then","else","with","without","like","has","have","had",
           "using","used","use","methods","targeting","targets","based","by","of","in","on","for","to"}
    verbs = {"employed","uses","using","installing","delivers","delivered","leverages","bypasses","encrypts","executes","downloads","spreads","runs","communicates","masquerades"}
    for t in toks:
        tl=t.lower().strip("-_/’'”")
        if tl in bad or tl in verbs:
            return False
    return True

def _normalize_defanged(s: str) -> str:
    if not s:
        return s
    out = s
    out = re.sub(r'hx{2}p(s?)://', r'http\1://', out, flags=re.I)   # hxxp/hxxps
    out = re.sub(r'hxxp(s?)://', r'http\1://', out, flags=re.I)
    out = out.replace('[.]', '.').replace('(.)', '.').replace('{.}', '.')
    out = re.sub(r'\[\s*\.\s*\]', '.', out)
    out = out.replace('[:]', ':').replace('[/:]', '/')
    out = re.sub(r'hxxps?\s*\[:\]\s*//', 'https://', out, flags=re.I)
    out = re.sub(r'hxxps?\s*://', 'https://', out, flags=re.I)
    return out

def _defang(domain: str) -> str:
    d = domain.lower()
    if d.startswith("www."): d = d[4:]
    return d.replace(".", "[.]")

# ==================== TEXT EXTRACTION (PDF/OCR) ====================
def read_text(path: Path, use_ocr: bool=False, ocr_dpi: int=200) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = (page.extract_text() or "").strip()
            if t:
                text_parts.append(t)

    if text_parts:
        return "\n\n".join(text_parts)

    if OCR_AVAILABLE and use_ocr:
        try:
            images = convert_from_path(str(path), dpi=ocr_dpi)
            ocr_text = []
            for img in images:
                ocr_text.append(pytesseract.image_to_string(img))
            joined = "\n\n".join(s.strip() for s in ocr_text if s and s.strip())
            if joined:
                return joined
        except Exception:
            return ""
    return ""

# ==================== ATT&CK (techniques) ====================
def attack_ids(text: str) -> List[str]:
    ids = {m.group(0).upper() for m in RE_ATTACK_ID.finditer(text)}
    def key(t):
        main=int(t[1:5]); sub=int(t.split('.',1)[1]) if '.' in t else 0
        return (main, sub)
    return sorted(ids, key=key)

def fetch_attack_enrichment(ids: List[str], timeout=20) -> Tuple[Dict[str,str], Dict[str,List[str]], Dict[str,str]]:
    """
    Fetch attack-pattern names and map techniques -> tactic IDs (TA000x).
    Returns: (names_map, technique_to_tactics, tactic_id_to_name)
    """
    if not ids:
        return {}, {}, {}
    try:
        url="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        r=requests.get(url, timeout=timeout); r.raise_for_status()
        data=r.json()
        want=set(id.upper() for id in ids)

        def _norm_tactic_name(s: str) -> str:
            return re.sub(r'\s+', ' ', s.replace('-', ' ').strip().lower())

        names: Dict[str, str] = {}
        tech_to_tactics: Dict[str, List[str]] = {}
        tactic_name_to_id: Dict[str, str] = {}
        tactic_id_to_name: Dict[str, str] = {}

        # collect tactics
        for obj in data.get("objects", []):
            if obj.get("type") == "x-mitre-tactic":
                name = obj.get("name")
                ext_id = None
                for ref in obj.get("external_references", []) or []:
                    if ref.get("source_name") == "mitre-attack":
                        ext_id = ref.get("external_id", "").upper()
                        break
                if ext_id and name:
                    tactic_name_to_id[_norm_tactic_name(name)] = ext_id
                    tactic_id_to_name[ext_id] = name

        # map techniques -> tactic IDs
        for obj in data.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            exid = None
            for ref in obj.get("external_references", []) or []:
                if ref.get("source_name") == "mitre-attack":
                    exid = ref.get("external_id", "").upper()
                    break
            if not exid or exid not in want:
                continue

            names[exid] = obj.get("name")
            ta_ids = []
            for kcp in obj.get("kill_chain_phases", []) or []:
                phase = kcp.get("phase_name")
                if not phase:
                    continue
                mapped = tactic_name_to_id.get(_norm_tactic_name(phase))
                if mapped:
                    ta_ids.append(mapped)
            tech_to_tactics[exid] = sorted(set(ta_ids))

        return names, tech_to_tactics, tactic_id_to_name
    except Exception:
        return {}, {}, {}



# ==================== DELIVERIES/SECTORS/INFRA ====================
def deliveries(text: str) -> List[str]:
    low=text.lower(); out=set()
    for kw in DELIVERY:
        if re.search(rf'\b{re.escape(kw)}\b', low):
            out.add("phishing" if "phish" in kw else kw)
    return sorted(out)

def sectors(text: str) -> List[str]:
    low=text.lower(); out=set()
    for kw in SECTORS:
        if kw in low: out.add(kw.capitalize())
    return sorted(out)

def _normalize_defanged_quick(s: str) -> str:
    if not s:
        return s
    out = s
    out = re.sub(r'hx{2}p(s?)://', r'http\1://', out, flags=re.I)   # hxxp/hxxps
    out = out.replace('[.]', '.').replace('(.)', '.').replace('{.}', '.')
    out = re.sub(r'\[\s*\.\s*\]', '.', out)
    out = re.sub(r'[:]\s*/\s*', '://', out)
    out = re.sub(r'\s+', ' ', out)   # collapse whitespace
    return out

def _flatten_tools_field(tools_field):
    # If tools_field is a dict of categories -> lists, flatten to a single list
    if isinstance(tools_field, dict):
        flat = []
        for v in tools_field.values():
            if isinstance(v, (list, tuple, set)):
                flat.extend(list(v))
            else:
                flat.append(v)
        return [str(x).strip() for x in flat if x]
    elif isinstance(tools_field, (list, tuple, set)):
        return [str(x).strip() for x in tools_field if x]
    else:
        # Single string
        return [str(tools_field).strip()]

def _compile_keyword_patterns(keywords):
    """
    Returns list of tuples (original_keyword, compiled_regex) sorted by keyword length descending
    so multi-word matches get tested first.
    """
    pats = []
    seen = set()
    for kw in sorted(set(keywords), key=len, reverse=True):
        if not kw: 
            continue
        orig = kw
        kw = kw.lower().strip()
        # escape then allow flexible whitespace between words
        # and accept defanged dots like [.] or (.)
        # transform '.' in kw to a pattern matching '.' or '[.]' or '(.)'
        esc = re.escape(kw)
        # replace escaped space sequences with \s+ (allow variable whitespace/punctuation)
        esc = esc.replace(r'\ ', r'\s+')
        # replace literal escaped dot '\.' back to a pattern
        esc = esc.replace(r'\.', r'(?:\.|\[\.\]|\(\.\))')
        # Build a safe regex with word boundaries for alphanumeric tokens to avoid mid-word matches
        if re.match(r'^[A-Za-z0-9\-\_\s\.\(\)\[\]]+$', kw):
            pattern = r'(?<![A-Za-z0-9])' + esc + r'(?![A-Za-z0-9])'
        else:
            pattern = esc
        try:
            pats.append((orig, re.compile(pattern, re.IGNORECASE)))
            seen.add(orig)
        except re.error:
            # fallback: literal substring matcher via simple escaped regex
            try:
                pats.append((orig, re.compile(re.escape(orig), re.IGNORECASE)))
            except re.error:
                # last resort: skip
                continue
    return pats

def infra_hints(text: str) -> Dict[str, List[str]]:
    """
    Returns {"types": [...], "tools": [...], "domains": [...]}
    Robust matching for defanged/OCR'd text and flexible INFRA_KEYWORDS shape.
    """
    if not text:
        return {"types": [], "tools": [], "domains": []}

    # normalize text once
    normalized = _normalize_defanged_quick(text).lower()

    out_types = set()
    out_tools = set()
    out_domains = set()

    # domains: reuse your earlier logic (simple defensive approach)
    for m in re.findall(r'([a-z0-9\-_]+\.[a-z]{2,})', normalized):
        tld = m.rsplit('.', 1)[-1]
        # use your COMMON_TLDS or a permissive check
        if tld.isalpha() and 2 <= len(tld) <= 24:
            base = m.lower()
            if base.startswith("www."): base = base[4:]
            if base not in DEFAULT_DOMAIN_ALLOWLIST:
                out_domains.add(_defang(base))

    # Flatten and prepare keyword sets from INFRA_KEYWORDS
    services_list = INFRA_KEYWORDS.get("services", []) or []
    tools_field = INFRA_KEYWORDS.get("tools", []) or []
    tools_list = _flatten_tools_field(tools_field)

    # compile patterns (multi-word first)
    service_patterns = _compile_keyword_patterns([s.lower().strip() for s in services_list if s])
    tool_patterns = _compile_keyword_patterns([t.lower().strip() for t in tools_list if t])

    # match services
    for kw, rx in service_patterns:
        if rx.search(normalized):
            out_types.add(kw)

    # match tools
    for kw, rx in tool_patterns:
        if rx.search(normalized):
            out_tools.add(kw)

    return {
        "types": sorted(out_types),
        "tools": sorted(out_tools),
        "domains": sorted(out_domains)
    }


# ==================== IOCS ====================
def is_domain_strict(d: str, loose: bool=False) -> bool:
    if RE_IPV4.fullmatch(d): return False
    if '.' not in d: return False
    d = d.lower()
    if d.startswith("www."): d = d[4:]
    if d in WINDOWS_PROCESSES: return False
    if d in DEFAULT_DOMAIN_ALLOWLIST: return False
    tld = d.rsplit('.',1)[-1]
    if tld in BANNED_TLDS: return False
    if loose:
        return 2 <= len(tld) <= 24 and tld.isalpha()
    else:
        return tld in COMMON_TLDS

def iocs(text: str, max_each=5, loose_domains: bool=False) -> Dict[str, List[str]]:
    norm_text = _normalize_defanged(text)

    hashes=[]
    hashes.extend([m.group(0).lower() for m in RE_SHA256.finditer(norm_text)])
    hashes.extend([m.group(0).lower() for m in RE_SHA1.finditer(norm_text)])
    hashes.extend([m.group(0).lower() for m in RE_MD5.finditer(norm_text)])
    hashes=dedupe(hashes)[:max_each]

    domains=[]
    for u in RE_URL.findall(norm_text):
        host=re.sub(r'^https?://','',u,flags=re.I).split('/')[0].split(':')[0].strip().lower()
        if host.startswith("www."): host = host[4:]
        if host and is_domain_strict(host, loose=loose_domains) and host not in domains:
            domains.append(host)
            if len(domains)>=max_each: break
    if len(domains)<max_each:
        for m in RE_DOMAIN.finditer(norm_text):
            d=m.group(1).lower()
            if d.startswith("www."): d = d[4:]
            if is_domain_strict(d, loose=loose_domains) and d not in domains:
                domains.append(d)
                if len(domains)>=max_each: break

    mutexes=[]
    for m in RE_MUTEX_KEYED.finditer(text):
        val=m.group(1).strip()
        if len(val)>=4 and val.lower() not in {"and","or","control","handling"}:
            mutexes.append(val)
    for m in RE_MUTEX_GLOBAL.finditer(text):
        mutexes.append(m.group(0))
    mutexes = [m for m in mutexes if not re.search(r'(?:^|\\)(?:temp|users|windows|program files)(?:\\|$)', m, re.I)]
    cleaned=[]
    for mu in mutexes:
        mu2=mu.strip().strip('\'"[]()'); mu2=re.sub(r'\\\\+', r'\\', mu2)
        if len(mu2)>=4 and not re.search(r'\s', mu2): cleaned.append(mu2)
    mutexes=dedupe(cleaned)[:max_each]

    return {"hashes": hashes, "domains": domains, "mutexes": mutexes}

def fetch_mitre_groups(timeout=20):
    """
    Returns:
      alias_index: dict[alias_lower] -> {"id": <stix_id>, "gid": <G#### or None>, "name": <canonical>}
      id_to_name:  dict[stix_id] -> canonical name
    """
    alias_index = {}
    id_to_name = {}
    try:
        url="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        r=requests.get(url, timeout=timeout); r.raise_for_status()
        data=r.json()

        for obj in data.get("objects", []):
            if obj.get("type") == "intrusion-set":
                stix_id = obj.get("id")
                name = (obj.get("name") or "").strip()

                # Extract MITRE external_id (G-IDs like G0010) if present
                gid = None
                for ref in obj.get("external_references", []) or []:
                    if ref.get("source_name") == "mitre-attack":
                        ext_id = (ref.get("external_id") or "").strip()
                        if ext_id and ext_id.upper().startswith("G"):
                            gid = ext_id.upper()
                            break

                aliases = set()
                for field in ("aliases", "x_mitre_aliases"):
                    for a in obj.get(field, []) or []:
                        a = (a or "").strip()
                        if a:
                            aliases.add(a)

                if name:
                    id_to_name[stix_id] = name
                    for a in {name, *aliases}:
                        alias_index[a.lower()] = {"id": stix_id, "gid": gid, "name": name}
    except Exception:
        pass
    return alias_index, id_to_name


def find_mitre_intrusion_set(text: str, mitre_alias_index: dict) -> Optional[dict]:
    low = text.lower()
    for alias, entry in mitre_alias_index.items():
        pat = r'(?<![A-Za-z0-9])' + re.escape(alias) + r'(?![A-Za-z0-9])'
        m = re.search(pat, low, re.IGNORECASE)
        if m:
            # Avoid software/browser/library false positives
            if alias.lower() in {"chromium", "android", "zeus", "mercury"}:
                continue
            # Skip if common tech names
            if re.search(r'\b(browser|engine|framework|project|library)\b', low[m.start():m.start()+60]):
                continue
            return {
                "id": entry["id"],
                "gid": entry.get("gid"),
                "name": entry["name"],
                "matched_alias": m.group(0)
            }
    return None



def _normalize_taxonomy_token(s: str) -> str:
    t = s.strip()
    rules = [
        (r'(?i)^APT[\s\-]?(\d{1,3})$',        r'APT\1'),
        (r'(?i)^UNC[\s\-]?(\d{1,5})$',        r'UNC\1'),
        (r'(?i)^(DEV)[\s\-]?(\d{3,5})$',      r'DEV-\2'),
        (r'(?i)^(Storm)[\s\-]?(\d{3,5})$',    r'Storm-\2'),
        (r'(?i)^(TG)[\s\-]?(\d{3,5})$',       r'TG-\2'),
        (r'(?i)^(TA)[\s\-]?([1-9]\d{2,4})$',  r'TA\2'),
    ]
    for pat, repl in rules:
        t2 = re.sub(pat, repl, t)
        if t2 != t:
            return t2
    return t

def extract_campaign(text: str) -> Optional[str]:
    """
    Heuristic: pick up 'campaign/operation/activity <Name>' style mentions.
    Returns a short, title-cased-ish string if it looks like a real name.
    """
    m = re.search(
        r'\b(?:campaign|operation|activity)\s+(?:known\s+as|called|titled)?\s*([A-Z][A-Za-z0-9\-\s]{3,80})',
        text,
        re.IGNORECASE
    )
    if m:
        cand = m.group(1).replace("\n"," ").strip()
        return cand if is_name_shape(cand) else None
    return None


# ==================== ACTOR EXTRACTION (now MITRE + optional MS + taxonomy) ====================
def extract_actor(text: str, doc, mitre_alias_index: dict, allow_fallback: bool=False):
    """
    Returns tuple: (display_actor: Optional[str], mitre_actor: Optional[dict], group_name: Optional[str])
    - display_actor: a human-readable string (prefer MITRE canonical; otherwise MS alias; otherwise taxonomy surface)
    - mitre_actor: {"id":..., "name":..., "matched_alias":...} if matched
    - group_name: canonical MITRE intrusion-set name if available, else None
    """
    # 1) MITRE intrusion-set alias match (preferred for canon)
    mitre_actor = find_mitre_intrusion_set(text, mitre_alias_index)
    if mitre_actor:
        return mitre_actor["name"], mitre_actor, mitre_actor["name"]

    # 2) Optional Microsoft alias map
    if _ALIASES_CI:
        for alias_low, canon in _ALIASES_CI.items():
            pat = r'(?<![A-Za-z0-9])' + re.escape(alias_low) + r'(?![A-Za-z0-9])'
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                # Return MS canonical as display; no MITRE object in this path
                return canon, None, None
        for canon in _MS_CANON:
            pat = r'(?<![A-Za-z0-9])' + re.escape(canon) + r'(?![A-Za-z0-9])'
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(0), None, None

    # 3) Explicit phrases
    for pat in [
        r'attribut(?:ed|ion)\s+to\s+([A-Z][A-Za-z0-9\-\s&]{2,80})',
        r'(?:tracked|track)\s+as\s+([A-Z][A-Za-z0-9\-\s&]{2,80})',
        r'(?:also\s+known\s+as|aka)\s+([A-Z][A-Za-z0-9\-\s&]{2,80})',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cand = re.split(r'[\(\[\,:;].*$', m.group(1).strip())[0].strip()
            if RE_TACTIC_TA000X.fullmatch(cand):
                continue
            if any(rx.search(cand) for rx in ACTOR_TAXONOMY) or is_name_shape(cand):
                return cand, None, None

    # 4) Taxonomy tokens
    for rx in ACTOR_TAXONOMY:
        m = rx.search(text)
        if m:
            cand = _normalize_taxonomy_token(m.group(0))
            if RE_TACTIC_TA000X.fullmatch(cand):
                continue
            return cand, None, None

    # 5) NER fallback
    if allow_fallback and hasattr(doc, "ents"):
        for ent in doc.ents:
            if ent.label_ in ("ORG","PRODUCT"):
                s = ent.text.replace("\n"," ").strip()
                if not is_name_shape(s): continue
                if RE_TACTIC_TA000X.fullmatch(s): continue
                if re.search(r'\b(group|team|unit|actor|collective)\b', s, re.I):
                    return s, None, None

    return None, None, None

# ==================== MALWARE ====================
MALWARE_PRIMARY_HINTS = ("ransomware","primary","main payload","campaign","encrypt","impact","payload")

def collect_malware_candidates(text: str, doc) -> List[Tuple[str,int,int]]:
    cands = []
    for m in re.finditer(r'\b([A-Z][A-Za-z0-9\-]{2,60})\s+(%s)(?:\s+(%s))?\b' % (MALWARE_SUFFIX, VERSION_TOK),
                         text, re.IGNORECASE):
        base = m.group(1).strip()
        kind = m.group(2)
        ver  = m.group(3) if (m.lastindex and m.lastindex >= 3) else None
        if not is_name_shape(base): continue
        name = " ".join([base, kind] + ([ver] if ver else []))
        cands.append((name, m.start(), 5))
    for m in re.finditer(r'\b([A-Z][A-Za-z0-9\-]{2,60})\s+(%s)\b' % MALWARE_SUFFIX, text, re.IGNORECASE):
        base=m.group(1).strip(); kind=m.group(2)
        if is_name_shape(base):
            cands.append((f"{base} {kind}", m.start(), 4))
    for m in re.finditer(r'\bmalware\s+(?:family|strain)\s*[:\-]?\s*([A-Z][A-Za-z0-9\-\s]{2,60})', text, re.IGNORECASE):
        cand = m.group(1).replace("\n"," ").strip()
        if is_name_shape(cand): cands.append((cand, m.start(), 3))
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT","ORG"):
            s = ent.text.replace("\n"," ").strip()
            if re.search(r'(ransom|trojan|backdoor|rat|loader|stealer|bot|miner|worm)', s, re.I) and is_name_shape(s):
                idx = text.find(ent.text)
                cands.append((s, idx if idx>=0 else 0, 2))

    window = 120
    scored=[]
    for name, pos, base_score in cands:
        lo=max(0, pos-window); hi=min(len(text), pos+window)
        ctx=text[lo:hi].lower()
        score=base_score + sum(1 for h in MALWARE_PRIMARY_HINTS if h in ctx)
        scored.append((name, pos, int(score)))

    best={}
    for name, pos, sc in scored:
        if name not in best or sc > best[name][1]:
            best[name]=(pos, sc)
    out=[(n, best[n][0], best[n][1]) for n in best]
    out.sort(key=lambda x: (-x[2], x[1]))
    return out

def choose_primary(cands: List[Tuple[str,int,int]]) -> Optional[str]:
    return cands[0][0] if cands else None

# ==================== NARRATIVE (with strong redaction) ====================
_MALWARE_CLASSES = ["ransomware","trojan","backdoor","loader","stealer","bot","worm","rat"]

def _expand_actor_terms(actor: str) -> set:
    terms = set()
    if not actor:
        return terms
    base = actor.strip()
    terms.add(base)
    for inner in re.findall(r'\(([^)]+)\)', base):
        t = inner.strip()
        if t:
            terms.add(t)
    for tok in re.findall(r'[A-Za-z][A-Za-z0-9\-]+', base):
        lo = tok.lower()
        if lo not in {"group","team","unit","actor","collective"}:
            terms.add(tok)
    return terms

def _expand_malware_terms(mal: str) -> set:
    terms = set()
    if not mal:
        return terms
    base = mal.strip()
    terms.add(base)
    head = base.split()[0]
    terms.add(head)
    for cls in _MALWARE_CLASSES:
        terms.add(f"{head} {cls}")
        terms.add(f"{cls} {head}")
        terms.add(f"{head}-{cls}")
        terms.add(f"{cls}-{head}")
    return terms

def _multi_redact(text: str, terms: set, replacement: str) -> str:
    for t in sorted(terms, key=len, reverse=True):
        pat = r'(?<![A-Za-z0-9])' + re.escape(t) + r'(?![A-Za-z0-9])'
        text = re.sub(pat, replacement, text, flags=re.IGNORECASE)
    return text

def build_narrative(
    doc,
    attack_ids_list: List[str],
    delivery: List[str],
    sectors_list: List[str],
    infra: Dict[str, List[str]],
    actor: Optional[str],
    malware: Optional[str],
    tactic_names: List[str] = None
) -> str:
    ACTION_VERBS = re.compile(
        r'\b(exfiltrat|encrypt|persist|bypass|elevat|inject|download|drop|beacon|'
        r'command|control|masquerad|steal|collect|discover|recon|communicat|'
        r'execute|obfuscat|delet|disable|wipe|lateral|enumerat)\w*\b',
        re.I
    )
    TECH_TERMS = re.compile(
        r'\b(socket|dll|process|registry|service|autorun|startup|persistence|'
        r'beacon|c2|command and control|mutex|payload|loader|backdoor|'
        r'credential|token|hook|injection|sandbox|vm)\b',
        re.I
    )
    IMPERATIVE_ANYWHERE = re.compile(
        r'(?:^|[:;\-\s])\s*(monitor|block|prevent|educate|implement|use|enable|disable|'
        r'apply|patch|update|restrict|enforc|harden)\b',
        re.I
    )
    ATTACK_LEADING = re.compile(r'^\s*T\d{4}(?:\.\d+)?\s+[A-Z][\w\-]+(?:\s*\([^)]+\))?\s*[:\-]?\s*')

    def preprocess(s: str) -> str:
        s2 = ATTACK_LEADING.sub('', s).strip()
        return re.sub(r'\s+', ' ', s2).strip()

    def too_many_titlecase_chunks(s: str) -> bool:
        tokens = re.findall(r'\b[A-Z][a-zA-Z]+\b', s)
        return len(tokens) >= 3 and re.search(r'(?:\b[A-Z][a-zA-Z]+\b\s+){2,}\b[A-Z][a-zA-Z]+\b', s) is not None

    def looks_like_ratio_or_date(s: str) -> bool:
        return bool(re.search(r'\b\d{1,4}/\d{1,2}(?:/\d{2,4})?\b', s))

    def bullet_like(s: str) -> bool:
        return bool(re.match(r'^\s*[-*•·]\s*\S+', s)) or s.endswith(':')

    def is_structural_noise(s: str) -> bool:
        sl = s.lower()
        if any(h in sl for h in HEADER_WORDS): return True
        if len(s) > 320: return True
        if s.count('|') + s.count('•') + s.count('·') > 2: return True
        if len(RE_SHA256.findall(s)) >= 2: return True
        if len(RE_URL.findall(s)) >= 2: return True
        if too_many_titlecase_chunks(s): return True
        if looks_like_ratio_or_date(s): return True
        if bullet_like(s): return True
        if IMPERATIVE_ANYWHERE.search(s): return True
        if re.search(r'\b(ID|Technique|Mitigation|Remediation)\b(?:\s+\b[A-Z][a-zA-Z]+\b){1,}', s): return True
        return False

    def sent_score(s: str) -> float:
        score = 0.0
        score += 2.0 * len(RE_ATTACK_ID.findall(s))
        score += 1.0 * len(RE_URL.findall(s))
        score += 0.7 * len(RE_DOMAIN.findall(s))
        score += 0.7 * len(RE_IPV4.findall(s))
        score += 1.2 * len(ACTION_VERBS.findall(s))
        if TECH_TERMS.search(s): score += 0.8
        L = len(s)
        if 60 <= L <= 240: score += 0.8
        elif L < 40 or L > 280: score -= 0.4
        if RE_ATTACK_ID.search(s): score += 0.5
        return score

    def norm_tokens(s: str) -> set:
        toks = re.findall(r'[a-z0-9]+', s.lower())
        return set(t for t in toks if len(t) > 2)

    def jaccard(a: set, b: set) -> float:
        if not a or not b: return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    candidates = []
    for span in doc.sents:
        raw = " ".join(span.text.split())
        if not raw:
            continue
        s = preprocess(raw)
        if not s or is_structural_noise(s):
            continue
        sc = sent_score(s)
        if sc > 0:
            candidates.append((sc, s))

    if not candidates:
        narrative = ""
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected, seen = [], []
        for sc, text in candidates:
            tset = norm_tokens(text)
            if any(jaccard(tset, prev) > 0.6 for prev in seen):
                continue
            selected.append(text)
            seen.append(tset)
            if len(selected) >= 3:
                break
        cleaned = []
        for s in selected:
            s = s.strip()
            if not s.endswith(('.', '!', '?')):
                s += '.'
            cleaned.append(s)
        narrative = " ".join(cleaned).strip()

    # ---- Strong redaction ----
    if malware:
        narrative = _multi_redact(narrative, _expand_malware_terms(malware), "[REDACTED_MALWARE]")
    if actor:
        narrative = _multi_redact(narrative, _expand_actor_terms(actor), "[REDACTED_ACTOR]")
    # redact taxonomy tokens that might leak attribution
    for rx in ACTOR_TAXONOMY:
        for m in rx.findall(narrative):
            narrative = re.sub(r'(?<!\w)'+re.escape(m)+r'(?!\w)', "[REDACTED_ACTOR]", narrative, flags=re.I)

    return narrative

# ==================== PIPELINE ====================
def process(path: Path, nlp, allow_actor_fallback: bool, allow_malware_fallback: bool, fetch_map: bool, use_ocr: bool, loose_domains: bool):
    text = read_text(path, use_ocr=use_ocr)
    if not text:
        raise ValueError(f"No text extracted: {path.name}")
    doc = nlp(text)

    # ATT&CK: techniques/tactics
    ttp = attack_ids(text)

    # Fetch enrichment (names + technique->tactics mapping) when requested
    attack_names = {}
    tech2tac = {}
    tac_id_to_name = {}
    if fetch_map and ttp:
        attack_names, tech2tac, tac_id_to_name = fetch_attack_enrichment(ttp)

    delivery = deliveries(text)
    secs = sectors(text)
    infra = infra_hints(text)
    ioc = iocs(text, loose_domains=loose_domains)

    # MITRE intrusion-set alias index
    mitre_alias_index, mitre_id_to_name = fetch_mitre_groups()

    # Actor/Group extraction (now MITRE-aware)
    display_actor, mitre_actor, group_name = extract_actor(text, doc, mitre_alias_index, allow_fallback=allow_actor_fallback)

    # Malware
    mal_cands = collect_malware_candidates(text, doc)
    primary_mal = choose_primary(mal_cands)
    if not primary_mal and allow_malware_fallback and mal_cands:
        primary_mal = mal_cands[0][0]

    attack_tactic_ids = sorted({ta for tas in tech2tac.values() for ta in tas}) if tech2tac else []
    attack_tactic_names = [tac_id_to_name.get(t, t) for t in attack_tactic_ids]

    public = {
        "attack_ttp_ids": ttp,
        "attack_tactic_ids": attack_tactic_ids,
        "attack_tactics": attack_tactic_names,
        "delivery_vectors": delivery,
        "sectors": secs,
        "infrastructure_hints": infra,
        "iocs": {"hashes": ioc["hashes"], "domains": ioc["domains"], "mutexes": ioc["mutexes"]},
        "narrative": build_narrative(doc, ttp, delivery, secs, infra, display_actor, primary_mal, tactic_names=attack_tactic_names)
    }
    gt = {
    "threat_actor": display_actor if display_actor else None,            
    "group_name": group_name if group_name else None,                    
    "group_id": (mitre_actor.get("gid") if mitre_actor else None),     
    "mitre_intrusion_set": (                                            
        {
            "id": mitre_actor["id"],
            "gid": mitre_actor.get("gid"),
            "name": mitre_actor["name"],
            "matched_alias": mitre_actor.get("matched_alias")
        } if mitre_actor else None
    ),
    "malware_family": primary_mal if primary_mal else None,
    "campaign_name": extract_campaign(text) or None,
    "attack_ttp_map": attack_names if fetch_map else {},
    "technique_to_tactics": tech2tac if fetch_map else {}
}

    return public, gt

def save(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Malware PDF extractor (v10): MITRE alias-aware actors/groups + strong redaction.")
    ap.add_argument("-i","--input-dir", required=True)
    ap.add_argument("-o","--output-dir", required=True)
    ap.add_argument("--allow-actor-fallback", action="store_true", help="Loosen actor detection (may increase false positives).")
    ap.add_argument("--allow-malware-fallback", action="store_true", help="Loosen malware detection (may increase false positives).")
    ap.add_argument("--no-attack-map", action="store_true", help="Skip MITRE ATT&CK name fetch.")
    ap.add_argument("--ocr", action="store_true", help="Enable OCR fallback (requires pytesseract/pdf2image).")
    ap.add_argument("--loose-domains", action="store_true", help="Allow non-standard TLDs (off by default).")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    indir=Path(args.input_dir); outdir=Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        nlp=spacy.load("en_core_web_sm")
    except Exception:
        raise SystemExit("spaCy model missing. Run: python -m spacy download en_core_web_sm")

    pdfs=sorted([p for p in indir.iterdir() if p.suffix.lower()==".pdf"])
    if not pdfs: raise SystemExit("No PDFs found.")

    if args.ocr and not OCR_AVAILABLE:
        print("[!] --ocr requested but pytesseract/pdf2image/Pillow not available. Proceeding without OCR.")

    for p in pdfs:
        try:
            if not args.quiet: print(f"[+] {p.name}")
            pub, gt = process(
                p, nlp,
                allow_actor_fallback=args.allow_actor_fallback,
                allow_malware_fallback=args.allow_malware_fallback,
                fetch_map=not args.no_attack_map,
                use_ocr=args.ocr and OCR_AVAILABLE,
                loose_domains=args.loose_domains
            )
            save(pub, outdir / f"{p.stem}_public.json")
            save(gt, outdir / f"{p.stem}_groundtruth.json")
        except Exception as e:
            print(f"[!] {p.name}: {e}")

if __name__ == "__main__":
    main()
