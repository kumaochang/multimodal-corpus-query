"""Shared helpers for the judged directive experiment (Amendment 9 item 5, Amendment 10)."""
import json
from pathlib import Path

def load_items(path):
    """Unique utterances of items.json (batch 3 repeats share the utt_id of their original)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    seen, out = set(), []
    for r in d["items"]:
        if r["utt_id"] in seen:
            continue
        seen.add(r["utt_id"]); out.append(r)
    return out

def find_file(dirs, name):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None
