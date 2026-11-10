"""Phone layer for Study 3: replaces moco2026/new_make_allosaurus.py.

What changes, and why (analysis/q24_pipeline_quality.md):
  - the inventory follows the clip's language (pipeline/inventories/<iso>.txt, derived from the
    語法概論 phonology chapters) instead of lang_id="cmn" for every clip;
  - emit = 1.0 instead of 2.0 (emit 2 more than doubles insertions: PER 1.29 against 0.50 at
    emit 1 under the same inventory on NTU Paiwan ASR);
  - every output carries a provenance sidecar naming the inventory file and its SHA-256.

Arm A is not changed. PREREGISTRATION.md §7 Amendment 3 item 2 fixes the detector as `ari` and
`maya` over Allosaurus `pwn` phone strings, the configuration whose operating point was measured
in analysis/q21 (shipped inventory, emit 1.0). For Paiwan clips this stage therefore writes a
second tier, `armA_pwn`, decoded exactly that way; the grammar-inventory tier `phones` feeds
everything else.

Usage:
  python phone_layer.py --manifest clips.json --inventories pipeline/inventories --out data/phones
manifest: [{"id": "news-Paiwan-20240803-02", "wav": ".../x.wav", "iso": "pwn"}, ...]
"""
import argparse, hashlib, json, platform
from pathlib import Path
import allosaurus
from allosaurus.app import read_recognizer
from praatio import textgrid

EMIT = 1.0

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def decode(rec, wav, lang_id):
    res = rec.recognize(wav, lang_id=lang_id, emit=EMIT, timestamp=True)
    entries = []
    for line in res.splitlines():
        start, dur, phone = line.split()
        s, d = float(start), float(dur)
        entries.append((round(s, 3), round(s + d, 3), phone))
    # allosaurus gives a fixed per-phone duration; clip each interval at the next onset
    fixed = []
    for i, (s, e, p) in enumerate(entries):
        nxt = entries[i + 1][0] if i + 1 < len(entries) else e
        fixed.append((s, max(s + 0.001, min(e, nxt)), p))
    return fixed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--inventories", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rec = read_recognizer()  # uni2005
    for m in json.loads(Path(a.manifest).read_text(encoding="utf-8")):
        tg_path = out / f"{m['id']}.phones.TextGrid"
        if tg_path.exists():
            continue
        inv = Path(a.inventories) / f"{m['iso']}.txt"
        fixed = decode(rec, m["wav"], str(inv))
        tiers = [("phones", fixed)]
        if m["iso"] == "pwn":
            tiers.append(("armA_pwn", decode(rec, m["wav"], "pwn")))
        end = max(t[-1][1] for _, t in tiers if t) if any(t for _, t in tiers) else 0.01
        tg = textgrid.Textgrid()
        for name, ents in tiers:
            tg.addTier(textgrid.IntervalTier(name, ents, 0, end))
        tg.save(str(tg_path), format="short_textgrid", includeBlankSpaces=False)
        (out / f"{m['id']}.phones.provenance.json").write_text(json.dumps({
            "stage": "phone_layer", "tool": "allosaurus", "version": allosaurus.__version__ if hasattr(allosaurus, "__version__") else "1.0.2",
            "model": "uni2005", "lang_id": f"inventory file {inv.name}", "inventory_sha256": sha256(inv),
            "emit": EMIT, "approximate": False,
            "armA_tier": "lang_id='pwn' (Allosaurus shipped inventory), emit 1.0, as registered" if m["iso"] == "pwn" else None, "input_wav": m["wav"], "input_sha256": sha256(m["wav"]),
            "python": platform.python_version()}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(m["id"], len(fixed))

if __name__ == "__main__":
    main()
