"""Summarize phone_qc.jsonl.

1. NTU Paiwan ASR (102 clips, gold PHON tier): phone error rate per configuration.
2. Study 1 broadcast clips (Amis, Paiwan, Truku; 10 each): inventory shape per configuration,
   i.e. number of phone types, share of output outside the grammar inventory, and the rate of
   each language's diagnostic segments (grammar segments the Mandarin inventory lacks).
3. Provenance: does cmn at emit=2 reproduce the phone-type set of data/emit2?
"""
import json, re, sys, unicodedata, collections
from pathlib import Path

# Usage: python phone_qc_analysis.py <phone_qc.jsonl> <ntu_gold_phon.json> <data/emit2 dir> <inventories dir> <out.json>

H = Path(__file__).resolve().parent
rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
gold = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
inv = {iso: [l.strip() for l in (Path(sys.argv[4]) / f"{iso}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
       for iso in ("ami", "pwn", "trv", "tay", "tsu", "tao")}
CONFIGS = ["cmn", "cmn_emit2", "ipa", "shipped", "grammar"]

def tnorm(p):
    p = p.replace("͡", "").replace("ɡ", "g")
    return {"ʦ": "ts", "ɉ": "ɟ"}.get(p, p)

# gold -> uni2005 space, using the inventory build's own nearest-phone mapping (report.json)
rep = json.loads((Path(sys.argv[4]) / "report.json").read_text(encoding="utf-8"))
G2U = {m["grammar"]: m["uni2005"][0] for m in rep["pwn"]["mapping"]}
G2U.update({"R": "r"})   # NTU gold writes a uvular rhotic as R; the grammar has one rhotic /r/

def gold_tokens(s):
    out = []
    for ch in s.replace(" ", ""):
        t = tnorm(ch)
        out.append(tnorm(G2U.get(t, t)))
    return out

def lenient(p):
    """Collapse diacritic-only distinctions (dental, retroflex hook kept as base letter)."""
    base = "".join(c for c in unicodedata.normalize("NFD", p) if not unicodedata.combining(c))
    return {"ɖ": "d", "ɭ": "l", "ɫ": "l", "ʈ": "t", "dʒ": "dʒ"}.get(base, base)

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

out = {}
ntu = [r for r in rows if r["set"] == "ntu"]
per = {}
for c in CONFIGS:
    e = n = el = 0
    for r in ntu:
        g = gold_tokens(gold[r["id"]]["phon"])
        h = [tnorm(p) for p in r[c]]
        e += lev(g, h); n += len(g)
        el += lev([lenient(x) for x in g], [lenient(x) for x in h])
    per[c] = {"PER": round(e / n, 4), "PER_lenient": round(el / n, 4), "gold_phones": n}
# diagnostic segment recall on NTU: count of q, ʔ, v, z, ɟ, c, ʎ in output vs gold
diag_pwn = ["q", "ʔ", "v", "z", "ɟ", "c", "ʎ", "r", "ə"]
gcount = collections.Counter(t for r in ntu for t in gold_tokens(gold[r["id"]]["phon"]))
per["_diagnostic_counts"] = {"gold": {d: gcount[d] for d in diag_pwn}}
for c in CONFIGS:
    hc = collections.Counter(tnorm(p) for r in ntu for p in r[c])
    per["_diagnostic_counts"][c] = {d: hc[d] for d in diag_pwn}
out["ntu_paiwan"] = per

DIAG = {
    "ami": ["ʔ", "r", "ɾ", "b", "v", "d", "ʊ", "t͡ɬ"],
    "pwn": ["q", "ʔ", "v", "z", "ɟ", "c", "ʎ", "l̪", "r", "b", "d", "g"],
    "trv": ["q", "ʔ", "ɣ", "b", "d", "ɟ", "ɫ", "ɾ"],
}
s1 = {}
for iso in ("ami", "pwn", "trv"):
    rr = [r for r in rows if r["set"] == "study1" and r["iso"] == iso]
    gi = set(tnorm(p) for p in inv[iso])
    s1[iso] = {}
    for c in CONFIGS:
        if c not in rr[0]:
            continue
        toks = [tnorm(p) for r in rr for p in r[c]]
        cnt = collections.Counter(toks)
        tot = len(toks)
        s1[iso][c] = {
            "phones": tot, "types": len(cnt),
            "share_outside_grammar_inventory": round(sum(v for k, v in cnt.items() if k not in gi) / tot, 4),
            "diagnostic_per_1000": {d: round(1000 * cnt[tnorm(d)] / tot, 2) for d in DIAG[iso]},
            "top10": cnt.most_common(10),
        }
out["study1_broadcast"] = s1

# provenance of data/emit2: its phone types and token count, read from the TextGrids themselves
emit2_counts = collections.Counter()
for tg in sorted(Path(sys.argv[3]).glob("*.allo.emit2.TextGrid")):
    for lab in re.findall(r'^"([^"]*)"\s*$', tg.read_text(encoding="utf-8"), flags=re.M):
        if lab not in ("IntervalTier", "allosaurus-emit2", "<blk>", ""):
            emit2_counts[lab] += 1
EMIT2_TYPES = set(emit2_counts)
s1all = [r for r in rows if r["set"] == "study1"]
for c in ("cmn", "cmn_emit2", "ipa"):
    types = set(p for r in s1all for p in r[c])
    out.setdefault("emit2_provenance", {})[c] = {
        "types": len(types), "jaccard_with_data_emit2": round(len(types & EMIT2_TYPES) / len(types | EMIT2_TYPES), 3),
        "in_emit2_not_here": sorted(EMIT2_TYPES - types), "here_not_in_emit2": sorted(types - EMIT2_TYPES),
        "tokens_here": sum(len(r[c]) for r in s1all), "tokens_data_emit2": sum(emit2_counts.values())}
Path(sys.argv[5]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
