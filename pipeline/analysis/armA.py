"""Arm A, the symbolic route: PREREGISTRATION.md Amendment 3 item 2 and §4 item 10.

The detector is `ari` and `maya` searched in Allosaurus `pwn` phone strings (shipped inventory,
emit 1.0), the configuration whose operating point analysis/q21 measured. Here it runs per
utterance instead of per NTU sentence clip:

  - phone string of an utterance: the phones of the `armA_pwn` tier (pipeline/stages/
    phone_layer.py, `<clip>.phones.TextGrid`) whose interval midpoint lies in
    [utterance start, utterance end), in time order, concatenated with no separator;
  - `ari`: occurrences of the regex `ari` in that string; unwindowed (primary) and over the
    first 25, 15 and 10 phones (sensitivity, §4 item 10);
  - `maya`: occurrences of the regex `ma[jy]a`, never windowed (§4 item 10);
  - an utterance is in Arm A when either target occurs, unwindowed; `ari` and `maya` are
    recorded separately so that they can be reported separately throughout.

Check of the implementation (2026-10-14): applied to Allosaurus `pwn` emit-1.0 output of the 102
NTU Paiwan ASR clips, each clip taken as one utterance, it reproduces q21's unwindowed table
exactly (ari: 84 detected, TP 20, FN 2, FP 26, TN 54; maya: 13, 4, 1, 8, 89). Separator or not
gives the same counts there.

Usage:
  python armA.py --phones <dir of *.phones.TextGrid> --trees <aligned_trees dir> \
      --clips <file with one Paiwan clip stem per line> --out <dir>
Writes <out>/armA_utterances.jsonl (one row per utterance: ids, phone count, counts per target
and window, in_A) and <out>/armA.provenance.json. Prints counts only.
"""
import argparse, hashlib, json, re, sys
from pathlib import Path
from praatio import textgrid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stages"))
from utterances import load_utterances  # noqa: E402

TIER = "armA_pwn"
ARI, MAYA = re.compile("ari"), re.compile("ma[jy]a")
WINDOWS = (25, 15, 10)

def phones_in(entries, s, e):
    return [p for a, b, p in entries if s <= (a + b) / 2 < e]

def detect(seq):
    row = {"n_phones": len(seq), "ari": len(ARI.findall("".join(seq))),
           "maya": len(MAYA.findall("".join(seq)))}
    for w in WINDOWS:
        row[f"ari_w{w}"] = len(ARI.findall("".join(seq[:w])))
    row["in_A"] = row["ari"] > 0 or row["maya"] > 0
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phones", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clips = [l.strip() for l in Path(a.clips).read_text(encoding="utf-8").splitlines() if l.strip()]
    tot = {"clips": 0, "clips_without_tree": 0, "utterances": 0, "utterances_without_phones": 0,
           "in_A": 0, "ari": 0, "maya": 0, **{f"ari_w{w}_utterances": 0 for w in WINDOWS}}
    inputs = {}
    with (out / "armA_utterances.jsonl").open("w", encoding="utf-8") as fh:
        for clip in clips:
            tree = Path(a.trees) / f"{clip}.aligntree.TextGrid"
            ph = Path(a.phones) / f"{clip}.phones.TextGrid"
            if not tree.exists():
                tot["clips_without_tree"] += 1
                continue
            tg = textgrid.openTextgrid(str(ph), includeEmptyIntervals=False)
            entries = [(e.start, e.end, e.label) for e in tg.getTier(TIER).entries]
            inputs[clip] = hashlib.sha256(ph.read_bytes()).hexdigest()
            tot["clips"] += 1
            for uid, s, e in load_utterances(tree, clip):
                r = {"utt_id": uid, "clip": clip, **detect(phones_in(entries, s, e))}
                fh.write(json.dumps(r) + "\n")
                tot["utterances"] += 1
                tot["utterances_without_phones"] += r["n_phones"] == 0
                tot["in_A"] += r["in_A"]; tot["ari"] += r["ari"] > 0; tot["maya"] += r["maya"] > 0
                for w in WINDOWS:
                    tot[f"ari_w{w}_utterances"] += r[f"ari_w{w}"] > 0
    (out / "armA.provenance.json").write_text(json.dumps({
        "rule": "PREREGISTRATION.md Amendment 3 item 2; §4 item 10; implementation in this file",
        "tier": TIER, "patterns": {"ari": ARI.pattern, "maya": MAYA.pattern}, "windows": WINDOWS,
        "phones_sha256": inputs, "totals": tot}, indent=1), encoding="utf-8")
    print(json.dumps(tot))

if __name__ == "__main__":
    main()
