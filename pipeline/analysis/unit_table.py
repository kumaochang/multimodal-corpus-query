"""Utterance-level study: kinematic and framing table for every labeled utterance, and the
eligible-unit counts of §6.2 and Amendment 9 item 8, computed before any outcome.

Units (Amendment 9 item 6 point 2): labeled utterances lying entirely within one camera shot
(no shot cut strictly inside [start, end], as analysis/q15) with the kinematic primary present
(item 6 point 1; Amendment 10 item 2): at least 1 s of tight pose records, i.e.
ceil(1000 / span) records (4 at Study 3's 266.9 ms; 8 at Study 1's 133.5 ms). The tight rule,
the 14-d vector and the 8 framing measures are imported from item5/feat_kin.py unchanged.

Eligibility
- §6.2 primary: unit and kept as Indigenous-language speech (`kept` in label_analysis units.tsv).
  Sensitivity: `kept_mixed`.
- Item 8 queries: unit and Gemini answer `mandarin` (not `mixed`); targets as §6.2. A community
  with fewer than 10 eligible Mandarin units is left out of D.
Counts are reported per language, per language and Haiku label, and per distance class (ordered
pairs; Short = Atayal-Truku, Long = Tao against the other five, Middle = the remaining nine pairs).

Prints counts only. Writes under --out (gitignored data/ tree):
  units_k.npz   ids, K (n x 14), k_missing, F (n x 8), n_records, n_tight, within_shot, shot_index
  eligibility.json  the printed object
No outcome (retrieval, similarity, lift) is computed here.

Usage (Study 3):
  python pipeline/analysis/unit_table.py --units data/study3_runs/label_analysis/units.tsv \
    --trees work/data/aligned_trees --shots data/study3_runs/shots.json \
    --pose-dirs work/data/pose --out data/study3_runs/unit_table
"""
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "item5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feat_kin import _pose_features, is_tight, framing, IDX_POSITION  # noqa: E402
from label_units import LABELS, load_utterances  # noqa: E402

LANGS = ["ami", "pwn", "trv", "tay", "tsu", "tao"]
MIN_MANDARIN = 10


def dist_class(a, b):
    if {a, b} == {"tay", "trv"}:
        return "Short"
    if "tao" in (a, b):
        return "Long"
    return "Middle"


def truthy(x):
    return str(x) == "True"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="label_analysis.py units.tsv")
    ap.add_argument("--trees", required=True)
    ap.add_argument("--shots", required=True)
    ap.add_argument("--pose-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.units, encoding="utf-8"), delimiter="\t"))
    by_clip = defaultdict(dict)
    for r in rows:
        by_clip[r["clip"]][r["unit_id"]] = r
    shots = json.loads(Path(a.shots).read_text(encoding="utf-8"))

    ids, K, km, F, nrec, ntight, within, shot_idx = [], [], [], [], [], [], [], []
    tot = Counter()
    for clip in sorted(by_clip):
        tree = Path(a.trees) / f"{clip}.aligntree.TextGrid"
        pose_p = next((Path(d) / f"{clip}.pose.json" for d in a.pose_dirs
                       if (Path(d) / f"{clip}.pose.json").exists()), None)
        recs = json.loads(pose_p.read_text()) if pose_p else []
        tot["clips_without_pose"] += pose_p is None
        if clip not in shots:
            tot["clips_without_shots"] += 1
        starts = np.array([s[0] for s in shots.get(clip, {}).get("shots", [])])
        span = float(np.median([f["span"] for f in recs])) if recs else float("nan")
        need = math.ceil(1000 / span) if span == span else 10 ** 9
        offs = np.array([f["offset"] for f in recs]) if recs else np.zeros(0)
        for uid, s, e in load_utterances(tree, clip):
            if uid not in by_clip[clip]:
                continue
            k_in = int(np.sum((starts > s) & (starts < e))) if len(starts) else -1
            idx = np.nonzero((offs >= s * 1000) & (offs < e * 1000))[0]
            sel = [recs[i] for i in idx]
            tf = [f for f in sel if is_tight(f)]
            if len(tf) >= need:
                K.append(np.delete(_pose_features(tf), IDX_POSITION)); km.append(0)
            else:
                K.append(np.full(14, np.nan)); km.append(1)
            F.append(framing(sel)); ids.append(uid); nrec.append(len(sel)); ntight.append(len(tf))
            within.append(k_in == 0)
            shot_idx.append(int(np.searchsorted(starts, s, side="right") - 1) if len(starts) else -1)
            tot[f"span_{round(span, 1)}"] += 1

    info = {u: (w, m) for u, w, m in zip(ids, within, km)}
    for r in rows:
        w, m = info.get(r["unit_id"], (False, 1))
        r["unit"] = w and m == 0
        r["within"] = w
        r["k_ok"] = m == 0

    def count(sel, name):
        out = {"n": len(sel), "by_language": {l: sum(r["lang"] == l for r in sel) for l in LANGS}}
        out["by_language_label"] = {l: dict(Counter(r["haiku"] for r in sel if r["lang"] == l)) for l in LANGS}
        out["clips"] = len({r["clip"] for r in sel}); out["days"] = len({r["day"] for r in sel})
        out["days_by_language"] = {l: len({r["day"] for r in sel if r["lang"] == l}) for l in LANGS}
        return out

    res = {"utterances_in_units_tsv": len(rows), "utterances_measured": len(ids),
           "clips": len(by_clip), "counts": dict(tot),
           "within_one_shot": int(sum(within)), "k_present": int(len(km) - sum(km)),
           "within_and_k": int(sum(1 for w, m in zip(within, km) if w and m == 0))}
    kept = [r for r in rows if truthy(r["kept"])]
    res["funnel_kept"] = {l: {"kept": sum(r["lang"] == l for r in kept),
                              "within_shot": sum(r["lang"] == l and r["within"] for r in kept),
                              "k_present": sum(r["lang"] == l and r["k_ok"] for r in kept),
                              "unit": sum(r["lang"] == l and r["unit"] for r in kept)} for l in LANGS}
    elig = [r for r in kept if r["unit"]]
    elig_mx = [r for r in rows if truthy(r["kept_mixed"]) and r["unit"]]
    res["s62_primary"] = count(elig, "primary")
    res["s62_kept_mixed"] = count(elig_mx, "kept_mixed")
    pairs = defaultdict(list)
    n_by = res["s62_primary"]["by_language"]
    for q, t in permutations(LANGS, 2):
        pairs[dist_class(q, t)].append({"pair": f"{q}->{t}", "queries": n_by[q], "targets": n_by[t]})
    res["s62_pairs_by_class"] = {c: {"ordered_pairs": len(v), "min_queries": min(p["queries"] for p in v),
                                     "min_targets": min(p["targets"] for p in v), "pairs": v}
                                 for c, v in pairs.items()}
    mand = [r for r in rows if r["lid_answer"] == "mandarin" and r["unit"]]
    res["item8_mandarin_queries"] = {l: sum(r["lang"] == l for r in mand) for l in LANGS}
    res["item8_communities_left_out"] = [l for l in LANGS if res["item8_mandarin_queries"][l] < MIN_MANDARIN]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "units_k.npz", ids=np.array(ids), K=np.array(K, float), k_missing=np.array(km),
             F=np.array(F, float), n_records=np.array(nrec), n_tight=np.array(ntight),
             within_shot=np.array(within), shot_index=np.array(shot_idx))
    (out / "eligibility.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
