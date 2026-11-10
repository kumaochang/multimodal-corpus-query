"""EXPLORATORY (not registered; added 2026-10-18 after the §6.2 primary results, run_log 26a347e).

Splits the §6.2 cross-lingual lift by the query's Haiku label, to see whether the above-chance
lift in every distance class comes from Reporting queries (largely anchor and studio shots) or
also from Explaining queries. Changes no registered result and decides nothing.

Same units, representations, retrieval, hit and chance as s62_lift.py (imported, unchanged).
Reported for 14d_tight (primary) and 14d_resid (framing residualized):
  for Explaining and Reporting queries separately: mean hit, mean chance, lift, per distance
  class and pooled over all 30 ordered pairs; and Explaining - Reporting pooled lift.
Intervals: query days resampled within language, 10,000, default_rng(20261023).
The three rare labels (20 queries) are counted but not analyzed.

Usage:
  python pipeline/analysis/s62_by_label.py --units data/study3_runs/label_analysis/units.tsv \
    --units-k data/study3_runs/unit_table/units_k.npz --trees work/data/aligned_trees \
    --pose-dirs work/data/pose --out data/study3_runs/s62/s62_by_label.json
"""
import argparse, csv, json
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

import s62_lift as S
from feat_kin import _pose_features, is_tight, IDX_POSITION  # noqa: E402  (path set by s62_lift)
from label_units import load_utterances  # noqa: E402

SEED = 20261023
GROUPS = ("Explaining", "Reporting")


def analyse(E, rows, rng):
    L = S.LANGS; tl = {t: i for i, t in enumerate(L)}; nL = len(L)
    per_lang, H, C = {}, defaultdict(list), defaultdict(list)
    for q in L:
        qi = [i for i, r in enumerate(rows) if r["lang"] == q]
        pos = {i: k for k, i in enumerate(qi)}
        M = np.full((len(qi), 2 * nL), np.nan)
        for t in L:
            if t == q:
                continue
            for i, d, h, c in S.pair_hits(E, rows, q, t):
                lab = rows[i]["label"]
                if lab in GROUPS:
                    g = GROUPS.index(lab)
                    M[pos[i], g * nL + tl[t]] = d
                    H[(lab, S.dist_class(q, t))].append(h); C[(lab, S.dist_class(q, t))].append(c)
        days = sorted({rows[i]["day"] for i in qi})
        per_lang[q] = (np.array([days.index(rows[i]["day"]) for i in qi]), M)

    def lift(means, g, cls=None):
        return np.mean([means[q][g * nL + tl[t]] for q, t in permutations(L, 2)
                        if cls is None or S.dist_class(q, t) == cls])

    out = {}
    for g, lab in enumerate(GROUPS):
        out[lab] = {"queries": sum(r["label"] == lab for r in rows)}
        for cls in ("Short", "Middle", "Long", None):
            o, b = S.day_boot(per_lang, lambda m, g=g, cls=cls: lift(m, g, cls), rng)
            key = cls or "all_pairs"
            out[lab][key] = {"lift": round(float(o), 4),
                             "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}
            if cls:
                out[lab][key]["hit"] = round(float(np.mean(H[(lab, cls)])), 4)
                out[lab][key]["chance"] = round(float(np.mean(C[(lab, cls)])), 4)
    o, b = S.day_boot(per_lang, lambda m: lift(m, 0) - lift(m, 1), rng)
    out["explaining_minus_reporting_all_pairs"] = {
        "diff": round(float(o), 4), "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--units-k", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--pose-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.units_k)
    zid = {u: k for k, u in enumerate(z["ids"])}
    ok = {u: bool(z["within_shot"][k]) and int(z["k_missing"][k]) == 0 for u, k in zid.items()}
    prim = [r for r in csv.DictReader(open(a.units, encoding="utf-8"), delimiter="\t")
            if ok.get(r["unit_id"]) and r["kept"] == "True"]
    need = {r["unit_id"] for r in prim}
    by_clip = defaultdict(set)
    for r in prim:
        by_clip[r["clip"]].add(r["unit_id"])
    full = {}
    for clip in sorted(by_clip):
        p = next(Path(d) / f"{clip}.pose.json" for d in a.pose_dirs if (Path(d) / f"{clip}.pose.json").exists())
        recs = json.loads(p.read_text()); offs = np.array([f["offset"] for f in recs])
        for uid, s, e in load_utterances(Path(a.trees) / f"{clip}.aligntree.TextGrid", clip):
            if uid in by_clip[clip]:
                full[uid] = _pose_features([recs[i] for i in np.nonzero((offs >= s * 1000) & (offs < e * 1000))[0]])
    rows = [dict(unit_id=r["unit_id"], clip=r["clip"], lang=r["lang"], day=r["day"], label=r["haiku"]) for r in prim]
    ids = [r["unit_id"] for r in rows]
    K14 = np.array([z["K"][zid[u]] for u in ids])
    F = np.array([z["F"][zid[u]] for u in ids])
    R14, _ = S.residualize(np.delete(np.array([full[u] for u in ids]), IDX_POSITION, 1), F)
    rng = np.random.default_rng(SEED)
    res = {"exploratory": True, "seed": SEED, "resamples": S.B, "units": len(rows),
           "query_labels": dict(Counter(r["label"] for r in rows)),
           "14d_tight": analyse(K14, rows, rng), "14d_resid": analyse(R14, rows, rng)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
