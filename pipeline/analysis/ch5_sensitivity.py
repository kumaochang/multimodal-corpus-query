"""Chapter 5: registered sensitivity analyses, the §5.4 composition matrix and the §6.2 pair table.

Requested by diss-writing 2026-10-19 before writing §5.1.2. Statistics only; no text, label or
transcript content is printed. None of these replaces a primary result.

Imports the frozen retrieval code of ch5_retrieval.py (Retriever, l2rows, summarize) unchanged and
rebuilds its representations with the same settings, so a sensitivity differs from the main run only
in the one thing named.

1. Amendment 4 B2 sensitivity (both text modalities): the XLM-R vectors (xlmr_sub, xlmr_tr)
   centered on the mean of each language's valid rows, then L2-normalized (corpus/build_similarity.py,
   scoped per language). Valid = non-empty input, as in the main run. The language mean is taken over
   all utterances of the language (before Mandarin exclusion), like the MFCC86 statistics (plan §10).
   TF-IDF is not centered: B2 names "the same vectors", i.e. the pooled XLM-R vectors, and
   build_similarity.py keeps TF-IDF uncentered. kept/own; W1-W3, plus X for xlmr_sub.
2. Amendment 8 item 2: utterances of duration in [29.9 s, 30.1 s] (aligned-tree boundaries), counted
   per language (all and kept). If any kept utterance falls in the range, every representation is
   rerun kept/own without them (removed as queries and as candidates); otherwise not rerun.
3. Analysis plan §5 rule 2: without the input-gap union (units.tsv `gap`, empty OCR or missing
   keyframe); every representation kept/own, W1-W3 and X where cross-language is defined.
   For 1-3: lift@1 of Explaining, remainder and Explaining - remainder with day-cluster 95%
   intervals, hit and chance beside them; the main run's values printed beside when --main is given.
4. §5.4 composition matrix (Amendment 9 item 6 point 4 measures; descriptive): on kept utterances,
   per language the mean of each of the 8 framing measures (unit_table.py F, over all pose records of
   the utterance) with day-cluster 95% intervals (days within language, ratio of sums over defined
   values); days, clips, utterances, utterances without pose records, defined n per measure. Second
   version on the §6.2 units (kept, within one shot, kinematic primary present: Amendment 10 item 2),
   with the kept utterances below the 1 s tight minimum counted per language (all kept, and kept
   within one shot).
5. §6.2 ordered pairs: from s62_results.json primary_14d_tight.pairs, per ordered pair the class,
   queries, targets (the target language's §6.2 units), hit, label-specific chance and pair lift.

Seed 20261021 (one generator, analyses in the order above), 10,000 resamples, percentile intervals.

Usage (item5 environment; numpy, scipy):
  python pipeline/analysis/ch5_sensitivity.py --features data/study3_runs/features_full \
    --units data/study3_runs/label_analysis/units.tsv --units-k data/study3_runs/unit_table/units_k.npz \
    --trees work/data/aligned_trees --s62 data/study3_runs/s62/s62_results.json \
    --main data/study3_runs/ch5/ch5_retrieval.json --out data/study3_runs/ch5/ch5_sensitivity.json
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_retrieval as cr  # noqa: E402

SEED, B = 20261021, 10_000
F_NAMES = ["bbox_h", "bbox_area", "face_extent", "pose_extent", "fallback_frac", "ppf_mean", "frac_1p", "frac_3plus"]
LOW, HIGH = 29.9, 30.1
GROUPS = ("Explaining", "remainder", "Explaining_minus_remainder")


def load(a):
    from scipy import sparse
    F = Path(a.features)
    z = np.load(F / "features.npz")
    ids = list(z["ids"])
    rows = {r["unit_id"]: r for r in csv.DictReader(open(a.units, encoding="utf-8"), delimiter="\t")}
    if set(rows) != set(ids):
        sys.exit("units.tsv and features disagree")
    U = {"lang": np.array([rows[u]["lang"] for u in ids]), "day": np.array([rows[u]["day"] for u in ids]),
         "clip": np.array([rows[u]["clip"] for u in ids]), "label": np.array([rows[u]["haiku"] for u in ids]),
         "kept": np.array([rows[u]["kept"] == "True" for u in ids]),
         "gap": np.array([rows[u]["gap"] == "True" for u in ids])}
    li = {l: i for i, l in enumerate(sorted(set(U["lang"])))}
    idx = {"lab": np.array([cr.LABELS.index(x) for x in U["label"]]),
           "clip": np.unique(U["clip"], return_inverse=True)[1],
           "day": np.unique(U["day"], return_inverse=True)[1],
           "lang": np.array([li[x] for x in U["lang"]])}
    # representations, exactly as ch5_retrieval.main
    W = z["whisper"].astype(np.float32)
    M = z["mfcc86"].astype(np.float64)
    mu, sd = M.mean(0), M.std(0)
    Mz = np.where(sd > 0, (M - mu) / np.where(sd > 0, sd, 1), 0.0)
    P = z["prosody"].astype(np.float64); Pz = np.zeros_like(P)
    for l in li:
        m = U["lang"] == l
        x = P[m]; lm = np.nanmean(x, 0); lm = np.where(np.isnan(lm), 0.0, lm)
        x = np.where(np.isnan(x), lm, x); s = x.std(0)
        Pz[m] = np.where(s > 0, (x - x.mean(0)) / np.where(s > 0, s, 1), 0.0)
    XS, XT = z["xlmr_sub"].astype(np.float32), z["xlmr_tr"].astype(np.float32)
    TS, TT = sparse.load_npz(F / "tfidf_sub.npz"), sparse.load_npz(F / "tfidf_tr.npz")
    sub_ok = np.abs(XS).sum(1) > 0; tr_ok = np.abs(XT).sum(1) > 0
    k = np.load(a.units_k)
    kpos = {u: i for i, u in enumerate(k["ids"])}
    K = np.zeros((len(ids), 14), np.float32); kin_ok = np.zeros(len(ids), bool)
    for i, u in enumerate(ids):
        j = kpos.get(u)
        if j is not None and bool(k["within_shot"][j]) and int(k["k_missing"][j]) == 0:
            K[i] = k["K"][j]; kin_ok[i] = True
    reps = {"whisper": (W, np.ones(len(ids), bool), True), "mfcc86": (Mz, np.ones(len(ids), bool), True),
            "prosody": (Pz, np.ones(len(ids), bool), True), "xlmr_sub": (XS, sub_ok, True),
            "tfidf_sub": (TS, sub_ok, True), "xlmr_tr": (XT, tr_ok, False), "tfidf_tr": (TT, tr_ok, False),
            "kin14": (K, kin_ok, True)}
    return ids, U, idx, reps, k


def center_by_language(X, ok, lang):
    Y = np.zeros_like(X, dtype=np.float64)
    for l in sorted(set(lang)):
        m = (lang == l) & ok
        if m.any():
            Y[m] = X[m] - X[m].mean(0, keepdims=True)
    return Y


def analyse(Xn, ok, cross, units_mask, U, idx, rng):
    """kept/own block as in ch5_retrieval: W1-W3 within language, X if cross."""
    rt = cr.Retriever(Xn, idx["lab"], idx["clip"], idx["day"], idx["lang"])
    units = np.nonzero(units_mask & ok)[0]
    block = {}
    for mode in ("W1", "W2", "W3"):
        parts, qs = [], []
        for l in sorted(set(U["lang"])):
            q = units[U["lang"][units] == l]
            if len(q) < 2:
                continue
            parts.append(rt.run(q, q, mode)); qs.append(q)
        r = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
        block[mode] = slim(cr.summarize(r, np.concatenate(qs), U, rng))
    if cross:
        parts, qs = [], []
        for l in sorted(set(U["lang"])):
            q = units[U["lang"][units] == l]; c = units[U["lang"][units] != l]
            if len(q) and len(c):
                parts.append(rt.run(q, c, "X")); qs.append(q)
        r = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
        block["X"] = slim(cr.summarize(r, np.concatenate(qs), U, rng))
    return block


def slim(s):
    L, D = s["lift_at_1"], s["descriptive"]
    return {"queries": s["queries"], "mean_candidates": s["mean_candidates"],
            "lift": {g: L[g] for g in GROUPS if g in L},
            "hit_chance": {g: {"hit": D[g].get("hit"), "chance": D[g].get("chance")} for g in ("Explaining", "remainder")}}


def comp_matrix(F, nrec, sel, U, rng, b):
    """Per language: mean of each framing measure over sel, day-cluster CI (ratio of sums)."""
    out = {}
    for l in sorted(set(U["lang"][sel])):
        m = sel & (U["lang"] == l)
        days = sorted(set(U["day"][m])); di = np.array([days.index(d) for d in U["day"][m]])
        X = F[m]; ok = ~np.isnan(X)
        S = np.zeros((len(days), X.shape[1])); C = np.zeros_like(S)
        for j in range(X.shape[1]):
            np.add.at(S[:, j], di[ok[:, j]], X[ok[:, j], j]); np.add.at(C[:, j], di[ok[:, j]], 1)
        ix = rng.integers(0, len(days), size=(b, len(days)))
        with np.errstate(invalid="ignore", divide="ignore"):
            bm = S[ix].sum(1) / C[ix].sum(1)
            pm = S.sum(0) / C.sum(0)
        row = {"days": len(days), "clips": len(set(U["clip"][m])), "utterances": int(m.sum()),
               "no_pose_records": int((nrec[m] == 0).sum()), "measures": {}}
        for j, n in enumerate(F_NAMES):
            row["measures"][n] = {"n": int(C[:, j].sum()),
                                  "mean": None if C[:, j].sum() == 0 else round(float(pm[j]), 4),
                                  "ci95": None if C[:, j].sum() == 0 else
                                  [round(float(np.nanpercentile(bm[:, j], 2.5)), 4), round(float(np.nanpercentile(bm[:, j], 97.5)), 4)]}
        out[l] = row
    return out


def fmt_ci(x):
    return "NA" if not x or x.get("ci95") is None else f"{x['mean']:.4f} [{x['ci95'][0]:.4f},{x['ci95'][1]:.4f}]"


def fmt_d(x):
    return "NA" if not x else f"{x['diff']:.4f} [{x['ci95'][0]:.4f},{x['ci95'][1]:.4f}]"


def print_block(title, results, main=None, main_key=None):
    print(f"\n## {title}")
    print("rep\tmode\tq\tExplaining lift\tremainder lift\tdiff\tE hit/chance\tR hit/chance" + ("\tmain diff" if main else ""))
    for rep, block in results.items():
        for mode, s in block.items():
            L, H = s["lift"], s["hit_chance"]
            line = (f"{rep}\t{mode}\t{s['queries']}\t{fmt_ci(L.get('Explaining'))}\t{fmt_ci(L.get('remainder'))}\t"
                    f"{fmt_d(L.get('Explaining_minus_remainder'))}\t{H['Explaining']['hit']}/{H['Explaining']['chance']}\t"
                    f"{H['remainder']['hit']}/{H['remainder']['chance']}")
            if main:
                mb = main["analyses"].get(main_key(rep), {}).get(mode)
                line += "\t" + (fmt_d(mb["lift_at_1"].get("Explaining_minus_remainder")) if mb else "NA")
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--units-k", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--s62", required=True)
    ap.add_argument("--main", help="ch5_retrieval.json, for side-by-side printing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resamples", type=int, default=B)
    a = ap.parse_args()
    cr.B = a.resamples                                    # summarize/day_boot read the module global
    ids, U, idx, reps, k = load(a)
    rng = np.random.default_rng(SEED)
    main_res = json.loads(Path(a.main).read_text(encoding="utf-8")) if a.main else None
    res = {"seed": SEED, "resamples": a.resamples, "utterances": len(ids), "kept": int(U["kept"].sum())}

    # 1. Amendment 4 B2: centered XLM-R
    b2 = {}
    for name in ("xlmr_sub", "xlmr_tr"):
        X, ok, cross = reps[name]
        Xc = cr.l2rows(center_by_language(X, ok, U["lang"]).astype(np.float32))
        Xc[~ok] = 0.0
        b2[name] = analyse(Xc, ok, cross, U["kept"], U, idx, rng)
    res["1_am4_b2_centered_xlmr"] = b2
    print(json.dumps({"done": "1_am4_b2"}), flush=True)

    # 2. Amendment 8 item 2: 29.9-30.1 s
    from label_units import load_utterances
    dur = {}
    for clip in sorted(set(U["clip"])):
        for uid, s, e in load_utterances(Path(a.trees) / f"{clip}.aligntree.TextGrid", clip):
            dur[uid] = e - s
    if any(u not in dur for u in ids):
        sys.exit("an utterance has no tree duration")
    d = np.array([dur[u] for u in ids])
    in_rng = (d >= LOW) & (d <= HIGH)
    langs = sorted(set(U["lang"]))
    res["2_am8_item2"] = {"range_s": [LOW, HIGH],
                          "count_all": {l: int((in_rng & (U["lang"] == l)).sum()) for l in langs},
                          "count_kept": {l: int((in_rng & U["kept"] & (U["lang"] == l)).sum()) for l in langs},
                          "total_all": int(in_rng.sum()), "total_kept": int((in_rng & U["kept"]).sum())}
    if (in_rng & U["kept"]).any():
        r2 = {}
        for name, (X, ok, cross) in reps.items():
            r2[name] = analyse(cr.l2rows(X), ok & ~in_rng, cross, U["kept"], U, idx, rng)
        res["2_am8_item2"]["results"] = r2
    else:
        res["2_am8_item2"]["results"] = "not rerun: no kept utterance in range"
    print(json.dumps({"done": "2_am8_item2"}), flush=True)

    # 3. input gaps
    r3 = {}
    for name, (X, ok, cross) in reps.items():
        r3[name] = analyse(cr.l2rows(X), ok & ~U["gap"], cross, U["kept"], U, idx, rng)
    res["3_no_input_gap"] = {"removed_all": int(U["gap"].sum()), "removed_kept": int((U["gap"] & U["kept"]).sum()),
                             "results": r3}
    print(json.dumps({"done": "3_no_input_gap"}), flush=True)

    # 4. composition matrix
    kpos = {u: i for i, u in enumerate(k["ids"])}
    if any(u not in kpos for u in ids):
        sys.exit("units_k.npz lacks an utterance")
    j = np.array([kpos[u] for u in ids])
    Fm = k["F"][j].astype(float); nrec = k["n_records"][j]; within = k["within_shot"][j].astype(bool)
    kmiss = k["k_missing"][j].astype(int) == 1
    unit62 = U["kept"] & within & ~kmiss
    res["4_composition"] = {
        "measures": F_NAMES,
        "kept": comp_matrix(Fm, nrec, U["kept"], U, rng, a.resamples),
        "s62_units": comp_matrix(Fm, nrec, unit62, U, rng, a.resamples),
        "s62_units_total": int(unit62.sum()),
        "below_tight_minimum_kept": {l: int((U["kept"] & kmiss & (U["lang"] == l)).sum()) for l in langs},
        "below_tight_minimum_kept_within_shot": {l: int((U["kept"] & within & kmiss & (U["lang"] == l)).sum()) for l in langs}}
    print(json.dumps({"done": "4_composition"}), flush=True)

    # 5. §6.2 pairs
    s62 = json.loads(Path(a.s62).read_text(encoding="utf-8"))
    P = s62["primary_14d_tight"]["pairs"]; nunits = s62["units"]
    res["5_s62_pairs"] = [{"pair": p, "class": v["class"], "queries": v["n"], "targets": nunits[p.split("->")[1]],
                           "hit": v["hit"], "chance": v["chance"], "lift": v["lift"]} for p, v in P.items()]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # printed tables (statistics only)
    print_block("1. Amendment 4 B2: XLM-R centered per language (kept/own)", b2, main_res, lambda r: f"{r}|kept|own")
    print("\n## 2. Amendment 8 item 2: utterances in [29.9, 30.1] s")
    print(json.dumps({k2: v for k2, v in res["2_am8_item2"].items() if k2 != "results"}, ensure_ascii=False))
    if isinstance(res["2_am8_item2"]["results"], dict):
        print_block("2. without [29.9, 30.1] s (kept/own)", res["2_am8_item2"]["results"], main_res, lambda r: f"{r}|kept|own")
    else:
        print(res["2_am8_item2"]["results"])
    print(f"\n## 3. input-gap union removed: all {res['3_no_input_gap']['removed_all']}, kept {res['3_no_input_gap']['removed_kept']}")
    print_block("3. without input-gap union (kept/own)", r3, main_res, lambda r: f"{r}|kept|own")
    for ver in ("kept", "s62_units"):
        print(f"\n## 4. composition matrix: {ver}" + (f" (total {res['4_composition']['s62_units_total']})" if ver == "s62_units" else ""))
        print("lang\tdays\tclips\tutterances\tno_pose\t" + "\t".join(F_NAMES))
        for l, row in res["4_composition"][ver].items():
            print(f"{l}\t{row['days']}\t{row['clips']}\t{row['utterances']}\t{row['no_pose_records']}\t" +
                  "\t".join(f"{fmt_ci(row['measures'][n])} n={row['measures'][n]['n']}" for n in F_NAMES))
    print("\nbelow 1 s tight minimum (kept): " + json.dumps(res["4_composition"]["below_tight_minimum_kept"]))
    print("below 1 s tight minimum (kept, within one shot): " + json.dumps(res["4_composition"]["below_tight_minimum_kept_within_shot"]))
    print("\n## 5. §6.2 ordered pairs (primary 14d tight)")
    print("pair\tclass\tqueries\ttargets\thit\tchance\tlift")
    for p in res["5_s62_pairs"]:
        print(f"{p['pair']}\t{p['class']}\t{p['queries']}\t{p['targets']}\t{p['hit']}\t{p['chance']}\t{p['lift']}")


if __name__ == "__main__":
    main()
