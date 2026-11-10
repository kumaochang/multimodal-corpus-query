"""§6.2: cross-lingual kinematic retrieval lift by genetic-distance class, and the same-community
contrast (PREREGISTRATION §2 items 5-6; Amendment 9 item 6 points 1, 2, 4 and item 8;
Amendment 10 item 2). Runs after s62_power.py (run_log 1532f36).

Units (item 6 point 2): labeled utterances lying within one shot whose kinematic primary exists
(unit_table.py flags), kept as Indigenous-language speech (Amendment 9 item 1). Every variant is
evaluated on the same units, so variants differ only in representation.

Representations (item 6 point 1)
  primary   14d_tight     Study 1 _pose_features on tight records minus 4 position dims
                          (must equal unit_table's K; asserted)
  secondary 18d_tight     the same with the position dims
            14d_resid     14d on all records of the utterance, residualized on the first 3 PCs of
            18d_resid     the 8 framing measures (q10 Rescue B: framing NaN -> 0, z-scored, PCA by
                          SVD on the analysis units, OLS with intercept)
            9d_gesture    q8 side_features on tight records; its per-hand minimum of 10 detected
                          frames at Study 1's 133.5 ms is applied as the same duration (1.335 s,
                          ceil(1335/span) records: 5 at 266.9 ms), as Amendment 10 item 2 did for
                          the tight minimum. Fixed here before any lift was computed.
            18d_full      18d on all records
Cosine on the raw vectors, as Study 1 (l2 normalization leaves cosine unchanged); a zero vector
has similarity 0 to everything.

Lift (item 6 point 2): for each ordered language pair (q, t), each unit of q retrieves its top-1
cosine neighbour among the units of t; hit = labels match; chance = share of the query's label
among t's units; pair lift = mean(hit - chance); class lift = mean over the class's ordered pairs.
Primary: Middle - Long, one-sided (prediction Middle >= Long), day-cluster bootstrap: query days
resampled within language, hits fixed, 10,000 resamples, default_rng(SEED_62); p = share of
resamples <= 0; 95% two-sided and 90% (one-sided 95% lower bound) intervals. Short descriptive.
Size of this test on these units is about 0.056 to 0.067 and MDE80 0.06 to 0.08 (s62_power).

Item 8: queries = units (as above) whose Gemini answer is `mandarin`; targets = the kept units of
each community. Lift_same(c): targets of c minus the query's own clip; Lift_other(c): mean of the
lifts with each other community's targets; D = mean over communities with >= 10 queries of
Lift_same - Lift_other; two-sided, day-cluster bootstrap over query days within community,
10,000, default_rng(20261019). Sensitivity: same-clip targets included. Reported with D: the 8
framing means for Mandarin and Indigenous-language units, and each class's share of units
meeting the tight minimum among its within-shot utterances.

Sensitivities of the primary (not decisions): kept_mixed; no Mandarin exclusion; without input
gaps; Gemini utterance labels (exploratory). The Middle subdivision of §2 item 7 is not run.

Prints the summary JSON and writes it, with per-pair tables, to --out.

Usage (Study 3):
  python pipeline/analysis/s62_lift.py --units data/study3_runs/label_analysis/units.tsv \
    --units-k data/study3_runs/unit_table/units_k.npz --trees work/data/aligned_trees \
    --pose-dirs work/data/pose --out data/study3_runs/s62
"""
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "item5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feat_kin import _pose_features, is_tight, IDX_POSITION, F_NAMES  # noqa: E402
from label_units import load_utterances  # noqa: E402

LANGS = ["ami", "pwn", "trv", "tay", "tsu", "tao"]
SEED_62, SEED_8, B = 20261021, 20261019, 10_000
MIN_MANDARIN = 10
GESTURE_MIN_MS = 10 * 133.47
MIN_RUN = 2


# ── q8 side_features, verbatim except that MIN_FRAMES is a parameter ─────────
def _centroid(pts):
    xs = [p[0] for p in pts if len(p) >= 2]; ys = [p[1] for p in pts if len(p) >= 2]
    if not xs: return None
    return float(np.mean(xs)), float(np.mean(ys))


def side_features(frames, side, MIN_FRAMES):
    body_h, body_cy = [], []
    tracks = {"left_hand": [], "right_hand": []}
    for fi, fr in enumerate(frames):
        s = fr[side]
        if s is None: continue
        pts = s.get("pose") or []
        ys = [p[1] for p in pts if len(p) >= 2]
        if len(ys) > 1:
            body_h.append(max(ys) - min(ys)); body_cy.append(float(np.mean(ys)))
        for hk in ("left_hand", "right_hand"):
            h = s.get(hk)
            if not h: continue
            c = _centroid(h)
            if c: tracks[hk].append((fi, c[0], c[1]))
    if not body_h:
        return dict(move=0.0, spread_x=0.0, spread_y=0.0, height=0.0, cover=0.0)
    bh = float(np.mean(body_h)); bcy = float(np.mean(body_cy))
    bh = bh if bh > 1e-6 else 1e-6
    moves, sx, sy, hts, covers = [], [], [], [], []
    for hk, tr in tracks.items():
        if len(tr) < MIN_FRAMES:
            continue
        arr = np.array(tr, dtype=float)
        fidx, cx, cy = arr[:, 0], arr[:, 1], arr[:, 2]
        adj = np.diff(fidx) == 1
        if adj.sum() >= MIN_RUN:
            d = np.hypot(np.diff(cx), np.diff(cy))[adj]
            moves.append(float(d.mean()) / bh)
        sx.append(float(cx.std()) / bh)
        sy.append(float(cy.std()) / bh)
        hts.append((bcy - float(cy.mean())) / bh)
        covers.append(len(tr) / len(frames))
    if not sx:
        return dict(move=0.0, spread_x=0.0, spread_y=0.0, height=0.0, cover=0.0)
    return dict(move=float(np.mean(moves)) if moves else 0.0,
                spread_x=float(np.mean(sx)), spread_y=float(np.mean(sy)),
                height=float(np.mean(hts)), cover=float(np.mean(covers)))
# ─────────────────────────────────────────────────────────────────────────────


def dist_class(a, b):
    if {a, b} == {"tay", "trv"}:
        return "Short"
    if "tao" in (a, b):
        return "Long"
    return "Middle"


def unit_norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > 0)


def residualize(K, F):
    S = np.nan_to_num(F)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
    U, s, Vt = np.linalg.svd(Sz - Sz.mean(0), full_matrices=False)
    PCs = (Sz - Sz.mean(0)) @ Vt[:3].T
    X = np.column_stack([np.ones(len(K)), PCs])
    beta, *_ = np.linalg.lstsq(X, K, rcond=None)
    ev = (s ** 2) / (s ** 2).sum()
    return K - X @ beta, [round(float(v), 4) for v in ev[:3]]


def pair_hits(E, rows, q, t, exclude_own_clip=False, targets=None):
    """d = hit - chance for every unit of q against target units of t."""
    qi = [i for i, r in enumerate(rows) if r["lang"] == q]
    ti = targets if targets is not None else [i for i, r in enumerate(rows) if r["lang"] == t]
    En = unit_norm(E)
    sim = En[qi] @ En[ti].T
    out = []
    tl = [rows[j]["label"] for j in ti]; tc = np.array([rows[j]["clip"] for j in ti])
    for a, i in enumerate(qi):
        s = sim[a].copy()
        mask = np.ones(len(ti), bool)
        if exclude_own_clip:
            mask = tc != rows[i]["clip"]
            s[~mask] = -np.inf
        n = mask.sum()
        if n == 0:
            continue
        j = int(np.argmax(s))
        lab = rows[i]["label"]
        chance = sum(1 for k in range(len(ti)) if mask[k] and tl[k] == lab) / n
        out.append((i, float(tl[j] == lab) - chance, float(tl[j] == lab), chance))
    return out


def day_boot(per_lang, stat_fn, rng):
    """per_lang: {lang: (day_index array, matrix of per-query contributions n x m)}. Resample query
    days within language; stat_fn maps {lang: column means} to a scalar."""
    pre = {}
    for l, (di, M) in per_lang.items():
        nd = di.max() + 1
        S = np.zeros((nd, M.shape[1])); C = np.zeros(nd)
        np.add.at(S, di, np.nan_to_num(M)); np.add.at(C, di, 1)
        cnt = np.zeros((nd, M.shape[1])); np.add.at(cnt, di, (~np.isnan(M)).astype(float))
        pre[l] = (S, cnt)
    obs = stat_fn({l: S.sum(0) / np.maximum(cnt.sum(0), 1) for l, (S, cnt) in pre.items()})
    boots = np.empty(B)
    idxs = {l: rng.integers(0, len(pre[l][0]), size=(B, len(pre[l][0]))) for l in pre}
    for b in range(B):
        boots[b] = stat_fn({l: pre[l][0][idxs[l][b]].sum(0) / np.maximum(pre[l][1][idxs[l][b]].sum(0), 1)
                            for l in pre})
    return obs, boots


def lift_analysis(E, rows, rng):
    """Returns pair table, class lifts, Middle - Long with bootstrap."""
    tl = {t: i for i, t in enumerate(LANGS)}
    per_lang, pairs = {}, {}
    for q in LANGS:
        qi = [i for i, r in enumerate(rows) if r["lang"] == q]
        pos = {i: k for k, i in enumerate(qi)}
        M = np.full((len(qi), len(LANGS)), np.nan)
        for t in LANGS:
            if t == q:
                continue
            res = pair_hits(E, rows, q, t)
            for i, d, h, c in res:
                M[pos[i], tl[t]] = d
            pairs[f"{q}->{t}"] = {"class": dist_class(q, t), "n": len(res),
                                  "hit": round(float(np.mean([r[2] for r in res])), 4),
                                  "chance": round(float(np.mean([r[3] for r in res])), 4),
                                  "lift": round(float(np.mean([r[1] for r in res])), 4)}
        days = sorted({rows[i]["day"] for i in qi})
        per_lang[q] = (np.array([days.index(rows[i]["day"]) for i in qi]), M)

    def cls_mean(means, c):
        return np.mean([means[q][tl[t]] for q, t in permutations(LANGS, 2) if dist_class(q, t) == c])

    out = {"pairs": pairs, "class_lift": {}}
    for c in ("Short", "Middle", "Long"):
        o, b = day_boot(per_lang, lambda m, c=c: cls_mean(m, c), rng)
        out["class_lift"][c] = {"lift": round(float(o), 4),
                                "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}
    o, b = day_boot(per_lang, lambda m: cls_mean(m, "Middle") - cls_mean(m, "Long"), rng)
    out["middle_minus_long"] = {"diff": round(float(o), 4), "p_one_sided": round(float(np.mean(b <= 0)), 4),
                                "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
                                "lower90": round(float(np.percentile(b, 5)), 4)}
    out["monotone_point_estimates"] = bool(out["class_lift"]["Short"]["lift"] >= out["class_lift"]["Middle"]["lift"]
                                           >= out["class_lift"]["Long"]["lift"])
    return out


def item8(E, rows, rng, include_same_clip=False):
    """rows include both queries (role 'q') and targets (role 't')."""
    tl = {t: i for i, t in enumerate(LANGS)}
    tgt = {c: [i for i, r in enumerate(rows) if r["role"] == "t" and r["lang"] == c] for c in LANGS}
    En = unit_norm(E)
    per_lang, n_q = {}, {}
    for c in LANGS:
        qi = [i for i, r in enumerate(rows) if r["role"] == "q" and r["lang"] == c]
        n_q[c] = len(qi)
        if len(qi) < MIN_MANDARIN:
            continue
        M = np.full((len(qi), len(LANGS)), np.nan)
        for a, i in enumerate(qi):
            for t in LANGS:
                ti = tgt[t]
                if not ti:
                    continue
                s = En[i] @ En[ti].T
                mask = np.ones(len(ti), bool)
                if t == c and not include_same_clip:
                    mask = np.array([rows[j]["clip"] != rows[i]["clip"] for j in ti])
                if mask.sum() == 0:
                    continue
                s = np.where(mask, s, -np.inf)
                j = ti[int(np.argmax(s))]
                lab = rows[i]["label"]
                chance = sum(1 for k, jj in enumerate(ti) if mask[k] and rows[jj]["label"] == lab) / mask.sum()
                M[a, tl[t]] = float(rows[j]["label"] == lab) - chance
        days = sorted({rows[i]["day"] for i in qi})
        per_lang[c] = (np.array([days.index(rows[i]["day"]) for i in qi]), M)
    comm = sorted(per_lang)

    def D(means):
        return np.mean([means[c][tl[c]] - np.mean([means[c][tl[t]] for t in LANGS if t != c]) for c in comm])

    o, b = day_boot(per_lang, D, rng)
    p2 = 2 * min(np.mean(b <= 0), np.mean(b >= 0))
    per_c = {}
    for c in comm:
        di, M = per_lang[c]
        m = np.nanmean(M, 0)
        per_c[c] = {"n_queries": n_q[c], "lift_same": round(float(m[tl[c]]), 4),
                    "lift_other": round(float(np.mean([m[tl[t]] for t in LANGS if t != c])), 4)}
    return {"D": round(float(o), 4), "p_two_sided": round(float(min(1.0, p2)), 4),
            "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "communities": per_c, "left_out": [c for c in LANGS if c not in comm]}


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
    within = {u: bool(z["within_shot"][k]) for u, k in zid.items()}
    allrows = list(csv.DictReader(open(a.units, encoding="utf-8"), delimiter="\t"))
    need = {r["unit_id"] for r in allrows if ok.get(r["unit_id"])}
    by_clip = defaultdict(set)
    for r in allrows:
        if r["unit_id"] in need:
            by_clip[r["clip"]].add(r["unit_id"])

    V = {k: {} for k in ("14d_tight", "18d_tight", "18d_full", "9d_gesture")}
    for clip in sorted(by_clip):
        p = next((Path(d) / f"{clip}.pose.json" for d in a.pose_dirs if (Path(d) / f"{clip}.pose.json").exists()))
        recs = json.loads(p.read_text())
        span = float(np.median([f["span"] for f in recs]))
        gmin = math.ceil(GESTURE_MIN_MS / span)
        offs = np.array([f["offset"] for f in recs])
        for uid, s, e in load_utterances(Path(a.trees) / f"{clip}.aligntree.TextGrid", clip):
            if uid not in by_clip[clip]:
                continue
            sel = [recs[i] for i in np.nonzero((offs >= s * 1000) & (offs < e * 1000))[0]]
            tf = [f for f in sel if is_tight(f)]
            k18t = _pose_features(tf)
            V["18d_tight"][uid] = k18t
            V["14d_tight"][uid] = np.delete(k18t, IDX_POSITION)
            V["18d_full"][uid] = _pose_features(sel)
            l = side_features(tf, "left", gmin); r = side_features(tf, "right", gmin)
            V["9d_gesture"][uid] = np.array([l["move"], l["spread_x"], l["spread_y"], l["height"],
                                             r["move"], r["spread_x"], r["spread_y"], r["height"],
                                             (l["move"] + r["move"]) / 2.0])
    Kz = {u: z["K"][zid[u]] for u in need}
    mism = sum(not np.allclose(V["14d_tight"][u], Kz[u], equal_nan=False) for u in need)
    if mism:
        raise SystemExit(f"14d_tight differs from unit_table K for {mism} units")
    Fz = {u: z["F"][zid[u]] for u in need}

    def build(sel_rows, label_key="haiku"):
        rows = [dict(unit_id=r["unit_id"], clip=r["clip"], lang=r["lang"], day=r["day"],
                     label=r[label_key]) for r in sel_rows]
        ids = [r["unit_id"] for r in rows]
        reps = {k: np.array([V[k][u] for u in ids]) for k in V}
        F = np.array([Fz[u] for u in ids])
        reps["14d_resid"], ev = residualize(np.delete(reps["18d_full"], IDX_POSITION, 1), F)
        reps["18d_resid"], _ = residualize(reps["18d_full"], F)
        return rows, reps, F, ev

    T = lambda r: r == "True"
    prim = [r for r in allrows if r["unit_id"] in need and T(r["kept"])]
    rng = np.random.default_rng(SEED_62)
    rows, reps, F, ev = build(prim)
    res = {"seed_62": SEED_62, "seed_8": SEED_8, "resamples": B,
           "units": dict(Counter(r["lang"] for r in rows)), "days": len({r["day"] for r in rows}),
           "label_counts": dict(Counter(r["label"] for r in rows)),
           "framing_pc_variance": ev, "gesture_min_records": "ceil(1335 ms / span)",
           "primary_14d_tight": lift_analysis(reps["14d_tight"], rows, rng),
           "secondary": {}, "sensitivity": {}}
    for k in ("18d_tight", "14d_resid", "18d_resid", "9d_gesture", "18d_full"):
        r = lift_analysis(reps[k], rows, rng)
        res["secondary"][k] = {"class_lift": r["class_lift"], "middle_minus_long": r["middle_minus_long"]}
    res["framing_means_by_language"] = {l: [round(float(v), 4) for v in np.nanmean(F[[i for i, r in enumerate(rows) if r["lang"] == l]], 0)]
                                        for l in LANGS}
    res["framing_names"] = list(F_NAMES)
    for name, sel, key in (
        ("kept_mixed", [r for r in allrows if r["unit_id"] in need and T(r["kept_mixed"])], "haiku"),
        ("no_mandarin_exclusion", [r for r in allrows if r["unit_id"] in need], "haiku"),
        ("no_input_gap", [r for r in prim if not T(r["gap"])], "haiku"),
        ("gemini_labels_exploratory", prim, "gemini")):
        rr, rp, _, _ = build(sel, key)
        r = lift_analysis(rp["14d_tight"], rr, rng)
        res["sensitivity"][name] = {"units": len(rr), "class_lift": r["class_lift"], "middle_minus_long": r["middle_minus_long"]}

    # item 8
    q = [dict(r, role="q") for r in allrows if r["unit_id"] in need and r["lid_answer"] == "mandarin"]
    t = [dict(r, role="t") for r in prim]
    r8 = [dict(unit_id=r["unit_id"], clip=r["clip"], lang=r["lang"], day=r["day"], label=r["haiku"], role=r["role"]) for r in q + t]
    E8 = np.array([V["14d_tight"][r["unit_id"]] for r in r8])
    rng8 = np.random.default_rng(SEED_8)
    res["item8"] = item8(E8, r8, rng8)
    res["item8_same_clip_included"] = item8(E8, r8, rng8, include_same_clip=True)
    Fq = np.array([Fz[r["unit_id"]] for r in q]); Ft = np.array([Fz[r["unit_id"]] for r in t])
    res["item8_framing_means"] = {"mandarin": [round(float(v), 4) for v in np.nanmean(Fq, 0)],
                                  "indigenous": [round(float(v), 4) for v in np.nanmean(Ft, 0)]}
    def tight_share(cond):
        w = [r for r in allrows if within.get(r["unit_id"]) and cond(r)]
        return round(sum(r["unit_id"] in need for r in w) / max(len(w), 1), 4)
    res["item8_tight_share_of_within_shot"] = {"mandarin": tight_share(lambda r: r["lid_answer"] == "mandarin"),
                                               "indigenous": tight_share(lambda r: T(r["kept"]))}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "s62_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    slim = {k: v for k, v in res.items() if k != "primary_14d_tight"}
    slim["primary_14d_tight"] = {k: v for k, v in res["primary_14d_tight"].items() if k != "pairs"}
    print(json.dumps(slim, ensure_ascii=False))


if __name__ == "__main__":
    main()
