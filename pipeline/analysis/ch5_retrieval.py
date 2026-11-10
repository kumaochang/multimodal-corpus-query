"""Chapter 5: single-representation retrieval and source-clustering diagnostics at the utterance.

Retrieval specification fixed 2026-10-18/19 before the Chapter 5 rankings were computed.

Units. Labeled utterances of the 214 core clips (label_analysis.py units.tsv). Primary: kept
(Mandarin excluded, Amendment 9 item 1). Sensitivity: all utterances.

Representations (one at a time; a unit enters a representation only where it is defined):
  whisper   Whisper large-v3 encoder mean; cosine on the raw vectors (as Study 1 and Arm B).
  mfcc86    the clip-level study's 86-d spectral summary; global per-dimension z-score with mean and
            sd estimated once on all 14,012 utterances (before Mandarin exclusion); zero-variance
            dimensions set to 0 (analysis plan §10).
  prosody   15 prosodic features; z-scored within language, NaN replaced by the language mean
            (the retrieval treatment stated in Ch. 4 §4.5).
  xlmr_sub  translation (subtitle) modality, XLM-R; defined where the subtitle is non-empty.
  tfidf_sub its char 1-3 TF-IDF reference; same units.
  xlmr_tr   transcription modality, XLM-R; five languages (no Tsou); non-empty transcripts.
  tfidf_tr  its TF-IDF reference. Transcription is within-language only (Amendment 4 B4).
  kin14     14-d kinematic primary (unit_table.py), units within one shot with the 1 s tight minimum.
Cosine; a zero vector has cosine 0 to every candidate. Ties in the top 1 are broken by candidate order.

Retrieval directions and candidate pools (per query):
  within-language: W1 all other defined units of the language; W2 excluding the query's clip;
  W3 excluding the query's broadcast day.
  cross-language (not for transcription): X all defined units of the other languages.
Relevance: the candidate's Haiku label equals the query's (five labels).

Per query: hit@1; chance = share of the query's label among its candidates; lift = hit - chance;
P@3 (share relevant in the top 3) and its chance; AP over the full ranking (base rate = chance).
Queries with no relevant candidate are kept for P@1 (chance 0) and dropped from AP (counted).

Groups (§1 item 3 as operationalized in plan §11): Explaining queries; all other queries pooled;
each of Reporting, Celebrating, Urging, Comforting singly, flagged tested_alone only if its queries
in that analysis span >= 15 broadcast days and >= 25 clips. Statistics: mean lift@1 per group and
Explaining - remainder, with day-cluster 95% intervals (query days resampled within language,
10,000 resamples, default_rng(SEED)); mean hit, chance, P@3, MAP and base rate per group,
descriptive. No test across representations.

Common subset: the same within-language W1-W3 and cross-language analyses restricted to units
defined in every non-transcription representation (whisper, mfcc86, prosody, xlmr_sub, tfidf_sub,
kin14), so representations are compared on identical queries and candidates.

Source-clustering diagnostic (plan §10), every representation, kept and all units: share of top-1
neighbours (a) in the query's clip under W1, (b) in the query's day under W2, (c) in the query's
language when the pool is all units of all languages outside the query's day; each beside its
chance (the share of such candidates in the pool) and beside the same-label rate under that pool.

Inputs: features.npz, tfidf_sub.npz, tfidf_tr.npz (feat_full.py), units.tsv (label_analysis.py),
units_k.npz (unit_table.py). Prints a compact summary; writes the full JSON to --out.

Usage (sub environment):
  python pipeline/analysis/ch5_retrieval.py --features data/study3_runs/features_full \
    --units data/study3_runs/label_analysis/units.tsv --units-k data/study3_runs/unit_table/units_k.npz \
    --out data/study3_runs/ch5/ch5_retrieval.json
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

SEED, B = 20261019, 10_000
LANG = {"Amis": "ami", "Paiwan": "pwn", "Truku": "trv", "Atayal": "tay", "Tsou": "tsu", "Yami": "tao"}
LABELS = ["Comforting", "Celebrating", "Explaining", "Urging", "Reporting"]
MIN_DAYS, MIN_CLIPS = 15, 25
CHUNK = 512


def l2rows(X):
    if hasattr(X, "multiply"):
        X = X.tocsr().astype(np.float32)
        n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        n[n == 0] = 1.0
        from scipy import sparse
        return sparse.diags(1.0 / n) @ X
    X = np.asarray(X, np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > 0)


def sims(Q, C):
    S = Q @ C.T
    return S.toarray() if hasattr(S, "toarray") else np.asarray(S)


class Retriever:
    """Scores queries against a candidate set; each query has its own exclusion mask."""

    def __init__(self, Xn, lab_idx, clip_idx, day_idx, lang_idx):
        self.X, self.lab, self.clip, self.day, self.lang = Xn, lab_idx, clip_idx, day_idx, lang_idx

    def run(self, q, cand, mode):
        """q, cand: index arrays. mode: W1, W2, W3, X, DIAG_LANG. Returns per-query dict arrays."""
        out = {k: [] for k in ("hit", "chance", "p3", "p3c", "ap", "has_rel", "top_clip", "top_day",
                               "top_lang", "c_clip", "c_day", "c_lang", "n_cand")}
        Cn = self.X[cand]
        for s in range(0, len(q), CHUNK):
            qs = q[s:s + CHUNK]
            S = sims(self.X[qs], Cn)
            for r, i in enumerate(qs):
                m = cand != i
                if mode == "W2":
                    m &= self.clip[cand] != self.clip[i]
                elif mode in ("W3", "DIAG_LANG"):
                    m &= self.day[cand] != self.day[i]
                idx = np.nonzero(m)[0]
                if len(idx) == 0:
                    for k in out: out[k].append(np.nan)
                    continue
                sc = S[r, idx]
                order = idx[np.argsort(-sc, kind="stable")]
                rel = (self.lab[cand[order]] == self.lab[i])
                nrel = rel.sum(); n = len(order)
                out["hit"].append(float(rel[0])); out["chance"].append(nrel / n)
                k3 = min(3, n); out["p3"].append(rel[:k3].mean()); out["p3c"].append(nrel / n)
                if nrel:
                    prec = np.cumsum(rel) / np.arange(1, n + 1)
                    out["ap"].append(float(prec[rel].mean()))
                else:
                    out["ap"].append(np.nan)
                out["has_rel"].append(float(nrel > 0))
                t = cand[order[0]]; pool = cand[idx]
                out["top_clip"].append(float(self.clip[t] == self.clip[i]))
                out["top_day"].append(float(self.day[t] == self.day[i]))
                out["top_lang"].append(float(self.lang[t] == self.lang[i]))
                out["c_clip"].append(float((self.clip[pool] == self.clip[i]).mean()))
                out["c_day"].append(float((self.day[pool] == self.day[i]).mean()))
                out["c_lang"].append(float((self.lang[pool] == self.lang[i]).mean()))
                out["n_cand"].append(n)
        return {k: np.array(v, float) for k, v in out.items()}


def day_boot(vals_by_group, q_lang, q_day, rng):
    """Mean of each group's values and Explaining - remainder, day-cluster CIs within language."""
    langs = sorted(set(q_lang))
    res = {}
    keys = list(vals_by_group)
    pre = {}
    for l in langs:
        ml = q_lang == l
        days = sorted(set(q_day[ml])); di = np.array([days.index(d) for d in q_day[ml]])
        S = np.zeros((len(days), len(keys))); C = np.zeros((len(days), len(keys)))
        for j, k in enumerate(keys):
            v, g = vals_by_group[k]
            vv, gg = v[ml], g[ml] & ~np.isnan(v[ml])
            np.add.at(S[:, j], di[gg], vv[gg]); np.add.at(C[:, j], di[gg], 1)
        pre[l] = (S, C)
    tS = sum(p[0].sum(0) for p in pre.values()); tC = sum(p[1].sum(0) for p in pre.values())
    bS = np.zeros((B, len(keys))); bC = np.zeros((B, len(keys)))
    for l, (S, C) in pre.items():
        idx = rng.integers(0, len(S), size=(B, len(S)))
        bS += S[idx].sum(1); bC += C[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = tS / tC; bm = bS / bC
    for j, k in enumerate(keys):
        res[k] = {"n": int(tC[j]), "mean": None if tC[j] == 0 else round(float(m[j]), 4),
                  "ci95": None if tC[j] == 0 else [round(float(np.nanpercentile(bm[:, j], 2.5)), 4),
                                                   round(float(np.nanpercentile(bm[:, j], 97.5)), 4)]}
    if "Explaining" in keys and "remainder" in keys:
        a, b = keys.index("Explaining"), keys.index("remainder")
        d = bm[:, a] - bm[:, b]
        res["Explaining_minus_remainder"] = {"diff": round(float(m[a] - m[b]), 4),
                                             "ci95": [round(float(np.nanpercentile(d, 2.5)), 4),
                                                      round(float(np.nanpercentile(d, 97.5)), 4)]}
    return res


def summarize(r, q, U, rng):
    lab = U["label"][q]; ql = U["lang"][q]; qd = U["day"][q]; qc = U["clip"][q]
    lift = r["hit"] - r["chance"]
    groups = {"Explaining": lab == "Explaining", "remainder": lab != "Explaining"}
    for c in LABELS:
        if c != "Explaining":
            groups[c] = lab == c
    lifts = day_boot({k: (lift, g) for k, g in groups.items()}, ql, qd, rng)
    desc = {}
    for k, g in groups.items():
        gg = g & ~np.isnan(r["hit"])
        if not gg.any():
            desc[k] = {"queries": 0}; continue
        apg = gg & ~np.isnan(r["ap"])
        desc[k] = {"queries": int(gg.sum()), "days": len(set(qd[gg])), "clips": len(set(qc[gg])),
                   "hit": round(float(r["hit"][gg].mean()), 4), "chance": round(float(r["chance"][gg].mean()), 4),
                   "p3": round(float(r["p3"][gg].mean()), 4), "p3_chance": round(float(r["p3c"][gg].mean()), 4),
                   "map": round(float(r["ap"][apg].mean()), 4) if apg.any() else None,
                   "ap_queries_dropped_no_relevant": int((gg & np.isnan(r["ap"])).sum())}
        if k in LABELS:
            desc[k]["tested_alone"] = desc[k]["days"] >= MIN_DAYS and desc[k]["clips"] >= MIN_CLIPS
    return {"lift_at_1": lifts, "descriptive": desc,
            "queries": int((~np.isnan(r["hit"])).sum()), "mean_candidates": round(float(np.nanmean(r["n_cand"])), 1)}


def diag(rt, units, U, rng):
    """Source-clustering rates under W1 (clip), W2 (day), DIAG_LANG (language)."""
    out = {}
    qd = U["day"][units]; ql = U["lang"][units]
    for mode, top, ch in (("W1", "top_clip", "c_clip"), ("W2", "top_day", "c_day"), ("DIAG_LANG", "top_lang", "c_lang")):
        rs = []
        if mode == "DIAG_LANG":
            r = rt.run(units, units, mode)
        else:
            parts = {}
            for l in sorted(set(ql)):
                q = units[ql == l]
                parts[l] = rt.run(q, q, mode)
            r = {k: np.concatenate([parts[l][k] for l in sorted(set(ql))]) for k in parts[next(iter(parts))]}
            order = np.concatenate([units[ql == l] for l in sorted(set(ql))])
            pos = {u: i for i, u in enumerate(order)}
            perm = np.array([pos[u] for u in units])
            r = {k: v[perm] for k, v in r.items()}
        same_label_rate = r["hit"]
        g = {"all": np.ones(len(units), bool)}
        stats = day_boot({"top_same_source": (r[top], g["all"]), "chance_same_source": (r[ch], g["all"]),
                          "same_label_hit": (same_label_rate, g["all"]), "same_label_chance": (r["chance"], g["all"])},
                         ql, qd, rng)
        out[{"W1": "clip_under_W1", "W2": "day_under_W2", "DIAG_LANG": "language_outside_day"}[mode]] = stats
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="feat_full.py output folder")
    ap.add_argument("--units", required=True)
    ap.add_argument("--units-k", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resamples", type=int, default=B)
    a = ap.parse_args()
    globals()["B"] = a.resamples
    from scipy import sparse
    F = Path(a.features)
    z = np.load(F / "features.npz")
    ids = list(z["ids"]); pos = {u: i for i, u in enumerate(ids)}
    rows = {r["unit_id"]: r for r in csv.DictReader(open(a.units, encoding="utf-8"), delimiter="\t")}
    if set(rows) != set(ids):
        sys.exit(f"units.tsv ({len(rows)}) and features ({len(ids)}) disagree")
    U = {"lang": np.array([rows[u]["lang"] for u in ids]), "day": np.array([rows[u]["day"] for u in ids]),
         "clip": np.array([rows[u]["clip"] for u in ids]), "label": np.array([rows[u]["haiku"] for u in ids]),
         "kept": np.array([rows[u]["kept"] == "True" for u in ids])}
    if not set(z["lang"]) <= set(LANG) or any(LANG[l] != c for l, c in zip(z["lang"], U["lang"])):
        sys.exit("language mismatch between features.npz and units.tsv")
    li = {l: i for i, l in enumerate(sorted(set(U["lang"])))}
    lab_i = np.array([LABELS.index(x) for x in U["label"]])
    clip_i = np.unique(U["clip"], return_inverse=True)[1]; day_i = np.unique(U["day"], return_inverse=True)[1]
    lang_i = np.array([li[x] for x in U["lang"]])

    # representations
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
    for u, i in pos.items():
        j = kpos.get(u)
        if j is not None and bool(k["within_shot"][j]) and int(k["k_missing"][j]) == 0:
            K[i] = k["K"][j]; kin_ok[i] = True
    reps = {"whisper": (W, np.ones(len(ids), bool), True), "mfcc86": (Mz, np.ones(len(ids), bool), True),
            "prosody": (Pz, np.ones(len(ids), bool), True), "xlmr_sub": (XS, sub_ok, True),
            "tfidf_sub": (TS, sub_ok, True), "xlmr_tr": (XT, tr_ok, False), "tfidf_tr": (TT, tr_ok, False),
            "kin14": (K, kin_ok, True)}
    common_ok = sub_ok & kin_ok
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "resamples": B, "utterances": len(ids),
           "defined": {n: {"all": int(d.sum()), "kept": int((d & U["kept"]).sum())} for n, (_, d, _) in reps.items()},
           "common_subset": {"all": int(common_ok.sum()), "kept": int((common_ok & U["kept"]).sum())},
           "analyses": {}, "diagnostics": {}}
    for name, (X, ok, cross) in reps.items():
        rt = Retriever(l2rows(X), lab_i, clip_i, day_i, lang_i)
        for version, vmask in (("kept", U["kept"]), ("all", np.ones(len(ids), bool))):
            for subset, smask in (("own", ok), ("common", common_ok & ok)):
                if name in ("xlmr_tr", "tfidf_tr") and subset == "common":
                    continue
                units = np.nonzero(vmask & smask)[0]
                key = f"{name}|{version}|{subset}"
                block = {}
                for mode in ("W1", "W2", "W3"):
                    parts, qs = [], []
                    for l in sorted(li):
                        q = units[U["lang"][units] == l]
                        if len(q) < 2:
                            continue
                        parts.append(rt.run(q, q, mode)); qs.append(q)
                    if not parts:
                        continue
                    r = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
                    block[mode] = summarize(r, np.concatenate(qs), U, rng)
                if cross:
                    parts, qs = [], []
                    for l in sorted(li):
                        q = units[U["lang"][units] == l]; c = units[U["lang"][units] != l]
                        if len(q) and len(c):
                            parts.append(rt.run(q, c, "X")); qs.append(q)
                    r = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
                    block["X"] = summarize(r, np.concatenate(qs), U, rng)
                res["analyses"][key] = block
                if subset == "own":
                    res["diagnostics"][f"{name}|{version}"] = diag(rt, units, U, rng)
        print(json.dumps({"done": name}), flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    compact = {"defined": res["defined"], "common_subset": res["common_subset"], "lift_at_1_kept_own": {}}
    for key, block in res["analyses"].items():
        n, v, s = key.split("|")
        if v == "kept" and s == "own":
            compact["lift_at_1_kept_own"][n] = {m: {g: block[m]["lift_at_1"][g] for g in ("Explaining", "remainder", "Explaining_minus_remainder")}
                                                for m in block}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
