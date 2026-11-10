"""Probe models and statistics of the judged directive experiment.

Registered in PREREGISTRATION.md Amendment 9 item 5 and Amendment 10 items 2 and 3. Nothing here
is tuned; every constant below is a registered value.

Labels: answer 1 = directive (positive), 2 = not directive, 3 = cannot tell (excluded, counted),
from the annotation kit CSVs of batches 1 and 2; batch 3 repeats give Cohen's kappa only.
Items: primary = strata marker, suffix_cue, suffix_nocue; none_cue enters no model.
Weights: 1 / inclusion probability (draw_annotation.py).
Sources (each standardized on the training fold; NaN -> training-fold mean; a source with more
than 16 columns reduced by PCA to min(16, n_train - 1) fitted on the training fold):
  T = xlmr + tfidf + form;  A = whisper + prosody;  K = kinematics + k_missing;
  Fa = face measures + face share + face missing;  F = 8 framing measures;  V = K + Fa.
Model: LogisticRegression(L2, C = 1.0), sample weights; leave one broadcast day out; out-of-fold
scores pooled. A training fold with one class predicts its weighted positive share (counted).
Primary: weighted AUC(T+A+V) - weighted AUC(T) on primary items; paired day-cluster bootstrap
(days resampled with replacement, 10,000, default_rng(20261017)) of the pooled scores; 95%
percentile interval; one-sided p = (#{delta_b <= 0} + 1) / (B + 1). Fewer than 15 positives in
batches 1-2: descriptive only (no p-values).
Output: --out JSON. Prints the summary.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

SEED_BOOT, B, C_REG, PCA_MAX, MIN_POS = 20261017, 10000, 1.0, 16, 15
PRIMARY = {"marker", "suffix_cue", "suffix_nocue"}
MODELS = {"T": ["T"], "T+A": ["T", "A"], "T+K": ["T", "K"], "T+Fa": ["T", "Fa"],
          "T+V": ["T", "K", "Fa"], "T+A+V": ["T", "A", "K", "Fa"], "F": ["F"]}
BLOCKS = {"T": ["xlmr", "tfidf", "form"], "A": ["whisper", "prosody"], "K": ["kin", "k_missing"],
          "Fa": ["face", "face_share", "face_missing"], "F": ["framing"]}

def load_sources(text, audio, kin, face, ids):
    S = {}
    def take(npz, key, name):
        z = np.load(npz, allow_pickle=True); pos = {u: i for i, u in enumerate(z["ids"])}
        arr = z[key]; arr = arr.reshape(len(arr), -1).astype(float)
        S[name] = np.array([arr[pos[u]] for u in ids])
    take(text, "xlmr", "xlmr"); take(text, "tfidf", "tfidf"); take(text, "form", "form")
    take(audio, "whisper", "whisper"); take(audio, "prosody", "prosody")
    take(kin, "K", "kin"); take(kin, "k_missing", "k_missing"); take(kin, "F", "framing")
    take(face, "face", "face"); take(face, "face_share", "face_share"); take(face, "face_missing", "face_missing")
    return S

def prep(Xtr, Xte, ntr):
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    m, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1.0
    Xtr, Xte = (Xtr - m) / s, (Xte - m) / s
    if Xtr.shape[1] > PCA_MAX:
        from sklearn.decomposition import PCA
        p = PCA(n_components=min(PCA_MAX, ntr - 1), random_state=0).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    return Xtr, Xte

def oof_scores(S, srcs, y, w, days):
    from sklearn.linear_model import LogisticRegression
    out = np.full(len(y), np.nan); onecls = 0
    for d in sorted(set(days)):
        te = days == d; tr = ~te
        parts = [prep(S[s][tr], S[s][te], tr.sum()) for s in srcs]
        Xtr = np.hstack([p[0] for p in parts]); Xte = np.hstack([p[1] for p in parts])
        if len(set(y[tr])) < 2:
            out[te] = np.average(y[tr], weights=w[tr]); onecls += 1; continue
        m = LogisticRegression(penalty="l2", C=C_REG, max_iter=5000).fit(Xtr, y[tr], sample_weight=w[tr])
        out[te] = m.predict_proba(Xte)[:, 1]
    return out, onecls

def wauc(y, s, w):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s, sample_weight=w)) if len(set(y)) == 2 else float("nan")

def wap(y, s, w):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s, sample_weight=w)) if len(set(y)) == 2 else float("nan")

def boot_diff(y, s1, s0, w, days, rng):
    ud = np.array(sorted(set(days))); idx = {d: np.where(days == d)[0] for d in ud}
    obs = wauc(y, s1, w) - wauc(y, s0, w); bs = []
    for _ in range(B):
        sel = np.concatenate([idx[d] for d in rng.choice(ud, size=len(ud), replace=True)])
        a, b = wauc(y[sel], s1[sel], w[sel]), wauc(y[sel], s0[sel], w[sel])
        if a == a and b == b: bs.append(a - b)
    bs = np.array(bs)
    return {"delta": obs, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_one_sided": float((np.sum(bs <= 0) + 1) / (len(bs) + 1)), "valid_resamples": int(len(bs))}

def kappa(a, b):
    a, b = np.array(a), np.array(b); cats = sorted(set(a) | set(b))
    po = np.mean(a == b); pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--judgements", nargs="+", required=True, help="results_batch_*.csv")
    ap.add_argument("--text", required=True); ap.add_argument("--audio", required=True)
    ap.add_argument("--kin", required=True); ap.add_argument("--face", required=True)
    ap.add_argument("--clm", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))["items"]
    by_item = {r["item_id"]: r for r in items}
    ans = {}
    for f in a.judgements:
        for row in csv.DictReader(open(f, encoding="utf-8-sig")):
            ans[row["item_id"]] = int(row["answer"])
    main_items = [r for r in items if r["batch"] in (1, 2) and r["item_id"] in ans]
    n3 = sum(ans[r["item_id"]] == 3 for r in main_items)
    lab = [r for r in main_items if ans[r["item_id"]] in (1, 2)]
    rep = [(ans[r["item_id"]], ans[[q for q in items if q["utt_id"] == r["utt_id"] and q["batch"] in (1, 2)][0]["item_id"]])
           for r in items if r["batch"] == 3 and r["item_id"] in ans
           and [q for q in items if q["utt_id"] == r["utt_id"] and q["batch"] in (1, 2)][0]["item_id"] in ans]
    res = {"judged_batches_1_2": len(main_items), "answer_3_excluded": n3,
           "positives_all": sum(ans[r["item_id"]] == 1 for r in lab),
           "kappa_batch3": kappa([x for x, _ in rep], [y for _, y in rep]) if rep else None,
           "kappa_n": len(rep)}
    # descriptive: directive share per stratum and form (weighted)
    desc = {}
    for st in sorted({r["stratum"] for r in lab}):
        g = [r for r in lab if r["stratum"] == st]
        desc[st] = {"n": len(g), "positives": sum(ans[r["item_id"]] == 1 for r in g)}
    res["by_stratum"] = desc
    prim = [r for r in lab if r["stratum"] in PRIMARY]
    ids = [r["utt_id"] for r in prim]
    y = np.array([1 if ans[r["item_id"]] == 1 else 0 for r in prim])
    w = np.array([1.0 / r["incl_prob"] for r in prim]); days = np.array([r["day"] for r in prim])
    res["primary_items"] = len(prim); res["primary_positives"] = int(y.sum())
    descriptive = y.sum() < MIN_POS; res["descriptive_only"] = bool(descriptive)
    S = load_sources(a.text, a.audio, a.kin, a.face, ids)
    scores, onecls = {}, {}
    for name, blocks in MODELS.items():
        srcs = [s for b in blocks for s in BLOCKS[b]]
        scores[name], onecls[name] = oof_scores(S, srcs, y, w, days)
    clm = {json.loads(l)["utt_id"]: json.loads(l) for l in open(a.clm, encoding="utf-8")}
    scores["CLM_a"] = np.array([clm[u]["clm_a"] for u in ids]); scores["CLM_b"] = np.array([clm[u]["clm_b"] for u in ids])
    res["auc"] = {k: wauc(y, v, w) for k, v in scores.items()}
    res["ap"] = {k: wap(y, v, w) for k, v in scores.items()}
    res["one_class_folds"] = onecls
    rng = np.random.default_rng(SEED_BOOT)
    comps = {"primary T+A+V - T": ("T+A+V", "T"), "T+A - T": ("T+A", "T"), "T+V - T": ("T+V", "T"),
             "T+K - T": ("T+K", "T"), "T+Fa - T": ("T+Fa", "T"), "T+V - F": ("T+V", "F"),
             "T+A+V - CLM_a": ("T+A+V", "CLM_a"), "T+A+V - CLM_b": ("T+A+V", "CLM_b")}
    res["contrasts"] = {}
    for k, (m1, m0) in comps.items():
        r = boot_diff(y, scores[m1], scores[m0], w, days, rng)
        if descriptive: r.pop("p_one_sided")
        res["contrasts"][k] = r
    # Subsets: the models are trained on all primary items; the contrasts are evaluated on the
    # subset's out-of-fold scores. suffix_only: the primary contrast (registered). face_present:
    # every contrast above (registered: "the same differences on the items with a face only").
    # CLM contrasts also by stratum (registered: "overall and by stratum").
    strat = np.array([r["stratum"] for r in prim])
    subsets = [("suffix_only", strat != "marker", ["primary T+A+V - T"]),
               ("face_present", S["face_missing"][:, 0] == 0, list(comps))]
    subsets += [(f"stratum={st}", strat == st, ["T+A+V - CLM_a", "T+A+V - CLM_b"]) for st in sorted(set(strat))]
    for sub, mask, keys in subsets:
        res[f"{sub}_n"] = int(mask.sum())
        if not mask.sum() or len(set(y[mask])) < 2:
            continue
        for k in keys:
            m1, m0 = comps[k]
            rr = boot_diff(y[mask], scores[m1][mask], scores[m0][mask], w[mask], days[mask], rng)
            if descriptive: rr.pop("p_one_sided")
            res["contrasts"][f"{sub}: {k}"] = rr
    # weighted directive share per form class (none_cue included, descriptive)
    res["weighted_share_by_form"] = {}
    for fm in sorted({r.get("form") for r in lab}):
        g = [r for r in lab if r.get("form") == fm]
        ww = np.array([1 / r["incl_prob"] for r in g]); yy = np.array([ans[r["item_id"]] == 1 for r in g])
        res["weighted_share_by_form"][fm] = {"n": len(g), "share": float(np.average(yy, weights=ww))}
    for st in desc:
        g = [r for r in lab if r["stratum"] == st]
        ww = np.array([1 / r["incl_prob"] for r in g]); yy = np.array([ans[r["item_id"]] == 1 for r in g])
        desc[st]["weighted_share"] = float(np.average(yy, weights=ww))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
