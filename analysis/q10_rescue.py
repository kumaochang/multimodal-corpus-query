"""
Q10 (W1 rescue): can the visual-channel result be recovered after the
shot-framing confound is removed?

Q7-Q9 established that the pooled Explaining result of q6 is carried by shot
framing: TITV frames explanatory segments as tight two-shots and other segments
as wide crowd shots, hand detection follows framing (presence_mean ~ persons per
frame, r = -0.86), and detection coverage alone separates the two label groups
(p = 0.0005, Cliff's delta = +0.75).

Two rescues are possible in principle and both are testable on the pilot:

  RESCUE A (frame conditioning). Make framing constant BY CONSTRUCTION rather
  than adjusting for it after the fact: recompute every feature using only
  frames that meet one fixed framing criterion (at most TIGHT_MAXP detected
  persons, and a tracked body occupying at least TIGHT_MINAREA of the frame).
  Every clip is then described from comparable shots. This is the stronger of
  the two, because it does not rely on a linear model of the confound.

  RESCUE B (covariate residualization). Keep all frames, but regress each
  kinematic dimension on the leading principal components of the shot-measure
  block and run retrieval on the residuals. Weaker, and at n = 30 it spends
  degrees of freedom that may take real signal with them; reported for
  completeness and because it is the adjustment a reviewer will ask about.

MULTIPLICITY WARNING. This is the third representation family tried on the same
30 clips after seeing that the first one was confounded. Any pass here is
provisional and is a hypothesis for the main corpus, not a confirmation. The
pilot's job is exploratory; the pre-registered rule tested on the 216-clip
corpus is what will settle it.

Run:  python analysis/q10_rescue.py
Outputs: analysis/q10_results.md (+ stdout)
"""
import json, importlib.util
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
from scipy.stats import bootstrap as scipy_bootstrap

ROOT = Path(__file__).resolve().parent.parent
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
q5 = _load("q5", "q5_baselines.py")
q7 = _load("q7", "q7_handpresence.py")
q8 = _load("q8", "q8_gesture_conditional.py")

POSE_DIR  = ROOT / "data" / "pose"
LABELFILE = ROOT / "pilot" / "labels.json"
OUT_MD    = ROOT / "analysis" / "q10_results.md"
CACHE     = ROOT / "analysis" / "_q8_cache.npz"

TIGHT_MAXP     = 2      # at most this many detected persons in the frame
TIGHT_MINAREA  = 0.12   # tracked body bbox must cover at least this share of the frame
MIN_TIGHT      = 120    # a clip needs at least this many qualifying frames to be used
FRAME_W, FRAME_H = 1280, 720
FALLBACK = {(0, 0, 640, 720), (640, 0, 1280, 720)}
SEED, NRES = 42, 100000


def tight_frames(frames):
    """Frames shot tightly enough that hands are in frame for structural reasons."""
    out = []
    for fr in frames:
        if len(fr.get("persons") or []) > TIGHT_MAXP:
            continue
        ok = False
        for side in ("left", "right"):
            s = fr[side]
            if s is None: continue
            b = s.get("bbox")
            if not isinstance(b, dict): continue
            key = (b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2"))
            if key in FALLBACK: continue
            if ((b["y2"] - b["y1"]) * (b["x2"] - b["x1"])) / (FRAME_H * FRAME_W) >= TIGHT_MINAREA:
                ok = True; break
        if ok: out.append(fr)
    return out


def pooled_test(E, labels, langs, rng, name, L, verdict):
    cl = q5.crossling_per_query(E, labels, langs)
    rows = [r for s in ["Amis", "Paiwan", "Truku"] for r in cl[s]]
    expl = [r for r in rows if r[1] == "Explaining"]
    nonx = [r for r in rows if r[1] != "Explaining"]
    for gname, grp in [("Explaining", expl), ("non-Explaining", nonx)]:
        h = sum(r[4] for r in grp); n = len(grp); ps = [r[5] for r in grp]
        pv = q7.pb_pval_ge(h, ps)
        L.append(f"| {name} | {gname} | {n} | {h} | {h/n:.3f} | {np.mean(ps):.3f} | "
                 f"{h/n-np.mean(ps):+.3f} | {pv:.4f} |")
        if gname == "Explaining": verdict[name] = (h, n, h/n, float(np.mean(ps)), pv)
    resid = np.array([r[4] - r[5] for r in expl + nonx], dtype=float)
    obs, pp = q7.perm_lift_diff(resid, len(expl), rng, nres=NRES)
    return f"- {name}: lift difference (Explaining - non-Explaining) = {obs:+.3f}, permutation one-sided p = {pp:.4f}"


def ci(a, nres=2000):
    a = np.asarray(a, float)
    r = scipy_bootstrap((a,), np.mean, n_resamples=nres, confidence_level=0.95,
                        method="percentile", random_state=SEED)
    return a.mean(), r.confidence_interval.low, r.confidence_interval.high


def main():
    files  = sorted(POSE_DIR.glob("*.pose.json"))
    ids    = [f.name.replace(".pose.json", "") for f in files]
    langs  = [i.split("-")[1] for i in ids]
    lmap   = json.load(open(LABELFILE, encoding="utf-8"))
    labels = [lmap[i] for i in ids]
    rng    = np.random.default_rng(SEED)

    K18_full, K18_tight, G_tight, ntight, ntot = [], [], [], [], []
    for f in files:
        frames = json.load(open(f))
        tf = tight_frames(frames)
        ntot.append(len(frames)); ntight.append(len(tf))
        K18_full.append(q5._pose_features(frames))
        if len(tf) >= MIN_TIGHT:
            K18_tight.append(q5._pose_features(tf))
            l = q8.side_features(tf, "left"); r = q8.side_features(tf, "right")
            G_tight.append([l["move"], l["spread_x"], l["spread_y"], l["height"],
                            r["move"], r["spread_x"], r["spread_y"], r["height"],
                            (l["move"] + r["move"]) / 2.0])
        else:
            K18_tight.append(np.full(18, np.nan)); G_tight.append(np.full(9, np.nan))
    K18_full = np.array(K18_full); K18_tight = np.array(K18_tight); G_tight = np.array(G_tight)
    ntight = np.array(ntight); ntot = np.array(ntot)
    keep = ~np.isnan(K18_tight).any(1)

    z = np.load(CACHE, allow_pickle=True)
    assert list(z["ids"]) == ids
    SHOT = z["SHOT"]; shotnames = list(z["shotnames"])

    L = ["# Q10: can the visual-channel result be rescued?\n"]
    L.append(f"- Framing criterion: at most {TIGHT_MAXP} detected persons AND a tracked body "
             f"covering at least {TIGHT_MINAREA:.0%} of the frame; a clip needs at least "
             f"{MIN_TIGHT} such frames.\n")
    L.append("\n**This is the third representation family tried on the same 30 clips. Any pass "
             "below is provisional and is a hypothesis for the 216-clip corpus, not a "
             "confirmation.**\n")

    # ── qualifying-frame audit ──────────────────────────────────────────────
    L.append("\n## 1. How many frames survive the framing criterion?\n")
    L.append("| clip | lang | label | total frames | tight frames | share | used |")
    L.append("|---|---|---|---|---|---|---|")
    for i in range(len(ids)):
        L.append(f"| {ids[i]} | {langs[i]} | {labels[i]} | {ntot[i]} | {ntight[i]} | "
                 f"{ntight[i]/ntot[i]:.3f} | {'yes' if keep[i] else 'NO'} |")
    ie = [i for i in range(len(ids)) if labels[i] == "Explaining"]
    io = [i for i in range(len(ids)) if labels[i] != "Explaining"]
    share = ntight / ntot
    U, p = mannwhitneyu(share[ie], share[io], alternative="two-sided")
    L.append(f"\n**Clips retained: {int(keep.sum())}/{len(ids)}** "
             f"(Explaining {sum(keep[i] for i in ie)}/{len(ie)}, "
             f"non-Explaining {sum(keep[i] for i in io)}/{len(io)})")
    L.append(f"\n**Share of tight frames by group: Explaining {share[ie].mean():.3f} vs "
             f"non-Explaining {share[io].mean():.3f}; Mann-Whitney p = {p:.4f}.** "
             "The criterion removes the difference in WHICH frames are measured; it cannot "
             "remove a difference in how many frames were available to measure, so a large "
             "gap here means the retained material is still not symmetric.")

    # ── RESCUE A ────────────────────────────────────────────────────────────
    idx = np.where(keep)[0]
    labs_k = [labels[i] for i in idx]; langs_k = [langs[i] for i in idx]
    reps = {
        "18d_full (confounded ref)":   (normalize(K18_full), labels, langs),
        "18d_tightframes":             (normalize(K18_tight[idx]), labs_k, langs_k),
        "14d_tightframes":             (normalize(np.delete(K18_tight[idx], q7.IDX_POSITION, 1)), labs_k, langs_k),
        "9d_gesture_cond_tightframes": (normalize(G_tight[idx]), labs_k, langs_k),
    }

    # ── RESCUE B ────────────────────────────────────────────────────────────
    S = np.nan_to_num(SHOT)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
    pca = PCA(n_components=3, random_state=SEED).fit(Sz)
    PCs = pca.transform(Sz)
    X = np.column_stack([np.ones(len(ids)), PCs])
    def residualize(K):
        beta, *_ = np.linalg.lstsq(X, K, rcond=None)
        return K - X @ beta
    L.append("\n## 2. Rescue B: shot-covariate residualization\n")
    L.append(f"Shot block = {shotnames}; first 3 PCs explain "
             f"{pca.explained_variance_ratio_[:3].sum():.1%} of its variance "
             f"({', '.join(f'{v:.1%}' for v in pca.explained_variance_ratio_[:3])}). "
             f"Each kinematic dimension is regressed on those 3 PCs plus an intercept "
             f"(4 parameters, n = 30, 26 residual df) and retrieval is run on the residuals.")
    reps["18d_residualized"] = (normalize(residualize(K18_full)), labels, langs)
    reps["14d_residualized"] = (normalize(residualize(np.delete(K18_full, q7.IDX_POSITION, 1))), labels, langs)

    # ── evaluation ──────────────────────────────────────────────────────────
    L.append("\n## 3. Full-pool retrieval\n")
    L.append("| representation | clips | dims | P@1 [95% CI] | P@3 | MAP | lab pur |")
    L.append("|---|---|---|---|---|---|---|")
    for name, (E, lb, lg) in reps.items():
        sim = cosine_similarity(E); res = q7.retrieval(sim, lb)
        m, lo, hi = ci(res["P@1"]); p3, *_ = ci(res["P@3"]); mp, *_ = ci(res["MAP"])
        s = sim.copy(); np.fill_diagonal(s, -1.0); nn = s.argmax(1)
        lp = np.mean([lb[i] == lb[nn[i]] for i in range(len(lb))])
        L.append(f"| {name} | {len(lb)} | {E.shape[1]} | {m:.3f} [{lo:.3f},{hi:.3f}] | "
                 f"{p3:.3f} | {mp:.3f} | {lp:.3f} |")

    L.append("\n## 4. Pooled-by-pairing cross-lingual test (q6 protocol)\n")
    L.append("| representation | group | n | hits | rate | weighted chance | lift | Poisson-binomial one-sided p |")
    L.append("|---|---|---|---|---|---|---|---|")
    perm, verdict = [], {}
    for name, (E, lb, lg) in reps.items():
        perm.append(pooled_test(E, lb, lg, rng, name, L, verdict))
    L.append("\n### Between-group difference in baseline-adjusted lift (permutation)\n")
    L += perm

    # does the framing confound survive inside the rescued representations?
    L.append("\n## 5. Did the rescue actually remove the confound?\n")
    L.append("| representation | presence_mean vs ppf_mean within retained clips | note |")
    L.append("|---|---|---|")
    from scipy.stats import pearsonr
    ppf = SHOT[:, shotnames.index("ppf_mean")]
    pm_full = K18_full[:, [4, 5, 12, 13]].mean(1)
    r0, p0 = pearsonr(pm_full, ppf)
    L.append(f"| 18d_full | r = {r0:+.3f} (p = {p0:.1e}) | the original confound |")
    pm_t = K18_tight[idx][:, [4, 5, 12, 13]].mean(1)
    r1, p1 = pearsonr(pm_t, ppf[idx])
    L.append(f"| 18d_tightframes | r = {r1:+.3f} (p = {p1:.3f}) | presence recomputed on tight frames only |")
    covm = z["cov"].mean(1)
    U2, p2 = mannwhitneyu(pm_t[[k for k, i in enumerate(idx) if labels[i] == 'Explaining']],
                          pm_t[[k for k, i in enumerate(idx) if labels[i] != 'Explaining']],
                          alternative="two-sided")
    L.append(f"\n**Tight-frame presence by group: Mann-Whitney U = {U2:.1f}, p = {p2:.4f}** "
             "(the full-clip version of this test was p = 0.0005).")

    L.append("\n## 6. Verdict (pooled Explaining pairing)\n")
    L.append("| representation | hits/n | rate | chance | lift | p | clears chance |")
    L.append("|---|---|---|---|---|---|---|")
    for name, (h, n, rate, ch, pv) in verdict.items():
        L.append(f"| {name} | {h}/{n} | {rate:.3f} | {ch:.3f} | {rate-ch:+.3f} | {pv:.4f} | "
                 f"{'YES' if pv < 0.05 else 'no'} |")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:1] + L[L.index("\n## 5. Did the rescue actually remove the confound?\n") - 40:]))
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
