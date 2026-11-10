"""
Q8 (W1 follow-up): a shot-invariant gesture representation.

Q7 established that the hand-presence dims of the 18-dim vector are a camerawork
variable: presence_mean correlates r = -0.86 with persons-per-frame and r = -0.87
with the fraction of 3+-person frames, and +0.69 with face extent. Hands are
detected when the shot is tight and undetected when it is wide. The 14-dim
"composition-robust" variant retains all five presence dims, so the pooled
Explaining result is not protected against this.

The fix is to stop measuring HOW OFTEN a hand is visible and measure WHAT THE
HAND DOES WHEN IT IS VISIBLE. Every feature below is computed conditional on
detection and divided by the speaker's own body extent, so it is invariant to
shot scale by construction:

  move      mean frame-to-frame displacement of the hand centroid,
            in units of body height  (gesture amplitude)
  spread_x  s.d. of hand centroid x, in units of body height
  spread_y  s.d. of hand centroid y, in units of body height
  height    mean (body centroid y - hand centroid y) / body height
            (gesture space: positive = hand carried above body centre)

Two slots (left, right) x 4 features + a pooled movement energy = 9 dims, matched
in dimensionality to the 9d_movement_only rep of Q7 so the comparison is fair.

Coverage is reported per clip: a clip whose hands are almost never detected
contributes a zero vector, and that is itself shot-driven, so the coverage column
must be inspected before the retrieval numbers are believed.

Run:  python analysis/q8_gesture_conditional.py  [--refresh]
Outputs: analysis/q8_results.md, analysis/_q8_cache.npz (+ stdout)
"""
import json, sys, importlib.util
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.stats import bootstrap as scipy_bootstrap

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("q5", ROOT / "analysis" / "q5_baselines.py")
q5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q5)
spec7 = importlib.util.spec_from_file_location("q7", ROOT / "analysis" / "q7_handpresence.py")
q7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(q7)

POSE_DIR  = ROOT / "data" / "pose"
LABELFILE = ROOT / "pilot" / "labels.json"
OUT_MD    = ROOT / "analysis" / "q8_results.md"
CACHE     = ROOT / "analysis" / "_q8_cache.npz"

MIN_RUN   = 2      # need >=2 consecutive detected frames to measure displacement
MIN_FRAMES = 10    # below this many detected frames a hand contributes nothing
NRES, SEED = 200000, 42

FEATNAMES = ["L.move", "L.spread_x", "L.spread_y", "L.height",
             "R.move", "R.spread_x", "R.spread_y", "R.height", "move_energy"]


def _centroid(pts):
    xs = [p[0] for p in pts if len(p) >= 2]; ys = [p[1] for p in pts if len(p) >= 2]
    if not xs: return None
    return float(np.mean(xs)), float(np.mean(ys))


def side_features(frames, side):
    """Conditional-on-detection hand kinematics for one slot, in body-height units."""
    body_h, body_cy = [], []
    # per-hand tracks: list of (frame_idx, cx, cy)
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
        # displacement only between frames that are adjacent in the source video
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


def build(files):
    G, cov = [], []
    for f in files:
        frames = json.load(open(f))
        l = side_features(frames, "left"); r = side_features(frames, "right")
        G.append([l["move"], l["spread_x"], l["spread_y"], l["height"],
                  r["move"], r["spread_x"], r["spread_y"], r["height"],
                  (l["move"] + r["move"]) / 2.0])
        cov.append([l["cover"], r["cover"]])
    return np.array(G), np.array(cov)


def pb_pval_ge(k, ps):
    return q7.pb_pval_ge(k, ps)


def ci(arr, nres=2000):
    arr = np.asarray(arr, dtype=float)
    r = scipy_bootstrap((arr,), np.mean, n_resamples=nres, confidence_level=0.95,
                        method="percentile", random_state=SEED)
    return float(arr.mean()), float(r.confidence_interval.low), float(r.confidence_interval.high)


def main():
    files  = sorted(POSE_DIR.glob("*.pose.json"))
    ids    = [f.name.replace(".pose.json", "") for f in files]
    langs  = [i.split("-")[1] for i in ids]
    lmap   = json.load(open(LABELFILE, encoding="utf-8"))
    labels = [lmap[i] for i in ids]

    refresh = "--refresh" in sys.argv
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        assert list(z["ids"]) == ids, "cache id mismatch; rerun with --refresh"
        G, cov, K18, SHOT, shotnames = z["G"], z["cov"], z["K18"], z["SHOT"], list(z["shotnames"])
    else:
        G, cov = build(files)
        K18, meas = [], []
        for f in files:
            frames = json.load(open(f))
            K18.append(q5._pose_features(frames)); meas.append(q7.clip_measures(frames))
        K18 = np.array(K18)
        shotnames = ["bbox_h", "bbox_area", "face_extent", "pose_extent",
                     "fallback_frac", "ppf_mean", "frac_1p", "frac_3plus"]
        SHOT = np.array([[m[v] for v in shotnames] for m in meas])
        np.savez(CACHE, ids=np.array(ids), G=G, cov=cov, K18=K18, SHOT=SHOT,
                 shotnames=np.array(shotnames))

    rng = np.random.default_rng(SEED)
    presence_mean = K18[:, [4, 5, 12, 13]].mean(1)

    reps = {
        "14d_comprobust (Q7 ref)": normalize(np.delete(K18, q7.IDX_POSITION, axis=1)),
        "9d_movement_only (Q7)":   normalize(K18[:, q7.IDX_MOVEMENT]),
        "9d_gesture_conditional":  normalize(G),
    }

    L = ["# Q8: shot-invariant conditional-gesture representation\n"]
    L.append(f"- Clips: {len(ids)} | min detected frames per hand: {MIN_FRAMES} | "
             f"all features divided by the speaker's own body height\n")

    # ── coverage ────────────────────────────────────────────────────────────
    L.append("\n## Detection coverage and raw features\n")
    L.append("| clip | lang | label | L cover | R cover | L.move | R.move | L.height | R.height |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i in range(len(ids)):
        L.append(f"| {ids[i]} | {langs[i]} | {labels[i]} | {cov[i,0]:.3f} | {cov[i,1]:.3f} | "
                 f"{G[i,0]:.4f} | {G[i,4]:.4f} | {G[i,3]:+.3f} | {G[i,7]:+.3f} |")
    zero_rows = int((np.abs(G).sum(1) == 0).sum())
    L.append(f"\n**Clips with an all-zero gesture vector (both slots below the "
             f"{MIN_FRAMES}-frame detection floor): {zero_rows}/{len(ids)}**")
    L.append(f"\n**Mean detection coverage: left slot {cov[:,0].mean():.3f}, "
             f"right slot {cov[:,1].mean():.3f}**")

    # ── decorrelation check ─────────────────────────────────────────────────
    L.append("\n## Is the new representation decorrelated from shot scale?\n")
    L.append("Pearson r (p) of each conditional-gesture dim against each shot variable, n=30. "
             "For reference, Q7's presence_mean reached |r| = 0.87.\n")
    L.append("| gesture dim | " + " | ".join(shotnames) + " |")
    L.append("|---" * (len(shotnames) + 1) + "|")
    worst = 0.0; worstcell = ""
    for j, fn in enumerate(FEATNAMES):
        cells = []
        for k, sn in enumerate(shotnames):
            xv = SHOT[:, k]; yv = G[:, j]
            ok = np.isfinite(xv) & np.isfinite(yv)
            if np.std(yv[ok]) < 1e-12 or np.std(xv[ok]) < 1e-12:
                cells.append("n/a"); continue
            r, p = pearsonr(yv[ok], xv[ok])
            cells.append(f"{'**' if p<0.05 else ''}{r:+.2f}{'**' if p<0.05 else ''} ({p:.3f})")
            if abs(r) > worst: worst, worstcell = abs(r), f"{fn} ~ {sn}: r={r:+.3f}, p={p:.4f}"
        L.append(f"| {fn} | " + " | ".join(cells) + " |")
    L.append(f"\n**Strongest remaining gesture~shot association: {worstcell}**")
    r_pm, p_pm = pearsonr(np.abs(G).sum(1), presence_mean)
    L.append(f"\n**Gesture-vector magnitude vs Q7 presence_mean: r = {r_pm:+.3f} (p = {p_pm:.4f})**")

    # ── retrieval ───────────────────────────────────────────────────────────
    L.append("\n## Full-pool retrieval and NN purity\n")
    ppf = SHOT[:, shotnames.index("ppf_mean")]
    order = np.argsort(ppf); group = np.empty(len(ids), dtype=int)
    for rank, idx in enumerate(order):
        group[idx] = 0 if rank < 10 else (1 if rank < 20 else 2)
    L.append("| representation | dims | P@1 [95% CI] | P@3 | MAP | lab pur | lang pur | comp pur |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, E in reps.items():
        sim = cosine_similarity(E); res = q7.retrieval(sim, labels)
        p1m, p1l, p1h = ci(res["P@1"]); p3m, *_ = ci(res["P@3"]); mm, *_ = ci(res["MAP"])
        s = sim.copy(); np.fill_diagonal(s, -1.0); nn = s.argmax(1)
        lp = np.mean([labels[i] == labels[nn[i]] for i in range(len(ids))])
        gp = np.mean([langs[i] == langs[nn[i]] for i in range(len(ids))])
        cp = np.mean([group[i] == group[nn[i]] for i in range(len(ids))])
        L.append(f"| {name} | {E.shape[1]} | {p1m:.3f} [{p1l:.3f},{p1h:.3f}] | {p3m:.3f} | "
                 f"{mm:.3f} | {lp:.3f} | {gp:.3f} | {cp:.3f} |")
    L.append("\n(baselines: label LOO=0.303 | language=0.310 | composition LOO=0.310)")

    # ── pooled-by-pairing test ──────────────────────────────────────────────
    L.append("\n## Pooled-by-pairing cross-lingual test (q6 protocol)\n")
    L.append("| representation | group | n | hits | rate | weighted chance | lift | Poisson-binomial one-sided p |")
    L.append("|---|---|---|---|---|---|---|---|")
    perm_lines, verdict = [], {}
    for name, E in reps.items():
        cl = q5.crossling_per_query(E, labels, langs)
        rows = [r for src in ["Amis", "Paiwan", "Truku"] for r in cl[src]]
        expl = [r for r in rows if r[1] == "Explaining"]
        nonx = [r for r in rows if r[1] != "Explaining"]
        for gname, grp in [("Explaining", expl), ("non-Explaining", nonx)]:
            h = sum(r[4] for r in grp); n = len(grp); ps = [r[5] for r in grp]
            pv = pb_pval_ge(h, ps)
            L.append(f"| {name} | {gname} | {n} | {h} | {h/n:.3f} | {np.mean(ps):.3f} | "
                     f"{h/n-np.mean(ps):+.3f} | {pv:.4f} |")
            if gname == "Explaining": verdict[name] = (h, n, h/n, float(np.mean(ps)), pv)
        resid = np.array([r[4] - r[5] for r in expl + nonx], dtype=float)
        obs, pp_ = q7.perm_lift_diff(resid, len(expl), rng)
        perm_lines.append(f"- {name}: lift difference (Explaining - non-Explaining) = {obs:+.3f}, "
                          f"permutation one-sided p = {pp_:.4f}")
    L.append("\n### Between-group difference in baseline-adjusted lift (permutation)\n")
    L += perm_lines

    L.append("\n### Cross-lingual P@1 by source language\n")
    L.append("| representation | Amis | Paiwan | Truku |")
    L.append("|---|---|---|---|")
    for name, E in reps.items():
        cl = q5.crossling_per_query(E, labels, langs)
        L.append(f"| {name} | " + " | ".join(
            f"{np.mean([r[4] for r in cl[s]]):.3f}" for s in ["Amis", "Paiwan", "Truku"]) + " |")

    L.append("\n## Verdict (pooled Explaining pairing)\n")
    L.append("| representation | hits/n | rate | chance | lift | p | clears chance at p<0.05 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, (h, n, rate, ch, pv) in verdict.items():
        L.append(f"| {name} | {h}/{n} | {rate:.3f} | {ch:.3f} | {rate-ch:+.3f} | {pv:.4f} | "
                 f"{'YES' if pv < 0.05 else 'no'} |")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-40:]))
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
