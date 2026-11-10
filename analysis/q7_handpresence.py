"""
Q7 (W1 go/no-go): is the kinematic signal gesture, or is it camerawork?

Two distinct worries sit inside the 18-dim pose vector:

  (a) POSITION  — dims 0,1,8,9 (L/R body_mean_x/y) encode where a body sits in
      the frame. Already addressed by the 14-dim composition-robust variant
      (q3q4/q5/q6), which drops them.

  (b) PRESENCE  — dims 4,5,12,13 (L/R lh_ratio, rh_ratio) and dim 17 (hand
      asymmetry) encode HOW OFTEN a hand is detected. A hand is detected when
      the speaker gestures, but also when the shot is framed so that the hands
      are inside the frame at a detectable size. Presence therefore may be a
      shot-scale proxy. The 14-dim variant RETAINS all five of these dims, so
      the strongest current result (pooled Explaining, 14-dim, p=0.0057) is not
      protected against (b) at all.

This script does three things:

  W1-a  Correlate the presence dims against direct shot-scale measures taken
        from the same pose files but independent of hand behaviour (slot bbox
        height/area, face-keypoint extent, body-keypoint vertical extent) and
        against the composition variables from q1q2.

  W1-b  Build a MOVEMENT-ONLY representation (9 dims: body_std, face_std,
        mean_body_energy) that drops position AND presence entirely, and rerun
        the full pooled-by-pairing test from q6 on it. Also runs a
        PRESENCE-ONLY representation (5 dims) as the adversarial complement:
        if presence-only alone reproduces the pooled Explaining effect, the
        effect is not safely attributable to gesture.

  W1-c  Language x shot-composition skew: is shot composition confounded with
        language? (If Truku clips are systematically framed differently, the
        Truku cross-lingual failure is a framing result, not a gesture result.)

Reuses _pose_features and crossling_per_query from q5_baselines.py verbatim, so
the 18-dim and 14-dim rows here must reproduce q3q4/q5/q6 exactly.

Run:  python analysis/q7_handpresence.py
Outputs: analysis/q7_results.md (+ stdout)
"""
import json, importlib.util
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr, kruskal, chi2_contingency
from scipy.stats import bootstrap as scipy_bootstrap

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("q5", ROOT / "analysis" / "q5_baselines.py")
q5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q5)

POSE_DIR  = ROOT / "data" / "pose"
LABELFILE = ROOT / "pilot" / "labels.json"
OUT_MD    = ROOT / "analysis" / "q7_results.md"

FRAME_W, FRAME_H = 1280, 720
FALLBACK_BOXES = {(0, 0, 640, 720), (640, 0, 1280, 720)}
NRES, SEED = 200000, 42

# dim indices in the 18-dim vector
IDX_POSITION = [0, 1, 8, 9]            # L/R body_mean_x, body_mean_y
IDX_PRESENCE = [4, 5, 12, 13, 17]      # L/R lh_ratio, rh_ratio, + hand asymmetry
IDX_MOVEMENT = [2, 3, 6, 7, 10, 11, 14, 15, 16]   # body_std, face_std, mean_body_energy


# ── per-clip shot-scale + composition measures ──────────────────────────────
def clip_measures(frames):
    n = len(frames)
    bbox_h, bbox_a, face_ext, pose_ext = [], [], [], []
    fallback = slotframes = facefilled = posefilled = 0
    ppf, n1, n2, n3 = [], 0, 0, 0
    lfill = rfill = 0
    for fr in frames:
        persons = fr.get("persons") or []
        k = len(persons); ppf.append(k)
        if k <= 1: n1 += 1
        elif k == 2: n2 += 1
        else: n3 += 1
        for side in ("left", "right"):
            s = fr[side]
            if s is None: continue
            if side == "left": lfill += 1
            else: rfill += 1
            slotframes += 1
            b = s.get("bbox")
            if isinstance(b, dict):
                key = (b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2"))
                if key in FALLBACK_BOXES:
                    fallback += 1
                else:
                    bbox_h.append((b["y2"] - b["y1"]) / FRAME_H)
                    bbox_a.append(((b["y2"] - b["y1"]) * (b["x2"] - b["x1"]))
                                  / (FRAME_H * FRAME_W))
            # face extent = shot scale, independent of hands
            pf = []
            for part in (s.get("face") or {}).values():
                if isinstance(part, list): pf.extend(part)
            xs = [p[0] for p in pf if len(p) >= 2]; ys = [p[1] for p in pf if len(p) >= 2]
            if len(xs) > 1:
                facefilled += 1
                face_ext.append(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
            # body vertical extent = how much of the frame the torso occupies
            pts = s.get("pose") or []
            ys2 = [p[1] for p in pts if len(p) >= 2]
            if len(ys2) > 1:
                posefilled += 1
                pose_ext.append(max(ys2) - min(ys2))
    ppf = np.array(ppf)
    m = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return {
        "bbox_h": m(bbox_h), "bbox_area": m(bbox_a),
        "face_extent": m(face_ext), "pose_extent": m(pose_ext),
        "fallback_frac": fallback / slotframes if slotframes else float("nan"),
        "face_fill": facefilled / slotframes if slotframes else float("nan"),
        "pose_fill": posefilled / slotframes if slotframes else float("nan"),
        "ppf_mean": float(ppf.mean()), "ppf_std": float(ppf.std()),
        "frac_1p": n1 / n, "frac_2p": n2 / n, "frac_3plus": n3 / n,
        "left_fill": lfill / n, "right_fill": rfill / n,
    }


# ── stats helpers ───────────────────────────────────────────────────────────
def pb_pmf(ps):
    pmf = np.zeros(len(ps) + 1); pmf[0] = 1.0
    for p in ps:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= (1 - p)
    return pmf

def pb_pval_ge(k, ps):
    return float(pb_pmf(ps)[k:].sum())

def perm_lift_diff(resid, ne, rng, nres=NRES, chunk=20000):
    obs = resid[:ne].mean() - resid[ne:].mean()
    n, cnt, done = len(resid), 0, 0
    while done < nres:
        c = min(chunk, nres - done)
        idx = np.argsort(rng.random((c, n)), axis=1)
        perm = resid[idx]
        stat = perm[:, :ne].mean(1) - perm[:, ne:].mean(1)
        cnt += int((stat >= obs).sum()); done += c
    return float(obs), cnt / nres

def loo_baseline(sizes):
    N = sum(sizes)
    return sum(g * (g - 1) for g in sizes) / (N * (N - 1))

def ci(arr, nres=2000):
    arr = np.asarray(arr, dtype=float)
    r = scipy_bootstrap((arr,), np.mean, n_resamples=nres, confidence_level=0.95,
                        method="percentile", random_state=SEED)
    return float(arr.mean()), float(r.confidence_interval.low), float(r.confidence_interval.high)

def retrieval(simM, labels, ks=(1, 3)):
    n = len(labels); out = {f"P@{k}": [] for k in ks}; aps = []
    for i in range(n):
        s = simM[i].copy(); s[i] = -1
        ranked = np.argsort(s)[::-1]
        for k in ks:
            out[f"P@{k}"].append(sum(1 for j in ranked[:k] if labels[j] == labels[i]) / k)
        rel = [1 if labels[j] == labels[i] else 0 for j in ranked]
        ap = nr = 0
        for rank, r in enumerate(rel, 1):
            if r: nr += 1; ap += nr / rank
        aps.append(ap / sum(rel) if sum(rel) else 0.0)
    out["MAP"] = aps
    return out


def main():
    files  = sorted(POSE_DIR.glob("*.pose.json"))
    ids    = [f.name.replace(".pose.json", "") for f in files]
    langs  = [i.split("-")[1] for i in ids]
    lmap   = json.load(open(LABELFILE, encoding="utf-8"))
    labels = [lmap[i] for i in ids]

    K18, meas = [], []
    for f in files:
        frames = json.load(open(f))
        K18.append(q5._pose_features(frames))
        meas.append(clip_measures(frames))
    K18 = np.array(K18)

    reps = {
        "18d_baseline":      normalize(K18),
        "14d_comprobust":    normalize(np.delete(K18, IDX_POSITION, axis=1)),
        "9d_movement_only":  normalize(K18[:, IDX_MOVEMENT]),
        "5d_presence_only":  normalize(K18[:, IDX_PRESENCE]),
    }
    rng = np.random.default_rng(SEED)

    L = ["# Q7 (W1): hand-presence vs shot scale, and the movement-only variant\n"]
    L.append(f"- Clips: {len(ids)} | frame {FRAME_W}x{FRAME_H} (bbox in px; pose/face/hand keypoints already normalized to [0,1]) | permutation resamples: {NRES:,} | seed {SEED}\n")
    L.append("\nDimension partition of the 18-dim vector:\n")
    L.append("| block | dims | content |")
    L.append("|---|---|---|")
    L.append(f"| position | {IDX_POSITION} | L/R body_mean_x, body_mean_y |")
    L.append(f"| presence | {IDX_PRESENCE} | L/R lh_ratio, rh_ratio, hand asymmetry |")
    L.append(f"| movement | {IDX_MOVEMENT} | L/R body_std_x/y, face_std_x/y, mean_body_energy |")

    # ── W1-a: presence vs shot scale ────────────────────────────────────────
    L.append("\n## W1-a. Do the hand-presence dims track shot scale?\n")
    presence = {
        "L.lh_ratio": K18[:, 4], "L.rh_ratio": K18[:, 5],
        "R.lh_ratio": K18[:, 12], "R.rh_ratio": K18[:, 13],
        "hand_asym(d17)": K18[:, 17],
        "presence_mean": K18[:, [4, 5, 12, 13]].mean(1),
    }
    shotvars = ["bbox_h", "bbox_area", "face_extent", "pose_extent", "fallback_frac", "face_fill",
                "ppf_mean", "frac_1p", "frac_3plus", "left_fill", "right_fill"]
    M = {v: np.array([m[v] for m in meas]) for v in shotvars}

    L.append("\nPer-clip shot-scale and composition measures (means over frames):\n")
    L.append("| clip | lang | label | bbox_h | face_ext | pose_ext | fallback | ppf | presence_mean |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i in range(len(ids)):
        L.append(f"| {ids[i]} | {langs[i]} | {labels[i]} | {M['bbox_h'][i]:.3f} | "
                 f"{M['face_extent'][i]:.4f} | {M['pose_extent'][i]:.3f} | "
                 f"{M['fallback_frac'][i]:.3f} | {M['ppf_mean'][i]:.2f} | "
                 f"{presence['presence_mean'][i]:.3f} |")

    L.append("\nPearson r (p) between each presence dim and each shot/composition variable, n=30:\n")
    L.append("| presence dim | " + " | ".join(shotvars) + " |")
    L.append("|---" * (len(shotvars) + 1) + "|")
    maxabs = 0.0; maxcell = ""
    for pname, pv in presence.items():
        cells = []
        for v in shotvars:
            xv = M[v]
            ok = np.isfinite(xv) & np.isfinite(pv)
            if ok.sum() < 3 or np.std(xv[ok]) < 1e-12:
                cells.append("n/a"); continue
            r, p = pearsonr(pv[ok], xv[ok])
            star = "**" if p < 0.05 else ""
            cells.append(f"{star}{r:+.2f}{star} ({p:.3f})")
            if abs(r) > maxabs: maxabs, maxcell = abs(r), f"{pname} ~ {v}: r={r:+.3f}, p={p:.4f}"
        L.append(f"| {pname} | " + " | ".join(cells) + " |")
    L.append(f"\n**Strongest presence~shot association:** {maxcell}")

    # presence-profile similarity vs shot-profile similarity, over all 435 pairs
    Pz = (K18[:, [4, 5, 12, 13, 17]] - K18[:, [4, 5, 12, 13, 17]].mean(0)) / (K18[:, [4, 5, 12, 13, 17]].std(0) + 1e-9)
    Sraw = np.nan_to_num(np.column_stack([M[v] for v in ["bbox_h", "bbox_area", "face_extent", "pose_extent"]]))
    Sz = (Sraw - Sraw.mean(0)) / (Sraw.std(0) + 1e-9)
    iu = np.triu_indices(len(ids), k=1)
    pv_ = cosine_similarity(Pz)[iu]; sv_ = cosine_similarity(Sz)[iu]
    r1, p1 = pearsonr(pv_, sv_); r2, p2 = spearmanr(pv_, sv_)
    L.append(f"\n**Presence-profile cosine vs shot-scale-profile cosine (n={len(pv_)} pairs):** "
             f"Pearson r={r1:.3f} (p={p1:.1e}), Spearman rho={r2:.3f} (p={p2:.1e})")

    # ── W1-b: movement-only and presence-only reps ──────────────────────────
    L.append("\n## W1-b. Movement-only and presence-only representations\n")

    lab_sizes = list(Counter(labels).values())
    ppf_mean = M["ppf_mean"]; order = np.argsort(ppf_mean)
    group = np.empty(len(ids), dtype=int)
    for rank, idx in enumerate(order):
        group[idx] = 0 if rank < 10 else (1 if rank < 20 else 2)
    gsizes = [int((group == g).sum()) for g in (0, 1, 2)]
    base_comp = loo_baseline(gsizes); base_lab = loo_baseline(lab_sizes); base_lang = 9 / 29

    L.append(f"Baselines: label LOO={base_lab:.3f} | language={base_lang:.3f} | composition LOO={base_comp:.3f}\n")
    L.append("\n### Full-pool retrieval and NN purity\n")
    L.append("| representation | dims | P@1 [95% CI] | P@3 | MAP | lab pur | lang pur | comp pur |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, E in reps.items():
        sim = cosine_similarity(E)
        res = retrieval(sim, labels)
        p1m, p1l, p1h = ci(res["P@1"]); p3m, *_ = ci(res["P@3"]); mapm, *_ = ci(res["MAP"])
        s = sim.copy(); np.fill_diagonal(s, -1.0); nn = s.argmax(1)
        lp = np.mean([labels[i] == labels[nn[i]] for i in range(len(ids))])
        gp = np.mean([langs[i] == langs[nn[i]] for i in range(len(ids))])
        cp = np.mean([group[i] == group[nn[i]] for i in range(len(ids))])
        L.append(f"| {name} | {E.shape[1]} | {p1m:.3f} [{p1l:.3f},{p1h:.3f}] | {p3m:.3f} | "
                 f"{mapm:.3f} | {lp:.3f} | {gp:.3f} | {cp:.3f} |")

    L.append("\n### Pooled-by-pairing cross-lingual test (q6 protocol)\n")
    L.append("| representation | group | n | hits | rate | weighted chance | lift | Poisson-binomial one-sided p |")
    L.append("|---|---|---|---|---|---|---|---|")
    perm_lines, verdicts = [], {}
    for name, E in reps.items():
        cl = q5.crossling_per_query(E, labels, langs)
        rows = [r for src in ["Amis", "Paiwan", "Truku"] for r in cl[src]]
        expl = [r for r in rows if r[1] == "Explaining"]
        nonx = [r for r in rows if r[1] != "Explaining"]
        for gname, grp in [("Explaining", expl), ("non-Explaining", nonx)]:
            hits = sum(r[4] for r in grp); n = len(grp); ps = [r[5] for r in grp]
            pval = pb_pval_ge(hits, ps)
            L.append(f"| {name} | {gname} | {n} | {hits} | {hits/n:.3f} | {np.mean(ps):.3f} | "
                     f"{hits/n-np.mean(ps):+.3f} | {pval:.4f} |")
            if gname == "Explaining":
                verdicts[name] = (hits, n, hits / n, float(np.mean(ps)), pval)
        resid = np.array([r[4] - r[5] for r in expl + nonx], dtype=float)
        obs, pp_ = perm_lift_diff(resid, len(expl), rng)
        perm_lines.append(f"- {name}: observed lift difference (Explaining - non-Explaining) = {obs:+.3f}, "
                          f"permutation one-sided p = {pp_:.4f}")
    L.append("\n### Between-group difference in baseline-adjusted lift (permutation)\n")
    L += perm_lines

    # per-language cross-lingual P@1 for each rep
    L.append("\n### Cross-lingual P@1 by source language\n")
    L.append("| representation | Amis | Paiwan | Truku |")
    L.append("|---|---|---|---|")
    for name, E in reps.items():
        cl = q5.crossling_per_query(E, labels, langs)
        vals = [np.mean([r[4] for r in cl[s]]) for s in ["Amis", "Paiwan", "Truku"]]
        L.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    # ── W1-c: language x composition skew ───────────────────────────────────
    L.append("\n## W1-c. Language x shot-composition skew\n")
    L.append("\n### Contingency: language x composition tertile (by mean persons/frame)\n")
    langset = ["Amis", "Paiwan", "Truku"]
    tab = np.zeros((3, 3), dtype=int)
    for i in range(len(ids)):
        tab[langset.index(langs[i]), group[i]] += 1
    L.append("| language | low | mid | high |")
    L.append("|---|---|---|---|")
    for li, lg in enumerate(langset):
        L.append(f"| {lg} | {tab[li,0]} | {tab[li,1]} | {tab[li,2]} |")
    chi2, pchi, dof, _ = chi2_contingency(tab + 0.0)
    cv = np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
    L.append(f"\nChi-square = {chi2:.3f}, df = {dof}, p = {pchi:.4f}, Cramer's V = {cv:.3f} "
             f"(small cells: exact interpretation only)")

    L.append("\n### Per-language means of each shot/composition variable (Kruskal-Wallis across 3 languages)\n")
    L.append("| variable | Amis | Paiwan | Truku | H | p |")
    L.append("|---|---|---|---|---|---|")
    for v in shotvars + ["presence_mean"]:
        xv = presence["presence_mean"] if v == "presence_mean" else M[v]
        groups = [xv[[i for i in range(len(ids)) if langs[i] == lg]] for lg in langset]
        groups = [g[np.isfinite(g)] for g in groups]
        try:
            H, pk = kruskal(*groups)
        except ValueError:
            H, pk = float("nan"), float("nan")
        star = "**" if pk < 0.05 else ""
        L.append(f"| {v} | {np.mean(groups[0]):.3f} | {np.mean(groups[1]):.3f} | "
                 f"{np.mean(groups[2]):.3f} | {H:.2f} | {star}{pk:.4f}{star} |")

    # within- vs between-language composition similarity
    comp = np.nan_to_num(np.column_stack([M[v] for v in shotvars]))
    compz = (comp - comp.mean(0)) / (comp.std(0) + 1e-9)
    simC = cosine_similarity(compz)
    same = np.array([simC[i, j] for i in range(len(ids)) for j in range(i + 1, len(ids)) if langs[i] == langs[j]])
    diff = np.array([simC[i, j] for i in range(len(ids)) for j in range(i + 1, len(ids)) if langs[i] != langs[j]])
    L.append(f"\n**Composition-profile cosine, within-language (n={len(same)}) mean = {same.mean():+.3f} "
             f"vs between-language (n={len(diff)}) mean = {diff.mean():+.3f}; difference = {same.mean()-diff.mean():+.3f}**")

    # ── verdict ─────────────────────────────────────────────────────────────
    L.append("\n## Go / no-go summary (pooled Explaining pairing)\n")
    L.append("| representation | hits/n | rate | chance | lift | p | clears chance at p<0.05 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, (h, n, rate, ch, pval) in verdicts.items():
        L.append(f"| {name} | {h}/{n} | {rate:.3f} | {ch:.3f} | {rate-ch:+.3f} | {pval:.4f} | "
                 f"{'YES' if pval < 0.05 else 'no'} |")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
