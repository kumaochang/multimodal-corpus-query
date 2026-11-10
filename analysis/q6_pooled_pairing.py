"""
Q6: pooled-by-pairing cross-lingual test (Explaining/demonstration vs non-Explaining).

Replaces the confounded language-level aggregate contrast: cross by PAIRING, pooling
all three source languages, each query judged against its own label-specific chance.

- Per-group significance: exact Poisson-binomial (each query carries its own chance p),
  one-sided P(X >= observed hits).
- Between-group difference in baseline-adjusted lift: Monte Carlo permutation test on
  per-query lift (hit - own chance), group labels permuted, 200,000 resamples, one-sided.

Reuses _pose_features + crossling_per_query from q5_baselines.py (verbatim pilot code).

Run:  python analysis/q6_pooled_pairing.py
Outputs: analysis/q6_results.md (+ stdout)
"""
import json, importlib.util
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("q5", ROOT / "analysis" / "q5_baselines.py")
q5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q5)

POSE_DIR = ROOT / "data" / "pose"
LABELFILE = ROOT / "pilot" / "labels.json"
OUT_MD = ROOT / "analysis" / "q6_results.md"
NRES = 200000
SEED = 42


def pb_pmf(ps):
    """Exact Poisson-binomial pmf via DP over independent per-trial probabilities."""
    pmf = np.zeros(len(ps) + 1); pmf[0] = 1.0
    for p in ps:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= (1 - p)
    return pmf

def pb_pval_ge(k, ps):
    return float(pb_pmf(ps)[k:].sum())


def main():
    files = sorted(POSE_DIR.glob("*.pose.json"))
    ids = [f.name.replace(".pose.json", "") for f in files]
    langs = [i.split("-")[1] for i in ids]
    lmap = json.load(open(LABELFILE, encoding="utf-8"))
    labels = [lmap[i] for i in ids]
    K18 = np.array([q5._pose_features(json.load(open(f))) for f in files])
    K14 = np.delete(K18, [0, 1, 8, 9], axis=1)
    reps = {"18-dim": normalize(K18), "14-dim": normalize(K14)}
    rng = np.random.default_rng(SEED)

    L = ["# Q6: pooled-by-pairing cross-lingual test\n"]

    # full per-cell grid
    L.append("## Per-cell grid (language × label): hits / n, chance, lift, Poisson-binomial p\n")
    L.append("| rep | source | label | n | hits | rate | chance | lift | binom p |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    from scipy.stats import binomtest
    for rep, E in reps.items():
        cl = q5.crossling_per_query(E, labels, langs)
        cells = defaultdict(lambda: [0, 0, 0.0])
        for src in ["Amis", "Paiwan", "Truku"]:
            for qi, ql, ti, tl, hit, base, ns in cl[src]:
                c = cells[(src, ql)]; c[0] += hit; c[1] += 1; c[2] = base
        for (src, ql), (h, n, base) in sorted(cells.items()):
            p = binomtest(h, n, base, alternative="greater").pvalue
            L.append(f"| {rep} | {src} | {ql} | {n} | {h} | {h/n:.3f} | {base:.3f} | {h/n-base:+.3f} | {p:.3f} |")

    # pooled Explaining vs non-Explaining
    L.append("\n## Pooled by pairing (all three source languages)\n")
    L.append("| rep | group | n | hits | rate | weighted chance | lift | Poisson-binomial one-sided p |")
    L.append("|---|---|---|---|---|---|---|---|")
    perm_lines = []
    for rep, E in reps.items():
        cl = q5.crossling_per_query(E, labels, langs)
        rows = [r for src in ["Amis", "Paiwan", "Truku"] for r in cl[src]]
        expl = [r for r in rows if r[1] == "Explaining"]
        nonx = [r for r in rows if r[1] != "Explaining"]
        for name, grp in [("Explaining", expl), ("non-Explaining", nonx)]:
            hits = sum(r[4] for r in grp); n = len(grp); ps = [r[5] for r in grp]
            L.append(f"| {rep} | {name} | {n} | {hits} | {hits/n:.3f} | {np.mean(ps):.3f} | "
                     f"{hits/n-np.mean(ps):+.3f} | {pb_pval_ge(hits, ps):.4f} |")
        # permutation test on baseline-adjusted lift
        resid = np.array([r[4] - r[5] for r in expl + nonx])
        ne = len(expl)
        obs = resid[:ne].mean() - resid[ne:].mean()
        cnt = 0
        for _ in range(NRES):
            perm = rng.permutation(resid)
            if perm[:ne].mean() - perm[ne:].mean() >= obs: cnt += 1
        perm_lines.append(f"- {rep}: observed lift difference (Explaining − non-Explaining) = {obs:+.3f}, "
                          f"permutation one-sided p = {cnt/NRES:.4f} ({NRES:,} resamples, per-query lift permuted)")

    L.append("\n## Between-group difference in baseline-adjusted lift (permutation test)\n")
    L += perm_lines

    # ── text vs kinematic on the pooled Explaining pairing (paired McNemar) ──
    from scipy.stats import binomtest
    z = np.load(ROOT / "analysis" / "_emb_cache.npz", allow_pickle=True)
    assert list(z["ids"]) == ids, "cache id mismatch; rerun step4_supplementary.py"
    T = normalize(z["T"])
    def expl_hits(E):
        cl = q5.crossling_per_query(E, labels, langs)
        rows = sorted([r for src in ["Amis", "Paiwan", "Truku"] for r in cl[src] if r[1] == "Explaining"], key=lambda r: r[0])
        return np.array([r[4] for r in rows])
    L.append("\n## Text vs kinematic on the pooled Explaining pairing (paired, 14 queries)\n")
    L.append("| representation | text hits | kin hits | discordant (text>kin / kin>text) | exact McNemar p |")
    L.append("|---|---|---|---|---|")
    tv = expl_hits(T)
    for rep in ["18-dim", "14-dim"]:
        kv = expl_hits(reps[rep])
        b = int(np.sum((tv == 1) & (kv == 0))); c = int(np.sum((tv == 0) & (kv == 1)))
        p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
        L.append(f"| {rep} | {int(tv.sum())}/14 | {int(kv.sum())}/14 | {b} / {c} | {p:.3f} |")
    L.append("\nText is a reference ceiling (inflated by the subtitle pivot and annotation circularity), not a competitor; the difference is not separable at this n.")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
