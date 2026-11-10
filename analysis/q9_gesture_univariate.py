"""
Q9 (W1, exploratory): univariate tests on the conditional-gesture dimensions.

Q8 showed that once hand behaviour is measured conditional on detection and
scaled by body height, nearest-neighbour retrieval on the resulting 9-dim vector
does not clear label-specific chance (6/14, p = 0.58). That is a null for the
RETRIEVAL framing, which asks a 9-dimensional cosine to rank 20 candidates from
30 clips. It is not the same question as whether any single interpretable gesture
parameter differs by illocutionary category.

This script runs that second, much better-powered question, and is EXPLORATORY:
the dimensions were inspected before the tests were chosen. Holm correction is
applied across the nine dimensions, and per-language breakdowns are reported so
that a result driven by one language is visible as such. Nothing here is
confirmatory evidence; it exists to supply the directional hypothesis that the
main study's pre-registered analysis rule will test on independent data.

Run:  python analysis/q9_gesture_univariate.py
Outputs: analysis/q9_results.md (+ stdout)
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import kruskal, mannwhitneyu

ROOT   = Path(__file__).resolve().parent.parent
CACHE  = ROOT / "analysis" / "_q8_cache.npz"
LABELF = ROOT / "pilot" / "labels.json"
OUT_MD = ROOT / "analysis" / "q9_results.md"

FEATNAMES = ["L.move", "L.spread_x", "L.spread_y", "L.height",
             "R.move", "R.spread_x", "R.spread_y", "R.height", "move_energy"]


def holm(pvals):
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        run = max(run, v)
        adj[i] = min(run, 1.0)
    return adj


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def main():
    z = np.load(CACHE, allow_pickle=True)
    ids = list(z["ids"]); G = z["G"]; cov = z["cov"]
    langs = [i.split("-")[1] for i in ids]
    lmap = json.load(open(LABELF, encoding="utf-8"))
    labels = [lmap[i] for i in ids]
    labset = sorted(set(labels), key=lambda x: -labels.count(x))

    # bilateral aggregates: average the two slots, since slot identity (questioner /
    # respondent) is not the variable of interest here
    AGG = {
        "move (both slots)":     (G[:, 0] + G[:, 4]) / 2,
        "spread_x (both slots)": (G[:, 1] + G[:, 5]) / 2,
        "spread_y (both slots)": (G[:, 2] + G[:, 6]) / 2,
        "height (both slots)":   (G[:, 3] + G[:, 7]) / 2,
    }

    L = ["# Q9 (exploratory): univariate tests on conditional-gesture dimensions\n"]
    L.append(f"- Clips: {len(ids)} | labels: {dict(Counter(labels))}\n")
    L.append("\n**These tests are exploratory.** The dimensions were inspected before the "
             "tests were chosen. Holm-corrected p-values are reported across the nine "
             "per-slot dimensions; the four bilateral aggregates are corrected separately. "
             "Treat every result here as a hypothesis for the main study, not as evidence.\n")

    # ── 1. Kruskal-Wallis across all four labels ────────────────────────────
    L.append("\n## 1. Kruskal-Wallis across the four pragmatic labels\n")
    L.append("| dimension | " + " | ".join(f"{l} (n={labels.count(l)})" for l in labset) +
             " | H | p | Holm p |")
    L.append("|---" * (len(labset) + 4) + "|")
    raw = []
    rows = []
    for j, fn in enumerate(FEATNAMES):
        groups = [G[[i for i in range(len(ids)) if labels[i] == l], j] for l in labset]
        H, p = kruskal(*groups)
        raw.append(p); rows.append((fn, [g.mean() for g in groups], H, p))
    adj = holm(np.array(raw))
    for (fn, means, H, p), a in zip(rows, adj):
        star = "**" if a < 0.05 else ""
        L.append(f"| {fn} | " + " | ".join(f"{m:+.4f}" for m in means) +
                 f" | {H:.2f} | {p:.4f} | {star}{a:.4f}{star} |")

    # ── 2. Explaining vs non-Explaining (the pairing used in q6/q7/q8) ───────
    L.append("\n## 2. Explaining vs non-Explaining (the q6 pairing)\n")
    L.append("| dimension | Explaining mean (n=%d) | non-Expl mean (n=%d) | U | p (two-sided) | Holm p | Cliff's delta |"
             % (labels.count("Explaining"), len(ids) - labels.count("Explaining")))
    L.append("|---|---|---|---|---|---|---|")
    ie = [i for i in range(len(ids)) if labels[i] == "Explaining"]
    io = [i for i in range(len(ids)) if labels[i] != "Explaining"]
    raw2, rows2 = [], []
    for j, fn in enumerate(FEATNAMES):
        a, b = G[ie, j], G[io, j]
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        raw2.append(p); rows2.append((fn, a.mean(), b.mean(), U, p, cliffs_delta(a, b)))
    adj2 = holm(np.array(raw2))
    for (fn, am, bm, U, p, d), a_ in zip(rows2, adj2):
        star = "**" if a_ < 0.05 else ""
        L.append(f"| {fn} | {am:+.4f} | {bm:+.4f} | {U:.1f} | {p:.4f} | {star}{a_:.4f}{star} | {d:+.3f} |")

    L.append("\n### Bilateral aggregates (corrected separately, 4 tests)\n")
    L.append("| aggregate | Explaining | non-Expl | U | p | Holm p | Cliff's delta |")
    L.append("|---|---|---|---|---|---|---|")
    raw3, rows3 = [], []
    for nm, v in AGG.items():
        a, b = v[ie], v[io]
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        raw3.append(p); rows3.append((nm, a.mean(), b.mean(), U, p, cliffs_delta(a, b)))
    adj3 = holm(np.array(raw3))
    for (nm, am, bm, U, p, d), a_ in zip(rows3, adj3):
        star = "**" if a_ < 0.05 else ""
        L.append(f"| {nm} | {am:+.4f} | {bm:+.4f} | {U:.1f} | {p:.4f} | {star}{a_:.4f}{star} | {d:+.3f} |")

    # ── 3. per-language check on the bilateral aggregates ───────────────────
    L.append("\n## 3. Per-language check (is any effect carried by one language?)\n")
    L.append("| aggregate | language | Explaining mean | non-Expl mean | n(E)/n(other) | difference |")
    L.append("|---|---|---|---|---|---|")
    for nm, v in AGG.items():
        for lg in ["Amis", "Paiwan", "Truku"]:
            e = [i for i in range(len(ids)) if langs[i] == lg and labels[i] == "Explaining"]
            o = [i for i in range(len(ids)) if langs[i] == lg and labels[i] != "Explaining"]
            if not e or not o: continue
            L.append(f"| {nm} | {lg} | {v[e].mean():+.4f} | {v[o].mean():+.4f} | "
                     f"{len(e)}/{len(o)} | {v[e].mean()-v[o].mean():+.4f} |")

    # ── 4. coverage as a control ────────────────────────────────────────────
    covm = cov.mean(1)
    U, p = mannwhitneyu(covm[ie], covm[io], alternative="two-sided")
    L.append(f"\n## 4. Control: detection coverage by group\n")
    L.append(f"\nMean hand-detection coverage: Explaining {covm[ie].mean():.3f} vs "
             f"non-Explaining {covm[io].mean():.3f}; Mann-Whitney U = {U:.1f}, p = {p:.4f}, "
             f"Cliff's delta = {cliffs_delta(covm[ie], covm[io]):+.3f}.")
    L.append("\nIf coverage itself separates the groups, then even the conditional features "
             "are sampled from systematically different portions of each clip, and any "
             "difference found above inherits that selection.")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
