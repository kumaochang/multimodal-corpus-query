"""
Q11: how many queries does the main corpus need to detect the de-confounded effect?

Q10 removed the framing confound by construction (presence vs persons-per-frame
falls from r = -0.86 to r = -0.10, and the group difference in hand presence from
p = 0.0005 to p = 0.92). The effect that survives is a lift of +0.150 over
label-specific chance (8/14 = 0.571 vs 0.421), which at n = 14 queries gives
p = 0.19. The question is not whether that is significant here. It is whether the
216-clip corpus can decide it.

Power is computed for a one-sided binomial test of rate vs label-specific chance,
then inflated by a design effect for clustering by broadcast day, since clips cut
from one broadcast are not independent.

Run:  python analysis/q11_power.py
Outputs: analysis/q11_results.md (+ stdout)
"""
import numpy as np
from pathlib import Path
from scipy.stats import binom

OUT_MD = Path(__file__).resolve().parent / "q11_results.md"

CHANCE   = 0.421          # weighted label-specific chance, pooled Explaining pairing
OBS_RATE = 0.571          # de-confounded observed rate (q10, 18d_tightframes)
ALPHA    = 0.05
POWER    = 0.80


def power_at(n, p1, p0=CHANCE, alpha=ALPHA):
    """Exact one-sided binomial power at sample size n."""
    crit = binom.ppf(1 - alpha, n, p0)
    while crit <= n and binom.sf(crit - 1, n, p0) > alpha:
        crit += 1
    return float(binom.sf(crit - 1, n, p1))


def n_for(p1, p0=CHANCE, alpha=ALPHA, power=POWER, nmax=2000):
    for n in range(10, nmax):
        if power_at(n, p1, p0, alpha) >= power:
            return n
    return None


def main():
    L = ["# Q11: power for the de-confounded visual-channel effect\n"]
    L.append(f"- Null (label-specific chance): {CHANCE:.3f}\n")
    L.append(f"- One-sided alpha = {ALPHA}, target power = {POWER:.0%}\n")
    L.append(f"- Pilot, de-confounded (q10 18d_tightframes): 8/14 = {OBS_RATE:.3f}, "
             f"lift = {OBS_RATE-CHANCE:+.3f}, p = 0.189\n")

    L.append("\n## 1. Independent queries needed, by assumed true rate\n")
    L.append("| assumed true rate | lift over chance | n queries for 80% power | n for 90% |")
    L.append("|---|---|---|---|")
    rates = [0.52, 0.55, 0.571, 0.60, 0.65, 0.70]
    for p1 in rates:
        n80 = n_for(p1); n90 = n_for(p1, power=0.90)
        mark = " (pilot point estimate)" if abs(p1 - OBS_RATE) < 1e-9 else ""
        L.append(f"| {p1:.3f}{mark} | {p1-CHANCE:+.3f} | {n80} | {n90} |")

    L.append("\n## 2. After inflation for clustering by broadcast day\n")
    L.append("Clips cut from one broadcast share camera crew, studio, anchor and editing "
             "decisions, so the effective sample is smaller than the nominal one. "
             "Design effect = 1 + (m - 1) x ICC, with m = clips contributing per day.\n")
    L.append("| assumed true rate | m clips/day | ICC 0.05 | ICC 0.10 | ICC 0.20 | ICC 0.30 |")
    L.append("|---|---|---|---|---|---|")
    for p1 in [0.55, 0.571, 0.60, 0.65]:
        n0 = n_for(p1)
        for m in (3, 6):
            cells = []
            for icc in (0.05, 0.10, 0.20, 0.30):
                deff = 1 + (m - 1) * icc
                cells.append(f"{int(np.ceil(n0 * deff))}")
            L.append(f"| {p1:.3f} | {m} | " + " | ".join(cells) + " |")

    L.append("\n## 3. What the 216-clip corpus actually supplies\n")
    L.append("216 clips = 6 languages x 36. Every clip is a query against the 180 clips of "
             "the other five languages. The number of queries in the target pairing is "
             "216 x (share of clips falling in that category), which is not known until the "
             "clips are labelled.\n")
    L.append("| category share of corpus | queries available | covers 80% power at true rate... |")
    L.append("|---|---|---|")
    for share in (0.20, 0.30, 0.40, 0.50):
        nq = int(round(216 * share))
        ok = [f"{p1:.2f}" for p1 in rates if n_for(p1) is not None and n_for(p1) <= nq]
        L.append(f"| {share:.0%} | {nq} | {', '.join(ok) if ok else 'none of the rates tested'} |")

    L.append("\n### With day clustering at m = 3, ICC = 0.10 (design effect 1.20)\n")
    L.append("| category share | nominal queries | effective queries | covers 80% power at true rate... |")
    L.append("|---|---|---|---|")
    for share in (0.20, 0.30, 0.40, 0.50):
        nq = int(round(216 * share)); eff = int(nq / 1.20)
        ok = [f"{p1:.2f}" for p1 in rates if n_for(p1) is not None and n_for(p1) <= eff]
        L.append(f"| {share:.0%} | {nq} | {eff} | {', '.join(ok) if ok else 'none of the rates tested'} |")

    L.append("\n## 4. Reading\n")
    L.append(f"\nAt the pilot's de-confounded point estimate ({OBS_RATE:.3f}), the test needs "
             f"{n_for(OBS_RATE)} independent queries for 80% power, or "
             f"{int(np.ceil(n_for(OBS_RATE) * 1.2))} after a design effect of 1.20. The 216-clip "
             f"corpus supplies that only if the target category covers roughly 40% of the "
             f"corpus or more. If the category runs at 20-30%, the clip-level design is "
             f"underpowered for this effect and a null result would be uninformative.")
    L.append("\nThe point estimate is itself from 14 queries and its confidence interval is "
             "wide, so these numbers set the order of magnitude, not a guarantee. The design "
             "consequence does not depend on the exact figure: a clip-level visual-channel "
             "test at this corpus size is marginal, and the way to stop it being marginal is "
             "to raise the number of independent units rather than the number of clips.")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
