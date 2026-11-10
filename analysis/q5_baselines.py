"""
Q5: label-specific chance baselines for the cross-lingual protocol + significance.

Addresses the base-rate flaw in the pairing-specificity claim: cross-lingual P@1
must be judged against a *label-specific* baseline = (#same-label clips in the
other two languages) / (cross-lingual pool size), which varies by label and by
source language, NOT against the global full-pool random 0.303.

Reuses:
  - cached text/mfcc/prosody embeddings from step4 (analysis/_emb_cache.npz)
  - VERBATIM _pose_features + retrieval conventions from pilot/experiment.py

Run:  python analysis/q5_baselines.py   (run step4_supplementary.py first to build cache)
Outputs: analysis/q5_results.md (+ stdout)
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import fisher_exact

ROOT      = Path(__file__).resolve().parent.parent
POSE_DIR  = ROOT / "data" / "pose"
LABELFILE = ROOT / "pilot" / "labels.json"
CACHE     = ROOT / "analysis" / "_emb_cache.npz"
OUT_MD    = ROOT / "analysis" / "q5_results.md"
LANGUAGES = ["Amis", "Paiwan", "Truku"]


def _pose_features(frames):  # verbatim from pilot/experiment.py
    total = len(frames); fps = {}
    for side in ("left", "right"):
        bx, by, fx, fy = [], [], [], []; lh = rh = 0
        for fr in frames:
            p = fr[side]
            if p is None: continue
            pts = p.get("pose") or []
            if pts:
                xs = [q[0] for q in pts if len(q) >= 2]; ys = [q[1] for q in pts if len(q) >= 2]
                if xs: bx.append(np.mean(xs)); by.append(np.mean(ys))
            if p.get("left_hand"): lh += 1
            if p.get("right_hand"): rh += 1
            face = p.get("face") or {}; pf = []
            for part in face.values():
                if isinstance(part, list): pf.extend(part)
            if pf:
                xf = [q[0] for q in pf if len(q) >= 2]; yf = [q[1] for q in pf if len(q) >= 2]
                if xf: fx.append(np.mean(xf)); fy.append(np.mean(yf))
        fps[side] = {"body_mean": [np.mean(bx) if bx else 0.5, np.mean(by) if by else 0.5],
                     "body_std": [np.std(bx) if len(bx) > 1 else 0.0, np.std(by) if len(by) > 1 else 0.0],
                     "lh_ratio": lh/total, "rh_ratio": rh/total,
                     "face_std": [np.std(fx) if len(fx) > 1 else 0.0, np.std(fy) if len(fy) > 1 else 0.0]}
    vec = []
    for side in ("left", "right"):
        f = fps[side]; vec += f["body_mean"] + f["body_std"] + [f["lh_ratio"], f["rh_ratio"]] + f["face_std"]
    mbe = np.mean([fps["left"]["body_std"], fps["right"]["body_std"]])
    lh_a = abs(fps["left"]["lh_ratio"] - fps["right"]["lh_ratio"])
    rh_a = abs(fps["left"]["rh_ratio"] - fps["right"]["rh_ratio"])
    vec += [mbe, lh_a + rh_a]
    return np.array(vec)


def crossling_per_query(embs_norm, labels, langs):
    """Return dict src_lang -> list of (idx, label, retrieved_idx, retrieved_label, hit, baseline, n_same)."""
    sim = cosine_similarity(embs_norm)
    out = {}
    for ql in LANGUAGES:
        q_idx = [i for i in range(len(labels)) if langs[i] == ql]
        r_idx = [i for i in range(len(labels)) if langs[i] != ql]
        pool = len(r_idx)
        rows = []
        for qi in q_idx:
            top = max(r_idx, key=lambda j: sim[qi, j])
            n_same = sum(1 for j in r_idx if labels[j] == labels[qi])
            rows.append((qi, labels[qi], top, labels[top], int(labels[top] == labels[qi]),
                         n_same / pool, n_same))
        out[ql] = rows
    return out


def main():
    files = sorted(POSE_DIR.glob("*.pose.json"))
    ids = [f.name.replace(".pose.json", "") for f in files]
    langs = [i.split("-")[1] for i in ids]
    lmap = json.load(open(LABELFILE, encoding="utf-8"))
    labels = [lmap[i] for i in ids]

    K18 = np.array([_pose_features(json.load(open(f))) for f in files])
    K14 = np.delete(K18, [0, 1, 8, 9], axis=1)
    reps = {"kin18": normalize(K18), "kin14": normalize(K14)}

    L = ["# Q5: label-specific cross-lingual baselines & significance\n"]

    # ── 1. label-specific baseline table (per source lang x label) ───────────
    L.append("## 1. Per-query label-specific chance baseline (pool = 20 candidates)\n")
    L.append("baseline = (#same-label clips in the OTHER two languages) / 20\n")
    L.append("| source lang | query label | #queries | #same-label candidates | chance P@1 |")
    L.append("|---|---|---|---|---|")
    label_counts = defaultdict(Counter)  # lang -> Counter(label)
    for i in range(len(ids)):
        label_counts[langs[i]][labels[i]] += 1
    all_labels = ["Explaining", "Urging", "Celebrating", "Reporting"]
    for ql in LANGUAGES:
        others = [x for x in LANGUAGES if x != ql]
        for lab in all_labels:
            nq = label_counts[ql][lab]
            if nq == 0: continue
            n_same = sum(label_counts[o][lab] for o in others)
            L.append(f"| {ql} | {lab} | {nq} | {n_same} | {n_same/20:.3f} |")

    # ── 2. observed vs expected per label (Amis, both reps) ───────────────────
    L.append("\n## 2. Amis cross-lingual: observed vs expected hits per label\n")
    L.append("| representation | label | #queries | observed hits | expected hits | lift (obs−exp) |")
    L.append("|---|---|---|---|---|---|")
    amis_contingency = {}
    for rep in ["kin18", "kin14"]:
        rows = crossling_per_query(reps[rep], labels, langs)["Amis"]
        by_lab = defaultdict(list)
        for r in rows: by_lab[r[1]].append(r)
        cont = {}  # label -> (hits, misses)
        for lab in ["Explaining", "Celebrating"]:
            rs = by_lab.get(lab, [])
            obs = sum(r[4] for r in rs); exp = sum(r[5] for r in rs)
            cont[lab] = (obs, len(rs) - obs)
            L.append(f"| {rep} | {lab} | {len(rs)} | {obs} | {exp:.2f} | {obs-exp:+.2f} |")
        # also non-Explaining/Celebrating labels if any (Amis has none)
        amis_contingency[rep] = cont

    # ── 3. Fisher exact on hit/miss x Explaining/Celebrating (Amis) ──────────
    L.append("\n## 3. Fisher's exact test: hit/miss × Explaining/Celebrating (Amis cross-lingual)\n")
    L.append("| representation | table [[E_hit,E_miss],[C_hit,C_miss]] | odds ratio | p (two-tailed) | p (one-tailed, greater) |")
    L.append("|---|---|---|---|---|")
    for rep in ["kin18", "kin14"]:
        eh, em = amis_contingency[rep]["Explaining"]
        ch, cm = amis_contingency[rep]["Celebrating"]
        table = [[eh, em], [ch, cm]]
        odr, p_two = fisher_exact(table, alternative="two-sided")
        _, p_greater = fisher_exact(table, alternative="greater")
        L.append(f"| {rep} | [[{eh},{em}],[{ch},{cm}]] | {odr if np.isfinite(odr) else 'inf'} | {p_two:.4f} | {p_greater:.4f} |")

    # ── 4. baseline-adjusted aggregate per source lang x rep ─────────────────
    L.append("\n## 4. Baseline-adjusted aggregate cross-lingual P@1\n")
    L.append("| representation | source lang | overall P@1 | query-weighted mean baseline | lift |")
    L.append("|---|---|---|---|---|")
    for rep in ["kin18", "kin14"]:
        cl = crossling_per_query(reps[rep], labels, langs)
        for ql in LANGUAGES:
            rows = cl[ql]
            p1 = np.mean([r[4] for r in rows])
            mb = np.mean([r[5] for r in rows])
            L.append(f"| {rep} | {ql} | {p1:.3f} | {mb:.3f} | {p1-mb:+.3f} |")

    # ── 5. all-pairs paired bootstrap among 4 modalities (backs §5.2) ────────
    L.append("\n## 5. All-pairs paired bootstrap on per-query P@1 (full-pool single-modality)\n")
    z = np.load(CACHE, allow_pickle=True)
    assert list(z["ids"]) == ids, "cache id mismatch; rerun step4_supplementary.py"
    mods = {"text": normalize(z["T"]), "audio": normalize(z["A"]),
            "prosody": normalize(z["P"]), "kinematic": normalize(K18)}
    def per_query_p1(e):
        sim = cosine_similarity(e); n = len(labels); out = []
        for i in range(n):
            s = sim[i].copy(); s[i] = -1; j = int(np.argmax(s)); out.append(1.0 if labels[j] == labels[i] else 0.0)
        return np.array(out)
    p1 = {m: per_query_p1(e) for m, e in mods.items()}
    def paired_boot(a, b, n_res=2000, seed=42):
        d = a - b; rng = np.random.default_rng(seed); n = len(d)
        means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_res)])
        return d.mean(), min(1.0, 2*min((means <= 0).mean(), (means >= 0).mean()))
    names = list(mods)
    L.append("| pair | mean P@1 diff | two-sided p | sig |")
    L.append("|---|---|---|---|")
    anysig = False
    for a in range(len(names)):
        for b in range(a+1, len(names)):
            md, pv = paired_boot(p1[names[a]], p1[names[b]])
            anysig |= (pv < 0.05)
            L.append(f"| {names[a]} vs {names[b]} | {md:+.3f} | {pv:.3f} | {'*' if pv<0.05 else 'ns'} |")
    L.append(f"\n**Any modality pair separated at p<0.05: {anysig}**")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
