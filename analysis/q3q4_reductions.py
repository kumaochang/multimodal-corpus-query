"""
Q3 (Analysis A consolidation + Analysis B reductions) + Q4 (composition-robust
variant) + Amis cross-lingual per-query breakdown.

Reuses:
  - _pose_features(), analyze_clip(), leave_one_out_baseline() from q1q2_composition
    (which itself carries the VERBATIM 18-dim extractor from pilot/experiment.py)
  - retrieval_eval / bootstrap_ci / cross_lingual_p1 copied VERBATIM from
    pilot/experiment.py (below) so all numbers are comparable to the paper.

18-dim layout (index -> meaning):
  0  L_body_mean_x   1  L_body_mean_y   2  L_body_std_x   3  L_body_std_y
  4  L_lh_ratio      5  L_rh_ratio      6  L_face_std_x   7  L_face_std_y
  8  R_body_mean_x   9  R_body_mean_y  10  R_body_std_x  11  R_body_std_y
 12  R_lh_ratio     13  R_rh_ratio     14  R_face_std_x  15  R_face_std_y
 16  G_body_energy  17  G_hand_asym
  (left slot = questioner, right slot = respondent; see Q1)

Run:  python analysis/q3q4_reductions.py
Outputs: analysis/q3q4_results.md (+ stdout)
"""
import sys, json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import bootstrap as scipy_bootstrap, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1q2_composition import _pose_features, analyze_clip, leave_one_out_baseline

ROOT      = Path(__file__).resolve().parent.parent
POSE_DIR  = ROOT / "data" / "pose"
LABELFILE = ROOT / "pilot" / "labels.json"
OUT_MD    = ROOT / "analysis" / "q3q4_results.md"
LANGUAGES = ["Amis", "Paiwan", "Truku"]


# ── VERBATIM from pilot/experiment.py ───────────────────────────────────────
def retrieval_eval(sim_matrix, labels, k_list=(1, 3)):
    n = len(labels)
    hit_arrays = {f"P@{k}": [] for k in k_list}
    aps = []
    for i in range(n):
        sims = sim_matrix[i].copy(); sims[i] = -1
        ranked = np.argsort(sims)[::-1]
        q_label = labels[i]
        for k in k_list:
            hits = sum(1 for j in ranked[:k] if labels[j] == q_label)
            hit_arrays[f"P@{k}"].append(hits / k)
        rel = [1 if labels[j] == q_label else 0 for j in ranked]
        ap, n_rel = 0.0, 0
        for rank, r in enumerate(rel, 1):
            if r: n_rel += 1; ap += n_rel / rank
        aps.append(ap / sum(rel) if sum(rel) > 0 else 0.0)
    hit_arrays["MAP"] = aps
    return {k: np.array(v) for k, v in hit_arrays.items()}

def bootstrap_ci(hit_array, n_resamples=2000):
    arr = np.array(hit_array)
    res = scipy_bootstrap((arr,), np.mean, n_resamples=n_resamples,
                          confidence_level=0.95, method="percentile",
                          random_state=42)
    return float(arr.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)

def cross_lingual_p1(embs, labels, segs):
    sim = cosine_similarity(embs)
    per_lang = {}
    for q_lang in LANGUAGES:
        q_idx = [i for i, s in enumerate(segs) if s["lang"] == q_lang]
        r_idx = [i for i, s in enumerate(segs) if s["lang"] != q_lang]
        hits = []
        for qi in q_idx:
            sims_row = {ri: sim[qi, ri] for ri in r_idx}
            top = sorted(sims_row, key=sims_row.get, reverse=True)[0]
            hits.append(1 if labels[top] == labels[qi] else 0)
        per_lang[q_lang] = float(np.mean(hits))
    return per_lang


# ── purity helper ───────────────────────────────────────────────────────────
def nn_purities(embs_norm, labels, langs, comp_group):
    sim = cosine_similarity(embs_norm)
    np.fill_diagonal(sim, -1.0)
    nn = sim.argmax(1)
    return {
        "label": float(np.mean([labels[i] == labels[nn[i]] for i in range(len(labels))])),
        "language": float(np.mean([langs[i] == langs[nn[i]] for i in range(len(labels))])),
        "composition": float(np.mean([comp_group[i] == comp_group[nn[i]] for i in range(len(labels))])),
    }

def battery(K_sub, labels, langs, segs, comp_group):
    """Full report battery for a (raw) sub-representation."""
    Kn = normalize(K_sub)
    sim = cosine_similarity(Kn)
    res = retrieval_eval(sim, labels)
    return {
        "P@1": bootstrap_ci(res["P@1"]),
        "P@3": bootstrap_ci(res["P@3"]),
        "MAP": bootstrap_ci(res["MAP"]),
        "cross_lingual": cross_lingual_p1(Kn, labels, segs),
        "purities": nn_purities(Kn, labels, langs, comp_group),
        "_Kn": Kn,
    }

def fmt(t): return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"


def main():
    files  = sorted(POSE_DIR.glob("*.pose.json"))
    ids    = [f.name.replace(".pose.json", "") for f in files]
    langs  = [i.split("-")[1] for i in ids]
    lmap   = json.load(open(LABELFILE, encoding="utf-8"))
    labels = [lmap[i] for i in ids]
    segs   = [{"id": i, "lang": l} for i, l in zip(ids, langs)]

    K_rows, comp_rows = [], []
    for f in files:
        frames = json.load(open(f))
        K_rows.append(_pose_features(frames))
        comp_rows.append(analyze_clip(frames))
    K = np.array(K_rows)  # 30 x 18 raw

    # composition tertile groups (same rule as q1q2)
    ppf = np.array([d["ppf_mean"] for d in comp_rows])
    order = np.argsort(ppf)
    comp_group = np.empty(len(ids), dtype=int)
    for rank, idx in enumerate(order):
        comp_group[idx] = 0 if rank < 10 else (1 if rank < 20 else 2)
    comp_z = np.array([[d["ppf_mean"], d["ppf_std"], d["frac_1p"], d["frac_2p"], d["frac_3plus"],
                        d["left_fill"], d["right_fill"],
                        np.nan_to_num(d["left_instab"]), np.nan_to_num(d["right_instab"])] for d in comp_rows])
    comp_z = (comp_z - comp_z.mean(0)) / (comp_z.std(0) + 1e-9)
    simC = cosine_similarity(comp_z)

    base_lab  = leave_one_out_baseline([14, 8, 5, 3])
    base_lang = 9 / 29
    base_comp = leave_one_out_baseline([int((comp_group == g).sum()) for g in (0, 1, 2)])

    # ── representations ──────────────────────────────────────────────────────
    reps = {}
    reps["18d_baseline"]     = K                                   # full
    reps["10d_respondent"]   = K[:, list(range(8, 16)) + [16, 17]] # role-based: right slot + global
    # activity-based: per clip pick slot with higher mean hand-presence
    act_rows = []
    act_choice = []
    for i in range(len(ids)):
        lh = (K[i, 4] + K[i, 5]) / 2.0
        rh = (K[i, 12] + K[i, 13]) / 2.0
        if lh >= rh:
            block = K[i, 0:8]; act_choice.append("L(questioner)")
        else:
            block = K[i, 8:16]; act_choice.append("R(respondent)")
        act_rows.append(np.concatenate([block, K[i, 16:18]]))
    reps["10d_activity"]     = np.array(act_rows)                  # activity-based + global
    reps["14d_comprobust"]   = np.delete(K, [0, 1, 8, 9], axis=1)  # Q4: drop L/R body_mean_x/y

    results = {name: battery(rep, labels, langs, segs, comp_group) for name, rep in reps.items()}

    # composition correlation (Pearson) for baseline + Q4 variant, with lang split
    iu = np.triu_indices(len(ids), k=1)
    same_lang = np.array([langs[a] == langs[b] for a, b in zip(*iu)])
    def corr_report(Kn):
        sim = cosine_similarity(Kn)
        kv, cv = sim[iu], simC[iu]
        out = {}
        out["all"]   = pearsonr(kv, cv)
        out["same"]  = pearsonr(kv[same_lang], cv[same_lang])
        out["cross"] = pearsonr(kv[~same_lang], cv[~same_lang])
        return out
    corr_base = corr_report(results["18d_baseline"]["_Kn"])
    corr_q4   = corr_report(results["14d_comprobust"]["_Kn"])

    # Amis cross-lingual per-query breakdown (18-dim baseline)
    Kn = results["18d_baseline"]["_Kn"]
    sim = cosine_similarity(Kn)
    amis_rows = []
    for qi in [i for i in range(len(ids)) if langs[i] == "Amis"]:
        r_idx = [j for j in range(len(ids)) if langs[j] != "Amis"]
        top = max(r_idx, key=lambda j: sim[qi, j])
        amis_rows.append((ids[qi], labels[qi], ids[top], labels[top], labels[top] == labels[qi]))

    # ── write markdown ───────────────────────────────────────────────────────
    L = []
    L.append("# Q3 (Analyses A & B) + Q4 + Amis per-query breakdown\n")
    L.append(f"Baselines: label LOO={base_lab:.3f} | language={base_lang:.3f} | composition LOO={base_comp:.3f}\n")

    L.append("\n## Analysis A (18-dim baseline) — consolidated\n")
    r = results["18d_baseline"]
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| P@1 | {fmt(r['P@1'])} |")
    L.append(f"| P@3 | {fmt(r['P@3'])} |")
    L.append(f"| MAP | {fmt(r['MAP'])} |")
    L.append(f"| cross-lingual P@1 | Amis={r['cross_lingual']['Amis']:.3f}  Paiwan={r['cross_lingual']['Paiwan']:.3f}  Truku={r['cross_lingual']['Truku']:.3f} |")
    L.append(f"| NN label purity | {r['purities']['label']:.3f} (base {base_lab:.3f}) |")
    L.append(f"| NN language purity | {r['purities']['language']:.3f} (base {base_lang:.3f}) |")
    L.append(f"| NN composition purity | {r['purities']['composition']:.3f} (base {base_comp:.3f}) |")
    L.append("\n**Kinematic cosine ~ composition cosine (Pearson):**")
    L.append(f"\n- all 435 pairs: r={corr_base['all'][0]:.3f} (p={corr_base['all'][1]:.1e})")
    L.append(f"\n- same-language pairs (n={int(same_lang.sum())}): r={corr_base['same'][0]:.3f} (p={corr_base['same'][1]:.1e})")
    L.append(f"\n- cross-language pairs (n={int((~same_lang).sum())}): r={corr_base['cross'][0]:.3f} (p={corr_base['cross'][1]:.1e})")

    L.append("\n## Analysis B — 10-dim reductions & Q4 14-dim variant (full battery)\n")
    L.append("| representation | dims | P@1 [95% CI] | P@3 [95% CI] | MAP [95% CI] | CL Amis | CL Paiwan | CL Truku | lab pur | lang pur | comp pur |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    dimcount = {"18d_baseline": 18, "10d_respondent": 10, "10d_activity": 10, "14d_comprobust": 14}
    for name in ["18d_baseline", "10d_respondent", "10d_activity", "14d_comprobust"]:
        r = results[name]; p = r["purities"]; cl = r["cross_lingual"]
        L.append(f"| {name} | {dimcount[name]} | {fmt(r['P@1'])} | {fmt(r['P@3'])} | {fmt(r['MAP'])} | "
                 f"**{cl['Amis']:.3f}** | {cl['Paiwan']:.3f} | {cl['Truku']:.3f} | "
                 f"{p['label']:.3f} | {p['language']:.3f} | {p['composition']:.3f} |")
    L.append(f"\nActivity-based slot choice per clip: {Counter(act_choice)}")

    L.append("\n## Q4 composition-robust variant (14-dim, dropped dims 0,1,8,9 = L/R body_mean_x/y)\n")
    L.append("**Kinematic cosine ~ composition cosine (Pearson), 14-dim variant:**")
    L.append(f"\n- all: r={corr_q4['all'][0]:.3f} (p={corr_q4['all'][1]:.1e})  (baseline was r={corr_base['all'][0]:.3f})")
    L.append(f"\n- same-language: r={corr_q4['same'][0]:.3f} | cross-language: r={corr_q4['cross'][0]:.3f}")

    L.append("\n## Amis cross-lingual per-query breakdown (18-dim baseline)\n")
    L.append("| # | query clip | query label | retrieved clip | retrieved label | hit? |")
    L.append("|---|---|---|---|---|---|")
    for k, (q, ql, rt, rl, hit) in enumerate(amis_rows, 1):
        L.append(f"| {k} | {q} | {ql} | {rt} | {rl} | {'HIT' if hit else 'miss'} |")
    nhit = sum(1 for *_, h in amis_rows if h)
    hit_labels = Counter(ql for _, ql, _, _, h in amis_rows if h)
    L.append(f"\n**Amis cross-lingual P@1 = {nhit}/10 = {nhit/10:.3f}**; hit query-labels: {dict(hit_labels)}")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    # ── stdout summary ───────────────────────────────────────────────────────
    print("== Analysis B / Q4 battery ==")
    print(f"{'rep':<16}{'P@1':>18}{'CL_Amis':>9}{'labP':>7}{'langP':>7}{'compP':>7}")
    for name in ["18d_baseline", "10d_respondent", "10d_activity", "14d_comprobust"]:
        r = results[name]
        print(f"{name:<16}{fmt(r['P@1']):>18}{r['cross_lingual']['Amis']:>9.3f}"
              f"{r['purities']['label']:>7.3f}{r['purities']['language']:>7.3f}{r['purities']['composition']:>7.3f}")
    print(f"\ncomposition corr  baseline all r={corr_base['all'][0]:.3f} (cross {corr_base['cross'][0]:.3f}) | "
          f"Q4 14d all r={corr_q4['all'][0]:.3f} (cross {corr_q4['cross'][0]:.3f})")
    print(f"activity slot choice: {Counter(act_choice)}")
    print(f"\nAmis per-query (18d): {nhit}/10 hits; hit labels {dict(hit_labels)}")
    for k, (q, ql, rt, rl, hit) in enumerate(amis_rows, 1):
        print(f"  {k:2d} {'HIT ' if hit else 'miss'} {q:<24} [{ql:<11}] -> {rt:<24} [{rl}]")
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
