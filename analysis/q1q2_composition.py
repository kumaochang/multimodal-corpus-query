"""
Q1 (slot assignment) + Q2 (shot-composition confound) diagnostics.

Investigates the reviewer-adjacent concern raised by the author: are the two
kinematic "left"/"right" slots filled by tracked speakers, or by arbitrary
detected persons (making the 18-dim vector partly a measure of camerawork /
crowd density rather than gesture)?

- Reuses the VERBATIM _pose_features() from pilot/experiment.py (copied below,
  unchanged) so the 18-dim kinematic representation is identical to the paper.
- Reuses the paper's retrieval/similarity conventions (L2-normalize rows, cosine,
  self excluded) for the composition nearest-neighbour purity.

Run:  python analysis/q1q2_composition.py
Outputs: analysis/q1q2_results.md  (+ stdout)
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr

ROOT      = Path(__file__).resolve().parent.parent
POSE_DIR  = ROOT / "data" / "pose"
LABELFILE = ROOT / "pilot" / "labels.json"
OUT_MD    = ROOT / "analysis" / "q1q2_results.md"

JUMP_PX = 150.0  # frame-to-frame body-centroid jump (px) counted as identity instability


# ── VERBATIM copy of _pose_features from pilot/experiment.py (do not modify) ────
def _pose_features(frames):
    total = len(frames)
    feats_per_side = {}
    for side in ("left", "right"):
        body_cx, body_cy, face_cx, face_cy = [], [], [], []
        lh_present = rh_present = 0
        for fr in frames:
            person = fr[side]
            if person is None: continue
            pts = person.get("pose") or []
            if pts:
                xs = [p[0] for p in pts if len(p) >= 2]
                ys = [p[1] for p in pts if len(p) >= 2]
                if xs:
                    body_cx.append(np.mean(xs)); body_cy.append(np.mean(ys))
            if person.get("left_hand"):  lh_present += 1
            if person.get("right_hand"): rh_present += 1
            face = person.get("face") or {}
            pts_f = []
            for part in face.values():
                if isinstance(part, list): pts_f.extend(part)
            if pts_f:
                xs_f = [p[0] for p in pts_f if len(p) >= 2]
                ys_f = [p[1] for p in pts_f if len(p) >= 2]
                if xs_f:
                    face_cx.append(np.mean(xs_f)); face_cy.append(np.mean(ys_f))
        feats_per_side[side] = {
            "body_mean": [np.mean(body_cx) if body_cx else 0.5,
                          np.mean(body_cy) if body_cy else 0.5],
            "body_std":  [np.std(body_cx) if len(body_cx) > 1 else 0.0,
                          np.std(body_cy) if len(body_cy) > 1 else 0.0],
            "lh_ratio":  lh_present / total,
            "rh_ratio":  rh_present / total,
            "face_std":  [np.std(face_cx) if len(face_cx) > 1 else 0.0,
                          np.std(face_cy) if len(face_cy) > 1 else 0.0],
        }
    vec = []
    for side in ("left", "right"):
        f = feats_per_side[side]
        vec.extend(f["body_mean"]); vec.extend(f["body_std"])
        vec.append(f["lh_ratio"]); vec.append(f["rh_ratio"]); vec.extend(f["face_std"])
    mean_body_energy = np.mean([feats_per_side["left"]["body_std"],
                                feats_per_side["right"]["body_std"]])
    lh_asym = abs(feats_per_side["left"]["lh_ratio"] - feats_per_side["right"]["lh_ratio"])
    rh_asym = abs(feats_per_side["left"]["rh_ratio"] - feats_per_side["right"]["rh_ratio"])
    vec.extend([mean_body_energy, lh_asym + rh_asym])
    return np.array(vec)


# ── helpers ─────────────────────────────────────────────────────────────────
def slot_cx(slot):
    if slot is None: return None
    b = slot.get("bbox")
    if isinstance(b, dict): return (b["x1"] + b["x2"]) / 2.0
    return None

def slot_role(slot, persons):
    """Match a slot back to its person entry (by exact bbox) to recover its role."""
    if slot is None: return "NONE"
    b = slot.get("bbox")
    if not isinstance(b, dict): return "no-dict-bbox"
    for p in persons:
        if all(abs(b.get(k, -9e9) - p.get(k, 9e9)) < 1e-6 for k in ("x1", "y1", "x2", "y2")):
            return p.get("role", "?")
    return "unmatched"


def analyze_clip(frames):
    n = len(frames)
    lrole, rrole = Counter(), Counter()
    lx_lt_rx = both = 0
    ppf = []                     # persons per frame
    n1 = n2 = n3plus = 0
    ljump = rjump = 0
    lpairs = rpairs = 0
    prev_lcx = prev_rcx = None
    lfill = rfill = 0
    for fr in frames:
        persons = fr.get("persons") or []
        k = len(persons); ppf.append(k)
        if k <= 1: n1 += 1
        elif k == 2: n2 += 1
        else: n3plus += 1
        L, R = fr["left"], fr["right"]
        lrole[slot_role(L, persons)] += 1
        rrole[slot_role(R, persons)] += 1
        lcx, rcx = slot_cx(L), slot_cx(R)
        if L is not None: lfill += 1
        if R is not None: rfill += 1
        if lcx is not None and rcx is not None:
            both += 1
            if lcx < rcx: lx_lt_rx += 1
        if lcx is not None and prev_lcx is not None:
            lpairs += 1
            if abs(lcx - prev_lcx) > JUMP_PX: ljump += 1
        if rcx is not None and prev_rcx is not None:
            rpairs += 1
            if abs(rcx - prev_rcx) > JUMP_PX: rjump += 1
        prev_lcx = lcx if lcx is not None else prev_lcx
        prev_rcx = rcx if rcx is not None else prev_rcx
    ppf = np.array(ppf)
    return {
        "n": n,
        "left_roles": dict(lrole), "right_roles": dict(rrole),
        "left_cx_lt_right_frac": (lx_lt_rx / both) if both else float("nan"),
        "left_fill": lfill / n, "right_fill": rfill / n,
        "ppf_mean": float(ppf.mean()), "ppf_std": float(ppf.std()),
        "frac_1p": n1 / n, "frac_2p": n2 / n, "frac_3plus": n3plus / n,
        "left_instab": (ljump / lpairs) if lpairs else float("nan"),
        "right_instab": (rjump / rpairs) if rpairs else float("nan"),
    }


def leave_one_out_baseline(group_sizes):
    N = sum(group_sizes)
    return sum(g * (g - 1) for g in group_sizes) / (N * (N - 1))


def main():
    files = sorted(POSE_DIR.glob("*.pose.json"))
    ids   = [f.name.replace(".pose.json", "") for f in files]
    langs = [i.split("-")[1] for i in ids]
    labels = [json.load(open(LABELFILE, encoding="utf-8"))[i] for i in ids]

    per = []
    K_rows = []
    for f in files:
        frames = json.load(open(f))
        per.append(analyze_clip(frames))
        K_rows.append(_pose_features(frames))
    K = np.array(K_rows)                 # 30 x 18, raw
    Kn = normalize(K)                    # paper convention
    simK = cosine_similarity(Kn)

    L = []
    L.append("# Q1 + Q2 diagnostics: slot assignment & shot-composition confound\n")
    L.append(f"- Clips: {len(ids)}  | JUMP_PX threshold = {JUMP_PX:.0f}px\n")

    # ── Q1: what occupies the slots ──────────────────────────────────────────
    L.append("\n## Q1. Slot occupancy: role, ordering, stability\n")
    L.append("| clip | left roles (filled) | right roles (filled) | L.cx<R.cx | L fill | R fill | L instab | R instab |")
    L.append("|---|---|---|---|---|---|---|---|")
    all_left_roles, all_right_roles = Counter(), Counter()
    for i, d in enumerate(per):
        lr = {k: v for k, v in d["left_roles"].items() if k != "NONE"}
        rr = {k: v for k, v in d["right_roles"].items() if k != "NONE"}
        for k, v in lr.items():  all_left_roles[k]  += v
        for k, v in rr.items():  all_right_roles[k] += v
        L.append(f"| {ids[i]} | {lr} | {rr} | {d['left_cx_lt_right_frac']:.2f} | "
                 f"{d['left_fill']:.2f} | {d['right_fill']:.2f} | {d['left_instab']:.3f} | {d['right_instab']:.3f} |")
    L.append(f"\n**Aggregate left-slot roles (filled frames):** {dict(all_left_roles)}")
    L.append(f"\n**Aggregate right-slot roles (filled frames):** {dict(all_right_roles)}")
    mean_lt = np.nanmean([d["left_cx_lt_right_frac"] for d in per])
    L.append(f"\n**Mean fraction of two-filled frames with left.cx < right.cx:** {mean_lt:.3f}")
    L.append(f"\n**Mean left instability:** {np.nanmean([d['left_instab'] for d in per]):.3f}  "
             f"| **Mean right instability:** {np.nanmean([d['right_instab'] for d in per]):.3f}")

    # ── Q2: composition structure ────────────────────────────────────────────
    L.append("\n## Q2. Shot-composition structure per clip\n")
    L.append("| clip | lang | label | ppf_mean | ppf_std | %1p | %2p | %3+p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, d in enumerate(per):
        L.append(f"| {ids[i]} | {langs[i]} | {labels[i]} | {d['ppf_mean']:.2f} | {d['ppf_std']:.2f} | "
                 f"{d['frac_1p']*100:.0f} | {d['frac_2p']*100:.0f} | {d['frac_3plus']*100:.0f} |")

    # composition profile vector (z-scored) for pairwise similarity
    comp = np.array([[d["ppf_mean"], d["ppf_std"], d["frac_1p"], d["frac_2p"], d["frac_3plus"],
                      d["left_fill"], d["right_fill"], d["left_instab"], d["right_instab"]]
                     for d in per])
    comp = np.nan_to_num(comp)
    comp_z = (comp - comp.mean(0)) / (comp.std(0) + 1e-9)
    simC = cosine_similarity(comp_z)

    # coarse 3-level grouping by mean persons-per-frame (tertiles)
    ppf_mean = np.array([d["ppf_mean"] for d in per])
    order = np.argsort(ppf_mean)
    group = np.empty(len(ids), dtype=int)
    for rank, idx in enumerate(order):
        group[idx] = 0 if rank < 10 else (1 if rank < 20 else 2)  # low / mid / high, 10 each
    gsizes = [int((group == g).sum()) for g in (0, 1, 2)]
    thr_lo = ppf_mean[order[9]]; thr_hi = ppf_mean[order[19]]
    L.append(f"\n**Coarse composition groups (tertiles of mean persons/frame):** "
             f"low<={thr_lo:.2f} (n={gsizes[0]}), mid<={thr_hi:.2f} (n={gsizes[1]}), high (n={gsizes[2]})")

    # NN composition purity for 18-dim kinematic
    sim = simK.copy(); np.fill_diagonal(sim, -1.0)
    nn = sim.argmax(1)
    comp_pur = np.mean([group[i] == group[nn[i]] for i in range(len(ids))])
    lab_pur  = np.mean([labels[i] == labels[nn[i]] for i in range(len(ids))])
    lang_pur = np.mean([langs[i]  == langs[nn[i]]  for i in range(len(ids))])
    base_comp = leave_one_out_baseline(gsizes)
    base_lang = 9 / 29
    base_lab  = leave_one_out_baseline([14, 8, 5, 3])

    L.append("\n## Q2 results: does 18-dim kinematic NN track composition?\n")
    L.append("| NN purity (18-dim kinematic) | value | LOO random baseline |")
    L.append("|---|---|---|")
    L.append(f"| shot-composition group | {comp_pur:.3f} | {base_comp:.3f} |")
    L.append(f"| pragmatic label | {lab_pur:.3f} | {base_lab:.3f} |")
    L.append(f"| language | {lang_pur:.3f} | {base_lang:.3f} |")

    # correlation kinematic-sim vs composition-sim across 435 pairs
    iu = np.triu_indices(len(ids), k=1)
    kv = simK[iu]; cv = simC[iu]
    pr, pp = pearsonr(kv, cv); sr, sp = spearmanr(kv, cv)
    L.append(f"\n**Kinematic cosine vs composition-profile cosine (n={len(kv)} pairs):**")
    L.append(f"\n- Pearson r = {pr:.3f} (p={pp:.1e})")
    L.append(f"\n- Spearman rho = {sr:.3f} (p={sp:.1e})")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    # concise stdout summary
    print("Aggregate LEFT-slot roles :", dict(all_left_roles))
    print("Aggregate RIGHT-slot roles:", dict(all_right_roles))
    print(f"left.cx<right.cx mean frac : {mean_lt:.3f}")
    print(f"instability L/R (mean)     : {np.nanmean([d['left_instab'] for d in per]):.3f} / "
          f"{np.nanmean([d['right_instab'] for d in per]):.3f}")
    print(f"ppf_mean range             : {ppf_mean.min():.2f} .. {ppf_mean.max():.2f}")
    print(f"composition groups (n)     : {gsizes}  thr_lo={thr_lo:.2f} thr_hi={thr_hi:.2f}")
    print(f"NN comp purity  {comp_pur:.3f}  (base {base_comp:.3f})")
    print(f"NN label purity {lab_pur:.3f}  (base {base_lab:.3f})")
    print(f"NN lang  purity {lang_pur:.3f}  (base {base_lang:.3f})")
    print(f"corr kinematic~composition : Pearson {pr:.3f} (p={pp:.1e}) | Spearman {sr:.3f} (p={sp:.1e})")
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
