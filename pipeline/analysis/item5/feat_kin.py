"""Blocks K (kinematics) and F (framing) per judged utterance (Amendment 9 item 5 and item 6
points 1 and 4; Amendment 10 items 1 and 2).

K: the 14-dimensional composition-robust variant on tight frames: Study 1's 18-dimensional
   `_pose_features` (copied verbatim from pilot/experiment.py via analysis/q1q2_composition.py)
   computed on the utterance's tight pose records, minus the 4 position dimensions
   (analysis/q7_handpresence.py IDX_POSITION). Tight record (analysis/q10_rescue.py): at most 2
   detected persons and a tracked, non-fallback body box covering at least 12% of the frame.
   Minimum: at least 1 s of tight records, i.e. ceil(1000 / span) records (4 at 266.9 ms);
   below it the utterance is missing for K (NaN row, `k_missing` = 1).
F: the 8 framing measures of analysis/q10 (shot block of q7 `clip_measures`: bbox_h, bbox_area,
   face_extent, pose_extent, fallback_frac, ppf_mean, frac_1p, frac_3plus) over all pose records
   of the utterance; NaN where undefined (handled by the probe as missing).
A pose record belongs to the utterance if its start time (`offset`, ms) lies in [start, end).
Output: --out .npz with ids, K (n x 14), k_missing, F (n x 8), n_records, n_tight, span_ms.
Prints counts only.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_items, find_file  # noqa: E402

FRAME_W, FRAME_H = 1280, 720
FALLBACK = {(0, 0, 640, 720), (640, 0, 1280, 720)}
TIGHT_MAXP, TIGHT_MINAREA = 2, 0.12
IDX_POSITION = [0, 1, 8, 9]
F_NAMES = ["bbox_h", "bbox_area", "face_extent", "pose_extent", "fallback_frac",
           "ppf_mean", "frac_1p", "frac_3plus"]

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
# ─────────────────────────────────────────────────────────────────────────────

def is_tight(fr):
    if len(fr.get("persons") or []) > TIGHT_MAXP:
        return False
    for side in ("left", "right"):
        s = fr[side]
        if s is None: continue
        b = s.get("bbox")
        if not isinstance(b, dict): continue
        if (b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")) in FALLBACK: continue
        if ((b["y2"] - b["y1"]) * (b["x2"] - b["x1"])) / (FRAME_H * FRAME_W) >= TIGHT_MINAREA:
            return True
    return False

def framing(frames):
    """The 8 shot-block measures of q7 clip_measures, over the given records."""
    n = len(frames)
    if n == 0:
        return [float("nan")] * 8
    bbox_h, bbox_a, face_ext, pose_ext = [], [], [], []
    fallback = slotframes = 0; ppf = []; n1 = n3 = 0
    for fr in frames:
        k = len(fr.get("persons") or []); ppf.append(k)
        if k <= 1: n1 += 1
        elif k >= 3: n3 += 1
        for side in ("left", "right"):
            s = fr[side]
            if s is None: continue
            slotframes += 1
            b = s.get("bbox")
            if isinstance(b, dict):
                if (b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")) in FALLBACK:
                    fallback += 1
                else:
                    bbox_h.append((b["y2"] - b["y1"]) / FRAME_H)
                    bbox_a.append(((b["y2"] - b["y1"]) * (b["x2"] - b["x1"])) / (FRAME_H * FRAME_W))
            pf = []
            for part in (s.get("face") or {}).values():
                if isinstance(part, list): pf.extend(part)
            xs = [p[0] for p in pf if len(p) >= 2]; ys = [p[1] for p in pf if len(p) >= 2]
            if len(xs) > 1:
                face_ext.append(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
            ys2 = [p[1] for p in (s.get("pose") or []) if len(p) >= 2]
            if len(ys2) > 1:
                pose_ext.append(max(ys2) - min(ys2))
    m = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return [m(bbox_h), m(bbox_a), m(face_ext), m(pose_ext),
            fallback / slotframes if slotframes else float("nan"),
            float(np.mean(ppf)), n1 / n, n3 / n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--pose-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = load_items(a.items)
    cache, ids, K, km, F, nrec, ntight, spans = {}, [], [], [], [], [], [], []
    missing_pose = 0
    for r in items:
        clip = r["clip"]
        if clip not in cache:
            p = find_file(a.pose_dirs, f"{clip}.pose.json")
            cache[clip] = json.loads(p.read_text()) if p else None
        recs = cache[clip]
        if recs is None:
            missing_pose += 1; recs = []
        span = float(np.median([f["span"] for f in recs])) if recs else float("nan")
        sel = [f for f in recs if r["start_s"] * 1000 <= f["offset"] < r["end_s"] * 1000]
        tf = [f for f in sel if is_tight(f)]
        need = math.ceil(1000 / span) if span == span else 10**9
        if len(tf) >= need:
            K.append(np.delete(_pose_features(tf), IDX_POSITION)); km.append(0)
        else:
            K.append(np.full(14, np.nan)); km.append(1)
        F.append(framing(sel)); ids.append(r["utt_id"])
        nrec.append(len(sel)); ntight.append(len(tf)); spans.append(span)
    np.savez(a.out, ids=np.array(ids), K=np.array(K, float), k_missing=np.array(km),
             F=np.array(F, float), F_names=np.array(F_NAMES), n_records=np.array(nrec),
             n_tight=np.array(ntight), span_ms=np.array(spans))
    print(json.dumps({"items": len(ids), "clips_without_pose": missing_pose,
                      "k_missing": int(sum(km)), "f_all_nan": int(np.isnan(np.array(F, float)).all(1).sum()),
                      "span_ms_values": sorted({round(s, 1) for s in spans if s == s})}))

if __name__ == "__main__":
    main()
