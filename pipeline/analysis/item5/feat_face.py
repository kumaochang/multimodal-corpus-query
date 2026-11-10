"""Block Fa per judged utterance (Amendment 9 item 5; Amendment 10 item 3).

OpenFace 3.0 (CMU-MultiComp-Lab, package `openface-test`, weights by `openface download`; code
version and weight files recorded) on frames sampled at 10 per second inside the utterance, from
the 720p video. In each frame: RetinaFace detection (FaceDetector), the largest detected face
(no active-speaker detection exists; the largest face is a proxy for the speaker), its 98 STAR
landmarks (LandmarkDetector; WFLW definition, `Landmark_98.pkl`, the package's own default) and the multitask outputs (MultitaskPredictor): action-unit
intensities and gaze yaw and pitch. Emotion logits are not used (not registered).

Head pose. OpenFace 3.0 outputs no head pose (its README and paper describe none), while
Amendment 9 lists head yaw, pitch and roll. They are therefore computed from OpenFace 3.0's own
98 landmarks with the standard perspective-n-point fit (cv2.solvePnP, iterative) of six points
(WFLW nose tip 54, chin 16, outer eye corners 60 and 72, mouth corners 76 and 82; these are the
68-point 30, 8, 36, 45, 48 and 54) to a fixed generic
3-D face model, camera focal length = frame width, principal point = frame centre, no lens
distortion. Decided before any face measure was computed; recorded as an implementation note.

Per utterance: mean and standard deviation over frames with a face of every AU intensity, gaze
yaw, gaze pitch, head yaw, pitch and roll; `face_share` = frames with a face / frames sampled;
`face_missing` = 1 if face_share < 0.5 (then the face columns are NaN; the probe imputes the
training-fold mean). Output: --out .npz (ids, face, face_names, face_share, face_missing).
Prints counts only. --dry-run samples frames and counts them without loading models.
The OpenFace calls follow its README (FaceDetector.get_face(image_path) -> (crop, dets);
LandmarkDetector.detect_landmarks(image, dets); MultitaskPredictor.predict(face) -> (emotion,
gaze, au)); the detection box format [x1, y1, x2, y2, score, ...] is checked on Study 1 frames
before any Study 3 run.
"""
import argparse, contextlib, json, logging, os, sys, tempfile
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_items, find_file  # noqa: E402

FPS = 10.0
MODEL_3D = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                     (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
LM_IDX = [54, 16, 60, 72, 76, 82]  # WFLW 98-point indices (68-point 30, 8, 36, 45, 48, 54)
N_LM = 98

def head_pose(lm, w, h):
    """Yaw, pitch, roll (degrees) from 98 WFLW landmarks by solvePnP against a generic face model."""
    if len(lm) != N_LM:
        raise ValueError(f"expected {N_LM} landmarks, got {len(lm)}")
    pts = np.array([lm[i] for i in LM_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return [np.nan] * 3
    R, _ = cv2.Rodrigues(rvec)
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return [yaw, pitch, roll]

def sample_frames(video, s, e):
    cap = cv2.VideoCapture(str(video)); out = []
    t = s
    while t < e:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            out.append(fr)
        t += 1.0 / FPS
    cap.release()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--det-weights"); ap.add_argument("--lm-weights"); ap.add_argument("--mtl-weights")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--openface-cwd", help="folder with a ./weights link (RetinaFace loads its "
                    "backbone init from ./weights/ relative to the working directory)")
    a = ap.parse_args()
    # absolute paths first, so that changing the working directory below breaks nothing
    a.items, a.out = os.path.abspath(a.items), os.path.abspath(a.out)
    a.videos = [os.path.abspath(v) for v in a.videos]
    for k in ("det_weights", "lm_weights", "mtl_weights"):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    items = load_items(a.items)
    if not a.dry_run:
        import torch
        # openface/Pytorch_Retinaface/detect.py calls parser.parse_args() at import time and
        # would parse this program's arguments and exit; give it an empty argument list while
        # importing, then restore. Behaviour of the models is unchanged.
        saved_argv, saved_cwd = sys.argv, os.getcwd()
        sys.argv = [sys.argv[0]]
        try:
            from openface.face_detection import FaceDetector
            from openface.landmark_detection import LandmarkDetector
            from openface.multitask_model import MultitaskPredictor
            import openface.multitask_model as of_mtm
        finally:
            sys.argv = saved_argv
        # openface/multitask_model.py calls cv2.cvtColor in preprocess() but never imports cv2
        # (openface-test 0.1.13); supply the same cv2 module this program uses. No other change.
        if not hasattr(of_mtm, "cv2"):
            of_mtm.cv2 = cv2
        # STAR's Alignment config hardcodes ckpt_dir = '/work/jiewenh/...' (its developer's
        # machine) and, while LandmarkDetector is built, opens a tensorboardX writer and a log
        # file under ckpt_dir/<data>/<folder>/log. These are training artefacts with no effect on
        # inference. Redirect them to a writable folder by wrapping the config factory, then
        # restore the factory and remove the logging handlers STAR adds to the root logger.
        from openface.STAR.lib import utility as star_utility
        star_root = os.path.abspath(os.path.join(a.openface_cwd or tempfile.mkdtemp(), "star_cfg"))
        orig_get_config = star_utility.get_config
        def get_config_writable(args):
            cfg = orig_get_config(args)
            cfg.ckpt_dir = star_root
            cfg.work_dir = os.path.join(cfg.ckpt_dir, cfg.data_definition, cfg.folder)
            cfg.model_dir = os.path.join(cfg.work_dir, "model")
            cfg.log_dir = os.path.join(cfg.work_dir, "log")
            os.makedirs(cfg.log_dir, exist_ok=True)
            return cfg
        root_log = logging.getLogger()
        saved_handlers, saved_level = list(root_log.handlers), root_log.level
        star_utility.get_config = get_config_writable
        if a.openface_cwd:
            os.chdir(a.openface_cwd)
        # OpenFace prints its arguments and model paths while loading; keep stdout to counts only
        quiet = open(os.devnull, "w")
        try:
            with contextlib.redirect_stdout(quiet):
                fd = FaceDetector(model_path=a.det_weights, device=a.device)
                # LandmarkDetector accepts only "cpu" or "cuda" and needs device_ids for cuda
                if a.device.startswith("cuda"):
                    gpu = int(a.device.split(":")[1]) if ":" in a.device else 0
                    ld = LandmarkDetector(model_path=a.lm_weights, device="cuda", device_ids=[gpu])
                else:
                    ld = LandmarkDetector(model_path=a.lm_weights, device="cpu")
                mt = MultitaskPredictor(model_path=a.mtl_weights, device=a.device)
        finally:
            os.chdir(saved_cwd)
            star_utility.get_config = orig_get_config
            for hd in list(root_log.handlers):
                if hd not in saved_handlers:
                    root_log.removeHandler(hd); hd.close()
            root_log.setLevel(saved_level)
    rows, shares, miss, ids, n_frames, n_au = [], [], [], [], 0, None
    tmp = tempfile.mkdtemp()
    for r in items:
        vid = find_file(a.videos, f"{r['clip']}.mp4")
        frames = sample_frames(vid, r["start_s"], r["end_s"]); n_frames += len(frames)
        per = []
        if not a.dry_run:
            for fr in frames:
                h, w = fr.shape[:2]
                p = os.path.join(tmp, "f.jpg"); cv2.imwrite(p, fr)
                cropped, dets = fd.get_face(p)
                if dets is None or len(dets) == 0:
                    continue
                # get_face() only checks that the top-scoring box reaches its vis_threshold (0.5);
                # dets still holds every box above the 0.02 candidate threshold. Choose the largest
                # face among boxes that pass the same 0.5 threshold (detect_landmarks uses it too),
                # so that a large low-confidence false positive is never taken as the speaker.
                vis = getattr(fd, "vis_threshold", 0.5)
                dets = [b for b in dets if b[4] >= vis]
                if not dets:
                    continue
                d = max(dets, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2 = [int(max(0, v)) for v in d[:4]]
                face = fr[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                _, gaze, au = mt.predict(face)
                with contextlib.redirect_stdout(quiet):  # it prints each box and score
                    lms = ld.detect_landmarks(fr, np.array([d]))
                hp = head_pose(lms[0], w, h) if lms else [np.nan] * 3
                au = au.detach().cpu().numpy().ravel(); gz = gaze.detach().cpu().numpy().ravel()[:2]
                n_au = len(au)
                per.append(np.concatenate([au, gz, hp]))
        share = len(per) / len(frames) if frames else 0.0
        shares.append(share); miss.append(int(share < 0.5)); ids.append(r["utt_id"])
        rows.append(np.concatenate([np.nanmean(per, 0), np.nanstd(per, 0)]) if per and share >= 0.5 else None)
    dim = next((len(x) for x in rows if x is not None), 0)
    F = np.array([x if x is not None else np.full(dim, np.nan) for x in rows]) if dim else np.zeros((len(ids), 0))
    names = ([f"au{i}" for i in range(n_au or 0)] + ["gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch", "head_roll"])
    names = [f"{n}_mean" for n in names] + [f"{n}_sd" for n in names]
    np.savez(a.out, ids=np.array(ids), face=F, face_names=np.array(names), face_share=np.array(shares),
             face_missing=np.array(miss))
    print(json.dumps({"items": len(ids), "frames_sampled": n_frames, "face_missing": int(sum(miss)),
                      "n_action_units": n_au, "face_columns": int(F.shape[1]),
                      "dry_run": a.dry_run}))

if __name__ == "__main__":
    main()
