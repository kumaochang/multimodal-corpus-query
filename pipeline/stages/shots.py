"""Shot segmentation for Study 3: the registered method of analysis/q14a_segment_all.py
(PySceneDetect ContentDetector, threshold 27.0, on the source video), with input and output
given as arguments instead of the Study 1 paths.

Usage: python pipeline/stages/shots.py --videos <dir of .mp4> --out <shots.json>
Output: the same structure as q14a, {clip_stem: {"fps", "duration_s", "shots": [[start, end], ...]}};
resumable; a sidecar <out>.provenance.json.
"""
import argparse, json, time
from pathlib import Path
import scenedetect
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

THRESHOLD = 27.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for f in sorted(Path(a.videos).glob("*.mp4")):
        if f.stem in done:
            continue
        t0 = time.time()
        v = open_video(str(f))
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=THRESHOLD))
        sm.detect_scenes(v, show_progress=False)
        scenes = sm.get_scene_list()
        done[f.stem] = {
            "fps": float(v.frame_rate),
            "duration_s": float(scenes[-1][1].seconds) if scenes else 0.0,
            "shots": [[float(x.seconds), float(y.seconds)] for x, y in scenes],
        }
        out.write_text(json.dumps(done), encoding="utf-8")
        print(f.stem, len(scenes), round(time.time() - t0, 1), flush=True)
    Path(str(out) + ".provenance.json").write_text(json.dumps({
        "stage": "shots", "tool": "PySceneDetect", "version": scenedetect.__version__,
        "detector": "ContentDetector", "threshold": THRESHOLD, "source": "analysis/q14a_segment_all.py",
        "clips": len(done)}, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
