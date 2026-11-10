"""Segment per-frame PaddleOCR output into subtitle-line spans.

TITV 《看見族語》 frames carry up to three text bands at the bottom of a 1280x720 frame:
  the speech subtitle (y-centre about 555-625), the red banner below it (headline or speaker
  tag, about 625-700), and the programme watermark with the language label at the left
  (x < 300), which OCR reads as 看見族語 fragments and must be excluded.

Output per clip: a list of {start_ms, end_ms, n_frames, text, banner} where consecutive frames
whose subtitle text is within a normalized edit distance of 0.3 belong to one line, and the
line's text is the most frequent frame reading (the one a VLM pass would re-read at mid-span).

Usage: python subtitle_lines.py <ocr_raw_dir> <out_dir>
"""
import json, re, sys, collections
from pathlib import Path

SUB_BAND, BANNER_BAND, WATERMARK_X = (555, 625), (625, 700), 300

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def band(dets, rng):
    boxes = []
    for box, text, conf, label in dets:
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        if max(xs) <= WATERMARK_X:
            continue
        yc = (min(ys) + max(ys)) / 2
        if rng[0] <= yc < rng[1]:
            boxes.append((min(xs), text))
    return re.sub(r"\s+", "", "".join(t for _, t in sorted(boxes)))

def segment(frames, thr=0.5):
    lines, cur = [], None
    for ts, dets in frames:
        t = band(dets, SUB_BAND)
        if cur and t and lev(t, cur["last"]) / max(len(t), len(cur["last"])) <= thr:
            cur["reads"].append(t); cur["end_ms"] = ts; cur["last"] = t
            cur["banners"].append(band(dets, BANNER_BAND))
            continue
        if cur:
            lines.append(cur); cur = None
        if t:
            cur = {"start_ms": ts, "end_ms": ts, "reads": [t], "last": t,
                   "banners": [band(dets, BANNER_BAND)]}
    if cur:
        lines.append(cur)
    out = []
    for l in lines:
        out.append({"start_ms": l["start_ms"], "end_ms": l["end_ms"], "n_frames": len(l["reads"]),
                    "text": collections.Counter(l["reads"]).most_common(1)[0][0],
                    "banner": collections.Counter(l["banners"]).most_common(1)[0][0]})
    return out

if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.ocr.json")):
        lines = segment(json.loads(f.read_text(encoding="utf-8")))
        (dst / f.name.replace(".ocr.json", ".lines.json")).write_text(
            json.dumps(lines, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f.name, len(lines))
