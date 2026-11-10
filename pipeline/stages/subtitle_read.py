"""Subtitle text per utterance, as PREREGISTRATION.md Amendment 7 item 1 specifies.

Steps (numbering as in the amendment):
  1. per-frame OCR: MOCO stage 5.1 output (`ocr_raw_results/<clip>.ocr.json`), uncorrected
  2. line segmentation: subtitle_lines.segment (as committed at b218de1)
  3. reading: one keyframe per line at the midpoint of its span, cropped to x 300-1280,
     y 540-640 of the 1280x720 frame at native resolution, read by claude-haiku-4-5-20251001 at
     temperature 0 with PROMPT below; an empty or "(none)" reading falls back to the line's most
     frequent OCR reading and the line is flagged. If ffmpeg returns no frame at the midpoint (a
     midpoint within one frame of the end of the video), no request is sent and the line takes the
     same fallback with flag "no_keyframe" (added 2026-10-13; 10 of 18,474 Study 3 lines)
  4. assignment: each line goes to the utterance it overlaps most; an utterance's subtitle text
     is its lines joined in time order
  5. the banner band is kept apart and is not part of the subtitle text

Usage:
  python pipeline/stages/subtitle_read.py --ocr <ocr_raw_results dir> --videos <h264 dir> \
      --trees <aligned_trees dir> --out <out dir> [--dry-run]

--dry-run skips the model and uses the OCR reading for every line (flag "dry_run"); it exists to
test everything else. Reported results must come from a run without it. Readings are cached per
line in <out>/_cache, so an interrupted run resumes without new API calls. Needs
ANTHROPIC_API_KEY unless --dry-run.
"""
import argparse, base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subtitle_lines import segment            # noqa: E402
from utterances import load_utterances        # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 0
CROP = "crop=980:100:300:540"                 # w:h:x:y in the 1280x720 frame
PROMPT = ("This image is the subtitle area of a frame from a Taiwanese Indigenous Television news "
          "broadcast. Transcribe the subtitle text exactly as shown, in the characters shown. Do not "
          "translate, correct or complete it. Output only the text. If there is no subtitle text, "
          "output exactly: (none)")

def keyframe_png(video, t_s):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t_s:.3f}", "-i", str(video),
                        "-frames:v", "1", "-vf", CROP, "-f", "image2pipe", "-vcodec", "png", "-"],
                       capture_output=True, check=True)
    return r.stdout

def read_line(client, png):
    msg = client.messages.create(
        model=MODEL, max_tokens=200, temperature=TEMPERATURE,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode()}},
            {"type": "text", "text": PROMPT}]}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

def assign(lines, utts, step_s):
    """A line's span runs from its first OCR frame to one OCR frame step past its last, so that a
    line seen on a single frame still has a duration; step_s is the clip's median frame step."""
    for l in lines:
        s, e = l["start_ms"] / 1000, l["end_ms"] / 1000 + step_s
        best, ov = None, 0.0
        for uid, a, b in utts:
            o = min(e, b) - max(s, a)
            if o > ov:
                best, ov = uid, o
        l["utt_id"] = best
    out = []
    for uid, a, b in utts:
        mine = [l for l in lines if l["utt_id"] == uid]
        out.append({"utt_id": uid, "start_s": a, "end_s": b,
                    "subtitle_text": "".join(l["text"] for l in mine), "n_lines": len(mine)})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); cache = out / "_cache"; cache.mkdir(parents=True, exist_ok=True)
    client = None
    if not a.dry_run:
        import anthropic
        client = anthropic.Anthropic()
    totals = {"clips": 0, "lines": 0, "flagged_empty_reading": 0, "flagged_no_keyframe": 0,
              "unassigned_lines": 0,
              "utterances": 0, "utterances_without_subtitle": 0}
    for ocr_file in sorted(Path(a.ocr).glob("*.ocr.json")):
        clip = ocr_file.name[:-len(".ocr.json")]
        tree = Path(a.trees) / f"{clip}.aligntree.TextGrid"
        video = Path(a.videos) / f"{clip}.mp4"
        if not tree.exists() or not video.exists():
            print(f"missing input for {clip}: tree={tree.exists()} video={video.exists()}", file=sys.stderr)
            continue
        frames = json.loads(ocr_file.read_text(encoding="utf-8"))
        ts = [f[0] for f in frames]
        steps = sorted(b - a for a, b in zip(ts, ts[1:]))
        step_s = steps[len(steps) // 2] / 1000 if steps else 1.0
        lines = segment(frames)
        for l in lines:
            l["ocr_text"] = l.pop("text")
            mid = (l["start_ms"] + l["end_ms"]) / 2000
            key = hashlib.sha256(f"{clip}|{mid:.3f}|{MODEL}|{PROMPT}".encode()).hexdigest()[:20]
            cf = cache / f"{key}.json"
            if a.dry_run:
                reading = None
            elif cf.exists():
                reading = json.loads(cf.read_text(encoding="utf-8"))["reading"]
            elif not (png := keyframe_png(video, mid)):
                reading = None
                l["vlm_text"], l["text"], l["flag"] = None, l["ocr_text"], "no_keyframe"
                totals["flagged_no_keyframe"] += 1
                continue
            else:
                reading = read_line(client, png)
                cf.write_text(json.dumps({"clip": clip, "mid_s": mid, "reading": reading},
                                         ensure_ascii=False), encoding="utf-8")
                time.sleep(0.2)
            l["vlm_text"] = reading
            if a.dry_run:
                l["text"], l["flag"] = l["ocr_text"], "dry_run"
            elif not reading or reading.strip() == "(none)":
                l["text"], l["flag"] = l["ocr_text"], "empty_reading"
                totals["flagged_empty_reading"] += 1
            else:
                l["text"], l["flag"] = reading.replace(" ", ""), None
        utts = load_utterances(tree, clip)
        per_utt = assign(lines, utts, step_s)
        totals["clips"] += 1; totals["lines"] += len(lines)
        totals["unassigned_lines"] += sum(l["utt_id"] is None for l in lines)
        totals["utterances"] += len(per_utt)
        totals["utterances_without_subtitle"] += sum(not u["subtitle_text"] for u in per_utt)
        (out / f"{clip}.subtitles.json").write_text(json.dumps(
            {"clip": clip, "lines": lines, "utterances": per_utt}, ensure_ascii=False, indent=0),
            encoding="utf-8")
        print(clip, len(lines), len(per_utt), flush=True)
    (out / "provenance.json").write_text(json.dumps({
        "stage": "subtitle_read", "amendment": "PREREGISTRATION.md Amendment 7 item 1",
        "model": None if a.dry_run else MODEL, "temperature": TEMPERATURE, "crop": CROP,
        "prompt": PROMPT, "dry_run": a.dry_run, "totals": totals}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(totals))

if __name__ == "__main__":
    main()
