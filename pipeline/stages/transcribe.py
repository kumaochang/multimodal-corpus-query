"""Transcription modality: PREREGISTRATION.md Amendment 4 Part B2, narrowed by Amendment 5.

- Model: omniASR_LLM_7B_v2 with the language code given (the ASR pilot's model and environment,
  scripts/asr_pilot/environment.txt).
- Languages: ami, pwn, tay, trv, tao. **No Tsou transcription is computed** (Amendment 5).
- Segmentation: the existing utterance boundaries (pipeline/stages/utterances.py). An utterance
  longer than 40 s is split at its longest internal pause (audio_utils.pauses; if it has none, at
  its midpoint) and the parts' transcripts are joined; the number of splits, and of midpoint
  splits, is reported. Splitting repeats until every part is 40 s or shorter.
- Normalization: as in the pilot (scripts/asr_pilot/omni_pilot.py normalize): NFC, lowercase,
  apostrophe variants unified, punctuation removed except ' and ^. Both raw and normalized text
  are stored.

Usage:
  python pipeline/stages/transcribe.py --manifest analysis/study3/manifest.json \
      --wav <dir of <clip>.wav> --trees <aligned_trees dir> --out data/study3_runs/transcripts \
      [--batch-size 8] [--dry-run]
--dry-run builds the segments and reports counts without loading the model.
Output: <out>/<clip>.jsonl, one line per utterance: utt_id, start_s, end_s, lang, parts, raw, text.
"""
import argparse, json, sys, tempfile
from pathlib import Path
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "asr_pilot"))
from utterances import load_utterances          # noqa: E402
from audio_utils import load_segment, pauses    # noqa: E402

MODEL_CARD = "omniASR_LLM_7B_v2"
MAX_S = 40.0
LANG = {"Amis": "ami_Latn", "Paiwan": "pwn_Latn", "Atayal": "tay_Latn", "Truku": "trv_Latn",
        "Yami": "tao_Latn"}          # Tsou deliberately absent

def split(wav, a, b, stats):
    if b - a <= MAX_S:
        return [(a, b)]
    seg, sr = load_segment(wav, a, b)
    ps = pauses(seg, sr)
    if ps:
        s, e = max(ps, key=lambda p: p[1] - p[0])
        cut = a + (s + e) / 2
    else:
        cut = (a + b) / 2
        stats["midpoint_splits"] += 1
    stats["splits"] += 1
    return split(wav, a, cut, stats) + split(wav, cut, b, stats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from omni_pilot import normalize            # the pilot's normalization, imported, not copied
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clips = [c for c in json.loads(Path(a.manifest).read_text(encoding="utf-8"))["clips"] if c["drawn"]]
    pipe = None
    if not a.dry_run:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        pipe = ASRInferencePipeline(model_card=MODEL_CARD)
    stats = {"clips": 0, "skipped_tsou_clips": 0, "utterances": 0, "splits": 0, "midpoint_splits": 0,
             "empty_transcripts": 0}
    for c in clips:
        clip = Path(c["file"]).stem
        if c["language"] not in LANG:
            stats["skipped_tsou_clips"] += 1
            continue
        dest = out / f"{clip}.jsonl"
        if dest.exists():
            continue
        lang = LANG[c["language"]]
        wav = Path(a.wav) / f"{clip}.wav"
        utts = load_utterances(Path(a.trees) / f"{clip}.aligntree.TextGrid", clip)
        jobs = [(uid, s, e, split(wav, s, e, stats)) for uid, s, e in utts]
        rows = []
        with tempfile.TemporaryDirectory() as td:
            paths, owners = [], []
            for k, (uid, s, e, parts) in enumerate(jobs):
                for j, (ps, pe) in enumerate(parts):
                    seg, sr = load_segment(wav, ps, pe)
                    p = Path(td) / f"{k:05d}_{j}.wav"
                    sf.write(str(p), seg, sr)
                    paths.append(str(p)); owners.append(k)
            hyps = [""] * len(paths)
            if pipe is not None:
                hyps = []
                for i in range(0, len(paths), a.batch_size):
                    b = paths[i:i + a.batch_size]
                    hyps += pipe.transcribe(b, lang=[lang] * len(b), batch_size=a.batch_size)
            for k, (uid, s, e, parts) in enumerate(jobs):
                raw = " ".join(h for h, o in zip(hyps, owners) if o == k).strip()
                if pipe is not None and not raw:
                    stats["empty_transcripts"] += 1
                rows.append({"utt_id": uid, "start_s": s, "end_s": e, "lang": lang,
                             "parts": len(parts), "raw": raw, "text": normalize(raw)})
        if not a.dry_run:
            dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        stats["clips"] += 1; stats["utterances"] += len(rows)
        print(clip, len(rows), flush=True)
    (out / ("provenance.dry_run.json" if a.dry_run else "provenance.json")).write_text(json.dumps({
        "stage": "transcribe", "amendments": "4 Part B2; 5", "model": MODEL_CARD,
        "max_segment_s": MAX_S, "dry_run": a.dry_run, "stats": stats}, indent=1), encoding="utf-8")
    print(json.dumps(stats))

if __name__ == "__main__":
    main()
