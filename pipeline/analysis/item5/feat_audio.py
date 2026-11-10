"""Block A per judged utterance (Amendment 9 item 5; Amendment 10 item 3).

Sources, kept separate:
  whisper : openai-whisper large-v3 encoder; the utterance audio (16 kHz) is padded or trimmed to
            30 s, encoded, and the first int(dur * 50) frames are averaged (Study 1's
            _encode_window and "audio_whisper_mean", pilot/experiment.py). Utterances longer than
            30 s use their first 30 s (counted).
  prosody : the 15 features of pipeline/stages/prosody.py, joined by utt_id from its JSONL (NaN
            kept; the probe imputes the training-fold mean).
Inputs: --items; --wav DIR [DIR ...] (<clip>.wav); --prosody JSONL [JSONL ...].
Output: --out .npz (ids, whisper n x 1280, prosody n x 15, prosody_names). Prints counts only.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_items, find_file  # noqa: E402

SR = 16000
PROSODY = ["dur_s", "voiced_ratio", "f0_med_hz", "f0_range_st", "f0_sd_st", "f0_slope_st_s",
           "f0_final_st", "f0_end_slope_st_s", "int_mean_db", "int_range_db", "int_end_drop_db",
           "pause_ratio", "pauses_per_s", "max_pause_s", "voiced_runs_per_s"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--wav", nargs="+", required=True)
    ap.add_argument("--prosody", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-whisper", action="store_true", help="plumbing tests only")
    a = ap.parse_args()
    import soundfile as sf
    items = load_items(a.items)
    pros = {}
    for p in a.prosody:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); pros[r["utt_id"]] = [r.get(k) for k in PROSODY]
    P = np.array([[np.nan if v is None else v for v in pros.get(r["utt_id"], [None] * 15)]
                  for r in items], float)
    Wv = np.zeros((len(items), 1280)); over30 = 0
    if not a.no_whisper:
        import torch, whisper
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("large-v3", device=dev).eval()
        for i, r in enumerate(items):
            wav = find_file(a.wav, f"{r['clip']}.wav")
            x, sr = sf.read(wav, start=int(r["start_s"] * SR), stop=int(r["end_s"] * SR), dtype="float32")
            assert sr == SR
            if x.ndim > 1: x = x.mean(1)
            dur = len(x) / SR; over30 += dur > 30
            chunk = whisper.pad_or_trim(torch.from_numpy(x))
            mel = whisper.log_mel_spectrogram(chunk, n_mels=model.dims.n_mels).to(dev)
            with torch.no_grad():
                enc = model.encoder(mel.unsqueeze(0)).squeeze(0).cpu().numpy()
            Wv[i] = enc[:max(1, int(min(dur, 30) * 50))].mean(0)
    np.savez(a.out, ids=np.array([r["utt_id"] for r in items]), whisper=Wv, prosody=P,
             prosody_names=np.array(PROSODY))
    print(json.dumps({"items": len(items), "prosody_missing_rows": int(sum(r["utt_id"] not in pros for r in items)),
                      "prosody_nan_cells": int(np.isnan(P).sum()), "over_30s": int(over30),
                      "whisper": "skipped" if a.no_whisper else "done"}))

if __name__ == "__main__":
    main()
