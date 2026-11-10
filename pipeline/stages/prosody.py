"""Prosodic features per utterance: the §5.1 representation built from F0, intensity, duration,
pause and boundary, which replaces Study 1's phone-segment statistics (analysis/q23). Amendment 6
item 1 puts "prosody, as computed for Chapter 5" into Arm B's fusion, so this feature set is fixed
here, in code, before any Study 3 output exists, and is not changed after.

Per utterance (Parselmouth/Praat; F0 floor 60 Hz, ceiling 500 Hz, time step 10 ms; semitones
relative to the utterance's median F0, so the features are speaker-normalized):
   1 dur_s               utterance duration
   2 voiced_ratio        voiced frames / all frames
   3 f0_med_hz           median F0 (Hz), the one non-normalized level, kept for register
   4 f0_range_st         95th minus 5th percentile, semitones
   5 f0_sd_st            standard deviation, semitones
   6 f0_slope_st_s       least-squares slope over voiced frames, semitones per second
   7 f0_final_st         mean of the last 25% of voiced frames minus mean of the first 25%
   8 f0_end_slope_st_s   slope over the voiced frames of the final 300 ms (boundary movement)
   9 int_mean_db         mean intensity
  10 int_range_db        95th minus 5th percentile intensity
  11 int_end_drop_db     mean intensity of the final 300 ms minus the utterance mean
  12 pause_ratio         share of time in internal pauses (audio_utils.pauses)
  13 pauses_per_s        internal pauses per second
  14 max_pause_s         longest internal pause
  15 voiced_runs_per_s   voiced stretches per second, a syllable-rate proxy
An utterance with fewer than 5 voiced frames gets NaN for the F0 features; their count is reported.
Utterances too short for Praat's analysis windows (added 2026-10-13, after the first Study 3 run
stopped on a 0.100 s utterance; no feature definition changed): shorter than 3/F0_FLOOR (pitch
window) gets NaN for voiced_ratio, the F0 features and voiced_runs_per_s; shorter than
6.4/F0_FLOOR (intensity window) gets NaN for the intensity features. Both counts are reported.

Usage: python pipeline/stages/prosody.py --manifest analysis/study3/manifest.json --wav <dir> \
          --trees <aligned_trees dir> --out data/study3_runs/prosody.jsonl
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import parselmouth

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utterances import load_utterances        # noqa: E402
from audio_utils import load_segment, pauses  # noqa: E402

F0_FLOOR, F0_CEIL, STEP = 60.0, 500.0, 0.01
MIN_PITCH_S = 3.0 / F0_FLOOR       # Praat's pitch window for the given floor
MIN_INT_S = 6.4 / F0_FLOOR         # Praat's intensity window for minimum_pitch = F0_FLOOR
F0_KEYS = ["f0_med_hz", "f0_range_st", "f0_sd_st", "f0_slope_st_s", "f0_final_st", "f0_end_slope_st_s"]
INT_KEYS = ["int_mean_db", "int_range_db", "int_end_drop_db"]
END_S = 0.30
FEATURES = ["dur_s", "voiced_ratio", "f0_med_hz", "f0_range_st", "f0_sd_st", "f0_slope_st_s",
            "f0_final_st", "f0_end_slope_st_s", "int_mean_db", "int_range_db", "int_end_drop_db",
            "pause_ratio", "pauses_per_s", "max_pause_s", "voiced_runs_per_s"]

def slope(t, y):
    return float(np.polyfit(t, y, 1)[0]) if len(t) >= 2 and np.ptp(t) > 0 else float("nan")

def features(seg, sr):
    dur = len(seg) / sr
    snd = parselmouth.Sound(seg.astype(np.float64), sampling_frequency=sr)
    nan = float("nan")
    out = {"dur_s": dur}
    if dur < MIN_PITCH_S:
        out.update({"voiced_ratio": nan, "voiced_runs_per_s": nan, **{k: nan for k in F0_KEYS}})
        v = None
    else:
        pitch = snd.to_pitch(time_step=STEP, pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEIL)
        f0 = pitch.selected_array["frequency"]
        t = pitch.xs()
        v = f0 > 0
        out["voiced_ratio"] = float(v.mean()) if len(v) else 0.0
    if v is None:
        pass
    elif v.sum() >= 5:
        med = float(np.median(f0[v]))
        st = 12 * np.log2(f0[v] / med)
        tv = t[v]
        q = max(1, len(st) // 4)
        endm = tv >= (dur - END_S)
        out.update({"f0_med_hz": med,
                    "f0_range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
                    "f0_sd_st": float(np.std(st)),
                    "f0_slope_st_s": slope(tv, st),
                    "f0_final_st": float(st[-q:].mean() - st[:q].mean()),
                    "f0_end_slope_st_s": slope(tv[endm], st[endm])})
    else:
        out.update({k: nan for k in F0_KEYS})
    if dur < MIN_INT_S:
        iv = np.array([])
    else:
        inten = snd.to_intensity(minimum_pitch=F0_FLOOR, time_step=STEP)
        iv, it = inten.values[0], inten.xs()
        ok = np.isfinite(iv)
        iv, it = iv[ok], it[ok]
    if len(iv):
        out["int_mean_db"] = float(iv.mean())
        out["int_range_db"] = float(np.percentile(iv, 95) - np.percentile(iv, 5))
        tail = iv[it >= dur - END_S]
        out["int_end_drop_db"] = float(tail.mean() - iv.mean()) if len(tail) else nan
    else:
        out.update({k: nan for k in INT_KEYS})
    ps = pauses(seg, sr)
    out["pause_ratio"] = sum(e - s for s, e in ps) / dur if dur else nan
    out["pauses_per_s"] = len(ps) / dur if dur else nan
    out["max_pause_s"] = max((e - s for s, e in ps), default=0.0)
    if v is not None:
        runs = int(np.sum(np.diff(np.concatenate([[0], v.astype(int)])) == 1)) if len(v) else 0
        out["voiced_runs_per_s"] = runs / dur if dur else nan
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", nargs="*", help="limit to these clip stems (testing)")
    a = ap.parse_args()
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["utt_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    clips = [Path(c["file"]).stem for c in json.loads(Path(a.manifest).read_text(encoding="utf-8"))["clips"]
             if c["drawn"]]
    if a.clips:
        clips = a.clips
    n_f0_nan = n_short_pitch = n_short_int = 0
    with out.open("a", encoding="utf-8") as fh:
        for clip in clips:
            for uid, s, e in load_utterances(Path(a.trees) / f"{clip}.aligntree.TextGrid", clip):
                if uid in done:
                    continue
                seg, sr = load_segment(Path(a.wav) / f"{clip}.wav", s, e)
                f = features(seg, sr)
                n_f0_nan += math.isnan(f["f0_med_hz"])
                n_short_pitch += f["dur_s"] < MIN_PITCH_S
                n_short_int += f["dur_s"] < MIN_INT_S
                fh.write(json.dumps({"utt_id": uid, **{k: f[k] for k in FEATURES}}) + "\n")
            print(clip, flush=True)
    Path(str(out) + ".provenance.json").write_text(json.dumps({
        "stage": "prosody", "tool": "praat-parselmouth", "version": parselmouth.VERSION,
        "praat": parselmouth.PRAAT_VERSION, "features": FEATURES, "f0_floor": F0_FLOOR,
        "f0_ceiling": F0_CEIL, "time_step": STEP, "end_window_s": END_S,
        "utterances_without_f0_this_run": n_f0_nan,
        "min_pitch_s": MIN_PITCH_S, "min_intensity_s": MIN_INT_S,
        "utterances_shorter_than_pitch_window_this_run": int(n_short_pitch),
        "utterances_shorter_than_intensity_window_this_run": int(n_short_int)}, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
