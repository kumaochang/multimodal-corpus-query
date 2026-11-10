# Prosodic augmentation and descriptive features

## Source contract and alignment

`INVENTORY.json` pins 13 inputs and verifies their hashes before/after. The fixed registered OOF archive provides the exact ordered primary IDs, y, inverse-inclusion weights and dates. Its SHA-256 is `f1d036398fa05e54ef0650923e547e395a4d4b4c4a7d0872a6add549d772223f`. The ordered-ID SHA-256 is `1f9a8ed24d6e425bb06fbf5453bbd8f98910fec08e55d8658f46280253e20b6d` (UTF-8 newline joined with final newline).

The 240-item frozen audio/text arrays align exactly with the first-round sampling metadata. The saved OOF archive supplies the judgment reference. All 240-by-15 prosody scalar values match their core/ext84 JSONL producer records exactly, including missing values. The input root is `data/study3_runs`; saved OOF inputs are at `data/study3_stats/registered/oof_scores.npz`.

## P1 models and preprocessing

The three models are T, T+P and T+W. T contains ASR XLM-R, frozen ASR TF-IDF and the four existing form indicators. P is only the 15-dimensional prosody block. W is only the 1,280-dimensional Whisper encoder block. The original A concatenates W and P as separate source transformations; retain that original T+A reference with its lineage, without introducing a new comparison family.

Use the original `probe.py` sourcewise training-fold procedure, not PCA of a single concatenated T matrix. For each held-out broadcast day, impute each source with training means, all-missing columns with zero, standardize with training population SD, setting zero SD to one. If source dimension exceeds 16, fit PCA with min(16,n_train-1), random_state=0, on training rows only. Then concatenate transformed sources. T+P keeps the same transformed T sources as T, adding P, and likewise for T+W. Cached transforms may be shared only when bitwise equal to independent frozen-function calls.

Use LogisticRegression with L2, C=1, max_iter=5000 and unchanged solver/default settings under an explicitly pinned sklearn version. Apply the frozen design weights to training. Leave one whole day out, sorted date order, all 12 days. A one-class training fold predicts its weighted positive share and is counted. Missing/nonfinite features follow the original training imputation; missing/nonfinite output predictions or solver nonconvergence stop the run rather than silently excluding cases. Record iterations and warnings. No tuning or outcome-driven model selection.

The local profile is Python 3.12.3, NumPy 1.26.4, SciPy 1.15.2 and scikit-learn 1.6.1. The registered server profile used SciPy 1.13.1 and scikit-learn 1.5.2. T, T+P and T+W form one adopted local numerical lineage. Paired comparisons use the saved local T control, not the server T scores. BASELINE_VERSIONS.json verifies the unchanged original control archive and records portable public program identities separately.

## Model evaluation and paired intervals

Report weighted AUC and weighted average precision (AP, not retrieval MAP) for each model, followed by T+P minus T, T+W minus T and T+P minus T+W, each with AUC/AP in that order, six paired coordinates. No threshold accuracy or fitted calibration model is added.

PCG64 seed 2026110502 generates 10,000 draws with one `rng.integers(0,12,size=(10000,12))` call over lexicographically ordered dates. Complete-day selections are shared by every model and metric. Saved OOF predictions are evaluated using inverse-inclusion weights multiplied by day multiplicities, without refitting models within each draw. Single-class draws have undefined AUC and AP and are not replaced by zero. Pointwise .025/.975 quantiles use finite draws and linear interpolation. All six coordinates also receive the max-standardized-deviation simultaneous rule, using SD with ddof=1. The whole family is suppressed if fewer than 99% of draws are jointly valid or any SD is zero; coordinates are never removed to obtain a band.

Saved computation products include OOF scores, folds, per-source transformations, PCA and model parameters, warnings, estimates, complete-day selections and undefined-draw counts.

## P2 coverage and display variables

All 15 frozen variables are retained with predefined roles, avoiding variable selection by observed effects.

| Family | Frozen variables | Units and evidence task |
|---|---|---|
| Duration and voicing | dur_s, voiced_ratio, voiced_runs_per_s | Seconds, fraction, runs/second; duration and voiced continuity |
| Pitch level and dispersion | f0_med_hz, f0_range_st, f0_sd_st | Hz, semitones; level and 95th-minus-5th/SD spread |
| Pitch change | f0_slope_st_s, f0_final_st, f0_end_slope_st_s | Semitones/second, semitones, semitones/second; overall and local change |
| Intensity | int_mean_db, int_range_db, int_end_drop_db | dB; mean, 95th-minus-5th spread and end-relative difference |
| Pausing | pause_ratio, pauses_per_s, max_pause_s | Fraction, pauses/second, seconds; temporal interruption |

Read definitions from the pinned producer, not variable-name intuition. Semitones are centered on each utterance's voiced median, not speaker-normalized. `f0_final_st` contrasts the last and first quarters of voiced frames, not necessarily chronological quarters of the utterance. `f0_end_slope_st_s` fits voiced frames within the final 300 ms. `int_end_drop_db` is end-window mean minus whole-utterance mean, so a negative value indicates lower final intensity. Do not reverse the sign to make the name read naturally. Preserve raw Hz/dB and their source/recording dependence.

The primary support is 192 for 14 variables, split 20/172. Only final F0 slope has missing values: 173 finite, split 18/155, with 19 excluded from that variable's descriptive denominator. Across all 240 items the same variable is finite for 209. Primary positive finite support spans 9 days and 17 clips for the other variables, and 9 days/15 clips for final slope. Negative support is 12 days/75 clips, or 12 days/71 clips for final slope. Report these supports alongside values. Do not impute P2 observed distributions from P1's modeling imputation.

The descriptive summaries retain raw units, finite n/day/clip counts, unweighted mean/SD and median/IQR, plus design-weighted means and positive-minus-negative differences. PCG64 seed 2026110503 generates 10,000 shared whole-day draws, with the same date ordering and integer draw construction as model evaluation. Each variable/group recomputes its finite weighted denominator; zero-denominator draws remain undefined. Intervals for weighted mean differences are pointwise descriptive percentile intervals, without variable selection or significance stars.

The primary display excludes cannot-tell and none_cue, retaining their counts from the accepted registered report. Any none_cue display is a separate enriched-sampling descriptive panel with its own support and no pooled primary rate; its one known positive does not support a positive-group distribution. 

## Actual time-series availability

The audio NPZ and core/ext84 JSONLs store fifteen scalar summaries per ID. These summaries support distributional comparisons; measuring F0 or intensity contours requires framewise tracks, which these files do not store.
