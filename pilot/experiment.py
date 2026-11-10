"""
MOCO2 Pilot Experiment
======================
在 Whisper ablation + bootstrap CI 的基礎上，新增：
  - BoW baseline：TF-IDF character n-gram on OCR Chinese subtitles

主要輸出：
  retrieval.json      完整結果（含 BoW）
  ablation.txt        §4.4 ablation table（含 BoW baseline 行）

Usage:
    python experiment.py
"""

import json, re, sys
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import torch
import librosa
import whisper
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import bootstrap as scipy_bootstrap

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = (Path(__file__).resolve().parents[1] / "data")
OCR_DIR     = ROOT / "ocr_corrected"
WAV_DIR     = ROOT / "wav"
EMIT2_DIR   = ROOT / "emit2"
POSE_DIR    = ROOT / "pose"
LABEL_FILE  = (Path(__file__).resolve().parents[1] / "pilot/labels.json")
OUT_JSON    = (Path(__file__).resolve().parents[1] / "pilot/retrieval.json")
OUT_TABLE   = (Path(__file__).resolve().parents[1] / "pilot/ablation.txt")
FEAT_CACHE  = (Path(__file__).resolve().parents[1] / "pilot/whisper_features_cache.npz")

LANGUAGES   = ["Amis", "Paiwan", "Truku"]
SAMPLE_RATE = 16000


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def get_segments():
    wavs = sorted(WAV_DIR.glob("*.wav"))
    return [{"id": w.stem, "lang": w.stem.split("-")[1], "wav": str(w)}
            for w in wavs]

def load_labels(segs):
    with open(LABEL_FILE, encoding="utf-8") as f:
        lmap = json.load(f)
    return [lmap.get(s["id"], "Explaining") for s in segs]


# ─────────────────────────────────────────────────────────────────────────────
# OCR text extraction (shared by both BoW and XLM-R)
# Priority: VLM-extracted clean text > noisy OCR+correction_map
# ─────────────────────────────────────────────────────────────────────────────
def extract_ocr_text(seg_id):
    p = OCR_DIR / f"{seg_id}.ocr.json"
    if not p.exists(): return ""
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    seen, texts = set(), []
    for _ts, dets in data:
        for det in dets:
            box, text, conf, label = det
            if conf < 0.8 or label == "logo": continue
            if not re.search(r"[一-鿿]", text): continue
            if text in seen or len(text) < 2: continue
            seen.add(text); texts.append(text)
    return " ".join(texts)


# ─────────────────────────────────────────────────────────────────────────────
# BoW/TF-IDF baseline (NEW in v4)
# ─────────────────────────────────────────────────────────────────────────────
def get_bow_embeddings(segs):
    """
    TF-IDF character n-grams (1–3) on OCR Chinese subtitle text.

    Why character n-grams instead of word-level:
      - Chinese text requires tokenisation for word-level; char n-grams work
        directly and capture sub-word morphological patterns.
      - sublinear_tf=True dampens high-frequency terms (same as BM25 analogue).
      - min_df=1 keeps all terms since N=30 is too small to prune.

    Returns: np.array (30, vocab_size), l2-normalized downstream.
    """
    print("[BoW/TF-IDF] Building TF-IDF features from OCR text...")
    texts = []
    for i, seg in enumerate(segs):
        t = extract_ocr_text(seg["id"])
        print(f"  [{i+1:02d}/30] {seg['id']}  ({len(t)} chars)")
        texts.append(t if t.strip() else "　")  # ideographic space for empty

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        norm="l2",          # already l2-normalised by vectorizer
    )
    X = vectorizer.fit_transform(texts)
    print(f"  Vocab size: {len(vectorizer.vocabulary_):,}  "
          f"Matrix: {X.shape}  nnz per row (avg): "
          f"{X.nnz / X.shape[0]:.0f}")
    return X.toarray()


# ─────────────────────────────────────────────────────────────────────────────
# Text (XLM-RoBERTa) — unchanged from v3
# ─────────────────────────────────────────────────────────────────────────────
def get_text_embeddings(segs):
    print("[Text] Loading XLM-RoBERTa...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    embs = []
    for i, seg in enumerate(segs):
        text = extract_ocr_text(seg["id"])
        print(f"  [{i+1:02d}/30] {seg['id']}")
        if not text.strip():
            embs.append(np.zeros(768)); continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        embs.append(emb.squeeze().cpu().numpy())
    return np.array(embs)


# ─────────────────────────────────────────────────────────────────────────────
# Audio-MFCC — unchanged from v3
# ─────────────────────────────────────────────────────────────────────────────
def get_mfcc_embeddings(segs):
    print("[Audio-MFCC] Extracting MFCC features...")
    embs = []
    for i, seg in enumerate(segs):
        print(f"  [{i+1:02d}/30] {seg['id']}")
        try:
            y, sr = librosa.load(seg["wav"], sr=SAMPLE_RATE, mono=True)
            mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            rms   = librosa.feature.rms(y=y)
            zcr   = librosa.feature.zero_crossing_rate(y)
            sc    = librosa.feature.spectral_centroid(y=y, sr=sr)
            vec   = np.concatenate([mfcc.mean(1), mfcc.std(1),
                                    delta.mean(1), delta.std(1),
                                    [rms.mean(), rms.std()],
                                    [zcr.mean(),  zcr.std()],
                                    [sc.mean(),   sc.std()]])
            embs.append(vec)
        except Exception as e:
            print(f"  ERROR: {e}"); embs.append(np.zeros(86))
    return np.array(embs)


# ─────────────────────────────────────────────────────────────────────────────
# Audio-Whisper encoder — unchanged from v3
# ─────────────────────────────────────────────────────────────────────────────
def _encode_window(model, audio_chunk, device):
    audio_chunk = whisper.pad_or_trim(audio_chunk)
    n_mels = model.dims.n_mels
    mel = whisper.log_mel_spectrogram(audio_chunk, n_mels=n_mels).to(device)
    with torch.no_grad():
        encoded = model.encoder(mel.unsqueeze(0))
    return encoded.squeeze(0).cpu().numpy()

def get_whisper_embeddings(segs, model_name="large-v3"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if FEAT_CACHE.exists():
        print(f"[Audio-Whisper] Loading from cache: {FEAT_CACHE}")
        cache = np.load(FEAT_CACHE, allow_pickle=True)
        cached_id_list = cache["ids"].tolist()
        ids_needed = {s["id"] for s in segs}
        if set(cached_id_list) == ids_needed:
            mean_embs    = np.array([cache[s["id"] + "_mean"]    for s in segs])
            meanstd_embs = np.array([cache[s["id"] + "_meanstd"] for s in segs])
            print(f"  Cache hit: {len(segs)} clips, "
                  f"dims mean={mean_embs.shape[1]}, meanstd={meanstd_embs.shape[1]}")
            return mean_embs, meanstd_embs
        print("  Cache mismatch — re-extracting.")

    print(f"[Audio-Whisper] Loading Whisper {model_name} on {device}...")
    model = whisper.load_model(model_name, device=device)
    model.eval()

    WINDOW = SAMPLE_RATE * 30
    STRIDE = SAMPLE_RATE * 25

    mean_list, meanstd_list, ids = [], [], []

    for i, seg in enumerate(segs):
        print(f"  [{i+1:02d}/30] {seg['id']} ", end="", flush=True)
        audio = whisper.load_audio(seg["wav"])
        dur   = len(audio) / SAMPLE_RATE
        all_frames = []

        if len(audio) <= WINDOW:
            frames = _encode_window(model, audio, device)
            actual_T = max(1, int(dur * 50))
            all_frames.append(frames[:actual_T])
        else:
            for start in range(0, len(audio), STRIDE):
                chunk = audio[start:start + WINDOW]
                if len(chunk) < SAMPLE_RATE * 2:
                    continue
                chunk_dur = len(chunk) / SAMPLE_RATE
                frames = _encode_window(model, chunk, device)
                actual_T = max(1, int(min(chunk_dur, 30) * 50))
                all_frames.append(frames[:actual_T])

        cat = np.concatenate(all_frames, axis=0)
        mean_v    = cat.mean(axis=0)
        meanstd_v = np.concatenate([mean_v, cat.std(axis=0)])

        mean_list.append(mean_v)
        meanstd_list.append(meanstd_v)
        ids.append(seg["id"])
        print(f"dur={dur:.0f}s  frames={cat.shape[0]}")

    mean_embs    = np.array(mean_list)
    meanstd_embs = np.array(meanstd_list)

    save_dict = {"ids": np.array(ids)}
    for sid, m, ms in zip(ids, mean_embs, meanstd_embs):
        save_dict[sid + "_mean"]    = m
        save_dict[sid + "_meanstd"] = ms
    np.savez(FEAT_CACHE, **save_dict)
    print(f"  Cached to {FEAT_CACHE}")

    return mean_embs, meanstd_embs


# ─────────────────────────────────────────────────────────────────────────────
# Prosody — unchanged from v3
# ─────────────────────────────────────────────────────────────────────────────
def _entropy(counter):
    total = sum(counter.values())
    if total == 0: return 0.0
    return -sum((c/total)*np.log2(c/total + 1e-12) for c in counter.values())

def parse_textgrid(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    intervals = []
    i = 0
    while i < len(lines):
        try:
            start = float(lines[i]); end = float(lines[i+1])
            label = lines[i+2].strip('"')
            if label and not re.fullmatch(r"[\d.]+", label):
                intervals.append((start, end, label))
            i += 3
        except (ValueError, IndexError):
            i += 1
    return intervals

def get_prosody_embeddings(segs):
    print("[Prosody] Extracting TextGrid features...")
    embs = []
    for i, seg in enumerate(segs):
        print(f"  [{i+1:02d}/30] {seg['id']}")
        tg = EMIT2_DIR / f"{seg['id']}.allo.emit2.TextGrid"
        if not tg.exists(): embs.append(np.zeros(8)); continue
        intervals = parse_textgrid(tg)
        if not intervals: embs.append(np.zeros(8)); continue
        total_dur = intervals[-1][1] - intervals[0][0]
        silence   = [iv for iv in intervals if iv[2] == "<blk>"]
        speech    = [iv for iv in intervals if iv[2] != "<blk>"]
        if not speech: embs.append(np.zeros(8)); continue
        sil_dur   = sum(e-s for s, e, _ in silence)
        sph_dur   = [e-s for s, e, _ in speech]
        phones    = [l for _, _, l in speech]
        n_bursts  = sum(1 for j in range(1, len(intervals))
                        if intervals[j-1][2] == "<blk>" and intervals[j][2] != "<blk>")
        max_sil   = max((e-s for s, e, _ in silence), default=0.0)
        vec = np.array([
            len(speech) / total_dur,
            sil_dur / total_dur,
            np.mean(sph_dur) * 1000,
            np.std(sph_dur) * 1000,
            _entropy(Counter(phones)),
            n_bursts / total_dur,
            max_sil,
            float(len(set(phones)))
        ])
        embs.append(vec)
    return np.array(embs)


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic — unchanged from v3
# ─────────────────────────────────────────────────────────────────────────────
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
                    body_cx.append(np.mean(xs))
                    body_cy.append(np.mean(ys))
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
                    face_cx.append(np.mean(xs_f))
                    face_cy.append(np.mean(ys_f))
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
        vec.append(f["lh_ratio"]); vec.append(f["rh_ratio"])
        vec.extend(f["face_std"])
    mean_body_energy = np.mean([feats_per_side["left"]["body_std"],
                                feats_per_side["right"]["body_std"]])
    lh_asym = abs(feats_per_side["left"]["lh_ratio"]
                  - feats_per_side["right"]["lh_ratio"])
    rh_asym = abs(feats_per_side["left"]["rh_ratio"]
                  - feats_per_side["right"]["rh_ratio"])
    vec.extend([mean_body_energy, lh_asym + rh_asym])
    return np.array(vec)

def get_kinematic_embeddings(segs):
    print("[Kinematic] Extracting pose features...")
    embs = []
    for i, seg in enumerate(segs):
        print(f"  [{i+1:02d}/30] {seg['id']}")
        pose_path = POSE_DIR / f"{seg['id']}.pose.json"
        if not pose_path.exists(): embs.append(np.zeros(18)); continue
        with open(pose_path) as f:
            frames = json.load(f)
        embs.append(_pose_features(frames))
    return np.array(embs)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval evaluation (exclude the query from every ranked metric)
# ─────────────────────────────────────────────────────────────────────────────
def retrieval_eval(sim_matrix, labels, k_list=(1, 3)):
    n = len(labels)
    hit_arrays = {f"P@{k}": [] for k in k_list}
    aps = []
    for i in range(n):
        sims = sim_matrix[i].copy(); sims[i] = -1
        ranked = np.argsort(sims)[::-1]
        ranked = ranked[ranked != i]
        q_label = labels[i]
        for k in k_list:
            hits = sum(1 for j in ranked[:k] if labels[j] == q_label)
            hit_arrays[f"P@{k}"].append(hits / k)
        rel = [1 if labels[j] == q_label else 0 for j in ranked]
        ap, n_rel = 0.0, 0
        for rank, r in enumerate(rel, 1):
            if r: n_rel += 1; ap += n_rel / rank
        aps.append(ap / sum(rel) if sum(rel) > 0 else 0.0)
    hit_arrays["MAP"] = aps
    return {k: np.array(v) for k, v in hit_arrays.items()}

def bootstrap_ci(hit_array, n_resamples=2000):
    arr = np.array(hit_array)
    res = scipy_bootstrap((arr,), np.mean, n_resamples=n_resamples,
                          confidence_level=0.95, method="percentile",
                          random_state=42)
    return float(arr.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)

def summary_stats(hit_dict):
    return {k: bootstrap_ci(v) for k, v in hit_dict.items()}

def cross_lingual_p1(embs, labels, segs):
    sim = cosine_similarity(embs)
    per_lang = {}
    for q_lang in LANGUAGES:
        q_idx = [i for i, s in enumerate(segs) if s["lang"] == q_lang]
        r_idx = [i for i, s in enumerate(segs) if s["lang"] != q_lang]
        hits = []
        for qi in q_idx:
            sims_row = {ri: sim[qi, ri] for ri in r_idx}
            top = sorted(sims_row, key=sims_row.get, reverse=True)[0]
            hits.append(1 if labels[top] == labels[qi] else 0)
        per_lang[q_lang] = float(np.mean(hits))
    return per_lang


# ─────────────────────────────────────────────────────────────────────────────
# Print and save ablation table (v4: adds BoW row)
# ─────────────────────────────────────────────────────────────────────────────
def fmt(mean, lo, hi):
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]"

def print_ablation_table(ablation_results, cl_results, out_path):
    sep = "-" * 88
    lines = []

    lines.append("=" * 88)
    lines.append("§4  Single-modality results + BoW baseline (v4)")
    lines.append("=" * 88)

    lines.append("\nTable A  Single-modality P@1 with bootstrap 95% CI (N=30)")
    lines.append(sep)
    lines.append(f"  {'Config':<30}  {'P@1 [95% CI]':>24}  {'P@3 [95% CI]':>24}  {'MAP':>6}")
    lines.append(sep)

    # BoW listed first as baseline, then others in descending P@1 order
    ordered_keys = [
        "bow",
        "text",
        "audio_mfcc",
        "audio_whisper_mean",
        "audio_whisper_meanstd",
        "prosody",
        "kinematic",
    ]
    for name in ordered_keys:
        if name not in ablation_results: continue
        s = ablation_results[name]
        marker = " *BoW baseline*" if name == "bow" else ""
        lines.append(f"  {name:<30}  {fmt(*s['P@1']):>24}  {fmt(*s['P@3']):>24}  {s['MAP'][0]:>6.3f}{marker}")
    lines.append(sep)
    lines.append(f"  {'majority-class baseline':<30}  {'~0.467':>24}")
    lines.append(f"  {'random baseline':<30}  {'~0.327':>24}")

    lines.append("\nTable B  Cross-lingual P@1 per source language")
    lines.append(sep)
    lines.append(f"  {'Config':<30}  {'Amis→other':>12}  {'Paiwan→other':>14}  {'Truku→other':>12}")
    lines.append(sep)
    for name in ordered_keys:
        if name not in cl_results: continue
        cl = cl_results[name]
        lines.append(f"  {name:<30}  {cl['Amis']:>12.3f}  {cl['Paiwan']:>14.3f}  {cl['Truku']:>12.3f}")
    lines.append(sep)

    lines.append("\nTable C  Key fusion comparisons (unchanged from v3, for reference)")
    lines.append(sep)
    lines.append(f"  {'Config':<38}  {'P@1 [95% CI]':>24}  {'P@3':>6}  {'MAP':>6}")
    lines.append(sep)
    fusion_keys = [
        "text+audio_mfcc",
        "text+audio_whisper_meanstd",
        "kinematic+audio_mfcc",
        "kinematic+audio_whisper_meanstd",
        "text+prosody",
        "text+kinematic",
    ]
    for name in fusion_keys:
        if name not in ablation_results: continue
        s = ablation_results[name]
        lines.append(f"  {name:<38}  {fmt(*s['P@1']):>24}  {s['P@3'][0]:>6.3f}  {s['MAP'][0]:>6.3f}")
    lines.append(sep)

    output = "\n".join(lines)
    print(output)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\n[Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MOCO2 Pilot Experiment v4  (+ BoW/TF-IDF baseline)")
    print("=" * 60)

    segs   = get_segments()
    labels = load_labels(segs)
    print(f"\nSegments: {len(segs)}   Labels: {Counter(labels)}")

    # ── Feature extraction ──────────────────────────────────────────────────
    print()
    bow_embs   = get_bow_embeddings(segs)   # NEW: BoW first (fast, no GPU)
    print()
    text_embs  = get_text_embeddings(segs)
    print()
    mfcc_embs  = get_mfcc_embeddings(segs)
    print()
    pros_embs  = get_prosody_embeddings(segs)
    print()
    kine_embs  = get_kinematic_embeddings(segs)
    print()
    whisper_mean_embs, whisper_meanstd_embs = get_whisper_embeddings(segs)

    # ── Normalize ───────────────────────────────────────────────────────────
    # Note: BoW TfidfVectorizer already outputs l2-normalized rows (norm="l2"),
    # so normalize() here is a no-op but kept for consistency.
    B   = normalize(bow_embs)
    T   = normalize(text_embs)
    A   = normalize(mfcc_embs)
    WM  = normalize(whisper_mean_embs)
    WMS = normalize(whisper_meanstd_embs)
    P   = normalize(pros_embs)
    K   = normalize(kine_embs)

    # ── Single-modality + ablation configs ──────────────────────────────────
    print("\n[Evaluating] All single-modality configs including BoW...")

    ablation_configs = {
        "bow":                            B,   # NEW
        "text":                           T,
        "audio_mfcc":                     A,
        "audio_whisper_mean":             WM,
        "audio_whisper_meanstd":          WMS,
        "prosody":                        P,
        "kinematic":                      K,
        "text+audio_mfcc":                normalize(np.hstack([T, A])),
        "text+audio_whisper_meanstd":     normalize(np.hstack([T, WMS])),
        "kinematic+audio_mfcc":           normalize(np.hstack([K, A])),
        "kinematic+audio_whisper_meanstd":normalize(np.hstack([K, WMS])),
        "text+prosody":                   normalize(np.hstack([T, P])),
        "text+kinematic":                 normalize(np.hstack([T, K])),
    }

    ablation_results = {}
    cl_results = {}
    for name, embs in ablation_configs.items():
        sim = cosine_similarity(embs)
        hit_dict = retrieval_eval(sim, labels)
        ablation_results[name] = summary_stats(hit_dict)
        if "+" not in name:
            cl_results[name] = cross_lingual_p1(embs, labels, segs)

    print_ablation_table(ablation_results, cl_results, OUT_TABLE)

    # ── Full fusion (all 15 combos, MFCC as canonical audio) ─────────────────
    print("\n[Full Fusion] All 15 combinations using audio_mfcc...")
    modalities_v4 = {"text": T, "audio": A, "prosody": P, "kinematic": K}
    all_combos = {}
    for r in range(1, 5):
        for combo in combinations(modalities_v4.keys(), r):
            name  = "+".join(combo)
            fused = normalize(np.hstack([modalities_v4[k] for k in combo]))
            all_combos[name] = fused

    full_results = {}
    print(f"  {'Combination':<42} {'P@1':>7}  {'P@3':>7}  {'MAP':>7}")
    for name, embs in all_combos.items():
        sim = cosine_similarity(embs)
        s = summary_stats(retrieval_eval(sim, labels))
        full_results[name] = s
        print(f"  {name:<42} {s['P@1'][0]:>7.3f}  {s['P@3'][0]:>7.3f}  {s['MAP'][0]:>7.3f}")

    # ── Save all results ─────────────────────────────────────────────────────
    def ser(d):
        return {k: {"mean": v[0], "ci_low": v[1], "ci_high": v[2]}
                for k, v in d.items()}

    out = {
        "ablation_audio_plus_bow": {k: ser(v) for k, v in ablation_results.items()},
        "cross_lingual":           cl_results,
        "full_fusion_whisper":     {k: ser(v) for k, v in full_results.items()},
        "label_distribution":      dict(Counter(labels)),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT_JSON}")
    print("Done!")


if __name__ == "__main__":
    main()
