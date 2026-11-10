"""
Build demo_data.json for the multimodal corpus demo (corpus/demo.html).

Re-uses the exact feature-extraction logic of pilot/experiment.py (v4) so that
similarity numbers shown in the demo are consistent with the paper, then adds
display-oriented data (subtitle text, ASR transcript, durations, PCA maps).

Outputs: corpus/demo_data.json

Usage:
    python corpus/build_demo_data.py
"""

import json, re, sys, wave
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT       = Path(__file__).resolve().parent.parent
DATA       = ROOT / "data"
OCR_DIR    = DATA / "ocr_corrected"
OCR_VLM    = DATA / "ocr_vlm"
WAV_DIR    = DATA / "wav"
EMIT2_DIR  = DATA / "emit2"
POSE_DIR   = DATA / "pose"
MERGED_DIR = DATA / "output_merged_dedup"
LABEL_FILE = ROOT / "pilot" / "labels.json"
FEAT_CACHE = ROOT / "pilot" / "whisper_features_cache.npz"
RETRIEVAL  = ROOT / "pilot" / "retrieval.json"
OUT_JSON   = Path(__file__).resolve().parent / "demo_data.json"

LABEL_ZH = {
    "Explaining":  "解說／教導",
    "Celebrating": "慶祝／歡慶",
    "Urging":      "呼籲／勸告",
    "Reporting":   "報導／陳述",
    "Comforting":  "關懷／慰問",
}


# ── segments ─────────────────────────────────────────────────────────────────
def get_segments():
    wavs = sorted(WAV_DIR.glob("*.wav"))
    return [{"id": w.stem, "lang": w.stem.split("-")[1], "wav": w} for w in wavs]


def wav_duration(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


# ── text sources ─────────────────────────────────────────────────────────────
def extract_ocr_text(seg_id):
    """Same as pilot/experiment.py — used for embeddings (paper-consistent)."""
    p = OCR_DIR / f"{seg_id}.ocr.json"
    if not p.exists():
        return ""
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    seen, texts = set(), []
    for _ts, dets in data:
        for det in dets:
            box, text, conf, label = det
            if conf < 0.8 or label == "logo":
                continue
            if not re.search(r"[一-鿿]", text):
                continue
            if text in seen or len(text) < 2:
                continue
            seen.add(text)
            texts.append(text)
    return " ".join(texts)


def vlm_subtitle_text(seg_id):
    """VLM-cleaned subtitle text — used for display / keyword search."""
    p = OCR_VLM / f"{seg_id}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return extract_ocr_text(seg_id)


def asr_transcript(seg_id):
    p = MERGED_DIR / f"{seg_id}_merged.json"
    if not p.exists():
        return ""
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("full_text", "").strip()


# ── feature extractors (identical logic to pilot/experiment.py) ─────────────
def get_bow_embeddings(segs):
    print("[BoW] TF-IDF char n-grams on OCR text")
    texts = [extract_ocr_text(s["id"]) or "　" for s in segs]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                          sublinear_tf=True, min_df=1, norm="l2")
    return vec.fit_transform(texts).toarray()


def get_text_embeddings(segs):
    print("[Text] XLM-RoBERTa embeddings")
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    embs = []
    for i, seg in enumerate(segs):
        text = extract_ocr_text(seg["id"])
        print(f"  [{i+1:02d}/30] {seg['id']}")
        if not text.strip():
            embs.append(np.zeros(768))
            continue
        inputs = tok(text, return_tensors="pt", truncation=True,
                     max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        embs.append(emb.squeeze().cpu().numpy())
    return np.array(embs)


def get_mfcc_embeddings(segs):
    print("[Audio-MFCC]")
    import librosa
    embs = []
    for i, seg in enumerate(segs):
        print(f"  [{i+1:02d}/30] {seg['id']}")
        y, sr = librosa.load(seg["wav"], sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        vec = np.concatenate([mfcc.mean(1), mfcc.std(1),
                              delta.mean(1), delta.std(1),
                              [rms.mean(), rms.std()],
                              [zcr.mean(), zcr.std()],
                              [sc.mean(), sc.std()]])
        embs.append(vec)
    return np.array(embs)


def get_whisper_embeddings(segs):
    print("[Audio-Whisper] from cache")
    cache = np.load(FEAT_CACHE, allow_pickle=True)
    assert set(cache["ids"].tolist()) == {s["id"] for s in segs}, "cache mismatch"
    return np.array([cache[s["id"] + "_meanstd"] for s in segs])


def _entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * np.log2(c / total + 1e-12) for c in counter.values())


def parse_textgrid(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    intervals = []
    i = 0
    while i < len(lines):
        try:
            start = float(lines[i]); end = float(lines[i + 1])
            label = lines[i + 2].strip('"')
            if label and not re.fullmatch(r"[\d.]+", label):
                intervals.append((start, end, label))
            i += 3
        except (ValueError, IndexError):
            i += 1
    return intervals


def get_prosody_embeddings(segs):
    print("[Prosody] TextGrid features")
    embs = []
    for seg in segs:
        tg = EMIT2_DIR / f"{seg['id']}.allo.emit2.TextGrid"
        if not tg.exists():
            embs.append(np.zeros(8)); continue
        intervals = parse_textgrid(tg)
        if not intervals:
            embs.append(np.zeros(8)); continue
        total_dur = intervals[-1][1] - intervals[0][0]
        silence = [iv for iv in intervals if iv[2] == "<blk>"]
        speech = [iv for iv in intervals if iv[2] != "<blk>"]
        if not speech:
            embs.append(np.zeros(8)); continue
        sil_dur = sum(e - s for s, e, _ in silence)
        sph_dur = [e - s for s, e, _ in speech]
        phones = [l for _, _, l in speech]
        n_bursts = sum(1 for j in range(1, len(intervals))
                       if intervals[j - 1][2] == "<blk>" and intervals[j][2] != "<blk>")
        max_sil = max((e - s for s, e, _ in silence), default=0.0)
        embs.append(np.array([
            len(speech) / total_dur, sil_dur / total_dur,
            np.mean(sph_dur) * 1000, np.std(sph_dur) * 1000,
            _entropy(Counter(phones)), n_bursts / total_dur,
            max_sil, float(len(set(phones)))]))
    return np.array(embs)


def _pose_features(frames):
    total = len(frames)
    feats_per_side = {}
    for side in ("left", "right"):
        body_cx, body_cy, face_cx, face_cy = [], [], [], []
        lh_present = rh_present = 0
        for fr in frames:
            person = fr[side]
            if person is None:
                continue
            pts = person.get("pose") or []
            if pts:
                xs = [p[0] for p in pts if len(p) >= 2]
                ys = [p[1] for p in pts if len(p) >= 2]
                if xs:
                    body_cx.append(np.mean(xs)); body_cy.append(np.mean(ys))
            if person.get("left_hand"):
                lh_present += 1
            if person.get("right_hand"):
                rh_present += 1
            face = person.get("face") or {}
            pts_f = []
            for part in face.values():
                if isinstance(part, list):
                    pts_f.extend(part)
            if pts_f:
                xs_f = [p[0] for p in pts_f if len(p) >= 2]
                ys_f = [p[1] for p in pts_f if len(p) >= 2]
                if xs_f:
                    face_cx.append(np.mean(xs_f)); face_cy.append(np.mean(ys_f))
        feats_per_side[side] = {
            "body_mean": [np.mean(body_cx) if body_cx else 0.5,
                          np.mean(body_cy) if body_cy else 0.5],
            "body_std": [np.std(body_cx) if len(body_cx) > 1 else 0.0,
                         np.std(body_cy) if len(body_cy) > 1 else 0.0],
            "lh_ratio": lh_present / total,
            "rh_ratio": rh_present / total,
            "face_std": [np.std(face_cx) if len(face_cx) > 1 else 0.0,
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
    lh_asym = abs(feats_per_side["left"]["lh_ratio"] - feats_per_side["right"]["lh_ratio"])
    rh_asym = abs(feats_per_side["left"]["rh_ratio"] - feats_per_side["right"]["rh_ratio"])
    vec.extend([mean_body_energy, lh_asym + rh_asym])
    return np.array(vec)


def get_kinematic_embeddings(segs):
    print("[Kinematic] pose features")
    embs = []
    for seg in segs:
        pose_path = POSE_DIR / f"{seg['id']}.pose.json"
        if not pose_path.exists():
            embs.append(np.zeros(18)); continue
        with open(pose_path) as f:
            frames = json.load(f)
        embs.append(_pose_features(frames))
    return np.array(embs)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    segs = get_segments()
    assert len(segs) == 30, f"expected 30 clips, got {len(segs)}"
    with open(LABEL_FILE, encoding="utf-8") as f:
        lmap = json.load(f)

    print(f"Clips: {len(segs)}")

    B = normalize(get_bow_embeddings(segs))
    T = normalize(get_text_embeddings(segs))
    A = normalize(get_mfcc_embeddings(segs))
    W = normalize(get_whisper_embeddings(segs))
    P = normalize(get_prosody_embeddings(segs))
    K = normalize(get_kinematic_embeddings(segs))
    F = normalize(np.hstack([T, A, P, K]))  # 4-modality late fusion (paper's canonical fusion)

    modalities = {
        "text":      {"emb": T, "name_zh": "文字 (XLM-R 語意向量)",   "name_en": "Text (XLM-RoBERTa)"},
        "bow":       {"emb": B, "name_zh": "文字 (詞袋 TF-IDF)",     "name_en": "Text (bag-of-words TF-IDF)"},
        "audio":     {"emb": A, "name_zh": "聲學 (MFCC)",            "name_en": "Audio (MFCC)"},
        "whisper":   {"emb": W, "name_zh": "聲學 (Whisper encoder)", "name_en": "Audio (Whisper encoder)"},
        "prosody":   {"emb": P, "name_zh": "韻律 (停頓/語速)",        "name_en": "Prosody (pause/rate)"},
        "kinematic": {"emb": K, "name_zh": "肢體動作 (骨架姿態)",      "name_en": "Kinematic (pose)"},
        "fusion":    {"emb": F, "name_zh": "多模態融合 (4 模態)",      "name_en": "Fusion (text+audio+prosody+kinematic)"},
    }

    sim, pca2d = {}, {}
    for key, m in modalities.items():
        S = cosine_similarity(m["emb"])
        # full precision: prosody/whisper/MFCC sims concentrate near 1.0, so
        # any rounding creates rank flips vs. the paper's numbers
        sim[key] = S.tolist()
        coords = PCA(n_components=2, random_state=42).fit_transform(m["emb"])
        # scale to [0,1] per axis for easy plotting
        lo, hi = coords.min(0), coords.max(0)
        coords = (coords - lo) / np.where(hi - lo == 0, 1, hi - lo)
        pca2d[key] = np.round(coords, 4).tolist()

    clips = []
    for seg in segs:
        sid = seg["id"]
        date = sid.split("-")[2]
        clips.append({
            "id": sid,
            "lang": seg["lang"],
            "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
            "label": lmap.get(sid, "Explaining"),
            "label_zh": LABEL_ZH.get(lmap.get(sid, "Explaining"), ""),
            "duration": round(wav_duration(seg["wav"]), 1),
            "video": f"../data/h264/{sid}.mp4",
            "zh_text": vlm_subtitle_text(sid),
            "asr_text": asr_transcript(sid),
        })

    metrics = {}
    if RETRIEVAL.exists():
        with open(RETRIEVAL, encoding="utf-8") as f:
            r = json.load(f)
        metrics = {
            "ablation": r.get("ablation_audio_plus_bow", {}),
            "cross_lingual": r.get("cross_lingual", {}),
        }

    out = {
        "clips": clips,
        "modalities": {k: {"name_zh": m["name_zh"], "name_en": m["name_en"], "dim": int(m["emb"].shape[1])}
                       for k, m in modalities.items()},
        "sim": sim,
        "pca2d": pca2d,
        "metrics": metrics,
        "label_zh": LABEL_ZH,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[Saved] {OUT_JSON}  ({OUT_JSON.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
