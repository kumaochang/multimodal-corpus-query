"""
Step 4 supplementary analyses:
  (1) paired bootstrap tests on per-query P@1 differences (text vs audio/prosody/kinematic)
  (2) complementarity table (per-query top-1 correctness) + oracle upper bound
  (3) weight sweep for text+{prosody,audio,kinematic}, alpha in {0.25,0.5,1,2,4}
      -- kinematic included as BOTH 18-dim and 14-dim composition-robust variants
  (4) silhouette scores per modality, by pragmatic label and by language

All feature extractors are VERBATIM copies from pilot/experiment.py, repointed at
dissertation/data. Retrieval / bootstrap conventions are identical to the paper.
A fidelity gate reproduces the paper's single-modality numbers before proceeding.

Run:  python analysis/step4_supplementary.py
Outputs: analysis/step4_results.md (+ stdout);  embeddings cached to analysis/_emb_cache.npz
"""
import sys, json, re
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.stats import bootstrap as scipy_bootstrap

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data"
OCR_DIR   = DATA / "ocr_corrected"
WAV_DIR   = DATA / "wav"
EMIT2_DIR = DATA / "emit2"
POSE_DIR  = DATA / "pose"
LABELFILE = ROOT / "pilot" / "labels.json"
REF_JSON  = ROOT / "pilot" / "retrieval.json"
CACHE     = ROOT / "analysis" / "_emb_cache.npz"
OUT_MD    = ROOT / "analysis" / "step4_results.md"
LANGUAGES = ["Amis", "Paiwan", "Truku"]
SAMPLE_RATE = 16000


# ── data / extractors (VERBATIM from pilot/experiment.py) ────────────────────
def get_segments():
    wavs = sorted(WAV_DIR.glob("*.wav"))
    return [{"id": w.stem, "lang": w.stem.split("-")[1], "wav": str(w)} for w in wavs]

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

def get_text_embeddings(segs):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    embs = []
    for seg in segs:
        text = extract_ocr_text(seg["id"])
        if not text.strip(): embs.append(np.zeros(768)); continue
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        embs.append(emb.squeeze().cpu().numpy())
    return np.array(embs)

def get_mfcc_embeddings(segs):
    import librosa
    embs = []
    for seg in segs:
        try:
            y, sr = librosa.load(seg["wav"], sr=SAMPLE_RATE, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            rms = librosa.feature.rms(y=y); zcr = librosa.feature.zero_crossing_rate(y)
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)
            vec = np.concatenate([mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1),
                                  [rms.mean(), rms.std()], [zcr.mean(), zcr.std()], [sc.mean(), sc.std()]])
            embs.append(vec)
        except Exception as e:
            print("  MFCC ERR", e); embs.append(np.zeros(86))
    return np.array(embs)

def _entropy(counter):
    total = sum(counter.values())
    if total == 0: return 0.0
    return -sum((c/total)*np.log2(c/total + 1e-12) for c in counter.values())

def parse_textgrid(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    intervals = []; i = 0
    while i < len(lines):
        try:
            start = float(lines[i]); end = float(lines[i+1]); label = lines[i+2].strip('"')
            if label and not re.fullmatch(r"[\d.]+", label): intervals.append((start, end, label))
            i += 3
        except (ValueError, IndexError): i += 1
    return intervals

def get_prosody_embeddings(segs):
    embs = []
    for seg in segs:
        tg = EMIT2_DIR / f"{seg['id']}.allo.emit2.TextGrid"
        if not tg.exists(): embs.append(np.zeros(8)); continue
        intervals = parse_textgrid(tg)
        if not intervals: embs.append(np.zeros(8)); continue
        total_dur = intervals[-1][1] - intervals[0][0]
        silence = [iv for iv in intervals if iv[2] == "<blk>"]
        speech = [iv for iv in intervals if iv[2] != "<blk>"]
        if not speech: embs.append(np.zeros(8)); continue
        sil_dur = sum(e-s for s, e, _ in silence); sph_dur = [e-s for s, e, _ in speech]
        phones = [l for _, _, l in speech]
        n_bursts = sum(1 for j in range(1, len(intervals))
                       if intervals[j-1][2] == "<blk>" and intervals[j][2] != "<blk>")
        max_sil = max((e-s for s, e, _ in silence), default=0.0)
        vec = np.array([len(speech)/total_dur, sil_dur/total_dur, np.mean(sph_dur)*1000,
                        np.std(sph_dur)*1000, _entropy(Counter(phones)), n_bursts/total_dur,
                        max_sil, float(len(set(phones)))])
        embs.append(vec)
    return np.array(embs)

def _pose_features(frames):
    total = len(frames); feats_per_side = {}
    for side in ("left", "right"):
        body_cx, body_cy, face_cx, face_cy = [], [], [], []
        lh_present = rh_present = 0
        for fr in frames:
            person = fr[side]
            if person is None: continue
            pts = person.get("pose") or []
            if pts:
                xs = [p[0] for p in pts if len(p) >= 2]; ys = [p[1] for p in pts if len(p) >= 2]
                if xs: body_cx.append(np.mean(xs)); body_cy.append(np.mean(ys))
            if person.get("left_hand"): lh_present += 1
            if person.get("right_hand"): rh_present += 1
            face = person.get("face") or {}; pts_f = []
            for part in face.values():
                if isinstance(part, list): pts_f.extend(part)
            if pts_f:
                xs_f = [p[0] for p in pts_f if len(p) >= 2]; ys_f = [p[1] for p in pts_f if len(p) >= 2]
                if xs_f: face_cx.append(np.mean(xs_f)); face_cy.append(np.mean(ys_f))
        feats_per_side[side] = {
            "body_mean": [np.mean(body_cx) if body_cx else 0.5, np.mean(body_cy) if body_cy else 0.5],
            "body_std": [np.std(body_cx) if len(body_cx) > 1 else 0.0, np.std(body_cy) if len(body_cy) > 1 else 0.0],
            "lh_ratio": lh_present/total, "rh_ratio": rh_present/total,
            "face_std": [np.std(face_cx) if len(face_cx) > 1 else 0.0, np.std(face_cy) if len(face_cy) > 1 else 0.0]}
    vec = []
    for side in ("left", "right"):
        f = feats_per_side[side]
        vec.extend(f["body_mean"]); vec.extend(f["body_std"]); vec.append(f["lh_ratio"]); vec.append(f["rh_ratio"]); vec.extend(f["face_std"])
    mean_body_energy = np.mean([feats_per_side["left"]["body_std"], feats_per_side["right"]["body_std"]])
    lh_asym = abs(feats_per_side["left"]["lh_ratio"] - feats_per_side["right"]["lh_ratio"])
    rh_asym = abs(feats_per_side["left"]["rh_ratio"] - feats_per_side["right"]["rh_ratio"])
    vec.extend([mean_body_energy, lh_asym + rh_asym])
    return np.array(vec)

def get_kinematic_embeddings(segs):
    embs = []
    for seg in segs:
        p = POSE_DIR / f"{seg['id']}.pose.json"
        if not p.exists(): embs.append(np.zeros(18)); continue
        embs.append(_pose_features(json.load(open(p))))
    return np.array(embs)


# ── retrieval / bootstrap (same contract as pilot/experiment.py) ────────────
def retrieval_eval(sim_matrix, labels, k_list=(1, 3)):
    n = len(labels); hit_arrays = {f"P@{k}": [] for k in k_list}; aps = []
    for i in range(n):
        sims = sim_matrix[i].copy(); sims[i] = -1; ranked = np.argsort(sims)[::-1]; q = labels[i]
        ranked = ranked[ranked != i]
        for k in k_list:
            hits = sum(1 for j in ranked[:k] if labels[j] == q); hit_arrays[f"P@{k}"].append(hits/k)
        rel = [1 if labels[j] == q else 0 for j in ranked]; ap = 0.0; nr = 0
        for rank, r in enumerate(rel, 1):
            if r: nr += 1; ap += nr/rank
        aps.append(ap/sum(rel) if sum(rel) > 0 else 0.0)
    hit_arrays["MAP"] = aps
    return {k: np.array(v) for k, v in hit_arrays.items()}

def bootstrap_ci(hit_array, n_resamples=2000):
    arr = np.array(hit_array)
    res = scipy_bootstrap((arr,), np.mean, n_resamples=n_resamples, confidence_level=0.95,
                          method="percentile", random_state=42)
    return float(arr.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)

def per_query_p1(embs_norm, labels):
    return retrieval_eval(cosine_similarity(embs_norm), labels)["P@1"]

def summary(embs_norm, labels):
    r = retrieval_eval(cosine_similarity(embs_norm), labels)
    return {k: bootstrap_ci(r[k]) for k in ("P@1", "P@3", "MAP")}


# ── paired bootstrap on per-query P@1 differences ───────────────────────────
def paired_bootstrap(p1_a, p1_b, n_resamples=2000, seed=42):
    d = np.asarray(p1_a) - np.asarray(p1_b)
    rng = np.random.default_rng(seed); n = len(d); means = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n); means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    frac_le0 = np.mean(means <= 0); frac_ge0 = np.mean(means >= 0)
    p_two = min(1.0, 2 * min(frac_le0, frac_ge0))
    return float(d.mean()), float(lo), float(hi), float(p_two)


def main():
    segs = get_segments()
    ids = [s["id"] for s in segs]; langs = [s["lang"] for s in segs]
    lmap = json.load(open(LABELFILE, encoding="utf-8"))
    labels = [lmap[i] for i in ids]
    print(f"Segments: {len(segs)}  labels: {Counter(labels)}")

    # ── extract (cache text/mfcc/prosody; kinematic cheap) ───────────────────
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["ids"]) == ids:
            T, A, P = z["T"], z["A"], z["P"]; print("[cache] loaded T/A/P")
        else:
            T = A = P = None
    else:
        T = A = P = None
    if T is None:
        print("[extract] text (XLM-R)..."); T = get_text_embeddings(segs)
        print("[extract] mfcc...");         A = get_mfcc_embeddings(segs)
        print("[extract] prosody...");      P = get_prosody_embeddings(segs)
        np.savez(CACHE, ids=np.array(ids), T=T, A=A, P=P); print(f"[cache] saved {CACHE}")
    K18 = get_kinematic_embeddings(segs)
    K14 = np.delete(K18, [0, 1, 8, 9], axis=1)  # composition-robust

    reps = {"text": normalize(T), "audio": normalize(A), "prosody": normalize(P),
            "kinematic": normalize(K18), "kinematic14": normalize(K14)}
    p1 = {k: per_query_p1(v, labels) for k, v in reps.items()}

    # ── fidelity gate vs paper ───────────────────────────────────────────────
    ref = json.load(open(REF_JSON, encoding="utf-8"))["ablation_audio_plus_bow"]
    L = ["# Step 4 supplementary analyses\n"]
    L.append("## Fidelity gate (re-extracted vs paper retrieval.json)\n")
    L.append("| modality | P@1 re-extracted | P@1 paper | match |")
    L.append("|---|---|---|---|")
    gate = {"text": "text", "audio": "audio_mfcc", "prosody": "prosody", "kinematic": "kinematic"}
    ok = True
    for r, key in gate.items():
        mine = summary(reps[r], labels)["P@1"][0]; paper = ref[key]["P@1"]["mean"]
        m = abs(mine - paper) < 1e-6; ok &= m
        L.append(f"| {r} | {mine:.4f} | {paper:.4f} | {'OK' if m else 'DIFF'} |")
    L.append(f"\nAll single-modality P@1 reproduce paper exactly: **{ok}**\n")

    # ── (1) paired bootstrap ─────────────────────────────────────────────────
    L.append("\n## (1) Paired bootstrap on per-query P@1 differences (n=2000, seed=42)\n")
    L.append("| contrast | mean diff (A-B) | 95% CI | two-sided p | sig |")
    L.append("|---|---|---|---|---|")
    contrasts = [("text", "audio"), ("text", "prosody"), ("text", "kinematic"),
                 ("kinematic", "audio"), ("kinematic14", "kinematic")]
    for a, b in contrasts:
        md, lo, hi, pv = paired_bootstrap(p1[a], p1[b])
        L.append(f"| {a} vs {b} | {md:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {pv:.3f} | {'*' if pv < 0.05 else 'ns'} |")

    # ── (2) complementarity + oracle ─────────────────────────────────────────
    L.append("\n## (2) Complementarity (per-query top-1 correctness) + oracle\n")
    base4 = ["text", "audio", "prosody", "kinematic"]
    correct = {m: (p1[m] == 1.0) for m in base4}
    L.append("| pair (A,B) | A right, B wrong | B right, A wrong | both right | both wrong |")
    L.append("|---|---|---|---|---|")
    for a in ["text"]:
        for b in ["audio", "prosody", "kinematic"]:
            ca, cb = correct[a], correct[b]
            L.append(f"| ({a},{b}) | {int(np.sum(ca & ~cb))} | {int(np.sum(~ca & cb))} | "
                     f"{int(np.sum(ca & cb))} | {int(np.sum(~ca & ~cb))} |")
    # also kinematic vs prosody/audio (non-text complementarity)
    for a, b in [("kinematic", "prosody"), ("kinematic", "audio"), ("prosody", "audio")]:
        ca, cb = correct[a], correct[b]
        L.append(f"| ({a},{b}) | {int(np.sum(ca & ~cb))} | {int(np.sum(~ca & cb))} | "
                 f"{int(np.sum(ca & cb))} | {int(np.sum(~ca & ~cb))} |")
    any4 = np.zeros(len(labels), dtype=bool)
    for m in base4: any4 |= correct[m]
    L.append(f"\nPer-modality top-1 correct counts (of 30): "
             f"{ {m: int(correct[m].sum()) for m in base4} }")
    L.append(f"\n**Oracle upper bound (any of 4 single modalities top-1 correct): "
             f"{any4.sum()}/30 = {any4.mean():.3f}**")
    # oracle for text + one other
    for b in ["audio", "prosody", "kinematic"]:
        o = correct["text"] | correct[b]
        L.append(f"\n- oracle(text, {b}) = {o.sum()}/30 = {o.mean():.3f}")

    # ── (3) weight sweep ─────────────────────────────────────────────────────
    L.append("\n## (3) Weight sweep: normalize(hstack([text, alpha*X])); text weight = 1\n")
    alphas = [0.25, 0.5, 1, 2, 4]
    sweep_specs = [("text+prosody", P), ("text+audio", A),
                   ("text+kinematic18", K18), ("text+kinematic14", K14)]
    Tn = normalize(T)
    L.append("| fusion | " + " | ".join(f"a={a}" for a in alphas) + " |")
    L.append("|" + "---|" * (len(alphas) + 1))
    for name, X in sweep_specs:
        Xn = normalize(X); row = []
        for a in alphas:
            fused = normalize(np.hstack([Tn, a * Xn]))
            row.append(f"{summary(fused, labels)['P@1'][0]:.3f}")
        L.append(f"| {name} (P@1) | " + " | ".join(row) + " |")
    L.append(f"\n(reference single-modality P@1: text={summary(Tn,labels)['P@1'][0]:.3f}, "
             f"kinematic18={summary(normalize(K18),labels)['P@1'][0]:.3f}, "
             f"kinematic14={summary(normalize(K14),labels)['P@1'][0]:.3f})")
    # also MAP sweep for the two kinematic variants (to see fusion quality shift)
    L.append("\nMAP sweep (kinematic variants):\n")
    L.append("| fusion | " + " | ".join(f"a={a}" for a in alphas) + " |")
    L.append("|" + "---|" * (len(alphas) + 1))
    for name, X in [("text+kinematic18", K18), ("text+kinematic14", K14)]:
        Xn = normalize(X); row = []
        for a in alphas:
            fused = normalize(np.hstack([Tn, a * Xn]))
            row.append(f"{summary(fused, labels)['MAP'][0]:.3f}")
        L.append(f"| {name} (MAP) | " + " | ".join(row) + " |")

    # ── (4) silhouette ───────────────────────────────────────────────────────
    L.append("\n## (4) Silhouette scores (cosine)\n")
    lab_id = {l: i for i, l in enumerate(sorted(set(labels)))}
    y_lab = [lab_id[l] for l in labels]
    y_lang = [LANGUAGES.index(l) for l in langs]
    L.append("| modality | silhouette by pragmatic label | silhouette by language |")
    L.append("|---|---|---|")
    for m in ["text", "audio", "prosody", "kinematic", "kinematic14"]:
        s_lab = silhouette_score(reps[m], y_lab, metric="cosine")
        s_lng = silhouette_score(reps[m], y_lang, metric="cosine")
        L.append(f"| {m} | {s_lab:+.3f} | {s_lng:+.3f} |")
    L.append("\n(higher by-label = clusters track pragmatics; higher by-language = clusters track language/session)")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    # stdout
    print("\n".join(x for x in L if not x.startswith("|") or x.count("|") < 3 or "---" not in x))
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
