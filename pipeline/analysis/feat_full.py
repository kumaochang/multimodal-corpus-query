"""Chapter 5 features for every utterance of the registered sample (214 clips with a tree).

Same settings as the frozen item-level programs and Study 1, applied to all utterances:
  whisper   openai-whisper large-v3 encoder, utterance audio padded/trimmed to 30 s, mean of the
            first int(min(dur, 30) * 50) frames (item5/feat_audio.py; Study 1 _encode_window).
  prosody   the 15 features of pipeline/stages/prosody.py, joined from prosody.jsonl (NaN kept).
  xlmr_sub  translation (subtitle) modality: FacebookAI/xlm-roberta-base, max_length 512, masked
            mean of last_hidden_state, on the Amendment 7 utterance subtitle text
            (<clip>.subtitles.json "subtitle_text"); empty -> zero vector (Amendment 4 B2, 7 item 2).
  xlmr_tr   transcription modality: the same model on the Omnilingual transcript
            (<clip>.jsonl "text"); five covered languages; Tsou and empty -> zero vector.
  tfidf_*   non-neural reference for each text modality: char_wb 1-3, sublinear_tf, l2, fitted
            once on all inputs of that modality in the sample (Amendment 4 B2; for transcripts the
            fit of item5/feat_text.py: registered sample, five covered languages). Saved sparse.
  mfcc86    the clip-level study's 86-d spectral summary (pilot/experiment.py get_mfcc_embeddings:
            20 MFCC mean/sd, their delta mean/sd, RMS mean/sd, ZCR mean/sd, spectral centroid
            mean/sd; 80 of the 86 dims are MFCC and delta), on the utterance audio, no per-utterance
            normalization. Two departures, needed only for short utterances and counted:
            (a) audio shorter than 2048 samples is zero-padded to 2048 (librosa's centred reflect
            padding fails below n_fft / 2); (b) when the MFCC sequence has fewer than 9 frames,
            delta uses mode="nearest" instead of the default interpolation. All other utterances
            use the recipe unchanged.
            Role in Chapter 5 (fixed 2026-10-18, before any Chapter 5 retrieval was computed): a
            stand-alone acoustic baseline and a source-clustering diagnostic beside Whisper; it
            enters no fusion and not the judged-directive probe. Standardization (global per-dim
            z-score) is applied in the evaluation, not here.
Kinematics are not recomputed: unit_table.py's units_k.npz holds them.

Resumable: per-clip parts under <out>/parts/ (whisper, mfcc86, xlmr_*); a finished clip is skipped.
After all clips: TF-IDF fit and assembly into <out>/features.npz (+ tfidf_sub.npz, tfidf_tr.npz).
Prints counts only, never text.

Usage (item5 environment, GPU; long run, use nohup):
  python pipeline/analysis/feat_full.py --clips data/study3_runs/labels/clips216.txt \
    --trees work/data/aligned_trees --wav work/data/wav --prosody data/study3_runs/prosody.jsonl \
    --subtitles data/study3_runs/subtitles --transcripts data/study3_runs/transcripts \
    --out data/study3_runs/features_full
Flags --no-whisper / --no-xlmr / --no-mfcc are for plumbing runs only.
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_units import load_utterances  # noqa: E402

SR = 16000
PROSODY = ["dur_s", "voiced_ratio", "f0_med_hz", "f0_range_st", "f0_sd_st", "f0_slope_st_s",
           "f0_final_st", "f0_end_slope_st_s", "int_mean_db", "int_range_db", "int_end_drop_db",
           "pause_ratio", "pauses_per_s", "max_pause_s", "voiced_runs_per_s"]
COVERED = {"Amis", "Atayal", "Paiwan", "Truku", "Yami"}
PLACEHOLDER = "　"
N_FFT, DELTA_W = 2048, 9


def mfcc86(y, tot):
    """Clip-level study recipe on one utterance; returns the 86-d vector."""
    import librosa
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y))); tot["mfcc_zero_padded"] += 1
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    if mfcc.shape[1] >= DELTA_W:
        delta = librosa.feature.delta(mfcc)
    else:
        delta = librosa.feature.delta(mfcc, width=DELTA_W, mode="nearest"); tot["mfcc_delta_nearest"] += 1
    rms = librosa.feature.rms(y=y); zcr = librosa.feature.zero_crossing_rate(y)
    sc = librosa.feature.spectral_centroid(y=y, sr=SR)
    return np.concatenate([mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1),
                           [rms.mean(), rms.std()], [zcr.mean(), zcr.std()], [sc.mean(), sc.std()]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--prosody", required=True)
    ap.add_argument("--subtitles", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-whisper", action="store_true")
    ap.add_argument("--no-xlmr", action="store_true")
    ap.add_argument("--no-mfcc", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    clips = [l.strip() for l in Path(a.clips).read_text(encoding="utf-8").splitlines() if l.strip()]
    tot = Counter()

    wmodel = tok = xmod = None; dev = "cpu"; rev = None
    if not (a.no_whisper and a.no_xlmr):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not a.no_whisper:
        import whisper
        wmodel = whisper.load_model("large-v3", device=dev).eval()
    if not a.no_xlmr:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        xmod = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base").eval().to(dev)
        rev = getattr(xmod.config, "_commit_hash", None)

    def xlmr(texts):
        import torch
        V = np.zeros((len(texts), 768), np.float32); trunc = 0
        for i, t in enumerate(texts):
            if not t.strip():
                continue
            trunc += len(tok(t)["input_ids"]) > 512
            inp = {k: v.to(dev) for k, v in tok(t, return_tensors="pt", truncation=True, max_length=512).items()}
            with torch.no_grad():
                o = xmod(**inp)
            m = inp["attention_mask"].unsqueeze(-1).float()
            V[i] = ((o.last_hidden_state * m).sum(1) / m.sum(1)).squeeze().cpu().numpy()
        return V, trunc

    def texts_for(clip, uids):
        sp = Path(a.subtitles) / f"{clip}.subtitles.json"
        sub = {u["utt_id"]: u["subtitle_text"] for u in json.loads(sp.read_text(encoding="utf-8"))["utterances"]} if sp.exists() else {}
        tp = Path(a.transcripts) / f"{clip}.jsonl"
        tr = {}
        if tp.exists():
            for l in tp.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l); tr[r["utt_id"]] = r.get("text") or ""
        return [sub.get(u, "") for u in uids], [tr.get(u, "") for u in uids], sp.exists(), tp.exists()

    t0 = time.time()
    for clip in clips:
        tree = Path(a.trees) / f"{clip}.aligntree.TextGrid"
        if not tree.exists():
            tot["clips_without_tree"] += 1; continue
        pf = parts / f"{clip}.npz"
        if pf.exists():
            tot["clips_skipped_done"] += 1; continue
        utts = load_utterances(tree, clip)
        uids = [u for u, _, _ in utts]
        W = np.zeros((len(utts), 1280), np.float32); M = np.zeros((len(utts), 86), np.float32)
        if not (a.no_whisper and a.no_mfcc):
            import soundfile as sf
            x_all, sr = sf.read(str(Path(a.wav) / f"{clip}.wav"), dtype="float32")
            assert sr == SR
            if x_all.ndim > 1: x_all = x_all.mean(1)
            for i, (u, s, e) in enumerate(utts):
                x = x_all[int(s * SR):int(e * SR)]
                dur = len(x) / SR; tot["over_30s"] += dur > 30
                if not a.no_mfcc:
                    M[i] = mfcc86(x, tot)
                if not a.no_whisper:
                    import torch, whisper
                    chunk = whisper.pad_or_trim(torch.from_numpy(np.ascontiguousarray(x)))
                    mel = whisper.log_mel_spectrogram(chunk, n_mels=wmodel.dims.n_mels).to(dev)
                    with torch.no_grad():
                        enc = wmodel.encoder(mel.unsqueeze(0)).squeeze(0).cpu().numpy()
                    W[i] = enc[:max(1, int(min(dur, 30) * 50))].mean(0)
        st, tt, has_s, has_t = texts_for(clip, uids)
        XS = np.zeros((len(utts), 768), np.float32); XT = XS.copy(); trs = trt = 0
        if not a.no_xlmr:
            XS, trs = xlmr(st); XT, trt = xlmr(tt)
        np.savez(pf, ids=np.array(uids), whisper=W, mfcc86=M, xlmr_sub=XS, xlmr_tr=XT,
                 trunc=np.array([trs, trt]))
        tot["clips_done"] += 1; tot["utterances_done"] += len(utts)
        tot["no_subtitle_file"] += not has_s; tot["no_transcript_file"] += not has_t
        if tot["clips_done"] % 10 == 0:
            print(json.dumps({"progress_clips": tot["clips_done"], "elapsed_s": round(time.time() - t0)}), flush=True)

    # assembly
    ids, lang, W, M, XS, XT, trunc = [], [], [], [], [], [], np.zeros(2, int)
    sub_txt, tr_txt = [], []
    for clip in clips:
        pf = parts / f"{clip}.npz"
        if not pf.exists():
            continue
        z = np.load(pf)
        uid = list(z["ids"]); ids += uid; lang += [clip.split("-")[1]] * len(uid)
        W.append(z["whisper"]); M.append(z["mfcc86"]); XS.append(z["xlmr_sub"]); XT.append(z["xlmr_tr"])
        trunc += z["trunc"]
        st, tt, _, _ = texts_for(clip, uid)
        sub_txt += st; tr_txt += tt
    n_expected = sum(len(load_utterances(Path(a.trees) / f"{c}.aligntree.TextGrid", c)) for c in clips
                     if (Path(a.trees) / f"{c}.aligntree.TextGrid").exists())
    if len(ids) != n_expected:
        print(json.dumps({"incomplete": True, "assembled": len(ids), "expected": n_expected})); return
    pros = {}
    for l in open(a.prosody, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); pros[r["utt_id"]] = [r.get(k) for k in PROSODY]
    P = np.array([[np.nan if v is None else v for v in pros.get(u, [None] * 15)] for u in ids], np.float32)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy import sparse
    vs = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, norm="l2")
    TS = vs.fit_transform([t if t.strip() else PLACEHOLDER for t in sub_txt])
    cov = [i for i, l in enumerate(lang) if l in COVERED]
    vt = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, norm="l2")
    vt.fit([tr_txt[i] or PLACEHOLDER for i in cov])
    TT = vt.transform([t if t.strip() else PLACEHOLDER for t in tr_txt])
    sparse.save_npz(out / "tfidf_sub.npz", TS.tocsr()); sparse.save_npz(out / "tfidf_tr.npz", TT.tocsr())
    np.savez(out / "features.npz", ids=np.array(ids), lang=np.array(lang),
             whisper=np.vstack(W), mfcc86=np.vstack(M), prosody=P, prosody_names=np.array(PROSODY),
             xlmr_sub=np.vstack(XS), xlmr_tr=np.vstack(XT))
    L = sorted(set(lang))
    res = {"utterances": len(ids), "counts": dict(tot), "xlmr_revision": rev,
           "xlmr_truncated": {"sub": int(trunc[0]), "tr": int(trunc[1])},
           "empty_subtitle_by_language": {l: sum(1 for i, x in enumerate(sub_txt) if lang[i] == l and not x.strip()) for l in L},
           "empty_transcript_by_language": {l: sum(1 for i, x in enumerate(tr_txt) if lang[i] == l and not x.strip()) for l in L},
           "prosody_missing_rows": sum(u not in pros for u in ids),
           "whisper_zero_rows": int((np.abs(np.vstack(W)).sum(1) == 0).sum()),
           "mfcc_nonfinite_rows": int((~np.isfinite(np.vstack(M))).any(1).sum()),
           "tfidf_vocab": {"sub": len(vs.vocabulary_), "tr": len(vt.vocabulary_)}}
    (out / "features.provenance.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
