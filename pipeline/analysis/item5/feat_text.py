"""Block T per judged utterance (Amendment 9 item 5; Amendment 10 item 3), plus the CLM inputs.

Sources, kept separate (the probe reduces each one on its own, Amendment 10 item 3):
  xlmr  : FacebookAI/xlm-roberta-base, max_length 512, mean of last_hidden_state masked by the
          attention mask; empty transcript -> zero vector (Amendment 4 B2, Study 1's
          get_text_embeddings). Revision recorded.
  tfidf : TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True,
          norm="l2"), fitted once on all transcription inputs of the registered 216-clip sample
          in the five covered languages (Amendment 4 B2, Amendment 5); the 84 added clips are
          transformed with it, not fitted on. Empty input -> the ideographic-space placeholder
          of Study 1.
  form  : 4 indicators: marker, suffix (none is the reference category), has `ari`, has
          `ma[jy]a` (form_candidates.py rules and stoplist).
Also writes, for clm_score.py, each item's transcript and its subtitle window (+- 3 s, the
window of form_candidates.py). These texts go only to a gitignored file and are never printed.
Inputs: --items; --transcripts DIR [DIR ...] (transcribe.py <clip>.jsonl); --fit-manifest and
--fit-transcripts for the TF-IDF fit; --subtitles DIR [DIR ...]. --no-xlmr skips the model
(plumbing tests). Output: --out .npz and --texts .jsonl. Prints counts only.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_items, find_file  # noqa: E402
from form_candidates import norm, LEX, STOP  # noqa: E402

W = 3.0
COVERED = {"Amis", "Atayal", "Paiwan", "Truku", "Yami"}

def transcript_index(dirs):
    idx = {}
    for d in dirs:
        for f in Path(d).glob("*.jsonl"):
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l); idx[r["utt_id"]] = r["text"]
    return idx

def form_vec(text):
    toks = [norm(t) for t in text.split()]
    ari = any(t == "ari" for t in toks)
    maya = any(re.match(r"^ma[jy]a$", t) for t in toks)
    marker = ari or maya
    suffix = (not marker) and any(len(t) >= 2 and t[-1] in "ui" and t not in STOP for t in toks)
    return [float(marker), float(suffix), float(ari), float(maya)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--transcripts", nargs="+", required=True)
    ap.add_argument("--fit-manifest", required=True)
    ap.add_argument("--fit-transcripts", required=True)
    ap.add_argument("--subtitles", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--texts", required=True)
    ap.add_argument("--no-xlmr", action="store_true")
    a = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    items = load_items(a.items)
    tr = transcript_index(a.transcripts)
    # TF-IDF fit: registered sample, five covered languages
    man = json.loads(Path(a.fit_manifest).read_text(encoding="utf-8"))
    fit_clips = {Path(c["file"]).stem for c in man["clips"] if c["drawn"] and c["language"] in COVERED}
    fit_texts = []
    for f in Path(a.fit_transcripts).glob("*.jsonl"):
        if f.stem in fit_clips:
            fit_texts += [json.loads(l)["text"] or "　" for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, norm="l2")
    vec.fit(fit_texts)
    texts = [tr.get(r["utt_id"], "") for r in items]
    tfidf = vec.transform([t if t.strip() else "　" for t in texts]).toarray()
    form = np.array([form_vec(t) for t in texts])
    xlmr = np.zeros((len(items), 768)); rev = None; trunc = 0
    if not a.no_xlmr:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        mod = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base").eval()
        rev = getattr(mod.config, "_commit_hash", None)
        dev = "cuda" if torch.cuda.is_available() else "cpu"; mod.to(dev)
        for i, t in enumerate(texts):
            if not t.strip():
                continue
            trunc += len(tok(t)["input_ids"]) > 512
            inp = {k: v.to(dev) for k, v in tok(t, return_tensors="pt", truncation=True,
                                                 max_length=512, padding=True).items()}
            with torch.no_grad():
                out = mod(**inp)
            m = inp["attention_mask"].unsqueeze(-1).float()
            xlmr[i] = ((out.last_hidden_state * m).sum(1) / m.sum(1)).squeeze().cpu().numpy()
    # subtitle windows for CLM input (a)
    sub_cache = {}
    with open(a.texts, "w", encoding="utf-8") as fh:
        for r, t in zip(items, texts):
            c = r["clip"]
            if c not in sub_cache:
                p = find_file(a.subtitles, f"{c}.subtitles.json")
                sub_cache[c] = json.loads(p.read_text(encoding="utf-8"))["lines"] if p else []
            s, e = r["start_s"], r["end_s"]
            sub = "".join(x["text"] for x in sub_cache[c]
                          if x["start_ms"] / 1000 < e + W and x["end_ms"] / 1000 > s - W)
            fh.write(json.dumps({"utt_id": r["utt_id"], "transcript": t, "subtitle_window": sub},
                                ensure_ascii=False) + "\n")
    np.savez(a.out, ids=np.array([r["utt_id"] for r in items]), xlmr=xlmr, tfidf=tfidf, form=form,
             form_names=np.array(["marker", "suffix", "has_ari", "has_maya"]))
    print(json.dumps({"items": len(items), "empty_transcripts": sum(not t.strip() for t in texts),
                      "no_transcript_record": sum(r["utt_id"] not in tr for r in items),
                      "tfidf_fit_inputs": len(fit_texts), "tfidf_vocab": len(vec.vocabulary_),
                      "xlmr": "skipped" if a.no_xlmr else "done", "xlmr_revision": rev,
                      "xlmr_truncated": int(trunc),
                      "form_counts": dict(zip(["marker", "suffix", "has_ari", "has_maya"],
                                              form.sum(0).astype(int).tolist()))}))

if __name__ == "__main__":
    main()
