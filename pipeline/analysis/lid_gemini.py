"""Spoken-language check by an audio-capable model (draft for Amendment 9, D0 validation).

D0 calls an utterance Mandarin when Whisper's Chinese text matches the subtitle
(mandarin_filter.py). That rule is untuned and has a known failure mode: Whisper forced to Chinese
on Indigenous-language speech can emit a paraphrase close to the subtitle. This program asks a
second, independent detector: a Gemini model hears the utterance audio (no subtitle, no
transcript, no video) and says what is spoken. It is not a human check; it is a third signal
beside the edit distance and Whisper P(zh), with different failure modes.

The question is binary at heart (Mandarin or not); the model is never asked to recognise Paiwan,
Amis or Truku, which it cannot be assumed to know. Hokkien and Hakka get their own answer because
they are not Mandarin and not the target language either.

Input : --wav-dir with <clip>.wav (16 kHz mono), --subtitles dir of <clip>.subtitles.json
        (utterance ids and boundaries, subtitle_read.py output; the subtitle text is not read).
Output: <out>/lid.jsonl (resumable), <out>/provenance.json. Prints counts only; with --filter
        (mandarin_filter.py JSONL) also the cross-table of D0 class x model answer.
Key   : --key-file (first line), else GEMINI_API_KEY in the environment; never printed.
--language restricts to clips of one language (clip stem news-<Language>-...).
"""
import argparse, collections, hashlib, io, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_units import with_retry  # noqa: E402  (same retry policy as the labeler)

ANSWERS = ["mandarin", "mixed", "other_chinese", "not_chinese", "no_speech"]
PROMPT = """You will hear a short audio segment from a Taiwanese television news programme.
Decide which language is spoken in it. Do not translate or transcribe.

Answer with one of:
- "mandarin": the speech is Mandarin Chinese (any accent), throughout.
- "mixed": Mandarin for at least one full clause AND another language for at least one full clause.
- "other_chinese": Taiwanese Hokkien or Hakka, without a full clause of Mandarin.
- "not_chinese": speech in a language that is none of these (for example an Austronesian language), possibly with isolated Mandarin loanwords or names.
- "no_speech": music, noise or silence only.

Reply with JSON only: {"answer": "<one of the five>", "confidence": <number from 0 to 1>}"""
MAX_S = 30.0

def segment_wav(wav, s, e):
    import soundfile as sf
    info = sf.info(wav)
    if info.samplerate != 16000 or info.channels != 1:
        sys.exit(f"{wav}: expected 16 kHz mono, got {info.samplerate} Hz x {info.channels}")
    e = min(e, s + MAX_S)
    a, _ = sf.read(wav, start=int(s * 16000), stop=int(e * 16000), dtype="int16")
    buf = io.BytesIO(); sf.write(buf, a, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue(), len(a) / 16000

def parse(raw):
    try:
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        r = json.loads(m.group())
        if r.get("answer") in ANSWERS:
            return r["answer"], r.get("confidence"), None
        return None, None, "invalid_answer"
    except Exception:
        return None, None, "unparsable"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--subtitles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="exact Gemini model version (required unless --dry-run)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--filter", help="mandarin_filter.py JSONL, for the cross-table")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="cut audio, count, no API call")
    ap.add_argument("--key-file")
    ap.add_argument("--language", help="only clips news-<Language>-*")
    a = ap.parse_args()
    if not a.dry_run and not a.model:
        sys.exit("--model must name an exact Gemini model version")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jl = out / "lid.jsonl"
    done = set()
    if jl.exists():
        done = {json.loads(l)["utt_id"] for l in open(jl, encoding="utf-8") if l.strip()}
    client = types = None
    if not a.dry_run:
        from google import genai
        from google.genai import types
        key = Path(a.key_file).expanduser().read_text().split()[0] if a.key_file else os.environ["GEMINI_API_KEY"]
        client = genai.Client(api_key=key)
    tot = collections.Counter(); n_new = 0
    with open(jl, "a", encoding="utf-8") as fh:
        for f in sorted(Path(a.subtitles).glob("*.subtitles.json")):
            d = json.loads(f.read_text(encoding="utf-8")); clip = d["clip"]
            if a.language and clip.split("-")[1] != a.language:
                continue
            wav = Path(a.wav_dir) / f"{clip}.wav"
            if not wav.exists():
                tot["clips_without_wav"] += 1; continue
            for u in d["utterances"]:
                if u["utt_id"] in done:
                    tot["skipped_done"] += 1; continue
                if a.limit is not None and n_new >= a.limit:
                    break
                data, dur = segment_wav(wav, u["start_s"], u["end_s"])
                tot["audio_s"] += dur
                tot["truncated_30s"] += (u["end_s"] - u["start_s"]) > MAX_S
                row = {"utt_id": u["utt_id"], "clip": clip, "dur_s": round(dur, 2)}
                if a.dry_run:
                    row.update(answer=None, confidence=None, fallback="dry_run")
                else:
                    cfg = types.GenerateContentConfig(temperature=a.temperature, max_output_tokens=64,
                          response_mime_type="application/json",
                          thinking_config=types.ThinkingConfig(thinking_budget=a.thinking_budget))
                    parts = [PROMPT, types.Part.from_bytes(data=data, mime_type="audio/wav")]
                    resp = with_retry(lambda: client.models.generate_content(
                                      model=a.model, contents=parts, config=cfg), tot)
                    um = resp.usage_metadata
                    tot["input_tokens"] += um.prompt_token_count or 0
                    tot["output_tokens"] += (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
                    ans, conf, fb = parse(resp.text)
                    row.update(answer=ans, confidence=conf, fallback=fb)
                    if fb:
                        tot[f"fallback_{fb}"] += 1
                    time.sleep(0.2)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                tot["utterances"] += 1; n_new += 1
    tot["audio_s"] = round(tot["audio_s"], 1)
    rows = [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]
    by_lang = collections.defaultdict(collections.Counter)
    for r in rows:
        by_lang[r["clip"].split("-")[1]][r["answer"] or "none"] += 1
    (out / "provenance.json").write_text(json.dumps({
        "program": "pipeline/analysis/lid_gemini.py", "model": a.model, "temperature": a.temperature,
        "thinking_budget": a.thinking_budget, "max_segment_s": MAX_S,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "subtitles_dir": a.subtitles, "wav_dir": a.wav_dir, "totals": dict(tot),
        "answers_by_language": {k: dict(v) for k, v in by_lang.items()}}, indent=1), encoding="utf-8")
    print(json.dumps(dict(tot)))
    for k in sorted(by_lang):
        print(json.dumps({"language": k, **by_lang[k]}))
    if a.filter:
        cls = {json.loads(l)["utt_id"]: json.loads(l)["class"] for l in open(a.filter, encoding="utf-8") if l.strip()}
        xt = collections.Counter((cls.get(r["utt_id"], "missing"), r["answer"] or "none") for r in rows)
        print("cross-table (D0 class -> model answer: n)")
        for c in sorted({k[0] for k in xt}):
            print(json.dumps({"d0": c, **{k[1]: v for k, v in xt.items() if k[0] == c}}))

if __name__ == "__main__":
    main()
