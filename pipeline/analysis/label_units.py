"""Pragmatic-function labels for clips and utterances (PREREGISTRATION.md §3 items 8-9).

Draft for Amendment 9; nothing here is in force until that amendment fixes it.

Instrument: scripts/label_with_claude.py, the Study 1 labeler whose clip-level labels were
validated against people (κ_inter 0.747, κ_avg(VLM) 0.765). This program keeps its parts and
changes only what the unit requires:
  - five keyframes at times start + dur * (i + OFFSET) / 5, i = 0..4, scaled to 640 px wide,
    JPEG q 3 (as Study 1; OFFSET 0.5 there);
  - the OCR summary of scripts/label_with_claude.extract_ocr_summary (ocr_corrected, confidence
    >= 0.8, not `logo`, contains a CJK character, at least 2 characters, de-duplicated in order,
    600 characters), restricted to OCR frames whose time lies in [start, end];
  - the prompt of Study 1, verbatim for --unit clip; for --unit utterance the two sentences that
    refer to a whole video are changed (UTT_EDITS below) and nothing else;
  - label set, JSON answer, and the Study 1 fallbacks (unparsable or invalid answer ->
    Reporting/low), each fallback counted.
Perturbations for the self-consistency measure of §3 item 9: --offset (keyframe sampling points)
and --order (a permutation of the five category lines, by seed).
Study 1 set no temperature (API default 1.0); --temperature makes that a choice to be fixed.
--provider gemini sends the same prompt, the same five JPEG keyframes and the same OCR summary,
in the same order, to a Gemini model (google-genai SDK, GEMINI_API_KEY); --model must then name
an exact Gemini model version. Token counts come from each provider's usage report.

Writes <out>/labels.jsonl (one row per unit, resumable) and <out>/provenance.json.
Prints counts only (units done, fallbacks), never labels or rationales.
"""
import argparse, base64, hashlib, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stages"))
from utterances import load_utterances  # noqa: E402

N_FRAMES, FRAME_W = 5, 640
LABELS = ["Comforting", "Celebrating", "Explaining", "Urging", "Reporting"]
CAT_LINES = {
    "Comforting": "- **Comforting**：慰問、關懷、安慰當事人（例如：災後探視、表達同情）",
    "Celebrating": "- **Celebrating**：慶祝、歡迎、歡慶（例如：豐年祭、頒獎、交車典禮）",
    "Explaining": "- **Explaining**：解說、教導、文化傳承、分享知識（例如：耆老講解傳統、老師示範）",
    "Urging": "- **Urging**：呼籲、勸告、政策訴求（例如：呼籲民眾參與、政策說明施壓）",
    "Reporting": "- **Reporting**：純粹的新聞播報，無明顯的上述互動性質",
}
PROMPT = """你正在協助一個多模態語料庫研究計畫標注原住民族電視台（TITV）的新聞影片片段。

以下是這支影片的：
1. 五張均勻抽樣的畫面截圖
2. OCR 擷取的中文字幕摘要

請判斷這支影片中，**主要互動者之間的互動性質**屬於哪一類：

{categories}

注意：
- 主播在攝影棚播報算 Reporting，除非明顯帶有上述互動功能
- 若影片含多個新聞，以**最主要**或**最具語用特色**的段落為準
- 若無法判斷，選 Reporting

請用以下 JSON 格式回答（只輸出 JSON，不要其他文字）：
{{
  "label": "Explaining",
  "confidence": "high",
  "rationale": "一句話說明判斷依據"
}}

confidence 值：high / medium / low"""
UTT_EDITS = [
    ("以下是這支影片的：", "以下是這支影片中一段話語（約 {dur:.1f} 秒）的："),
    ("請判斷這支影片中，**主要互動者之間的互動性質**屬於哪一類：",
     "請判斷這段話語中，**主要互動者之間的互動性質**屬於哪一類："),
    ("- 若影片含多個新聞，以**最主要**或**最具語用特色**的段落為準\n", ""),
]

def prompt_for(unit, order, dur):
    p = PROMPT.format(categories="\n".join(CAT_LINES[c] for c in order))
    if unit == "utterance":
        for a, b in UTT_EDITS:
            assert a in p, a
            p = p.replace(a, b.format(dur=dur))
    return p

def keyframes(video, start, dur, offset):
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(N_FRAMES):
            t = start + dur * (i + offset) / N_FRAMES
            o = os.path.join(tmp, f"f{i}.jpg")
            subprocess.run(["ffmpeg", "-ss", f"{t:.3f}", "-i", str(video), "-vframes", "1",
                            "-vf", f"scale={FRAME_W}:-1", "-q:v", "3", o, "-y", "-loglevel", "error"],
                           check=True)
            frames.append(Path(o).read_bytes() if os.path.exists(o) else b"")
    return frames

def ocr_summary(ocr_frames, start, end, max_chars=600):
    seen, texts = set(), []
    for ts, dets in ocr_frames:
        if not (start <= ts / 1000 <= end):
            continue
        for box, text, conf, label in dets:
            if conf < 0.8 or label == "logo" or not re.search(r"[一-鿿]", text):
                continue
            if text in seen or len(text) < 2:
                continue
            seen.add(text); texts.append(text)
    s = " / ".join(texts)
    return (s[:max_chars] + ("…" if len(s) > max_chars else "")) if s else "(無 OCR 資料)"

TRANSIENT = ("503", "529", "429", "500", "UNAVAILABLE", "overloaded", "RESOURCE_EXHAUSTED",
             "rate", "timeout", "Timeout", "Connection")

def with_retry(fn, tot, attempts=8):
    """Retry transient provider errors (overload, rate limit, network) with backoff 5, 10, 20 ...
    s, at most 5 minutes; anything else, or the last attempt, raises."""
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:
            msg = f"{type(e).__name__} {e}"
            if k == attempts - 1 or not any(t in msg for t in TRANSIENT):
                raise
            tot["retries"] += 1
            time.sleep(min(300, 5 * 2 ** k))

def parse(raw):
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        r = json.loads(m.group()) if m else json.loads(raw)
    except Exception:
        return {"label": "Reporting", "confidence": "low", "fallback": "unparsable"}
    if r.get("label") not in LABELS:
        return {"label": "Reporting", "confidence": "low", "fallback": "invalid_label"}
    return {"label": r["label"], "confidence": r.get("confidence"), "rationale": r.get("rationale"),
            "fallback": None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["clip", "utterance"], required=True)
    ap.add_argument("--clips", required=True, help="file, one clip stem per line")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--ocr", required=True, help="ocr_corrected dir")
    ap.add_argument("--trees", help="aligned_trees dir (utterance unit)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic")
    ap.add_argument("--model", default=None, help="exact model id; anthropic default claude-haiku-4-5-20251001")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="gemini only: thinking token budget (0 = off, as Haiku has none)")
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--order-seed", type=int, default=None, help="permute category lines")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--units", default=None, help="JSON with utterance_ids (draw_perturbation_sample.py output): label only these "
                    "utterances; input selection only, prompts and frames unchanged (added 2026-10-17, deviation m)")
    ap.add_argument("--dry-run", action="store_true", help="build inputs, call nothing")
    a = ap.parse_args()
    import random
    order = list(LABELS)
    if a.order_seed is not None:
        random.Random(a.order_seed).shuffle(order)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "labels.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(l)["unit_id"] for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    if a.model is None:
        if a.provider == "gemini":
            sys.exit("--provider gemini needs --model with an exact Gemini model version")
        a.model = "claude-haiku-4-5-20251001"
    client = None
    if not a.dry_run:
        if a.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
        else:
            from google import genai
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    clips = [l.strip() for l in Path(a.clips).read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = None
    if a.units:
        if a.unit != "utterance":
            sys.exit("--units applies to --unit utterance only")
        keep = set(json.loads(Path(a.units).read_text(encoding="utf-8"))["utterance_ids"])
    tot = {"units": 0, "skipped_done": len(done), "fallback_unparsable": 0, "fallback_invalid_label": 0,
           "empty_ocr": 0, "missing_keyframe": 0, "clips_without_tree": 0,
           "input_tokens": 0, "output_tokens": 0, "retries": 0}
    with rows_path.open("a", encoding="utf-8") as fh:
        for clip in clips:
            video = Path(a.videos) / f"{clip}.mp4"
            ocr = json.loads((Path(a.ocr) / f"{clip}.ocr.json").read_text(encoding="utf-8"))
            if a.unit == "clip":
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout)
                units = [(clip, 0.0, dur)]
            else:
                tree = Path(a.trees) / f"{clip}.aligntree.TextGrid"
                if not tree.exists():
                    tot["clips_without_tree"] += 1; continue
                units = load_utterances(tree, clip)
            for uid, s, e in units:
                if keep is not None and uid not in keep:
                    continue
                if uid in done:
                    continue
                if a.limit is not None and tot["units"] >= a.limit:
                    break
                summ = ocr_summary(ocr, s, e) if a.unit == "utterance" else ocr_summary(ocr, -1, 1e9)
                tot["empty_ocr"] += summ == "(無 OCR 資料)"
                frames = keyframes(video, s, e - s, a.offset)
                tot["missing_keyframe"] += sum(not f for f in frames)
                prompt = prompt_for(a.unit, order, e - s)
                row = {"unit_id": uid, "clip": clip, "start_s": s, "end_s": e,
                       "n_frames": sum(bool(f) for f in frames), "ocr_chars": len(summ)}
                if not a.dry_run:
                    ocr_text = f"【OCR 中文字幕摘要】\n{summ}"
                    if a.provider == "anthropic":
                        content = [{"type": "text", "text": prompt}] + [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                             "data": base64.standard_b64encode(f).decode()}} for f in frames if f] + [
                            {"type": "text", "text": ocr_text}]
                        kw = {} if a.temperature is None else {"temperature": a.temperature}
                        msg = with_retry(lambda: client.messages.create(model=a.model, max_tokens=256,
                                         messages=[{"role": "user", "content": content}], **kw), tot)
                        tot["input_tokens"] += msg.usage.input_tokens
                        tot["output_tokens"] += msg.usage.output_tokens
                        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                    else:
                        from google.genai import types
                        parts = [prompt] + [types.Part.from_bytes(data=f, mime_type="image/jpeg")
                                            for f in frames if f] + [ocr_text]
                        cfg = types.GenerateContentConfig(max_output_tokens=256,
                              thinking_config=types.ThinkingConfig(thinking_budget=a.thinking_budget),
                              **({} if a.temperature is None else {"temperature": a.temperature}))
                        resp = with_retry(lambda: client.models.generate_content(
                                          model=a.model, contents=parts, config=cfg), tot)
                        um = resp.usage_metadata
                        tot["input_tokens"] += um.prompt_token_count or 0
                        tot["output_tokens"] += (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
                        raw = resp.text or ""
                    row.update(parse(raw))
                    if row["fallback"]:
                        tot[f"fallback_{row['fallback']}"] += 1
                    time.sleep(0.2)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                tot["units"] += 1
    (out / "provenance.json").write_text(json.dumps({
        "program": "pipeline/analysis/label_units.py", "unit": a.unit, "provider": a.provider,
        "model": None if a.dry_run else a.model,
        "temperature": a.temperature, "units_file": a.units, "thinking_budget": a.thinking_budget if a.provider == "gemini" else None,
        "offset": a.offset, "category_order": order,
        "prompt_sha256": hashlib.sha256(prompt_for(a.unit, order, 0.0).encode()).hexdigest(),
        "dry_run": a.dry_run, "totals": tot}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tot))

if __name__ == "__main__":
    main()
