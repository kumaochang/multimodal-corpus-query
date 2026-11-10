"""Candidate pool for the directive judgement (Amendment 9 item 5, Study 3 Paiwan).

Unit: a Paiwan utterance that survives Amendment 9 item 1 (Gemini answer `not_chinese` or
`other_chinese`, and subtitle/Whisper distance not <= 0.4). Its form class, from the whitespace
tokens of its transcription (transcribe.py `text`):
  marker : a token equal to `ari` (hortative) or matching `ma[jy]a` (prohibitive)
  suffix : otherwise, a token of two or more letters ending in `u` or `i` (the imperative
           suffixes; Amendment 6 item 3 extended to -i) that is not in the function-word stoplist
           `paiwan_suffix_stoplist.json` (paiwan_stoplist.py, from One Hundred Paiwan Texts:
           directive-glossed share of u/i-final tokens 1:20 before, about 1:5 after)
  none   : otherwise
Cue, for enrichment only (never a label, never a model input): the Chinese subtitle text
overlapping [start - W, end + W] contains a loose directive marker (MARKERS). The window absorbs
the subtitle offset found on Study 1 (Amendment 7 applies none).
Stratum = marker | suffix_cue | suffix_nocue | none_cue | none_nocue (the last is not drawn).

Inputs : one or more --set MANIFEST:TRANSCRIPTS:SUBTITLES:LID_JSONL:MANDARIN_FILTER_JSONL.
Output : --out JSONL (utt_id, clip, day, half, start_s, end_s, form, cue, stratum); gitignored,
         never printed. Prints attrition and counts per stratum x half only.
"""
import argparse, collections, json, re
from pathlib import Path

W = 3.0
MARKERS = re.compile(r"請|要|吧|不要|別|一起|應該|必須|務必|記得|趕快|快點|讓我們|來|可以|不可以|不能|千萬")
HAN = re.compile(r"[一-鿿]")
LEX = re.compile(r"^(ari|ma[jy]a)$")
STOP = set(json.loads((Path(__file__).parent / "paiwan_suffix_stoplist.json").read_text(encoding="utf-8"))["stoplist"])

def norm(t):
    return re.sub(r"[^a-z']", "", t.lower())

def form_of(text):
    toks = [norm(t) for t in text.split()]
    if any(LEX.match(t) for t in toks):
        return "marker"
    if any(len(t) >= 2 and t[-1] in "ui" and t not in STOP for t in toks):
        return "suffix"
    return "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help="MANIFEST:TRANSCRIPTS_DIR:SUBTITLES_DIR:LID_JSONL:MANDARIN_FILTER_JSONL")
    ap.add_argument("--split", default="analysis/study3/split.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp = json.loads(Path(a.split).read_text(encoding="utf-8"))
    half = {d: "seed" for d in sp["seed_half"]} | {d: "scoring" for d in sp["scoring_half"]}
    n = collections.Counter(); cells = collections.Counter(); rows = []
    for spec in a.set:
        man, tdir, sdir, lid, mfp = spec.split(":")
        gem = {json.loads(l)["utt_id"]: json.loads(l)["answer"] for l in open(lid, encoding="utf-8") if l.strip()}
        mf = {json.loads(l)["utt_id"]: json.loads(l).get("dist") for l in open(mfp, encoding="utf-8") if l.strip()}
        for c in json.loads(Path(man).read_text(encoding="utf-8"))["clips"]:
            if not c["drawn"] or c["language"] != "Paiwan":
                continue
            stem = Path(c["file"]).stem
            tf, sf = Path(tdir) / f"{stem}.jsonl", Path(sdir) / f"{stem}.subtitles.json"
            if not tf.exists():
                n["clips_without_transcript"] += 1; continue
            if not sf.exists():
                n["clips_without_subtitles"] += 1
            lines = json.loads(sf.read_text(encoding="utf-8"))["lines"] if sf.exists() else []
            for l in tf.read_text(encoding="utf-8").splitlines():
                if not l.strip():
                    continue
                u = json.loads(l); n["utterances"] += 1
                g = gem.get(u["utt_id"])
                if g is None:
                    n["no_lid"] += 1; continue
                if g not in ("not_chinese", "other_chinese"):
                    n[f"dropped_{g}"] += 1; continue
                d = mf.get(u["utt_id"])
                if d is not None and d <= 0.4:
                    n["dropped_textmatch"] += 1; continue
                n["kept"] += 1
                s, e = u["start_s"], u["end_s"]
                sub = "".join(x["text"] for x in lines
                              if x["start_ms"] / 1000 < e + W and x["end_ms"] / 1000 > s - W)
                cue = bool(MARKERS.search(sub))
                f = form_of(u["text"])
                st = "marker" if f == "marker" else f"{f}_{'cue' if cue else 'nocue'}"
                h = half.get(c["broadcast_date"], "none")
                rows.append({"utt_id": u["utt_id"], "clip": stem, "day": c["broadcast_date"], "half": h,
                             "start_s": s, "end_s": e, "form": f, "cue": cue,
                             "has_sub": bool(HAN.search(sub)), "stratum": st})
                cells[(st, h)] += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(dict(n)))
    print(json.dumps({f"{k[0]}|{k[1]}": v for k, v in sorted(cells.items())}))

if __name__ == "__main__":
    main()
