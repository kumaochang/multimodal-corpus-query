"""Which utterances are Mandarin speech? (draft for Amendment 9, D0)

In TITV news the Chinese subtitle translates Indigenous-language speech, but when an interviewee
speaks Mandarin it is a transcript of that speech. MOCO stage 4.2 transcribes every clip with
Whisper large-v2 (Chinese prompt). On Mandarin speech its text matches the subtitle almost
character for character; on Indigenous-language speech it does not (it hallucinates, analysis/q17).
On Study 1 the distribution of the distance is bimodal.

Rule, per utterance (ids as pipeline/stages/utterances.py):
  sub = Han characters of the utterance's subtitle text (subtitle_read.py output)
  wh  = Han characters of the Whisper segments overlapping the utterance, in order
  if len(sub) >= MIN_SUB: d = Levenshtein(sub, wh) / max(len(sub), len(wh))   (d = 1 if wh empty)
      d <= MANDARIN_MAX            -> "mandarin"
      MANDARIN_MAX < d <= AMBIG_MAX -> "ambiguous"
      d > AMBIG_MAX                -> "indigenous"
  else (no usable subtitle): if --lid given, Whisper P(zh) >= LID_ZH -> "mandarin_lid",
      otherwise "no_subtitle".
Writes <out> JSONL (utt_id, clip, sub_len, dist, class); prints class counts per language only.
"""
import argparse, json, re, collections
from pathlib import Path

MIN_SUB, MANDARIN_MAX, AMBIG_MAX, LID_ZH = 4, 0.4, 0.6, 0.5
HAN = re.compile(r"[一-鿿]")

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def han(s):
    return "".join(HAN.findall(s or ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtitles", required=True, help="dir of <clip>.subtitles.json")
    ap.add_argument("--whisper", required=True, help="dir of <clip>_whisper.json")
    ap.add_argument("--lid", help="lid_eval.py JSONL with whisper_p_zh per utt id (optional)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lid = {}
    if a.lid:
        for l in open(a.lid, encoding="utf-8"):
            r = json.loads(l)
            if "whisper_p_zh" in r:
                lid[r["id"]] = r["whisper_p_zh"]
    counts = collections.defaultdict(collections.Counter)
    with open(a.out, "w", encoding="utf-8") as fh:
        for f in sorted(Path(a.subtitles).glob("*.subtitles.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            clip = d["clip"]; lang = clip.split("-")[1]
            wf = Path(a.whisper) / f"{clip}_whisper.json"
            segs = json.loads(wf.read_text(encoding="utf-8"))["segments"] if wf.exists() else []
            for u in d["utterances"]:
                sub = han(u["subtitle_text"])
                row = {"utt_id": u["utt_id"], "clip": clip, "sub_len": len(sub), "dist": None}
                if len(sub) >= MIN_SUB:
                    wh = han("".join(s["text"] for s in segs if s["start"] < u["end_s"] and s["end"] > u["start_s"]))
                    dist = 1.0 if not wh else lev(sub, wh) / max(len(sub), len(wh))
                    row["dist"] = round(dist, 4)
                    row["class"] = ("mandarin" if dist <= MANDARIN_MAX else
                                    "ambiguous" if dist <= AMBIG_MAX else "indigenous")
                else:
                    p = lid.get(u["utt_id"])
                    row["class"] = "mandarin_lid" if p is not None and p >= LID_ZH else "no_subtitle"
                    row["p_zh"] = p
                counts[lang][row["class"]] += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for lang in sorted(counts):
        print(json.dumps({"language": lang, **counts[lang]}))

if __name__ == "__main__":
    main()
