"""Directive criterion over Chinese subtitle text (Amendment 9 D1, option A; draft).

Study 2's criterion (IMP, HORT, NEG.IMP, DEONTIC) is a closed rule over English sentence mood and
cannot run on Chinese. This is its category-for-category Chinese counterpart, keyed on explicit
markers only. There is no subject-drop test: Chinese drops subjects freely, so "no overt subject"
would fire on most clauses. As in Study 2, only the subtitle's own sentence mood counts: a
directive inside reported speech (X 呼籲/提醒/希望 ...) is not directive, and questions are not.

Segments: each subtitle line (a screen of text), further split at punctuation. A line is the
nearest available clause boundary; subtitles carry almost no punctuation.
  C        optional discourse connective before a clause-initial marker (所以, 那, 也 ...)
  NEG.IMP  ^C (請勿|勿|不要|別|不准|不可以|千萬不要|千萬別), or ^C ADDR (都|也|千萬|就)? (不要|別|不可以|不准)
  IMP      ^C (請|敬請) + V, not 請問/請教/請假/請客/請求/請示/請託/請願
  HORT     ^C (讓我們|讓咱們)一起; or a segment with 我們/咱們/大家/一起 that ends in 吧 and has no
           epistemic marker (應該 可能 大概 也許 或許 好像), since 吧 is also epistemic
  DEONTIC  (ADDR|PLURAL) (都|也|一定|還是|就|更|真的)? (應該|必須|務必|一定要|得要); or PLURAL ... 要,
           要 not followed by 求/不要/是/的; or ^C (務必|千萬要). Bare 要 after singular 你/妳/您 is
           not counted: in speech it is mostly generic or conditional ("你要做一個作品 ...", Study 1)
  ADDR     你們 妳們 你 妳 您 大家 各位
  PLURAL   你們 妳們 大家 各位 民眾 族人 鄉親 (plural addressee and audience nouns)
  HORT     讓我們/讓咱們 only as 讓我們一起, or in a segment ending in 吧: bare 讓我們 + V is mostly
           causative in these subtitles ("讓我們更了解 ...", Study 1), and precision is preferred
  NEG.IMP  不要 not followed by 緊/那麼/這麼/太 (degree: "不要那麼多的壓力")
Excluded: a segment ending in 嗎/呢 or containing ？?, 要不要/是不是/有沒有/會不會/為什麼, or a modal
directly followed by a wh-word (你要怎麼換算; but not the embedded 一定要懂得要怎麼樣織布) (question);
a hit whose segment, or the segment before it in the same utterance, contains a reporting verb
(呼籲 提醒 要求 希望 表示 指出 強調 建議 請求 勸 告訴 叮嚀 叮囑 期許 期盼 鼓勵 盼).
An utterance is directive if at least one segment has a non-excluded hit.

Input : --subtitles dir of <clip>.subtitles.json (subtitle_read.py); --filter mandarin_filter.py
        JSONL (D0): utterances classed mandarin / mandarin_lid are skipped and counted.
Output: --out JSONL per utterance (utt_id, clip, d0, directive, categories, n_excluded_*);
        --hits optional TSV of hit segments, for reading on Study 1 only. Prints counts only.
"""
import argparse, collections, json, re
from pathlib import Path

C = r"(?:那麼|那|所以|因此|然後|也|但是|但|可是|還有|而且|現在|最後|另外|因此)?"
ADDR = r"(?:你們|妳們|你|妳|您|大家|各位)"
PLURAL = r"(?:你們|妳們|大家|各位|民眾|族人|鄉親)"
RULES = [
    ("NEG.IMP", re.compile(rf"^{C}(?:請勿|勿|不要(?!緊|那麼|這麼|太)|別|不准|不可以|千萬不要|千萬別)")),
    ("NEG.IMP", re.compile(rf"^{C}{ADDR}(?:都|也|千萬|就)?(?:不要(?!緊)|別|不可以|不准)")),
    ("IMP", re.compile(rf"^{C}(?:請|敬請)(?!問|教|假|客|求|示|託|願)")),
    ("HORT", re.compile(rf"^{C}(?:讓我們|讓咱們)一起")),
    ("DEONTIC", re.compile(rf"(?:{ADDR}|{PLURAL})(?:都|也|一定|還是|就|更|真的)?(?:應該|必須|務必|一定要|得要)")),
    ("DEONTIC", re.compile(rf"{PLURAL}(?:都|也|一定|還是|就|更|真的)?要(?!求|不要|是|的)")),
    ("DEONTIC", re.compile(rf"^{C}(?:務必|千萬要)")),
]
HORT_SUBJ = re.compile(r"我們|咱們|大家|一起")
EPISTEMIC = re.compile(r"應該|可能|大概|也許|或許|好像")
QUESTION = re.compile(r"[？?]|要不要|是不是|有沒有|會不會|為什麼|(?:要|應該|該|能|可以)(?:怎麼|如何|什麼|去哪)|(?:嗎|呢)$")
REPORT = re.compile(r"呼籲|提醒|要求|希望|表示|指出|強調|建議|請求|勸|告訴|叮嚀|叮囑|期許|期盼|鼓勵|盼")
EMBED = ("懂得", "知道", "學會", "了解", "曉得", "學")

def is_question(seg):
    """A modal + wh-word right after 懂得/知道 ... is an embedded question, not a question."""
    for m in QUESTION.finditer(seg):
        if not seg[:m.start()].endswith(EMBED):
            return True
    return False

SPLIT = re.compile(r"[。！!，,；;：:、\s「」『』]+")

def segments(lines):
    for l in lines:
        for s in SPLIT.split(l):
            if s:
                yield s

def hits(seg):
    out = [cat for cat, rx in RULES if rx.search(seg)]
    if seg.endswith("吧") and HORT_SUBJ.search(seg) and not EPISTEMIC.search(seg):
        out.append("HORT")
    return sorted(set(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtitles", required=True)
    ap.add_argument("--filter", required=True, help="D0 output of mandarin_filter.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hits", help="TSV of hit segments (Study 1 only)")
    a = ap.parse_args()
    d0 = {}
    for l in open(a.filter, encoding="utf-8"):
        r = json.loads(l); d0[r["utt_id"]] = r["class"]
    cnt = collections.defaultdict(collections.Counter)
    hf = open(a.hits, "w", encoding="utf-8") if a.hits else None
    if hf:
        hf.write("utt_id\td0\tcategory\tstatus\tsegment\tutterance_text\n")
    with open(a.out, "w", encoding="utf-8") as fh:
        for f in sorted(Path(a.subtitles).glob("*.subtitles.json")):
            d = json.loads(f.read_text(encoding="utf-8")); clip = d["clip"]; lang = clip.split("-")[1]
            by_utt = collections.defaultdict(list)
            for ln in d["lines"]:
                if ln.get("utt_id"):
                    by_utt[ln["utt_id"]].append(ln["text"])
            for u in d["utterances"]:
                c = d0.get(u["utt_id"], "missing"); cn = cnt[lang]
                cn[f"d0_{c}"] += 1
                if c in ("mandarin", "mandarin_lid", "missing"):
                    continue
                row = {"utt_id": u["utt_id"], "clip": clip, "d0": c, "categories": [],
                       "n_excluded_question": 0, "n_excluded_reported": 0}
                segs = list(segments(by_utt.get(u["utt_id"]) or [u["subtitle_text"]]))
                for i, s in enumerate(segs):
                    h = hits(s)
                    if not h:
                        continue
                    if is_question(s):
                        status = "question"; row["n_excluded_question"] += 1
                    elif REPORT.search(s) or (i and REPORT.search(segs[i - 1])):
                        status = "reported"; row["n_excluded_reported"] += 1
                    else:
                        status = "hit"; row["categories"] += h
                    if hf:
                        hf.write(f"{u['utt_id']}\t{c}\t{'+'.join(h)}\t{status}\t{s}\t{u['subtitle_text']}\n")
                row["categories"] = sorted(set(row["categories"]))
                row["directive"] = bool(row["categories"])
                cn["scored"] += 1
                cn["directive"] += row["directive"]
                cn["directive_excl_ambiguous"] += row["directive"] and c != "ambiguous"
                for k in row["categories"]:
                    cn[k] += 1
                cn["excluded_question"] += row["n_excluded_question"] > 0
                cn["excluded_reported"] += row["n_excluded_reported"] > 0
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for lang in sorted(cnt):
        print(json.dumps({"language": lang, **cnt[lang]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
