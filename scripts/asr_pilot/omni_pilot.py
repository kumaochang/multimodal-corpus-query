"""
ASR pilot for the Tsou -> Bunun decision (see README.md in this folder).

Runs Meta Omnilingual ASR on gold read-speech / NTU Paiwan ASR clips,
computes per-utterance CER, and applies the decision rule fixed in README.md.

Usage (on a GPU machine with network access to dl.fbaipublicfiles.com):

    pip install omnilingual-asr
    # 1. decision gold: 族語E樂園 情境族語 S1, sampled per README.md
    python omni_pilot.py klokah-manifest --out-dir klokah/ --manifest-dir manifests/
    # 2. decision run
    python omni_pilot.py run --manifests manifests/klokah_*.csv --out results/
    # 3. secondary check on NTU Paiwan ASR (not part of the decision)
    python omni_pilot.py ntu-manifest --ntu-dir ../../data/ntu_paiwan_asr --out manifests_secondary/pwn_ntu.csv
    python omni_pilot.py run --manifests manifests_secondary/pwn_ntu.csv --out results_secondary/

Manifest CSV columns:
    audio  path to a wav/flac file, < 40 s
    lang   Omnilingual code: ami_Latn, pwn_Latn, tay_Latn, trv_Latn, tao_Latn, bnn_Latn
    ref    gold transcription in the official orthography
"""
import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

# ---- decision rule (must match README.md; do not edit after results are seen) ----
MODEL_CARD = "omniASR_LLM_7B_v2"
THRESHOLD = 0.25          # median CER per language
RETAINED = ["ami_Latn", "pwn_Latn", "tay_Latn", "trv_Latn", "tao_Latn"]
NEEDED_RETAINED = 4       # at least 4 of the 5 retained languages must pass
CANDIDATE = "bnn_Latn"    # must pass on its own
# -------------------------------------------------------------------------------

APOSTROPHES = "’‘ʼʻ`´"


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    for a in APOSTROPHES:
        s = s.replace(a, "'")
    s = re.sub(r"[^\w\s'^]", " ", s)      # keep letters, digits, ' and ^
    s = re.sub(r"\s+", " ", s).strip()
    return s


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return float("nan")
    return edit_distance(r, h) / len(r)


def cmd_ntu_manifest(args):
    ntu = Path(args.ntu_dir)
    clips = json.loads((ntu / "clips.json").read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio", "lang", "ref"])
        for c in clips:
            w.writerow([str((ntu / "clips" / c["clip"]).resolve()), "pwn_Latn", c["form"]])
    print(f"wrote {len(clips)} rows to {out}")


KLOKAH = "https://web.klokah.tw"
SEED_BASE = 20261007
N_PER_LANG = 50
MIN_CHARS = 10
DIALECTS = {   # language -> dialect ids on 族語E樂園, ascending
    "ami_Latn": [1, 2, 3, 4, 5],
    "tay_Latn": [6, 7, 8, 9, 10, 11],
    "trv_Latn": [33],
    "bnn_Latn": [18, 19, 20, 21, 22],
    "tao_Latn": [42],
    "pwn_Latn": [23, 24, 25, 26],
}


def allocation(k: int, total: int = N_PER_LANG):
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def cmd_klokah_manifest(args):
    import datetime
    import random
    import subprocess
    import urllib.request

    def get_json(path):
        with urllib.request.urlopen(KLOKAH + path, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    out_dir = Path(args.out_dir)
    man_dir = Path(args.manifest_dir)
    man_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mp3").mkdir(parents=True, exist_ok=True)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    log = {"retrieved": datetime.date.today().isoformat(), "dialects": {}}

    for lang, dids in DIALECTS.items():
        rows = []
        for did, n in zip(dids, allocation(len(dids))):
            meta = get_json(f"/dialogue/json/SN112{did:02d}.json")
            eligible, seen = [], set()
            for tid in meta["S1"]:
                for s in get_json(f"/dialogue/php/getDiaData.php?tid={tid}"):
                    text = normalize(s.get("ab") or "")
                    if not s.get("snd") or len(text) < MIN_CHARS or text in seen:
                        continue
                    seen.add(text)
                    eligible.append(s)
            picked = random.Random(SEED_BASE + did).sample(eligible, n)
            log["dialects"][did] = {"title": meta.get("title"), "eligible": len(eligible), "picked": n}
            for s in picked:
                tid, sid = s["sn"].split("/")
                mp3 = out_dir / "mp3" / f"{tid}_{sid}.mp3"
                wav = out_dir / "wav" / f"{tid}_{sid}.wav"
                if not mp3.exists():
                    urllib.request.urlretrieve(f"{KLOKAH}/text/sound/{tid}/{sid}.mp3", mp3)
                if not wav.exists():
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3),
                                    "-ac", "1", "-ar", "16000", str(wav)], check=True)
                rows.append([str(wav.resolve()), lang, s["ab"], did, s["sn"], s.get("ch", "")])
            print(f"{lang} did={did} {meta.get('title')}: {len(eligible)} eligible, picked {n}")
        with (man_dir / f"klokah_{lang.split('_')[0]}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["audio", "lang", "ref", "did", "sn", "zh"])
            w.writerows(rows)
    (out_dir / "retrieval_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifests in {man_dir}, retrieval log in {out_dir / 'retrieval_log.json'}")


def cmd_run(args):
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    rows = []
    for m in args.manifests:
        with open(m, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["manifest"] = Path(m).name
                rows.append(r)
    print(f"{len(rows)} utterances from {len(args.manifests)} manifests; model {MODEL_CARD}")

    pipe = ASRInferencePipeline(model_card=MODEL_CARD)
    hyps = []
    bs = args.batch_size
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        hyps += pipe.transcribe([r["audio"] for r in batch], lang=[r["lang"] for r in batch], batch_size=bs)
        print(f"  {min(i + bs, len(rows))}/{len(rows)}", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_lang = {}
    with (out / "utterances.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["manifest", "lang", "audio", "ref", "hyp", "cer"])
        for r, h in zip(rows, hyps):
            c = cer(r["ref"], h)
            w.writerow([r["manifest"], r["lang"], r["audio"], r["ref"], h, f"{c:.4f}"])
            if c == c:  # not nan
                per_lang.setdefault(r["lang"], []).append(c)

    summary = {lang: {"n": len(v), "median_cer": statistics.median(v), "mean_cer": statistics.fmean(v)}
               for lang, v in sorted(per_lang.items())}

    passed = {lang: s["median_cer"] <= THRESHOLD for lang, s in summary.items()}
    retained_pass = sum(passed.get(l, False) for l in RETAINED)
    missing = [l for l in RETAINED + [CANDIDATE] if l not in summary]
    switch = (not missing) and passed.get(CANDIDATE, False) and retained_pass >= NEEDED_RETAINED
    decision = {
        "model": MODEL_CARD, "threshold": THRESHOLD,
        "retained_passing": retained_pass, "needed": NEEDED_RETAINED,
        "candidate_passes": passed.get(CANDIDATE, False),
        "missing_languages": missing,
        "decision": "SWITCH Tsou -> Bunun" if switch else ("INCOMPLETE" if missing else "KEEP Tsou"),
    }
    (out / "summary.json").write_text(json.dumps({"per_language": summary, "decision": decision},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    for lang, s in summary.items():
        print(f"{lang}: n={s['n']} median CER={s['median_cer']:.3f} {'PASS' if passed[lang] else 'fail'}")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ntu-manifest")
    a.add_argument("--ntu-dir", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_ntu_manifest)
    k = sub.add_parser("klokah-manifest")
    k.add_argument("--out-dir", required=True)
    k.add_argument("--manifest-dir", required=True)
    k.set_defaults(func=cmd_klokah_manifest)
    b = sub.add_parser("run")
    b.add_argument("--manifests", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--batch-size", type=int, default=4)
    b.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
