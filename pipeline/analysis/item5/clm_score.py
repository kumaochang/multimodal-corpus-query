"""CLM-8B text baseline (Amendment 9 item 5, "Text-classifier baselines").

One typed Choice question, fixed wording, no examples or tuning. Two inputs scored separately:
(a) the Chinese subtitle window (+- 3 s), (b) the Paiwan transcription, from feat_text.py's
--texts file. Score = probability of "yes, a directive". An empty input is not sent; it gets
the uninformative score 0.5 and is counted (the registration is silent on empty inputs; this
choice favours neither model and is reported).
Requires the CLM server: vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling
--max-model-len 2048 --port 8090, then clm-serve (weights Contrastive-LM/CLM-v0.1-8B).
Output: --out JSONL (utt_id, clm_a, clm_b) and provenance; prints counts only.
"""
import argparse, json
from importlib import metadata
from pathlib import Path

INSTRUCTIONS = ("Is the speaker asking the listener (or a group that includes the speaker) to do "
                "something or not to do something, as in a command, request, urging, invitation, "
                "reminder or prohibition?")
YES, NO = "yes, a directive", "no, not a directive"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from clm import CLMClient, Choice
    client = CLMClient()
    q = {"directive": Choice(instructions=INSTRUCTIONS, criteria={YES: YES, NO: NO})}
    n = {"items": 0, "empty_a": 0, "empty_b": 0}
    with open(a.out, "w", encoding="utf-8") as fh:
        for l in open(a.texts, encoding="utf-8"):
            r = json.loads(l); n["items"] += 1; row = {"utt_id": r["utt_id"]}
            for key, field in (("clm_a", "subtitle_window"), ("clm_b", "transcript")):
                txt = (r[field] or "").strip()
                if not txt:
                    row[key] = 0.5; n["empty_" + key[-1]] += 1; continue
                ans = client.system_one(txt, q)
                row[key] = float(ans.answers["directive"].probabilities[YES])
            fh.write(json.dumps(row) + "\n")
    try:
        ver = metadata.version("contrastive-lm")
    except Exception:
        ver = None
    Path(a.out + ".provenance.json").write_text(json.dumps(
        {"program": "pipeline/analysis/item5/clm_score.py", "contrastive_lm_version": ver,
         "weights": "Contrastive-LM/CLM-v0.1-8B", "backbone": "Qwen/Qwen3-8B",
         "instructions": INSTRUCTIONS, "options": [YES, NO], **n}, indent=1))
    print(json.dumps(n))

if __name__ == "__main__":
    main()
