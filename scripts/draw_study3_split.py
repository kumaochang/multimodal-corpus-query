"""Draw the Paiwan seed/scoring split of PREREGISTRATION.md Amendment 3 item 4.

"The twelve Paiwan broadcast days are divided into a seed half and a scoring half, six and six,
drawn by continuing the single default_rng(20261005) stream after every draw of Amendment 1,
taking 6 of the 12 Paiwan days without replacement in date order."

Implementation, fixed in this file before it is run (commit before the first run):
  1. A fresh default_rng(seed from draw.json's design) replays every recorded step of
     draw.json's stream with the recorded call; each replayed result must equal the recorded
     one, or the script stops. This puts the generator where Amendment 1's draw left it.
  2. The twelve Paiwan drawn days are put in date order (index 0 = earliest).
  3. rng.choice(12, size=6, replace=False) is drawn once. The six days it returns are the
     SEED half; the other six are the SCORING half. (Amendment 3 names the seed half first and
     describes the draw as taking six days; the days taken are the seed half.)
  4. The step is appended to the stream as step 151 in the output record; draw.json itself is
     not edited.
Reads only analysis/study3/draw.json. Writes --out (analysis/study3/split.json), no timestamp.
"""
import argparse, hashlib, json, re, sys
import numpy as np

CALL = re.compile(r"^rng\.choice\((\d+)(?:, size=(\d+), replace=False)?\)$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", default="analysis/study3/draw.json")
    ap.add_argument("--out", default="analysis/study3/split.json")
    a = ap.parse_args()
    raw = open(a.draw, "rb").read()
    d = json.loads(raw.decode("utf-8"))
    seed = d["design"]["seed"]
    rng = np.random.default_rng(seed)
    for s in d["stream"]:
        m = CALL.match(s["call"])
        if not m:
            sys.exit(f"step {s['step']}: unrecognised call {s['call']!r}")
        n, size = int(m[1]), m[2]
        got = [int(rng.choice(n))] if size is None else [int(i) for i in rng.choice(n, size=int(size), replace=False)]
        if got != s["result"]:
            sys.exit(f"step {s['step']}: replay gave {got}, draw.json has {s['result']}; stopping")
    days = sorted(day["broadcast_date"] for day in d["languages"]["Paiwan"]["drawn_days"])
    if len(days) != 12:
        sys.exit(f"Paiwan has {len(days)} drawn days, not 12; stopping")
    take = [int(i) for i in rng.choice(12, size=6, replace=False)]
    seed_days = sorted(days[i] for i in take)
    scoring_days = sorted(x for x in days if x not in seed_days)
    out = {
        "rule": "PREREGISTRATION.md Amendment 3 item 4; implementation in scripts/draw_study3_split.py",
        "inputs": {"draw": {"path": a.draw.replace("\\", "/"), "sha256": hashlib.sha256(raw).hexdigest()}},
        "numpy_version": np.__version__,
        "numpy_version_of_draw": d["numpy_version"],
        "replayed_steps": len(d["stream"]),
        "step": {"step": len(d["stream"]) + 1, "language": "Paiwan", "purpose": "seed half of the 12 days",
                 "call": "rng.choice(12, size=6, replace=False)", "result": take},
        "paiwan_days_in_date_order": days,
        "seed_half": seed_days,
        "scoring_half": scoring_days,
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1); f.write("\n")
    print("replayed", len(d["stream"]), "steps OK; seed half", seed_days, "; scoring half", scoring_days)

if __name__ == "__main__":
    main()
