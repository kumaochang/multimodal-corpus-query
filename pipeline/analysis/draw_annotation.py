"""Draw the items the Paiwan annotator judges (Amendment 9 item 5; one annotator, small load).

Design (fixed before any item is judged):
- Strata from form_candidates.py; quotas QUOTA (240 in all): marker 50, suffix_cue 100,
  suffix_nocue 60, none_cue 30; none_nocue is not drawn. A stratum smaller than its quota is
  taken whole; the shortfall goes, in the order of SPILL, to strata that still have items.
  Inclusion probability per stratum = drawn / available; analysis weights are its inverse.
- Batch 1 = 50 items keeping the stratum shares of the 240 (go/no-go: batch 2 only if batch 1 has
  >= 3 "directive" answers). Batch 2 = the other 190. Batch 3 = 40 repeats, simple random from
  batches 1-2, judged no earlier than 7 days after batch 2 (intra-rater agreement).
- Order within each batch is random. numpy default_rng(SEED).
Output: --out JSON (items with batch, position, stratum, inclusion probability). Prints counts.
"""
import argparse, collections, json
from pathlib import Path
import numpy as np

SEED, N_B1, N_REP = 20261018, 50, 40
QUOTA = {"marker": 50, "suffix_cue": 100, "suffix_nocue": 60, "none_cue": 30}
SPILL = ["suffix_cue", "suffix_nocue", "marker", "none_cue"]

def alloc(sizes, total):
    tot = sum(sizes.values())
    raw = {k: total * v / tot for k, v in sizes.items()}
    base = {k: min(int(raw[k]), sizes[k]) for k in sizes}
    for k in sorted(sizes, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if sum(base.values()) >= total:
            break
        if base[k] < sizes[k]:
            base[k] += 1
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()]
    pool.sort(key=lambda r: r["utt_id"])
    rng = np.random.default_rng(SEED)
    by = collections.defaultdict(list)
    for r in pool:
        by[r["stratum"]].append(r)
    take = {k: min(q, len(by[k])) for k, q in QUOTA.items()}
    short = sum(QUOTA.values()) - sum(take.values())
    for k in SPILL:
        add = min(short, len(by[k]) - take[k])
        take[k] += add; short -= add
    take = {k: v for k, v in take.items() if v}
    drawn = []
    for k, m in take.items():
        idx = rng.choice(len(by[k]), size=m, replace=False)
        p = m / len(by[k])
        drawn += [dict(by[k][i], incl_prob=p) for i in sorted(idx)]
    b1n = alloc(collections.Counter(r["stratum"] for r in drawn), min(N_B1, len(drawn)))
    b1 = set()
    for k, m in b1n.items():
        ids = [r["utt_id"] for r in drawn if r["stratum"] == k]
        b1 |= {ids[i] for i in rng.choice(len(ids), size=m, replace=False)}
    for r in drawn:
        r["batch"] = 1 if r["utt_id"] in b1 else 2
    rep = [dict(drawn[i], batch=3, repeat_of=drawn[i]["utt_id"])
           for i in sorted(rng.choice(len(drawn), size=min(N_REP, len(drawn)), replace=False))]
    items = drawn + rep
    for b in (1, 2, 3):
        sel = [r for r in items if r["batch"] == b]
        for pos, j in enumerate(rng.permutation(len(sel)), 1):
            sel[j]["position"] = pos
    for r in items:
        r["item_id"] = f"b{r['batch']}_{r['position']:03d}"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": SEED, "numpy_version": np.__version__, "design": {"quota": QUOTA, "spill": SPILL, "batch1": N_B1, "repeats": N_REP},
               "pool_sizes": {k: len(v) for k, v in by.items()}, "taken": take,
               "items": sorted(items, key=lambda r: r["item_id"])},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    c = collections.Counter((r["batch"], r["stratum"]) for r in items)
    print(json.dumps({"pool": {k: len(v) for k, v in by.items()}, "taken": take,
                      "batch_x_stratum": {f"{b}|{s}": v for (b, s), v in sorted(c.items())}}))

if __name__ == "__main__":
    main()
