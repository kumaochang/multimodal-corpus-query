"""Recalculate finite-bank summaries from a completed controlled numeric cache."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contracts import create_json, json_load, require, sha_file
from numerics import comparison_summary, completion_bounds, return_mask, route_scores


def verify(folder):
    provenance = json_load(folder / "provenance.json")
    require(provenance["status"] == "computed_pending_independent_cache_validation", "wrong run status")
    for name, record in provenance["outputs"].items():
        p = folder / name
        require(p.is_file() and p.stat().st_size == record["bytes"] and sha_file(p) == record["sha256"],
                "saved output differs from provenance")
    stored = json_load(folder / "summary.json")
    with (folder / "known_positive_cases.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    case_index = {(r["bank"], r["baseline"], r["route"], r["utt_id"]): r for r in cases}
    require(len(case_index) == len(cases), "duplicate case identity")
    case_count = 0
    with np.load(folder / "numeric_cache.npz", allow_pickle=False) as z:
        q, inclusive = z["query_indices"], z["inclusive_indices"]
        for name in ("whisper", "prosody", "K", "xlmr", "tfidf"):
            expected = z["normalized_" + name][q] @ z["normalized_" + name][inclusive].T
            np.testing.assert_array_equal(z["inclusive_raw_" + name], expected)
        for bank in ("main", "inclusive"):
            idx = z[bank + "_indices"]
            y, weights = z["y"][idx], z["weights"][idx]
            raw = {name: z[bank + "_raw_" + name] for name in ("whisper", "prosody", "K", "xlmr", "tfidf")}
            positions = np.flatnonzero(np.isin(inclusive, idx))
            for name in raw:
                np.testing.assert_array_equal(raw[name], z["inclusive_raw_" + name][:, positions])
            valid = z["k_present"][q, None] & z["k_present"][idx][None, :]
            scores, rankings, detail = route_scores(raw, valid)
            np.testing.assert_array_equal(detail["audio_channel_count"], z[bank + "_audio_channel_count"])
            masks = {name: z[bank + "_mask_" + name] for name in ("Phone", "Forms")}
            pairs = []
            for row in stored[bank]["comparisons"]:
                name, baseline = row["route"], row["baseline"]
                np.testing.assert_array_equal(scores[name], z[bank + "_aggregate_" + name])
                np.testing.assert_array_equal(rankings[name], z[bank + "_ranks_" + name])
                budget = int(masks[baseline].sum())
                key = name + "_at_" + baseline
                masks[key] = return_mask(rankings[name], budget, len(idx))
                np.testing.assert_array_equal(masks[key], z[bank + "_mask_" + key])
                result = comparison_summary(masks[baseline], masks[key], y, weights)
                require(all(row[k] == v for k, v in result.items()), "summary does not match cache")
                pairs.append((key, baseline))
                for i in np.flatnonzero(y == 1):
                    case = case_index[(bank, baseline, name, str(z["ids"][idx[i]]))]
                    require(case["symbolic"] == str(bool(masks[baseline][i])) and
                            case["similarity"] == str(bool(masks[key][i])), "case membership differs")
                    require(case["day"] == str(z["days"][idx[i]]) and case["clip"] == str(z["clips"][idx[i]]),
                            "case metadata differs")
                    case_count += 1
            if bank == "inclusive":
                bounds, bits = completion_bounds(y, weights, masks, pairs)
                require(bounds == stored[bank]["logical_bounds"], "completion bounds differ")
                np.testing.assert_array_equal(bits, z["unknown_completion_bits"])
    require(case_count == len(cases), "extra case rows")
    return {"status": "PASS", "scope": provenance["scope"], "comparisons": 12,
            "case_rows_checked": case_count, "shared_completions_checked": 8192,
            "scores_rankings_masks_summaries_recomputed": True,
            "new_research_rankings_created": False,
            "provenance_sha256": sha_file(folder / "provenance.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.folder.resolve())
    create_json(args.out, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
