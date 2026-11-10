"""Reproduce the accepted reference-impact aggregates and random-rank context.

Seven explicitly supplied, SHA-256/size-bound inputs only. Research IDs remain
in memory. No text/media/features/models, fitting, resampling or reranking.
Output is aggregate-only and is created exclusively, never overwritten.

Inputs are supplied explicitly; hashes and sizes are checked against the fixed source contract.
An optional saved aggregate summary is checked without changing the computations.
"""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import tempfile

import numpy as np

EXPECTED = {
    "items": (88870, "99bedb757963ea071d71c58254a7ea2e5d5a7ade6c046c64b05fb83cbfae67e2"),
    "primary_oof": (17795, "f1d036398fa05e54ef0650923e547e395a4d4b4c4a7d0872a6add549d772223f"),
    "saved_s": (4343284, "c8519e769954ade2e398cd01232674c0745b6c3ebd70903b2da123f84135d4d6"),
    "positive_cases": (15493, "1630fe51cc3b6a7455b2cafc55db1213e5269b44f6758c18fd10f14e3b697833"),
    "answers_1": (1958, "f494fa565d55a17d38eae50afa82852f26bc769c458d37372f799d3c1f20460b"),
    "answers_2": (7374, "5173dbafb6f1b301f85c503292966560aacdf956840ab5fa84c16d92b08ab3af"),
    "answers_3": (1576, "b06dfaf8b191531a712d59d355035c22fce66847663079c883b3cbd72ab076d6"),
}
ROUTES = ("Audio", "ASR_XLMR", "ASR_TFIDF")
PRIOR_SUMMARY_SHA = "834ef218f6fb863757e68f494abc2c9822d1e8a0a2c9a11183ae67d4c0c4d139"
METADATA = ("item_id", "utt_id", "batch", "day", "clip", "stratum", "incl_prob", "form")


def need(ok, message):
    # Deliberately no individual research identities in errors or stdout.
    if not ok:
        raise ValueError(message)


def record(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def guard(path, size, digest):
    value = record(path)
    need(value == {"bytes": size, "sha256": digest}, "Bound source hash/size mismatch")
    return value


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unique(rows, key):
    found = {}
    for row in rows:
        need(row[key] not in found, "Duplicate ID")
        found[row[key]] = row
    return found


def validate_codes(rows, expected_ids):
    found = {}
    for key, code in rows:
        need(key not in found, "Duplicate answer ID")
        need(code in ("1", "2", "3"), "Illegal response code")
        found[key] = int(code)
    need(set(found) == expected_ids, "Wrong/missing/extra batch IDs")
    return found


def read_codes(path, expected_ids):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        need(next(reader) == ["item_id", "answer", "seconds", "time"], "Wrong CSV schema")
        # Time and seconds are never interpreted or exported.
        return validate_codes(((r[0], r[1]) for r in reader), expected_ids)


def cross_tabs(pairs, groups):
    result = []
    for role, ids in groups:
        chosen = {i: pair for i, pair in pairs.items() if i in ids}
        cc = Counter(chosen.values())
        result.append({
            "role": role, "set_size": len(ids), "repeat_intersection_total": len(chosen),
            "first_affirmative_in_repeat_intersection": sum(a == 1 for a, b in chosen.values()),
            "retained_first_affirmatives": cc[1, 1],
            "changed_first_affirmatives": sum(a == 1 and b != 1 for a, b in chosen.values()),
            "affirmative_to_negative": cc[1, 2], "affirmative_to_cannot_tell": cc[1, 3],
            "all_changes_in_repeat_intersection": sum(a != b for a, b in chosen.values()),
            "confusion_first_by_repeat": [[cc[a, b] for b in (1, 2, 3)] for a in (1, 2, 3)],
        })
    return result


def independent_cross_check(pairs, groups, actual):
    # Independently partition unit IDs by transition and reconstruct each matrix.
    partitions = {(a, b): {uid for uid, pair in pairs.items() if pair == (a, b)}
                  for a in (1, 2, 3) for b in (1, 2, 3)}
    for (role, ids), row in zip(groups, actual):
        mat = [[len(ids & partitions[a, b]) for b in (1, 2, 3)] for a in (1, 2, 3)]
        need(row["role"] == role and row["set_size"] == len(ids), "Role population mismatch")
        need(row["repeat_intersection_total"] == len(set(pairs) & ids) == sum(map(sum, mat)),
             "Independent intersection mismatch")
        need(mat == row["confusion_first_by_repeat"], "Independent matrix mismatch")
        need(row["first_affirmative_in_repeat_intersection"] == sum(mat[0]) and
             row["retained_first_affirmatives"] == mat[0][0] and
             row["changed_first_affirmatives"] == sum(mat[0][1:]) and
             row["affirmative_to_negative"] == mat[0][1] and
             row["affirmative_to_cannot_tell"] == mat[0][2] and
             row["all_changes_in_repeat_intersection"] == sum(mat[a][b] for a in range(3) for b in range(3) if a != b),
             "Independent transition count mismatch")


def exact(value):
    return {"numerator": value.numerator, "denominator": value.denominator, "decimal": float(value)}


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def pmf(N, M, K, h):
    return Fraction(choose(M, h) * choose(N - M, K - h), choose(N, K))


def tail(N, M, K, h):
    return sum((pmf(N, M, K, j) for j in range(h, M + 1)), Fraction())


def independent_tail(N, M, K, h):
    # Count positives among the omitted items, not selected items.
    omitted = N - K
    return sum((Fraction(choose(M, a) * choose(N - M, omitted - a), choose(N, omitted))
                for a in range(M - h + 1)), Fraction())


def rank_context(N, M, observations, ranks):
    distributions = []
    for K in (8, 90):
        probabilities = [pmf(N, M, K, j) for j in range(M + 1)]
        need(sum(probabilities, Fraction()) == 1, "PMF normalization mismatch")
        need(sum((j * p for j, p in enumerate(probabilities)), Fraction()) == Fraction(M * K, N),
             "PMF mean mismatch")
        distributions.append({"K": K, "mean_hits": exact(Fraction(M * K, N)),
                              "probability_by_hits": [{"hits": j, "probability": exact(p)}
                                                      for j, p in enumerate(probabilities)],
                              "mean_precision": exact(Fraction(M, N))})
    comparisons = []
    for baseline, route, K, hS, hG in observations:
        need(tail(N, M, K, hS) == independent_tail(N, M, K, hS) and
             tail(N, M, K, hG) == independent_tail(N, M, K, hG), "Exact complement check failed")
        comparisons.append({"baseline": baseline, "route": route, "K": K,
                            "observed_symbolic_hits": hS, "observed_similarity_hits": hG,
                            "expected_random_hits": exact(Fraction(M * K, N)),
                            "random_upper_tail_at_symbolic_hits": exact(tail(N, M, K, hS)),
                            "random_upper_tail_at_similarity_hits": exact(tail(N, M, K, hG)),
                            "source_arrays": "saved_s::main_indices,y,main_mask_<baseline>,main_mask_<route>_at_<baseline>",
                            "status": "conditional_descriptive_tail_not_new_confirmatory_p_value"})
    backgrounds = []
    for route, r in ranks.items():
        need(Fraction(sum(j <= r for j in range(1, N + 1)), N) == Fraction(r, N), "CDF enumeration mismatch")
        backgrounds.append({"role": "R3", "route": route, "saved_first_rank": r,
                            "uniform_fixed_target_rank_CDF": exact(Fraction(r, N)),
                            "is_prespecified_significance_p": False,
                            "postselected_target_and_stopping_point": True})
    return {"bank": {"N": N, "M_known_positive": M, "K": [8, 90]},
            "full_six_comparisons": comparisons, "complete_two_budget_distributions": distributions,
            "R3_posthoc_background": backgrounds,
            "tail_probabilities": {"K90_ge10": exact(tail(N, M, 90, 10)),
                                   "K90_all11": exact(tail(N, M, 90, 11)),
                                   "K8_ge1": exact(tail(N, M, 8, 1)),
                                   "K8_ge2": exact(tail(N, M, 8, 2)),
                                   "fixed_target_top90_probability": exact(Fraction(90, 99))}}


def compare_summary(result, summary):
    old = read_json(summary)
    reference = old["reference_stability"]
    need(result["reference_stability"]["aggregate_cross_tabs"] == reference["aggregate_cross_tabs"],
         "Accepted summary cross-tabs differ")
    saved = old["conditional_random_ranking"]
    current = result["conditional_random_ranking"]
    for key in ("bank", "complete_two_budget_distributions"):
        need(current[key] == saved[key], "Accepted summary arithmetic differs")
    need(len(current["full_six_comparisons"]) == len(saved["full_six_comparisons"]) == 6, "Six comparison count differs")
    for new, previous in zip(current["full_six_comparisons"], saved["full_six_comparisons"]):
        # Locator naming changes from original summary to explicit saved arrays;
        # every scientific field, exact fraction and descriptive status must agree.
        need({k: v for k, v in new.items() if k != "source_arrays"} ==
             {k: v for k, v in previous.items() if k != "source_pointer"}, "Accepted six-row arithmetic differs")


def compute(paths):
    records = {role: guard(paths[role], *expected) for role, expected in EXPECTED.items()}
    items = [{k: r[k] for k in METADATA} for r in read_json(paths["items"])["items"]]
    main = [r for r in items if r["batch"] in (1, 2)]
    repeated = [r for r in items if r["batch"] == 3]
    need(len(main) == 240 and len(repeated) == 40, "Wrong frozen metadata size")
    first, repeats = unique(main, "utt_id"), unique(repeated, "utt_id")
    unique(items, "item_id")
    codes = {b: read_codes(paths["answers_" + str(b)], {str(r["item_id"]) for r in items if r["batch"] == b})
             for b in (1, 2, 3)}
    code = lambda r: codes[r["batch"]][str(r["item_id"])]
    for uid, row in repeats.items():
        need(uid in first and all(row[k] == first[uid][k] for k in ("day", "clip", "stratum", "incl_prob")),
             "Frozen repeat metadata differs")
    pairs = {uid: (code(first[uid]), code(row)) for uid, row in repeats.items()}
    counts = Counter(pairs.values())
    confusion = [[counts[a, b] for b in (1, 2, 3)] for a in (1, 2, 3)]
    need(confusion == [[1, 4, 0], [0, 30, 0], [0, 0, 5]], "Accepted overall confusion differs")
    with np.load(paths["primary_oof"], allow_pickle=False) as z:
        primary_ids, primary_y = list(map(str, z["ids"])), z["y"].copy()
    need(len(primary_ids) == len(set(primary_ids)) == 192 and int(primary_y.sum()) == 20, "Wrong primary reference size")
    need(all(code(first[uid]) == (1 if y else 2) for uid, y in zip(primary_ids, primary_y)), "Primary labels differ")
    positive20 = {uid for uid, y in zip(primary_ids, primary_y) if y == 1}
    observations = []
    with np.load(paths["saved_s"], allow_pickle=False) as z:
        ids, y = list(map(str, z["ids"])), z["y"]
        need(ids == [r["utt_id"] for r in main], "S unit order differs")
        need(all(int(v) == {1: 1, 2: 0, 3: -1}[code(first[uid])] for uid, v in zip(ids, y)), "S labels differ")
        Q = {ids[int(i)] for i in z["query_indices"]}
        C = {ids[int(i)] for i in z["main_indices"] if int(y[int(i)]) == 1}
        need(len(Q) == 10 and len(C) == 11 and not Q & C, "Wrong seed/target identity")
        need(Q | C == {uid for uid, r in first.items() if code(r) == 1}, "Seed/target partition differs")
        idx = z["main_indices"]
        yp = y[idx] == 1
        N, M = len(idx), int(yp.sum())
        need((N, M) == (99, 11), "Wrong finite bank")
        for baseline in ("Phone", "Forms"):
            mask = z["main_mask_" + baseline]
            K, hS = int(mask.sum()), int((mask & yp).sum())
            need(K == (8 if baseline == "Phone" else 90), "Wrong fixed budget")
            for route in ROUTES:
                gm = z["main_mask_" + route + "_at_" + baseline]
                need(gm.shape == mask.shape == (99,) and int(gm.sum()) == K, "Wrong saved membership mask")
                observations.append((baseline, route, K, hS, int((gm & yp).sum())))
        actual_r3_ids = {ids[int(idx[j])] for j in range(N) if yp[j] and not z["main_mask_Forms"][j]}
    cases = read_json(paths["positive_cases"])
    r3 = [r for r in cases if all(not m["symbolic"] for m in r["all6_memberships"] if m["baseline"] == "Forms")]
    need(len(r3) == 1 and actual_r3_ids == {r3[0]["utt_id"]}, "R3 unique role differs")
    r3 = r3[0]
    ranks = {m["route"]: m["first_similarity_rank"] for m in r3["all6_memberships"]}
    need(ranks == {"Audio": 24, "ASR_XLMR": 27, "ASR_TFIDF": 50}, "R3 accepted ranks differ")
    groups = [("primary_all_decidable", set(primary_ids)), ("primary_positive_reference", positive20),
              ("Q_seed_positive_reference", Q), ("C_scoring_positive_reference", C)]
    cross = cross_tabs(pairs, groups)
    independent_cross_check(pairs, groups, cross)
    uid = r3["utt_id"]
    flags = {"is_in_retest": uid in pairs,
             "first_positive_changed_on_retest": None if uid not in pairs else pairs[uid][0] != pairs[uid][1],
             "first_positive_retained_on_retest": None if uid not in pairs else pairs[uid][0] == pairs[uid][1],
             "role_only_no_ID_or_answer_export": True}
    context = rank_context(N, M, observations, {"Non-text": ranks["Audio"], "ASR_XLMR": ranks["ASR_XLMR"],
                                               "ASR_TFIDF": ranks["ASR_TFIDF"]})
    for role, expected in EXPECTED.items():
        need(guard(paths[role], *expected) == records[role], "Bound source changed during computation")
    return {"schema": "reference-and-rank-context/1", "status": "REPRODUCED_AGGREGATES_ONLY",
            "accepted_prior_summary_sha256": PRIOR_SUMMARY_SHA,
            "sources": {role: {"filename": paths[role].name, **rec} for role, rec in records.items()},
            "reference_stability": {"response_codes": {"1": "directive", "2": "not directive", "3": "cannot tell"},
                                    "matrix_axes": "Rows first, columns repeat, code order1/2/3",
                                    "repeat_n": len(pairs), "confusion_first_by_repeat": confusion,
                                    "aggregate_cross_tabs": cross, "R3_role_flags": flags},
            "conditional_random_ranking": context,
            "scope": {"IDs_or_individual_answers_exported": False,
                      "CSV_fields_interpreted": ["item_id", "answer"], "metadata_fields": list(METADATA),
                      "sources_unchanged": True, "reference_and_seeds_unchanged": True,
                      "models_scores_features_or_new_ranks_computed": False,
                      "RNG_bootstrap_API_or_media_used": False,
                      "interpretation": "Fixed-bank, fixed-K descriptive null context; R3 is postselected. No joint independent-routes null or weighted-metric null is claimed. No repeat evidence for R3 is not evidence of stability."},
            "validation": {"independent_transition_sets_and_matrices": True, "exact_PMF_sum_and_mean": True,
                           "exact_omitted_sample_tail_agreement": True, "uniform_rank_CDF_enumeration": True}}


def self_test():
    def rejects(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("Expected rejection")
    rejects(lambda: unique([{"id": "a"}, {"id": "a"}], "id"))
    rejects(lambda: validate_codes([("a", "1"), ("a", "2")], {"a"}))
    rejects(lambda: validate_codes([("wrong_batch", "1")], {"a"}))
    rejects(lambda: validate_codes([("a", "9")], {"a"}))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "synthetic"
        path.write_bytes(b"synthetic")
        rejects(lambda: guard(path, 9, "0" * 64))
    pairs = {"a": (1, 2), "b": (1, 1), "c": (3, 3)}
    groups = [("synthetic", {"a", "b"}), ("empty", set())]
    independent_cross_check(pairs, groups, cross_tabs(pairs, groups))
    need(tail(5, 2, 2, 1) == independent_tail(5, 2, 2, 1) == Fraction(7, 10), "Synthetic tail")
    return {"status": "PASS", "synthetic_checks": 7,
            "checks": ["duplicate metadata", "duplicate answer", "wrong batch", "illegal code", "wrong hash",
                       "independent transitions including empty group", "exact tail"], "research_sources_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for role in EXPECTED:
        parser.add_argument("--" + role.replace("_", "-"), type=Path)
    parser.add_argument("--verify-summary", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    need(args.out is not None and args.verify_summary is not None and
         all(getattr(args, role) is not None for role in EXPECTED), "Supply all seven sources, summary and a new output path")
    need(not args.out.exists(), "Output exists; preserve original result")
    paths = {role: getattr(args, role).resolve() for role in EXPECTED}
    result = compute(paths)
    compare_summary(result, args.verify_summary)
    result["validation"]["accepted_summary_all_cross_tabs_and_six_arithmetic_rows_exact"] = True
    result["validation"]["synthetic"] = self_test()
    result["program"] = {"filename": Path(__file__).name, **record(Path(__file__))}
    # x mode also protects against a concurrent file appearing after the check.
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "repeat_n": result["reference_stability"]["repeat_n"],
                      "role_groups": 4, "rank_comparisons": 6, "output": record(args.out)}))


if __name__ == "__main__":
    main()
