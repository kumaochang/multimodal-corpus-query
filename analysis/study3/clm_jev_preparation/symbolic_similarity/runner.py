"""Gated finite-bank exploration. Default operations never rank real materials."""

import argparse
import csv
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
from contracts import create_json, fingerprint, json_load, now, require, sha_file
from numerics import (comparison_summary, completion_bounds, normalized_rows,
                      prosody_transform, return_mask, route_scores)


def environment():
    return {"python": platform.python_version(),
            **{name: metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}}


def code_records():
    return {name: sha_file(HERE / name) for name in
            ("runner.py", "numerics.py", "verify_saved.py", "../contracts.py", "SOURCE_CONTRACT.json", "PLAN.md",
             "PROFILES.json", "LOCAL_EXECUTION_ADDENDUM.md")}


def expected_environment(profile):
    profiles = json_load(HERE / "PROFILES.json")["profiles"]
    require(profile in profiles, "unknown numerical consumer profile")
    return profiles[profile]


def input_paths(args, contract):
    paths = {}
    for role, record in contract["sources"].items():
        if role.startswith("answers"):
            paths[role] = args.answer_root / Path(record["path_suffix"]).name
        elif role in ("split", "registered_audit", "coverage"):
            paths[role] = args.repo_root / record["path_suffix"]
        else:
            paths[role] = args.data_root / record["path_suffix"]
    return {role: path.resolve() for role, path in paths.items()}


def verified_sources(paths, contract):
    records = {}
    for role, path in paths.items():
        require(path.is_file(), f"missing explicit source: {role}")
        actual = {"sha256": sha_file(path), "bytes": path.stat().st_size}
        expected = contract["sources"][role]
        require(all(actual[key] == expected[key] for key in actual), f"source hash/size differs: {role}")
        records[role] = {"path": str(path), **actual}
    return records


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unique_index(rows, key):
    values = [row[key] for row in rows]
    require(all(isinstance(value, str) and value for value in values), "invalid identifier")
    require(len(values) == len(set(values)), "duplicate identifier")
    return dict(zip(values, rows))


def prepare(args):
    """Source, metadata and numeric checks, without pair scores/ranks/hits."""
    contract = json_load(HERE / "SOURCE_CONTRACT.json")
    actual_environment = environment()
    require(actual_environment == expected_environment(args.profile), "fixed consumer environment mismatch")
    require(all(pool["num_threads"] == 1 for pool in threadpool_info()), "numerical libraries must use one thread")
    paths = input_paths(args, contract)
    source_records = verified_sources(paths, contract)
    split = json_load(paths["split"])
    seed_days, scoring_days = set(split["seed_half"]), set(split["scoring_half"])
    require(len(seed_days) == len(scoring_days) == 6 and not seed_days & scoring_days, "split differs")
    all_items = json_load(paths["items"])["items"]
    unique_index(all_items, "item_id")
    items = [row for row in all_items if row["batch"] in (1, 2)]
    ids = list(unique_index(items, "utt_id"))
    require(len(items) == 240, "unique first-round population differs")
    answer = {}
    for batch in (1, 2):
        with paths[f"answers{batch}"].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        indexed = unique_index(rows, "item_id")
        expected = {row["item_id"] for row in items if row["batch"] == batch}
        require(set(indexed) == expected, "first-round batch ID set differs")
        require(all(row["answer"] in ("1", "2", "3") for row in rows), "illegal answer")
        answer.update({key: int(row["answer"]) for key, row in indexed.items()})
    pool = unique_index(json_lines(paths["pool"]), "utt_id")
    phone = unique_index(json_lines(paths["phone_core"]) + json_lines(paths["phone_ext"]), "utt_id")
    for row in items:
        require(row["day"] in seed_days | scoring_days, "unexpected item day")
        half = "seed" if row["day"] in seed_days else "scoring"
        require(row["half"] == half, "item half differs from saved split")
        require(row["utt_id"] in pool and row["utt_id"] in phone, "missing membership metadata")
        require(row["form"] == pool[row["utt_id"]]["form"], "saved form classification differs")
    y = np.array([1 if answer[row["item_id"]] == 1 else 0 if answer[row["item_id"]] == 2 else -1
                  for row in items], dtype=np.int8)
    weights = np.array([1 / float(row["incl_prob"]) for row in items], dtype=np.float64)
    require(np.isfinite(weights).all() and (weights > 0).all(), "invalid inclusion weights")
    q = np.array([i for i, row in enumerate(items) if row["day"] in seed_days and y[i] == 1])
    inclusive = np.array([i for i, row in enumerate(items) if row["day"] in scoring_days])
    main = inclusive[y[inclusive] >= 0]
    indices = {"main": main, "inclusive": inclusive}
    expected_bindings = contract["ordered_bindings"]
    for key, idx in (("all_positive_seed_queries", q), ("decidable_scoring_bank", main),
                     ("unknown_inclusive_scoring_bank", inclusive)):
        selected = [items[i] for i in idx]
        expected = expected_bindings[key]
        require(len(selected) == expected["n"], "ordered population count differs")
        require(fingerprint([row["utt_id"] for row in selected]) == expected["ordered_ids_sha256"],
                "ordered population ID hash differs")
        require(len({row["day"] for row in selected}) == expected["days"], "population day count differs")
        require(len({row["clip"] for row in selected}) == expected["clips"], "population clip count differs")
    require(not set(q) & set(inclusive), "query/candidate overlap")
    require(int((y[main] == 1).sum()) == 11 and int((y[inclusive] < 0).sum()) == 13, "reference support differs")
    arrays = {}
    for role in ("text", "audio", "kin"):
        with np.load(paths[role], allow_pickle=False) as source:
            require(source["ids"].tolist() == ids, f"feature IDs misaligned: {role}")
            for key in ({"text": ("xlmr", "tfidf"), "audio": ("whisper", "prosody"),
                         "kin": ("K", "k_missing")}[role]):
                values = source[key]
                expected = contract["feature_headers"][role][key]
                require(list(values.shape) == expected["shape"] and str(values.dtype) == expected["dtype"],
                        f"feature shape/dtype differs: {role}/{key}")
                arrays[key] = np.array(values, order="C", copy=True)
    require(np.isin(arrays["k_missing"], [0, 1]).all(), "illegal k_missing")
    k_present = arrays["k_missing"] == 0
    require(np.isfinite(arrays["K"][k_present]).all(), "observed K is nonfinite")
    arrays["K"][~k_present] = 0
    for name in ("xlmr", "tfidf", "whisper"):
        require(np.isfinite(arrays[name]).all(), f"unexpected nonfinite values: {name}")
    prosody, transform = prosody_transform(arrays["prosody"])
    normalized = {name: normalized_rows(arrays[name]) for name in ("xlmr", "tfidf", "whisper", "K")}
    normalized["prosody"] = prosody
    symbolic, budgets = {}, {}
    for name, idx in indices.items():
        phone_values = [phone[ids[i]]["in_A"] for i in idx]
        require(all(type(value) is bool for value in phone_values), "phone membership must be boolean")
        symbolic[name] = {"Phone": np.array(phone_values),
                          "Forms": np.array([items[i]["form"] in ("marker", "suffix") for i in idx])}
        budgets[name] = {route: int(mask.sum()) for route, mask in symbolic[name].items()}
        require(budgets[name] == contract["budgets"][name], "matched budgets differ")
    evidence = {"schema": "symbolic-similarity-preflight/1", "status": "PASS", "profile": args.profile,
                "scope": contract["scope"], "sources": source_records,
                "code_records": code_records(), "environment": actual_environment,
                "threadpools": threadpool_info(), "ordered_bindings": expected_bindings,
                "budgets": budgets, "support": contract["counts"],
                "numeric": {name: {"shape": list(values.shape), "dtype": str(values.dtype),
                                    "zero_rows": int(np.count_nonzero(np.linalg.norm(values, axis=1) == 0)),
                                    "C_contiguous": bool(values.flags.c_contiguous)}
                            for name, values in normalized.items()},
                "prosody_nan_cells": int(transform["nan_mask"].sum()),
                "unexpected_nonfinite": 0, "cosines_rankings_hits_computed": False}
    data = {"ids": np.array(ids), "items": items, "q": q, "banks": indices, "y": y,
            "weights": weights, "normalized": normalized, "k_present": k_present,
            "prosody_transform": transform, "symbolic": symbolic}
    return evidence, data


def execute(args):
    started = time.perf_counter()
    require(args.preflight is not None and args.preflight_sha256 is not None, "verified preflight required")
    require(sha_file(args.preflight) == args.preflight_sha256, "preflight hash differs")
    previous = json_load(args.preflight)
    require(previous.get("status") == "PASS" and previous.get("schema") == "symbolic-similarity-preflight/1",
            "not a numeric research preflight")
    current, data = prepare(args)
    for field in ("scope", "profile", "sources", "code_records", "environment", "ordered_bindings", "budgets", "numeric"):
        require(current[field] == previous.get(field), f"preflight no longer matches: {field}")
    require(args.out.is_absolute(), "execution output must be an explicit absolute controlled path")
    protected_paths = [Path(record["path"]) for record in current["sources"].values()]
    require(not args.out.exists(), "output exists; do not overwrite")
    require(not any(args.out == path or args.out in path.parents for path in protected_paths),
            "output would contain a protected source")
    args.out.mkdir(parents=True)
    archive = {"ids": data["ids"], "query_indices": data["q"], "y": data["y"], "weights": data["weights"],
               "k_present": data["k_present"],
               "days": np.array([row["day"] for row in data["items"]]),
               "clips": np.array([row["clip"] for row in data["items"]])}
    for name, values in data["normalized"].items():
        archive["normalized_" + name] = values
    archive.update({"prosody_" + name: values for name, values in data["prosody_transform"].items()})
    all_candidates = data["banks"]["inclusive"]
    q = data["q"]
    raw_all = {name: values[q] @ values[all_candidates].T for name, values in data["normalized"].items()}
    direct_all = {name: np.einsum("ik,jk->ij", values[q], values[all_candidates], optimize=False)
                  for name, values in data["normalized"].items()}
    score_errors = {name: float(np.max(np.abs(values - direct_all[name]))) for name, values in raw_all.items()}
    require(all(value <= 1e-12 for value in score_errors.values()), "independent raw cosine check failed")
    for name, values in raw_all.items():
        archive["inclusive_raw_" + name] = values
    summaries, case_rows, numerical_checks = {}, [], {}
    for bank, idx in data["banks"].items():
        positions = np.flatnonzero(np.isin(all_candidates, idx))
        raw = {name: values[:, positions] for name, values in raw_all.items()}
        k_valid = data["k_present"][q, None] & data["k_present"][idx][None, :]
        scores, rankings, detail = route_scores(raw, k_valid)
        alternative_scores, alternative_ranks, _ = route_scores(
            {name: values[:, positions] for name, values in direct_all.items()}, k_valid)
        numerical_checks[bank] = {}
        for name, ranking in rankings.items():
            check = {"aggregate_max_difference": float(np.max(np.abs(scores[name] - alternative_scores[name]))),
                     "rank_positions_different": int(np.count_nonzero(ranking != alternative_ranks[name])),
                     "topK_set_difference": {}}
            for baseline, mask in data["symbolic"][bank].items():
                budget = int(mask.sum())
                delta = int(np.count_nonzero(return_mask(ranking, budget, len(idx)) !=
                                              return_mask(alternative_ranks[name], budget, len(idx))))
                check["topK_set_difference"][baseline] = delta
                require(delta == 0, "numeric implementation changes top-K membership; stop before route hits")
            numerical_checks[bank][name] = check
        archive[bank + "_indices"] = idx
        archive[bank + "_audio_channel_count"] = detail["audio_channel_count"]
        archive[bank + "_per_seed_audio"] = detail["per_seed_audio"]
        for name in raw:
            archive[bank + "_raw_" + name] = raw[name]
            archive[bank + "_z_" + name] = detail["z"][name]
            archive[bank + "_valid_" + name] = detail["valid"][name]
            for field, values in detail["parameters"][name].items():
                archive[bank + "_zparam_" + name + "_" + field] = values
        masks = dict(data["symbolic"][bank])
        pairs, comparisons = [], []
        for name, values in scores.items():
            archive[bank + "_aggregate_" + name] = values
            archive[bank + "_ranks_" + name] = rankings[name]
        for baseline, symbolic in data["symbolic"][bank].items():
            archive[bank + "_mask_" + baseline] = symbolic
            budget = int(symbolic.sum())
            for name, ranking in rankings.items():
                key = name + "_at_" + baseline
                masks[key] = return_mask(ranking, budget, len(idx))
                archive[bank + "_mask_" + key] = masks[key]
                pairs.append((key, baseline))
                comparisons.append({"baseline": baseline, "route": name, "K": budget,
                                    "full_bank_returned": budget == len(idx),
                                    **comparison_summary(symbolic, masks[key], data["y"][idx], data["weights"][idx])})
                for local in np.flatnonzero(data["y"][idx] == 1):
                    row = data["items"][idx[local]]
                    case_rows.append({"bank": bank, "baseline": baseline, "route": name,
                                      "utt_id": row["utt_id"], "day": row["day"], "clip": row["clip"],
                                      "symbolic": bool(symbolic[local]), "similarity": bool(masks[key][local])})
        summaries[bank] = {"n": len(idx), "known_positives": int((data["y"][idx] == 1).sum()),
                           "unknown": int((data["y"][idx] < 0).sum()), "comparisons": comparisons}
        if bank == "inclusive":
            bounds, completions = completion_bounds(data["y"][idx], data["weights"][idx], masks, pairs)
            summaries[bank]["logical_bounds"] = bounds
            archive["unknown_completion_bits"] = completions
    np.savez_compressed(args.out / "numeric_cache.npz", **archive)
    with (args.out / "known_positive_cases.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    require(verified_sources(input_paths(args, json_load(HERE / "SOURCE_CONTRACT.json")),
                             json_load(HERE / "SOURCE_CONTRACT.json")) == current["sources"], "input changed during run")
    create_json(args.out / "summary.json", summaries)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        try:
            import psutil
            memory = psutil.Process().memory_info()
            peak_bytes = int(getattr(memory, "peak_wset", memory.rss))
        except ImportError:
            peak_bytes = None
    provenance = {"scope": current["scope"], "status": "computed_pending_independent_cache_validation",
                  "timestamp_utc": now(), "wall_seconds": time.perf_counter() - started,
                  "process_peak_RSS_bytes": peak_bytes, "preflight": {"path": str(args.preflight), "sha256": args.preflight_sha256},
                  "inputs": current["sources"], "code_records": current["code_records"],
                  "profile": current["profile"], "environment": current["environment"], "threadpools": current["threadpools"],
                  "independent_numeric_check": {"raw_cosine_max_errors": score_errors, "banks": numerical_checks},
                  "resource_budget": {"seconds": 600, "bytes": 4 * 1024**3, "OS_hard_memory_cap": False},
                  "outputs": {name: {"sha256": sha_file(args.out / name), "bytes": (args.out / name).stat().st_size}
                              for name in ("numeric_cache.npz", "known_positive_cases.csv", "summary.json")}}
    create_json(args.out / "provenance.json", provenance)
    print(json.dumps({"status": provenance["status"], "out": str(args.out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("environment", "preflight", "execute"))
    parser.add_argument("--profile", default="item5")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--answer-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.operation == "environment":
        expected = expected_environment(args.profile)
        print(json.dumps({"profile": args.profile, "actual": environment(), "expected": expected, "matches": environment() == expected,
                          "real_materials_read": False}, ensure_ascii=False))
        return
    require(args.data_root is not None and args.answer_root is not None and args.out is not None,
            "explicit data, answer and output paths are required")
    with threadpool_limits(limits=1):
        if args.operation == "preflight":
            require(not args.execute, "preflight cannot execute rankings")
            evidence, _ = prepare(args)
            evidence["timestamp_utc"] = now()
            ref = create_json(args.out, evidence)
            print(json.dumps({"status": "numeric_preflight_passed_no_ranking", **ref}))
        else:
            require(args.execute, "real execution is off; explicit --execute required")
            try:
                execute(args)
            except Exception as exc:
                if args.out.is_dir() and not (args.out / "provenance.json").exists():
                    failure = args.out / "INCOMPLETE.json"
                    if not failure.exists():
                        create_json(failure, {"status": "incomplete", "timestamp_utc": now(),
                                              "error_type": type(exc).__name__,
                                              "reason": str(exc), "usable_as_completed_result": False})
                raise


if __name__ == "__main__":
    main()
