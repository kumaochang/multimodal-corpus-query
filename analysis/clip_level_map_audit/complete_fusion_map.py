"""Complete the original 15 clip configurations from saved numeric inputs only."""

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD_COMMIT = "95dfc842c3888765c4d8e19ddff8c6a9bfea0ec5"
CHANNELS = ("text", "audio", "prosody", "kinematic")
INPUTS = (
    "analysis/_emb_cache.npz", "analysis/_q8_cache.npz",
    "analysis/q8_gesture_conditional.py", "analysis/q5_baselines.py",
    "analysis/step4_supplementary.py", "pilot/experiment.py",
    "pilot/retrieval.json", "pilot/labels.json", "pilot/labels_log.json",
    "corpus/demo_data.json", "corpus/build_demo_data.py",
    "analysis/clip_level_map_audit/retrieval_results.json",
    "data/clip_level_map_audit/saved_matrix_ranks.npz",
)


def file_record(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def id_permutation(source_ids, target_ids):
    source = [str(value) for value in source_ids]
    target = [str(value) for value in target_ids]
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("duplicate ID")
    if set(source) != set(target):
        raise ValueError("ID sets differ")
    index = {value: i for i, value in enumerate(source)}
    return np.array([index[value] for value in target], dtype=np.int64)


def score_metrics(relevance):
    """Return precision and AP, checking AP by a separate rank-position formula."""
    rel = np.asarray(relevance, dtype=bool)
    if rel.size == 0:
        return {"P1": None, "P3": None, "AP": None, "n_rel": 0, "n_cand": 0}
    n_rel = int(rel.sum())
    ap = None
    if n_rel:
        hits = 0
        precision_sum = 0.0
        for rank, relevant in enumerate(rel, start=1):
            if relevant:
                hits += 1
                precision_sum += hits / rank
        ap = precision_sum / n_rel
        positions = np.flatnonzero(rel) + 1
        independent = float(np.mean(np.arange(1, n_rel + 1) / positions))
        np.testing.assert_allclose(ap, independent, atol=1e-15, rtol=0)
    return {"P1": float(rel[0]), "P3": float(rel[:3].mean()), "AP": ap,
            "n_rel": n_rel, "n_cand": int(rel.size)}


def saved_order(scores, query):
    """Preserve the historical quicksort tie order before removing self."""
    values = np.asarray(scores).copy()
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite row")
    values[query] = -1
    legacy = np.argsort(values, kind="quicksort")[::-1]
    return legacy, legacy[legacy != query]


def pure_evaluator(source):
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "retrieval_eval")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pure retrieval_eval", "exec"),
         namespace)
    return namespace["retrieval_eval"]


def deterministic_npz(path, arrays):
    """Fix ZIP metadata so a second run can verify the complete cache bytes."""
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())


def load_vectors(ids, demo):
    with np.load(ROOT / INPUTS[0], allow_pickle=False) as source:
        source_ids = source["ids"]
        permutation = id_permutation(source_ids, ids)
        vectors = {name: source[key][permutation] for name, key in
                   (("text", "T"), ("audio", "A"), ("prosody", "P"))}
        arrays = {"tap_source_ids": source_ids, "tap_permutation": permutation}
    with np.load(ROOT / INPUTS[1], allow_pickle=False) as source:
        source_ids = source["ids"]
        permutation = id_permutation(source_ids, ids)
        vectors["kinematic"] = source["K18"][permutation]
        arrays.update(k_source_ids=source_ids, k_permutation=permutation)
    expected = {"text": 768, "audio": 86, "prosody": 8, "kinematic": 18}
    checks = []
    normalized = {}
    for name, values in vectors.items():
        if values.shape != (30, expected[name]) or not np.isfinite(values).all():
            raise ValueError(f"{name}: invalid frozen vector shape or values")
        normalized[name] = normalize(values)
        arrays[name + "_raw"] = values
        arrays[name + "_normalized"] = normalized[name]
        reference = np.asarray(demo["sim"][name])
        scores = cosine_similarity(normalized[name])
        difference = float(np.max(np.abs(scores - reference)))
        if difference > 5e-7:
            raise ValueError(f"{name}: saved matrix differs by {difference}")
        checks.append({"channel": name, "shape": list(values.shape),
                       "raw_dtype": str(values.dtype), "normalized_dtype": str(normalized[name].dtype),
                       "C_contiguous": bool(values.flags.c_contiguous),
                       "F_contiguous": bool(values.flags.f_contiguous),
                       "zero_rows": int(np.count_nonzero(np.linalg.norm(values, axis=1) == 0)),
                       "max_saved_score_difference": difference})
    return normalized, arrays, checks


def run(output, cache):
    output = output.resolve()
    cache = cache.resolve()
    output.relative_to(ROOT)
    cache.relative_to(ROOT)
    names = ("fusion15_results.json", "fusion15_results.csv", "fusion15_reconciliation.csv",
             "fusion15_validation.json", "fusion15_manifest.json")
    if cache.exists() or any((output / name).exists() for name in names):
        raise FileExistsError("F0 outputs already exist; use another isolated output/cache path")
    sources = {name: file_record(ROOT / name) for name in INPUTS}
    demo = read_json(ROOT / "corpus/demo_data.json")
    ids = [row["id"] for row in demo["clips"]]
    adopted = read_json(ROOT / "pilot/labels.json")
    original = read_json(ROOT / "pilot/labels_log.json")
    id_permutation(ids, list(adopted))
    id_permutation(ids, list(original))
    labels = np.array([adopted[value] for value in ids])
    if len(ids) != 30 or [row["label"] for row in demo["clips"]] != list(labels):
        raise ValueError("frozen IDs or adopted labels differ")
    corrections = [value for value in ids if original[value]["label"] != adopted[value]]
    expected_corrections = {"news-Amis-20260302-05", "news-Truku-20260305-02"}
    if set(corrections) != expected_corrections:
        raise ValueError("adopted correction identity differs")
    reference = read_json(ROOT / "pilot/retrieval.json")["full_fusion_whisper"]
    old_source = (HERE / "historical_evaluator.py").read_text(encoding="utf-8")
    old_eval = pure_evaluator(old_source)
    repaired = [pure_evaluator((ROOT / name).read_text(encoding="utf-8")) for name in
                ("pilot/experiment.py", "analysis/step4_supplementary.py")]
    vectors, arrays, vector_checks = load_vectors(ids, demo)
    arrays.update(ids=np.array(ids), labels=labels)
    with np.load(ROOT / "data/clip_level_map_audit/saved_matrix_ranks.npz", allow_pickle=False) as old:
        regression_archive = {name: old[name] for name in old.files}
    if list(regression_archive["ids"]) != ids or not np.array_equal(regression_archive["labels"], labels):
        raise ValueError("existing seven-route archive identity differs")
    summaries, reconciliation, tie_rows, matrix_checks = [], [], [], []
    for size in range(1, 5):
        for channels in combinations(CHANNELS, size):
            config = "+".join(channels)
            fused = normalize(np.hstack([vectors[name] for name in channels]))
            scores = cosine_similarity(fused)
            historical = old_eval(scores, list(labels))
            orders, legacy_orders, ranked_scores, relevance, metrics, old_aps = [], [], [], [], [], []
            for i in range(30):
                full, order = saved_order(scores[i], i)
                if int(full[-1]) != i or len(order) != 29 or i in order:
                    raise ValueError(f"{config}: self removal contract failed")
                rel = labels[order] == labels[i]
                row = score_metrics(rel)
                legacy_ap = score_metrics(labels[full] == labels[i])["AP"]
                if row["AP"] is None:
                    raise ValueError("unexpected zero-relevant query")
                expected_old = (row["n_rel"] * row["AP"] + (row["n_rel"] + 1) / 30) / (row["n_rel"] + 1)
                np.testing.assert_allclose(legacy_ap, expected_old, atol=1e-15, rtol=0)
                np.testing.assert_allclose(legacy_ap, historical["MAP"][i], atol=1e-15, rtol=0)
                if row["P1"] != historical["P@1"][i] or row["P3"] != historical["P@3"][i]:
                    raise ValueError("self exclusion changed precision")
                unique, counts = np.unique(scores[i, order], return_counts=True)
                tied = unique[counts > 1]
                for value in tied:
                    positions = np.flatnonzero(scores[i, order] == value)
                    tie_rows.append({"configuration": config, "query_row": i,
                                     "score": float(value), "ranks": (positions + 1).tolist(),
                                     "candidate_rows": order[positions].tolist()})
                orders.append(order)
                legacy_orders.append(full)
                ranked_scores.append(scores[i, order])
                relevance.append(rel)
                metrics.append(row)
                old_aps.append(legacy_ap)
            for evaluator in repaired:
                result = evaluator(scores, list(labels))
                for key, field in (("P@1", "P1"), ("P@3", "P3"), ("MAP", "AP")):
                    np.testing.assert_allclose(result[key], [row[field] for row in metrics], atol=1e-15, rtol=0)
            summary = {"configuration": config, "queries": 30, "candidates_per_query": 29,
                       "P1_denominator": 30, "P3_denominator": 30, "MAP_denominator": 30,
                       "no_relevant_queries": 0, "no_candidate_queries": 0,
                       "P1": float(np.mean([row["P1"] for row in metrics])),
                       "P3": float(np.mean([row["P3"] for row in metrics])),
                       "MAP": float(np.mean([row["AP"] for row in metrics])),
                       "historical_MAP_reconstructed": float(np.mean(old_aps)),
                       "fused_dtype": str(fused.dtype)}
            summaries.append(summary)
            for field, saved_key in (("P1", "P@1"), ("P3", "P@3"), ("historical_MAP_reconstructed", "MAP")):
                observed, saved = summary[field], reference[config][saved_key]["mean"]
                delta = observed - saved
                reconciliation.append({"configuration": config, "metric": saved_key,
                                       "reconstructed_historical": observed, "saved_historical": saved,
                                       "difference": delta, "within_1e_12": bool(abs(delta) <= 1e-12)})
                if abs(delta) > 1e-12:
                    raise ValueError(f"{config} {saved_key}: historical mismatch {delta}")
            rank_array = np.asarray(orders, dtype=np.int16)
            prefix = config + "__"
            arrays.update({prefix + "fused_vectors": fused, prefix + "scores": scores,
                           prefix + "legacy_ranks": np.asarray(legacy_orders, dtype=np.int16),
                           prefix + "ranks": rank_array, prefix + "ranked_scores": np.asarray(ranked_scores),
                           prefix + "relevance": np.asarray(relevance),
                           prefix + "P1": np.array([row["P1"] for row in metrics]),
                           prefix + "P3": np.array([row["P3"] for row in metrics]),
                           prefix + "AP": np.array([row["AP"] for row in metrics]),
                           prefix + "legacy_AP": np.asarray(old_aps),
                           prefix + "n_rel": np.array([row["n_rel"] for row in metrics], dtype=np.int16),
                           prefix + "candidate_mask": ~np.eye(30, dtype=bool)})
            route = channels[0] if len(channels) == 1 else "fusion" if len(channels) == 4 else None
            if route:
                saved_scores = np.asarray(demo["sim"][route])
                saved_orders = np.array([saved_order(saved_scores[i], i)[1] for i in range(30)])
                differences = int(np.count_nonzero(rank_array != saved_orders))
                np.testing.assert_allclose(arrays[prefix + "AP"], regression_archive[route + "_corrected_AP"], atol=1e-15, rtol=0)
                score_delta = float(np.max(np.abs(scores - saved_scores)))
                if score_delta > 5e-7:
                    raise ValueError(f"{route}: saved-matrix score difference exceeds tolerance")
                rank_changes = []
                for i in np.flatnonzero(np.any(rank_array != saved_orders, axis=1)):
                    positions = np.flatnonzero(rank_array[i] != saved_orders[i])
                    rank_changes.append({"query_row": int(i), "ranks": (positions + 1).tolist(),
                                         "candidate_rows_new": rank_array[i, positions].tolist(),
                                         "candidate_rows_saved": saved_orders[i, positions].tolist(),
                                         "scores_new": scores[i, rank_array[i, positions]].tolist(),
                                         "scores_saved": saved_scores[i, rank_array[i, positions]].tolist()})
                for i in range(30):
                    old_metrics = score_metrics(labels[saved_orders[i]] == labels[i])
                    for field in ("P1", "P3", "AP"):
                        np.testing.assert_allclose(metrics[i][field], old_metrics[field], atol=1e-15, rtol=0)
                matrix_checks.append({"route": route, "max_score_difference": float(np.max(np.abs(scores - saved_scores))),
                                      "rank_positions_different": differences, "rank_changes": rank_changes,
                                      "per_query_P1_P3_AP_match": True})
    if {row["configuration"] for row in summaries} != set(reference):
        raise ValueError("15-configuration set differs")
    # Also verify the two existing single routes that are outside the 15-config family.
    regression = read_json(ROOT / "analysis/clip_level_map_audit/retrieval_results.json")
    for route in ("bow", "whisper"):
        scores = np.asarray(demo["sim"][route])
        rows = []
        for i in range(30):
            order = regression_archive[route + "_ranks"][i]
            np.testing.assert_array_equal(order, saved_order(scores[i], i)[1])
            rows.append(score_metrics(labels[order] == labels[i]))
        for field, key in (("P1", "P@1"), ("P3", "P@3"), ("AP", "MAP")):
            np.testing.assert_allclose(np.mean([row[field] for row in rows]),
                                       regression["point_estimates"][route][key], atol=1e-15, rtol=0)
    if sources != {name: file_record(ROOT / name) for name in INPUTS}:
        raise ValueError("protected source changed during execution")
    output.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    deterministic_npz(cache, arrays)
    with np.load(cache, allow_pickle=False) as saved:
        for name, value in arrays.items():
            np.testing.assert_array_equal(saved[name], value)
        for summary in summaries:
            prefix = summary["configuration"] + "__"
            for i, order in enumerate(saved[prefix + "ranks"]):
                row = score_metrics(saved["labels"][order] == saved["labels"][i])
                for field, key in (("P1", "P1"), ("P3", "P3"), ("AP", "AP")):
                    np.testing.assert_allclose(row[field], saved[prefix + key][i], atol=1e-15, rtol=0)
    write_csv(output / names[1], summaries)
    write_csv(output / names[2], reconciliation)
    write_json(output / names[0], {"status": "verified_descriptive_point_estimates",
                                  "queries": 30, "candidates_per_query": 29,
                                  "reference_key": "full_fusion_whisper (MFCC, not Whisper)",
                                  "configuration_order": [row["configuration"] for row in summaries],
                                  "results": summaries, "numeric_cache": relative(cache),
                                  "intervals": None, "new_bootstrap_or_model_runs": False})
    validation = {"status": "PASS", "historical_fields_matched": len(reconciliation),
                  "precision_fields_matched": 30, "historical_MAP_fields_matched": 15,
                  "saved_matrix_regression": matrix_checks,
                  "additional_single_route_regressions": ["bow", "whisper"],
                  "vectors": vector_checks, "adopted_label_corrections": 2,
                  "query_orders_verified": True, "all_AP_defined": True,
                  "AP_two_formulas_and_legacy_identity_verified": True,
                  "repaired_pure_evaluators_verified": 2, "cache_readback_verified": True,
                  "exact_ties": tie_rows, "protected_inputs_unchanged": True,
                  "historical_source_commit": OLD_COMMIT,
                  "historical_source_sha256": hashlib.sha256(old_source.encode("utf-8")).hexdigest(),
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                              "platform": platform.platform(), "threadpools": threadpool_info(),
                              "thread_environment": {name: os.environ.get(name) for name in
                                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
                  "sorting": "mask self=-1; quicksort descending; remove self; save full and filtered orders",
                  "reproduction_scope": "saved numeric reconstruction; historical rankings were not recovered"}
    write_json(output / names[3], validation)
    manifest = {"schema": "clip-fusion15-audit/1", "input_records": sources,
                "code": {relative(Path(__file__)): file_record(Path(__file__)),
                         relative(HERE / "PLAN_F0.md"): file_record(HERE / "PLAN_F0.md")},
                "outputs": {relative(output / name): file_record(output / name) for name in names[:-1]},
                "numeric_cache": {"path": relative(cache), **file_record(cache)}}
    write_json(output / names[4], manifest)
    print(json.dumps({"status": "PASS", "configurations": len(summaries),
                      "historical_fields_matched": len(reconciliation), "cache": relative(cache),
                      "cache_record": file_record(cache)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/clip_level_map_audit/fusion15_rankings.npz")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.cache)


if __name__ == "__main__":
    main()
