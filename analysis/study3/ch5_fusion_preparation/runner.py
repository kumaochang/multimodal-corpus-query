"""F1 frozen numerical adapter, native baseline gate and finite common-bank fusion.

No model inference. Original feature producers/retrieval scripts remain unchanged.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from execution_support import Budget, check_profile, check_sources, clean, extract_functions, record, write_json
from kernels import candidate_mask, day_weights, fuse_normalized, intervals, means_by_day, query_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "analysis/study3/ulfr_local"))
from ulfr_reader import Release

SOURCES = ["ami", "tay", "pwn", "tao", "trv", "tsu"]
POOLS = ["W1", "W2", "W3", "X"]
ROUTES = ["xlmr_sub", "tfidf_sub", "whisper", "mfcc86", "prosody", "kin14"]
METRICS = ["P1", "P3", "MAP", "chance", "lift1"]


def prepare_arrays(z, k, U, l2):
    """Frozen full-population preprocessing before selection, original float precision."""
    M = z["mfcc86"].astype(np.float64)
    mu, sd = M.mean(0), M.std(0)
    Mz = np.where(sd > 0, (M - mu) / np.where(sd > 0, sd, 1), 0.0)
    P = z["prosody"].astype(np.float64)
    Pz = np.zeros_like(P)
    parameters = {"mfcc_mean": mu, "mfcc_sd": sd}
    for lang in sorted(set(U["lang"])):
        mask = U["lang"] == lang
        x = P[mask]
        lm = np.nanmean(x, 0)
        lm = np.where(np.isnan(lm), 0.0, lm)
        x = np.where(np.isnan(x), lm, x)
        m, s = x.mean(0), x.std(0)
        Pz[mask] = np.where(s > 0, (x - m) / np.where(s > 0, s, 1), 0.0)
        parameters.update({f"prosody_{lang}_impute": lm, f"prosody_{lang}_mean": m, f"prosody_{lang}_sd": s})
    # Original K was assigned to a float32 zero matrix only on eligible rows.
    K = np.zeros((len(U["lang"]), 14), np.float32)
    kin_ok = k["within_shot"] & (k["k_missing"] == 0)
    K[kin_ok] = k["K"][kin_ok]
    raw = {"whisper": z["whisper"].astype(np.float32), "mfcc86": Mz, "prosody": Pz,
           "xlmr_sub": z["xlmr_sub"].astype(np.float32), "kin14": K}
    return {name: l2(x) for name, x in raw.items()}, parameters, kin_ok


def score_result(score, labels, masks):
    n = len(labels)
    result = {}
    for pool, mask in masks.items():
        ranks = np.full((n, int(mask.sum(1).max())), -1, dtype=np.int32)
        values = np.full((n, 5), np.nan)
        counts = mask.sum(1).astype(np.int32)
        for i in range(n):
            cand = np.flatnonzero(mask[i])
            order, m = query_metrics(score[i, cand], cand, labels, i)
            ranks[i, :len(order)] = order
            values[i] = m
        result[pool] = (ranks, values, counts)
    return result


def load_inputs(root, restore, release, l2):
    with np.load(restore / "features_full/features.npz", allow_pickle=False) as f:
        z = {k: f[k] for k in ("ids", "whisper", "mfcc86", "prosody", "xlmr_sub")}
    with np.load(restore / "unit_table/units_k.npz", allow_pickle=False) as f:
        k = {name: f[name] for name in ("ids", "K", "within_shot", "k_missing")}
    assert np.array_equal(z["ids"], k["ids"]) and np.array_equal(z["ids"], release.ids)
    U = release.units
    arrays, parameters, kin_ok = prepare_arrays(z, k, U, l2)
    arrays["tfidf_sub"] = l2(sparse.load_npz(restore / "features_full/tfidf_sub.npz"))
    common = (np.abs(z["xlmr_sub"]).sum(1) > 0) & kin_ok
    assert np.array_equal(common, np.load(root / "data/study3_stats/ch5_fusion/common_ok.npy", allow_pickle=False))
    selected = np.flatnonzero(common & (U["kept"] == "True"))
    inv = json.loads((HERE / "INVENTORY.json").read_bytes())
    assert len(selected) == 384
    ordered = ("\n".join(release.ids[selected]) + "\n").encode()
    assert hashlib.sha256(ordered).hexdigest() == inv["population"]["ordered_ids_sha256"]
    local = {key: U[key][selected] for key in ("lang", "day", "clip", "haiku")}
    masks = {p: np.stack([candidate_mask(i, local["lang"], local["clip"], local["day"], p) for i in range(len(selected))]) for p in POOLS}
    return arrays, parameters, selected, local, masks


def native_scores(X, selected, local, pool, sims):
    """Use original per-source Q/C multiplication shapes, rather than full-bank GEMM."""
    n = len(selected)
    scores = np.full((n, n), np.nan, dtype=np.float32)
    for lang in SOURCES:
        q = np.flatnonzero(local["lang"] == lang)
        c = np.flatnonzero(local["lang"] != lang) if pool == "X" else q
        scores[np.ix_(q, c)] = sims(X[selected[q]], X[selected[c]])
    return scores


def native_reference(release, name, pool, selected, local, mask, ranks, values, count):
    original = np.full_like(values, np.nan)
    old_ranks = np.full_like(ranks, -1)
    score_error, changed_rank_queries = 0., 0
    for lang in SOURCES:
        cid = f"ch5_retrieval|analysis|{name}|kept|common|{pool}|{lang}"
        entry = release.condition(cid)
        path = release._numeric_file("rankings/" + entry["perquery_file"], "F1 native per-query reference")
        with np.load(path, allow_pickle=False) as z:
            expected_q = selected[local["lang"] == lang]
            assert np.array_equal(z["query_rows"], expected_q)
            positions = np.flatnonzero(local["lang"] == lang)
            for col, key in enumerate(("hit", "p3", "ap", "chance")):
                original[positions, col] = z[key]
            original[positions, 4] = z["hit"] - z["chance"]
            assert np.array_equal(count[positions], z["n_cand"])
        block = release.block(cid)
        assert np.array_equal(block.query_rows, expected_q)
        c = selected[local["lang"] != lang] if pool == "X" else expected_q
        assert np.array_equal(block.bank_rows, c)
        to_local = {int(row): i for i, row in enumerate(selected)}
        for j, pos in enumerate(positions):
            a, b = block.offsets[j:j+2]
            ranked = np.array([to_local[int(r)] for r in block.rank_rows[a:b]], dtype=np.int32)
            assert np.array_equal(np.sort(ranked), np.flatnonzero(mask[pos]))
            old_ranks[pos, :len(ranked)] = ranked
            changed_rank_queries += int(not np.array_equal(ranked, ranks[pos, :len(ranked)]))
    close = np.isclose(original, values, atol=1e-12, rtol=0, equal_nan=True)
    return original, old_ranks, {"representation": name, "pool": pool,
        "metric_mismatches": {metric: int((~close[:, j]).sum()) for j, metric in enumerate(METRICS)},
        "rank_changed_queries": changed_rank_queries,
        "max_metric_error": float(np.nanmax(np.abs(values - original))),
        "query_candidate_denominators_equal": True}


def baseline(args, budget, profile, pins):
    release = Release()
    release.verify_unit_rows_against_units_tsv()
    frozen = extract_functions(args.root / "pipeline/analysis/ch5_retrieval.py", ["l2rows", "sims"])
    arrays, parameters, selected, local, masks = load_inputs(args.root, args.restore, release, frozen["l2rows"])
    args.out.mkdir(parents=True)
    write_json(args.out / "FREEZE.json", {"profile": profile, "sources": pins, "programs": program_pins(),
                                         "adoption": record(HERE / "EXECUTION_ADOPTION.md")})
    np.savez_compressed(args.out / "preprocessing.npz", **parameters)
    cache = {"selected_rows": selected, "ids": release.ids[selected], **local,
             **{p + "__mask": m for p, m in masks.items()},
             **{p + "__n_rel": (m & (local["haiku"][:, None] == local["haiku"][None, :])).sum(1) for p, m in masks.items()}}
    rows, native_ok, single_ok = [], True, True
    for name in ROUTES:
        X = arrays[name][selected]
        if sparse.issparse(X):
            X = X.toarray()
        # Fixed new fusion recipe, including final normalization for a single block.
        one = fuse_normalized([X])
        one_scores = one @ one.T
        one_result = score_result(one_scores, local["haiku"], masks)
        cache[name + "__single_scores"] = one_scores
        for pool in POOLS:
            score = native_scores(arrays[name], selected, local, pool, frozen["sims"])
            ranks, values, count = score_result(score, local["haiku"], {pool: masks[pool]})[pool]
            original, saved_ranks, row = native_reference(release, name, pool, selected, local, masks[pool], ranks, values, count)
            rr, vv, cc = one_result[pool]
            close = np.isclose(values, vv, atol=1e-12, rtol=0, equal_nan=True)
            row["oneblock_metric_mismatches"] = {metric: int((~close[:, j]).sum()) for j, metric in enumerate(METRICS)}
            row["oneblock_rank_changed_queries"] = int(np.any(rr != ranks, axis=1).sum())
            native_ok &= not any(row["metric_mismatches"].values())
            single_ok &= bool(close.all() and np.array_equal(cc, count))
            prefix = name + "__" + pool + "__"
            cache.update({prefix + "native_scores": score, prefix + "native_ranks": ranks,
                          prefix + "native_metrics": values, prefix + "saved_metrics": original,
                          prefix + "saved_ranks": saved_ranks, prefix + "single_ranks": rr,
                          prefix + "single_metrics": vv, prefix + "n_cand": count})
            rows.append(row)
            budget.tick()
        print(json.dumps({"baseline_route_completed": name}), flush=True)
    passed = bool(native_ok and single_ok)
    np.savez_compressed(args.out / "cache.npz", **cache)
    result = {"status": "baseline_pass" if passed else "baseline_stop_before_multiblock",
              "native_metrics_gate": bool(native_ok), "oneblock_metrics_gate": bool(single_ok),
              "comparisons": rows, "full_perquery_differences": "cache.npz", "new_multiblock_computed": False,
              "resources": budget.tick(), "formal_read_log": release.read_log}
    for path, pin in release.read_log.items():
        assert record(args.root / path)["sha256"] == pin["sha256"]
    return result


def program_pins():
    return {p.name: record(p) for p in (Path(__file__), HERE / "kernels.py", HERE / "execution_support.py")}


def fusion(args, budget, profile, pins):
    gate = json.loads((args.gate / "RESULTS.json").read_bytes())
    freeze = json.loads((args.gate / "FREEZE.json").read_bytes())
    if gate["status"] != "baseline_pass" or freeze["programs"] != program_pins() or freeze["sources"] != pins:
        raise RuntimeError("unpassed/stale baseline gate")
    gate_manifest = json.loads((args.gate / "MANIFEST.json").read_bytes())
    for name, pin in gate_manifest.items():
        if record(args.gate / name) != pin:
            raise RuntimeError("baseline artifact mismatch")
    release = Release()
    frozen = extract_functions(args.root / "pipeline/analysis/ch5_retrieval.py", ["l2rows"])
    arrays, parameters, selected, local, masks = load_inputs(args.root, args.restore, release, frozen["l2rows"])
    blocks = {name: x[selected].toarray() if sparse.issparse(x) else x[selected] for name, x in arrays.items()}
    del arrays
    configs = json.loads((HERE / "INVENTORY.json").read_bytes())["configurations"]
    args.out.mkdir(parents=True)
    write_json(args.out / "FREEZE.json", {"profile": profile, "sources": pins, "programs": program_pins(),
                                         "gate": record(args.gate / "RESULTS.json"), "adoption": record(HERE / "EXECUTION_ADOPTION.md")})
    metrics, rank_arrays = [], {"selected_rows": selected, "ids": release.ids[selected], **local,
                               **{p + "__mask": m for p, m in masks.items()},
                               **{p + "__n_rel": (m & (local["haiku"][:, None] == local["haiku"][None, :])).sum(1) for p, m in masks.items()}}
    with np.load(args.gate / "cache.npz", allow_pickle=False) as baseline_cache:
        for cfg in configs:
            name = "+".join(cfg)
            if len(cfg) == 1:
                scores = baseline_cache[cfg[0] + "__single_scores"]
                result = {p: (baseline_cache[f"{cfg[0]}__{p}__single_ranks"], baseline_cache[f"{cfg[0]}__{p}__single_metrics"],
                              baseline_cache[f"{cfg[0]}__{p}__n_cand"]) for p in POOLS}
            else:
                fused = fuse_normalized([blocks[k] for k in cfg])
                scores = fused @ fused.T
                result = score_result(scores, local["haiku"], masks)
            rank_arrays[name + "__scores"] = scores
            for pool in POOLS:
                ranks, values, count = result[pool]
                rank_arrays[name + "__" + pool + "__ranks"] = ranks
                rank_arrays[name + "__" + pool + "__metrics"] = values
                metrics.append(values)
            budget.tick()
    np.savez_compressed(args.out / "rankings.npz", **rank_arrays)
    all_values = np.stack(metrics, axis=1)  # query, configuration*pool, metric
    keys, selected_days, mult = day_weights(local["lang"], local["day"], SOURCES, 10000, 2026110501)
    query_keys = list(zip(local["lang"].tolist(), local["day"].tolist()))
    table, draw_arrays = [], {}
    for group, group_mask in (("all", np.ones(len(selected), bool)), ("Explaining", local["haiku"] == "Explaining"),
                              ("remainder", local["haiku"] != "Explaining")):
        values = all_values.copy()
        values[~group_mask] = np.nan
        observed, draws, denom, _ = means_by_day(values.reshape(len(selected), -1), query_keys, keys, mult)
        draw_arrays[group + "__draws"] = draws
        for ci, cfg in enumerate(configs):
            for pi, pool in enumerate(POOLS):
                base = (ci * 4 + pi) * 5
                for mi, metric in enumerate(METRICS):
                    coordinate = base + mi
                    finite = np.isfinite(draws[:, coordinate])
                    vals = values[:, ci*4+pi, mi]
                    valid = np.isfinite(vals)
                    table.append({"configuration": "+".join(cfg), "pool": pool, "version": "kept", "subset": "common384",
                                  "group": group, "metric": metric, "queries": int(group_mask.sum()), "effective_n": int(denom[coordinate]),
                                  "days": len(set(zip(local["lang"][valid], local["day"][valid]))), "clips": len(set(local["clip"][valid])),
                                  "candidate_min": int(masks[pool][group_mask].sum(1).min()) if group_mask.any() else None,
                                  "candidate_max": int(masks[pool][group_mask].sum(1).max()) if group_mask.any() else None,
                                  "no_candidates": int((masks[pool][group_mask].sum(1) == 0).sum()),
                                  "zero_relevant": int((rank_arrays[pool + "__n_rel"][group_mask] == 0).sum()),
                                  "estimate": observed[coordinate], "ci95_pointwise": np.quantile(draws[finite, coordinate], [.025, .975], method="linear") if finite.any() else None,
                                  "undefined_draws": int((~finite).sum()), "coverage_status": "pooled exploratory" if group == "all" else "descriptive model-class group"})
        budget.tick()
    names = ["+".join(cfg) for cfg in configs]
    baseline = names.index("xlmr_sub") * 4 + POOLS.index("W3")
    full = ["xlmr_sub+whisper", "xlmr_sub+prosody", "xlmr_sub+kin14", "xlmr_sub+whisper+prosody+kin14"]
    contrasts, contrast_draws, contrast_obs = [], [], []
    obs = np.nanmean(all_values, axis=0)
    bdraw = draw_arrays["all__draws"].reshape(10000, len(configs)*4, 5)
    for cfg in full:
        j = names.index(cfg) * 4 + POOLS.index("W3")
        for metric in range(3):
            contrast_obs.append(obs[j, metric] - obs[baseline, metric])
            contrast_draws.append(bdraw[:, j, metric] - bdraw[:, baseline, metric])
            contrasts.append({"configuration": cfg, "baseline": "xlmr_sub", "pool": "W3", "metric": METRICS[metric]})
    contrast_draws = np.column_stack(contrast_draws)
    family = intervals(contrast_obs, contrast_draws)
    for j, row in enumerate(contrasts):
        row.update(estimate=contrast_obs[j], ci95_pointwise=family["pointwise"][j],
                   ci95_simultaneous=family["simultaneous"][j] if family["simultaneous"] is not None else None)
    np.savez_compressed(args.out / "bootstrap.npz", keys=np.array(keys), multiplicities=mult,
                        **{s + "__selections": a for s, a in selected_days.items()}, **draw_arrays,
                        contrast_draws=contrast_draws, contrast_estimates=np.array(contrast_obs))
    return {"status": "completed", "seed": 2026110501, "resamples": 10000, "configurations": configs,
            "summary_rows": table, "contrasts": contrasts, "family": family, "resources": budget.tick()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=("baseline", "fusion"))
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--restore", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gate", type=Path)
    args = p.parse_args()
    if args.root.resolve() != ROOT.resolve() or args.restore.resolve() != (args.root / "data/study3_runs").resolve():
        raise ValueError("input roots must match this local frozen source contract")
    if args.out.exists():
        raise FileExistsError(args.out)
    budget = Budget()
    with threadpool_limits(limits=1):
        profile = check_profile()
        pins = check_sources(args.root, HERE / "INVENTORY.json")
        result = baseline(args, budget, profile, pins) if args.task == "baseline" else fusion(args, budget, profile, pins)
        for relative, pin in pins.items():
            if record(args.root / relative) != pin:
                raise RuntimeError("protected source changed during execution")
        result["protected_source_hashes_unchanged"] = True
        result["resources"] = budget.tick()
        write_json(args.out / "RESULTS.json", result)
        write_json(args.out / "MANIFEST.json", {p.name: record(p) for p in args.out.iterdir() if p.is_file()})
        print(json.dumps(clean({k: v for k, v in result.items() if k not in ("summary_rows", "comparisons", "formal_read_log", "configurations")}), ensure_ascii=False))


if __name__ == "__main__":
    main()
