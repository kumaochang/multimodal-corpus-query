"""Explicitly adopted new F1 lineage. Historical stopped gates remain untouched."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

import runner as base
from execution_support import Budget, check_profile, check_sources, clean, extract_functions, record, write_json
from kernels import day_weights, fuse_normalized, intervals, means_by_day

HERE, ROOT = base.HERE, base.ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restore", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.restore.resolve() != (ROOT / "data/study3_runs").resolve():
        raise ValueError("unadopted input root")
    budget = Budget()
    with threadpool_limits(limits=1):
        profile = check_profile()
        pins = check_sources(ROOT, HERE / "INVENTORY.json")
        manifest = json.loads((args.baseline / "MANIFEST.json").read_bytes())
        assert record(args.baseline / "MANIFEST.json")["sha256"] == "68cb71cfc7a19e8aed19fcb76092675f730e05b0bf1fd4cac5d667a1fb5faa44"
        for name, pin in manifest.items():
            assert record(args.baseline / name) == pin
        freeze = json.loads((args.baseline / "FREEZE.json").read_bytes())
        versions = json.loads((HERE / "BASELINE_VERSIONS.json").read_bytes())
        assert record(args.baseline / "FREEZE.json")["sha256"] == versions["original_freeze_sha256"]
        assert base.program_pins() == versions["public_programs"] and pins == versions["public_sources"]
        historical = json.loads((args.baseline / "RESULTS.json").read_bytes())
        assert historical["status"] == "baseline_stop_before_multiblock"
        release = base.Release()
        frozen = extract_functions(ROOT / "pipeline/analysis/ch5_retrieval.py", ["l2rows"])
        arrays, parameters, selected, local, masks = base.load_inputs(ROOT, args.restore, release, frozen["l2rows"])
        blocks = {name: x[selected].toarray() if sparse.issparse(x) else x[selected] for name, x in arrays.items()}
        del arrays
        # No multi-block scores exist until every same-recipe one-block control passes.
        controls, tests = {}, []
        with np.load(args.baseline / "cache.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["ids"], release.ids[selected])
            for field in ("lang", "day", "clip", "haiku"):
                np.testing.assert_array_equal(z[field], local[field])
            for pool in base.POOLS:
                np.testing.assert_array_equal(z[pool+"__mask"], masks[pool])
            for name in base.ROUTES:
                one = fuse_normalized([blocks[name]])
                scores = one @ one.T
                np.testing.assert_array_equal(scores, z[name+"__single_scores"])
                result = base.score_result(scores, local["haiku"], masks)
                for pool, (ranks, values, counts) in result.items():
                    np.testing.assert_array_equal(ranks, z[f"{name}__{pool}__single_ranks"])
                    np.testing.assert_array_equal(values, z[f"{name}__{pool}__single_metrics"])
                    np.testing.assert_array_equal(counts, z[f"{name}__{pool}__n_cand"])
                controls[name] = (scores, result)
                tests.append({"representation": name, "full_float32_scores_bitwise_equal": True,
                              "ranks_perquery_metrics_masks_and_denominators_exact": True,
                              "score_payload_sha256": hashlib.sha256(scores.tobytes()).hexdigest()})
                budget.tick()
        args.out.mkdir(parents=True)
        write_json(args.out / "CONTROL_PREFLIGHT.json", {"status": "PASS_SAVED_ONEBLOCK_REPRODUCTION",
                   "historical_automatic_gate_remains": historical["status"], "controls": tests,
                   "authority": "LOCAL_LINEAGE_ADOPTION.md and coordination report §14", "resources": budget.tick()})
        write_json(args.out / "FREEZE.json", {"profile": profile, "sources": pins,
                   "core_programs": base.program_pins(), "new_program": record(Path(__file__)),
                   "baseline_manifest": record(args.baseline / "MANIFEST.json"),
                   "adoption": record(HERE / "LOCAL_LINEAGE_ADOPTION.md"), "plan": record(HERE / "PLAN.md")})
        np.savez_compressed(args.out / "preprocessing.npz", **parameters)
        configs = json.loads((HERE / "INVENTORY.json").read_bytes())["configurations"]
        saved = {"selected_rows": selected, "ids": release.ids[selected], **local,
                 **{p+"__mask": m for p, m in masks.items()},
                 **{p+"__n_rel": (m & (local["haiku"][:, None] == local["haiku"][None, :])).sum(1) for p, m in masks.items()}}
        all_metrics = []
        for cfg in configs:
            name = "+".join(cfg)
            if len(cfg) == 1:
                scores, result = controls[cfg[0]]
            else:
                fused = fuse_normalized([blocks[k] for k in cfg])
                scores = fused @ fused.T
                result = base.score_result(scores, local["haiku"], masks)
            saved[name+"__scores"] = scores
            for pool in base.POOLS:
                ranks, values, counts = result[pool]
                saved[f"{name}__{pool}__ranks"] = ranks
                saved[f"{name}__{pool}__metrics"] = values
                saved[f"{name}__{pool}__n_cand"] = counts
                all_metrics.append(values)
            budget.tick()
        np.savez_compressed(args.out / "rankings.npz", **saved)
        values = np.stack(all_metrics, axis=1)
        keys, selections, mult = day_weights(local["lang"], local["day"], base.SOURCES, 10000, 2026110501)
        query_keys = list(zip(local["lang"].tolist(), local["day"].tolist()))
        rows, draws_by_group = [], {}
        for group, mask in (("all", np.ones(len(selected), bool)), ("Explaining", local["haiku"] == "Explaining"), ("remainder", local["haiku"] != "Explaining")):
            v = values.copy()
            v[~mask] = np.nan
            obs, draws, den, _ = means_by_day(v.reshape(384, -1), query_keys, keys, mult)
            draws_by_group[group+"__draws"] = draws
            for ci, cfg in enumerate(configs):
                for pi, pool in enumerate(base.POOLS):
                    for mi, metric in enumerate(base.METRICS):
                        j = (ci*4+pi)*5+mi
                        finite = np.isfinite(draws[:, j])
                        valid = np.isfinite(v[:, ci*4+pi, mi])
                        rows.append({"configuration": "+".join(cfg), "pool": pool, "version": "kept", "subset": "common384", "group": group,
                                     "metric": metric, "language_scope": "Amis,Atayal,Paiwan,Tao,Truku,Tsou",
                                     "queries": int(mask.sum()), "effective_n": int(den[j]),
                                     "days": len(set(zip(local["lang"][valid], local["day"][valid]))), "clips": len(set(local["clip"][valid])),
                                     "candidate_min": int(masks[pool][mask].sum(1).min()), "candidate_max": int(masks[pool][mask].sum(1).max()),
                                     "no_candidates": int((masks[pool][mask].sum(1) == 0).sum()), "zero_relevant": int((saved[pool+"__n_rel"][mask] == 0).sum()),
                                     "estimate": obs[j], "ci95_pointwise": np.quantile(draws[finite, j], [.025, .975], method="linear") if finite.any() else None,
                                     "undefined_draws": int((~finite).sum()), "coverage_status": "pooled exploratory" if group == "all" else "descriptive model-class group"})
            budget.tick()
        names = ["+".join(cfg) for cfg in configs]
        j0 = names.index("xlmr_sub")*4+2
        means = np.nanmean(values, axis=0)
        draws = draws_by_group["all__draws"].reshape(10000, 140, 5)
        contrasts, delta, bdelta = [], [], []
        for cfg in ("xlmr_sub+whisper", "xlmr_sub+prosody", "xlmr_sub+kin14", "xlmr_sub+whisper+prosody+kin14"):
            j = names.index(cfg)*4+2
            for metric in range(3):
                delta.append(means[j, metric]-means[j0, metric])
                bdelta.append(draws[:, j, metric]-draws[:, j0, metric])
                contrasts.append({"configuration": cfg, "baseline": "xlmr_sub", "pool": "W3", "metric": base.METRICS[metric]})
        bdelta = np.column_stack(bdelta)
        family = intervals(delta, bdelta)
        for j, row in enumerate(contrasts):
            row.update(estimate=delta[j], ci95_pointwise=family["pointwise"][j], ci95_simultaneous=family["simultaneous"][j] if family["simultaneous"] is not None else None)
        np.savez_compressed(args.out / "bootstrap.npz", keys=np.array(keys), multiplicities=mult,
                            **{s+"__selections": a for s, a in selections.items()}, **draws_by_group,
                            contrast_draws=bdelta, contrast_estimates=np.array(delta))
        for name, pin in manifest.items():
            assert record(args.baseline / name) == pin
        check_sources(ROOT, HERE / "INVENTORY.json")
        result = {"status": "completed_adopted_local_lineage", "seed": 2026110501, "resamples": 10000,
                  "configurations": configs, "summary_rows": rows, "contrasts": contrasts, "family": family,
                  "historical_gate_unchanged": True, "protected_sources_unchanged": True, "resources": budget.tick()}
        write_json(args.out / "RESULTS.json", result)
        write_json(args.out / "MANIFEST.json", {p.name: record(p) for p in args.out.iterdir() if p.is_file()})
        print(json.dumps(clean({k:v for k,v in result.items() if k not in ("configurations", "summary_rows")}), ensure_ascii=False))


if __name__ == "__main__":
    main()
