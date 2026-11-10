"""Two explicitly adopted exploratory models, paired only to saved local T."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import runner as base
from execution_support import Budget, check_profile, check_sources, clean, record, write_json
from kernels import intervals

HERE, ROOT, SUPPORT = base.HERE, base.ROOT, base.SUPPORT


def paired_evaluation(y, w, days, scores, n=10000):
    dates, selected, mult, idx = base.selections(days, 2026110502, n)
    names = ["local_T", "T_plus_P", "T_plus_W"]
    observed = np.array([base.weighted_metrics(y, scores[name], w) for name in names])
    draws = np.full((n, 3, 2), np.nan)
    for b in range(n):
        weights = w * mult[b, idx]
        for j, name in enumerate(names):
            draws[b, j] = base.weighted_metrics(y, scores[name], weights)
    pairs = [(1, 0), (2, 0), (1, 2)]
    delta = np.concatenate([observed[a]-observed[b] for a,b in pairs])
    bdelta = np.concatenate([draws[:,a]-draws[:,b] for a,b in pairs], axis=1)
    return dates, selected, mult, observed, draws, delta, bdelta


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
        assert record(args.baseline / "MANIFEST.json")["sha256"] == "e3c0ad3028019d02c693f47800d8b2e09d465a8ba854a0650bea689e14eb12f5"
        baseline_manifest = json.loads((args.baseline / "MANIFEST.json").read_bytes())
        for name, pin in baseline_manifest.items():
            assert record(args.baseline / name) == pin
        freeze = json.loads((args.baseline / "FREEZE.json").read_bytes())
        versions = json.loads((HERE / "BASELINE_VERSIONS.json").read_bytes())
        assert record(args.baseline / "FREEZE.json")["sha256"] == versions["original_freeze_sha256"]
        for path, pin in versions["public_programs"].items():
            assert record(ROOT / path) == pin
        assert versions["public_sources"] == pins
        assert json.loads((args.baseline / "RESULTS.json").read_bytes())["status"] == "baseline_stop_pending_lineage_review"
        primary, audio, prosody_names, idx = base.load_primary(ROOT, args.restore)
        y, w, days = (primary[k] for k in ("y", "w", "days"))
        with np.load(args.baseline / "cache.npz", allow_pickle=False) as z:
            for key in ("ids", "y", "w", "days", "clips"):
                np.testing.assert_array_equal(primary[key], z[key])
            local_T = z["local_T"]
        with np.load(args.restore / "item5/text.npz", allow_pickle=False) as z:
            with np.load(args.restore / "item5/audio.npz", allow_pickle=False) as a:
                np.testing.assert_array_equal(z["ids"], a["ids"])
            sources = {key: z[key][idx] for key in ("xlmr", "tfidf", "form")}
        sources.update(audio)
        args.out.mkdir(parents=True)
        write_json(args.out / "FREEZE.json", {"profile": profile, "sources": pins, "new_program": record(Path(__file__)),
                   "core_programs": freeze["programs"], "adoption": record(SUPPORT / "LOCAL_LINEAGE_ADOPTION.md"),
                   "baseline_manifest": record(args.baseline / "MANIFEST.json"), "plan": record(HERE / "PLAN.md")})
        scores, logs = {"local_T": local_T}, {}
        for name, channel in (("T_plus_P", "prosody"), ("T_plus_W", "whisper")):
            out = args.out / name
            out.mkdir()
            scores[name], logs[name] = base.oof(sources, ["xlmr", "tfidf", "form", channel], y, w, days, out, budget)
            for j in range(12):
                with np.load(out / f"fold_{j:02d}.npz", allow_pickle=False) as new, np.load(args.baseline / f"fold_{j:02d}.npz", allow_pickle=False) as old:
                    for key in old.files:
                        if key in ("train", "test") or key.startswith(("xlmr__", "tfidf__", "form__")):
                            np.testing.assert_array_equal(new[key], old[key])
            print(json.dumps({"model_completed": name, "text_fold_transforms_equal_adopted_local_baseline": True}), flush=True)
        dates, selected, mult, observed, draws, delta, bdelta = paired_evaluation(y, w, days, scores)
        family = intervals(delta, bdelta)
        model_rows = []
        names = ["local_T", "T_plus_P", "T_plus_W"]
        for j, name in enumerate(names):
            for m, metric in enumerate(("AUC", "AP")):
                finite = np.isfinite(draws[:, j, m])
                model_rows.append({"model": name, "metric": metric, "estimate": observed[j,m],
                                   "ci95_pointwise": np.quantile(draws[finite,j,m], [.025,.975], method="linear") if finite.any() else None,
                                   "undefined_draws": int((~finite).sum())})
        contrasts = []
        for j, (a,b) in enumerate((("T_plus_P","local_T"),("T_plus_W","local_T"),("T_plus_P","T_plus_W"))):
            for m, metric in enumerate(("AUC", "AP")):
                k = j*2+m
                contrasts.append({"model": a, "baseline": b, "metric": metric, "estimate": delta[k],
                                  "ci95_pointwise": family["pointwise"][k],
                                  "ci95_simultaneous": family["simultaneous"][k] if family["simultaneous"] is not None else None,
                                  "undefined_draws": family["invalid_draws"][k]})
        np.savez_compressed(args.out / "cache.npz", **{k:primary[k] for k in ("ids","y","w","days","clips")}, **scores,
                            dates=dates, selections=selected, multiplicities=mult, metric_draws=draws, contrast_draws=bdelta,
                            contrast_estimates=delta)
        for name, pin in baseline_manifest.items():
            assert record(args.baseline / name) == pin
        check_sources(ROOT, HERE / "INVENTORY.json")
        result = {"status": "completed_adopted_local_lineage", "n": len(y), "positives": int(y.sum()), "days": len(set(days)),
                  "clips": len(set(primary["clips"])), "seed": 2026110502, "resamples": 10000,
                  "models": model_rows, "contrasts": contrasts, "family": family, "folds": logs,
                  "all_text_fold_transforms_exactly_match_adopted_baseline": True,
                  "historical_gate_unchanged": True, "protected_sources_unchanged": True, "resources": budget.tick()}
        write_json(args.out / "RESULTS.json", result)
        write_json(args.out / "MANIFEST.json", {p.relative_to(args.out).as_posix(): record(p) for p in args.out.rglob("*") if p.is_file()})
        print(json.dumps(clean({k:v for k,v in result.items() if k != "folds"}), ensure_ascii=False))


if __name__ == "__main__":
    main()
