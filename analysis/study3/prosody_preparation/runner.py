"""Independent P1 baseline/paired exploration and P2 scalar descriptions.

Reads frozen arrays and accepted saved OOF reference only, never raw answer files.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SUPPORT = HERE.parent / "ch5_fusion_preparation"
sys.path.insert(0, str(SUPPORT))
from execution_support import Budget, check_profile, check_sources, clean, record, write_json
from kernels import intervals


def validate_arrays(ids, expected_ids, y, w, days):
    if list(ids) != list(expected_ids) or len(set(ids)) != len(ids):
        raise ValueError("duplicate/misordered IDs")
    if not set(y) <= {0, 1} or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("invalid reference/weights")
    if not (len(ids) == len(y) == len(w) == len(days)):
        raise ValueError("misaligned arrays")


def prep_recorded(Xtr, Xte):
    """Same operations and defaults as pinned probe.prep, with transformation parameters."""
    mu = np.nanmean(Xtr, 0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr)
    Xte = np.where(np.isnan(Xte), mu, Xte)
    m, s = Xtr.mean(0), Xtr.std(0)
    s[s == 0] = 1.0
    Xtr, Xte = (Xtr - m) / s, (Xte - m) / s
    params = {"imputation_mean": mu, "scale_mean": m, "scale_SD": s}
    if Xtr.shape[1] > 16:
        p = PCA(n_components=min(16, len(Xtr) - 1), random_state=0).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        params.update(PCA_components=p.components_, PCA_mean=p.mean_,
                      PCA_singular_values=p.singular_values_, PCA_variance=p.explained_variance_)
    return Xtr, Xte, params


def oof(sources, names, y, w, days, out, budget):
    scores = np.full(len(y), np.nan)
    fold_log = []
    for j, day in enumerate(sorted(set(days))):
        te, tr = days == day, days != day
        parameters, parts = {}, []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for name in names:
                a, b, params = prep_recorded(sources[name][tr], sources[name][te])
                parts.append((a, b))
                parameters.update({name + "__" + key: value for key, value in params.items()})
            Xtr = np.hstack([p[0] for p in parts])
            Xte = np.hstack([p[1] for p in parts])
            one_class = len(set(y[tr])) < 2
            if one_class:
                scores[te] = np.average(y[tr], weights=w[tr])
                iterations = []
                model_params = None
            else:
                model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000).fit(Xtr, y[tr], sample_weight=w[tr])
                scores[te] = model.predict_proba(Xte)[:, 1]
                parameters.update(model_coef=model.coef_, model_intercept=model.intercept_)
                iterations = model.n_iter_.tolist()
                model_params = model.get_params()
        notes = [str(r.message) for r in caught]
        if any(r.category.__name__ == "ConvergenceWarning" for r in caught):
            raise RuntimeError("logistic did not converge")
        if out is not None:
            np.savez_compressed(out / f"fold_{j:02d}.npz", train=np.flatnonzero(tr), test=np.flatnonzero(te), **parameters)
        fold_log.append({"day": str(day), "training_n": int(tr.sum()), "test_n": int(te.sum()),
                         "training_positive": int(y[tr].sum()), "one_class": one_class,
                         "n_iter": iterations, "warnings": notes, "model_parameters": model_params})
        budget.tick()
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise RuntimeError("incomplete or illegal OOF probabilities")
    return scores, fold_log


def weighted_metrics(y, scores, weights):
    """Weighted ROC AUC and threshold-grouped average precision; no sklearn dependency here."""
    y, scores, weights = np.asarray(y), np.asarray(scores), np.asarray(weights)
    if (not (y.shape == scores.shape == weights.shape) or not np.isfinite(scores).all()
            or not np.isfinite(weights).all() or np.any(weights < 0) or not set(y) <= {0, 1}):
        raise ValueError("invalid metric arrays")
    keep = weights > 0
    y, scores, weights = y[keep], scores[keep], weights[keep]
    pos, neg = weights[y == 1].sum(), weights[y == 0].sum()
    if not pos or not neg:
        return np.array([np.nan, np.nan])
    order = np.argsort(scores, kind="stable")
    scores, y, weights = scores[order], y[order], weights[order]
    starts = np.r_[0, 1 + np.flatnonzero(scores[1:] != scores[:-1])]
    wp = np.add.reduceat(weights * (y == 1), starts)
    wn = np.add.reduceat(weights * (y == 0), starts)
    auc = (wp * (np.cumsum(wn) - wn + .5 * wn)).sum() / (pos * neg)
    wp, wn = wp[::-1], wn[::-1]
    ap = ((wp / pos) * (np.cumsum(wp) / np.cumsum(wp + wn))).sum()
    return np.array([auc, ap])


def selections(days, seed, n=10000):
    unique = np.array(sorted(set(days)))
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = rng.integers(0, len(unique), size=(n, len(unique)))
    mult = np.zeros_like(selected)
    np.add.at(mult, (np.arange(n)[:, None], selected), 1)
    idx = np.searchsorted(unique, days)
    return unique, selected, mult, idx


def describe(x, y, w, days, clips, mult, day_idx):
    finite = np.isfinite(x)
    rows, draws = [], []
    for group in (0, 1):
        mask = finite & (y == group)
        vals, weights = x[mask], w[mask]
        sums = np.bincount(day_idx[mask], weights=vals*weights, minlength=mult.shape[1])
        denoms = np.bincount(day_idx[mask], weights=weights, minlength=mult.shape[1])
        num, den = mult @ sums, mult @ denoms
        means = np.divide(num, den, out=np.full(len(mult), np.nan), where=den > 0)
        rows.append({"group": "positive" if group else "negative", "n": int(mask.sum()),
                     "missing": int(((y == group) & ~finite).sum()), "days": len(set(days[mask])),
                     "clips": len(set(clips[mask])), "mean": float(vals.mean()) if len(vals) else np.nan,
                     "SD": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                     "median": float(np.median(vals)) if len(vals) else np.nan,
                     "q25": float(np.quantile(vals, .25, method="linear")) if len(vals) else np.nan,
                     "q75": float(np.quantile(vals, .75, method="linear")) if len(vals) else np.nan,
                     "weight_sum": float(weights.sum()),
                     "weighted_mean": float(np.average(vals, weights=weights)) if len(vals) else np.nan,
                     "undefined_draws": int((~np.isfinite(means)).sum())})
        draws.append(means)
    difference = draws[1] - draws[0]
    valid = np.isfinite(difference)
    return rows, {"difference": rows[1]["weighted_mean"] - rows[0]["weighted_mean"],
                  "ci95_pointwise": np.quantile(difference[valid], [.025, .975], method="linear").tolist() if valid.any() else None,
                  "valid_draws": int(valid.sum()), "undefined_draws": int((~valid).sum())}, np.column_stack(draws + [difference])


def load_primary(root, restore):
    inv = json.loads((HERE / "INVENTORY.json").read_bytes())
    with np.load(root / inv["saved_oof_source"], allow_pickle=False) as z:
        saved = {k: z[k] for k in ("ids", "y", "w", "days", "T")}
    items = json.loads((restore / "annotation/items.json").read_bytes())["items"]
    main = [r for r in items if r["batch"] in (1, 2)]
    lookup = {r["utt_id"]: r for r in main}
    if len(lookup) != len(main) or len(main) != 240:
        raise RuntimeError("sampling ID mismatch")
    expected = [r["utt_id"] for r in main if r["utt_id"] in set(saved["ids"])]
    validate_arrays(saved["ids"], expected, saved["y"], saved["w"], saved["days"])
    assert len(expected) == 192 and int(saved["y"].sum()) == 20
    digest = hashlib.sha256(("\n".join(expected) + "\n").encode()).hexdigest()
    assert digest == inv["primary"]["ordered_ids_sha256"]
    for i, uid in enumerate(expected):
        assert saved["days"][i] == lookup[uid]["day"] and saved["w"][i] == 1 / lookup[uid]["incl_prob"]
    with np.load(restore / "item5/audio.npz", allow_pickle=False) as z:
        assert z["ids"].tolist() == [r["utt_id"] for r in main]
        pos = {u: i for i, u in enumerate(z["ids"])}
        idx = np.array([pos[u] for u in expected])
        audio = {k: z[k][idx] for k in ("whisper", "prosody")}
        names = z["prosody_names"].tolist()
    for j, coverage in enumerate(inv["coverage"]):
        finite = np.isfinite(audio["prosody"][:, j])
        assert finite.sum() == coverage["primary_192_finite"]
        assert (finite & (saved["y"] == 1)).sum() == coverage["primary_positive_finite"]
    saved["clips"] = np.array([lookup[u]["clip"] for u in expected])
    return saved, audio, names, idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=("p2", "p1-baseline"))
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--restore", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.restore.resolve() != (args.root / "data/study3_runs").resolve():
        raise ValueError("restore argument must resolve to the pinned restore within root")
    if args.out.exists():
        raise FileExistsError(args.out)
    budget = Budget()
    with threadpool_limits(limits=1):
        profile = check_profile()
        pins = check_sources(args.root, HERE / "INVENTORY.json")
        primary, audio, names, idx = load_primary(args.root, args.restore)
        args.out.mkdir(parents=True)
        write_json(args.out / "FREEZE.json", {"task": args.task, "profile": profile, "sources": pins,
                   "programs": {str(p.relative_to(ROOT)): record(p) for p in (Path(__file__), SUPPORT / "kernels.py", SUPPORT / "execution_support.py")},
                   "plan": record(HERE / "PLAN.md"), "adoption": record(SUPPORT / "EXECUTION_ADOPTION.md")})
        y, w, days, clips = (primary[k] for k in ("y", "w", "days", "clips"))
        if args.task == "p2":
            dates, selected, mult, di = selections(days, 2026110503)
            all_rows, effects, all_draws = [], [], []
            for j, name in enumerate(names):
                rows, effect, draws = describe(audio["prosody"][:, j], y, w, days, clips, mult, di)
                all_rows.extend({"feature": name, **r} for r in rows)
                effects.append({"feature": name, **effect})
                all_draws.append(draws)
                budget.tick()
            np.savez_compressed(args.out / "cache.npz", **primary, values=audio["prosody"], names=np.array(names),
                                dates=dates, selections=selected, multiplicities=mult, draws=np.stack(all_draws, axis=1))
            result = {"status": "completed_descriptive", "seed": 2026110503, "n": len(y), "positives": int(y.sum()),
                      "days": len(set(days)), "clips": len(set(clips)), "descriptions": all_rows, "differences": effects,
                      "interval_scope": "pointwise conditional whole-day percentile, no hypothesis tests or fitted calibration",
                      "scope": "primary only; cannot-tell excluded, none_cue separate in original report; no contour"}
        else:
            with np.load(args.restore / "item5/text.npz", allow_pickle=False) as z:
                with np.load(args.restore / "item5/audio.npz", allow_pickle=False) as a:
                    assert np.array_equal(z["ids"], a["ids"])
                sources = {k: z[k][idx] for k in ("xlmr", "tfidf", "form")}
            scores, folds = oof(sources, ["xlmr", "tfidf", "form"], y, w, days, args.out, budget)
            saved = primary["T"]
            old, new = weighted_metrics(y, saved, w), weighted_metrics(y, scores, w)
            sign_old = np.sign(saved[:, None] - saved[None, :])
            sign_new = np.sign(scores[:, None] - scores[None, :])
            changed = int(np.count_nonzero(np.triu(sign_old != sign_new, 1)))
            error = float(np.max(np.abs(saved - scores)))
            passed = error <= 1e-10 and changed == 0 and np.allclose(old, new, atol=1e-12, rtol=0)
            np.savez_compressed(args.out / "cache.npz", **primary, local_T=scores, difference=scores-saved,
                                pair_sign_saved=sign_old, pair_sign_local=sign_new)
            result = {"status": "baseline_pass" if passed else "baseline_stop_pending_lineage_review",
                      "max_score_error": error, "changed_pair_rank_or_tie": changed,
                      "saved_AUC_AP": old, "local_AUC_AP": new, "metric_difference": new-old,
                      "folds": folds, "augmented_models_fitted": False,
                      "diagnostic": "same pinned data/reference/weights, original sourcewise operations and default PCA/logistic under adopted local library versions; compare stored fold parameters; version causation not proven"}
        for relative, pin in pins.items():
            if record(args.root / relative) != pin:
                raise RuntimeError("protected source changed during execution")
        result["resources"] = budget.tick()
        result["protected_source_hashes_unchanged"] = True
        write_json(args.out / "RESULTS.json", result)
        write_json(args.out / "MANIFEST.json", {p.name: record(p) for p in args.out.iterdir() if p.is_file()})
        print(json.dumps(clean({k: v for k, v in result.items() if k not in ("descriptions", "differences", "folds")}), ensure_ascii=False))


if __name__ == "__main__":
    main()
