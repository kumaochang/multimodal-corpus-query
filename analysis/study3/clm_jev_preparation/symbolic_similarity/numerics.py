"""Pure finite-bank numerical operations. This module performs no file reads."""

import numpy as np


def normalized_rows(values):
    values = np.array(values, dtype=np.float64, order="C", copy=True)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("nonfinite or invalid feature matrix")
    lengths = np.linalg.norm(values, axis=1)
    return np.ascontiguousarray(values / np.where(lengths > 0, lengths, 1)[:, None])


def prosody_transform(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError("invalid prosody matrix")
    counts = np.isfinite(values).sum(axis=0)
    means = np.nansum(values, axis=0) / np.maximum(counts, 1)
    filled = np.where(np.isnan(values), means, values)
    scales = filled.std(axis=0, ddof=0)
    transformed = (filled - means) / np.where(scales > 0, scales, 1)
    transformed[:, scales == 0] = 0
    return normalized_rows(transformed), {"means": means, "scales": scales,
                                          "finite_counts": counts, "nan_mask": np.isnan(values)}


def cosine_rows(query, candidates):
    return normalized_rows(query) @ normalized_rows(candidates).T


def row_zscore(scores, valid):
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if scores.shape != valid.shape or scores.ndim != 2 or not np.isfinite(scores[valid]).all():
        raise ValueError("invalid score/mask matrix")
    counts = valid.sum(axis=1)
    means = np.where(valid, scores, 0).sum(axis=1) / np.maximum(counts, 1)
    centered = np.where(valid, scores - means[:, None], 0)
    sd = np.sqrt((centered ** 2).sum(axis=1) / np.maximum(counts, 1))
    z = centered / np.where(sd > 0, sd, 1)[:, None]
    z[~valid] = np.nan
    return z, {"counts": counts, "means": means, "sd": sd, "zero_sd": sd == 0}


def route_scores(raw, k_valid):
    """Standardize within this bank, average available audio channels, then seeds."""
    shape = raw["whisper"].shape
    valid = {name: np.ones(shape, dtype=bool) for name in raw}
    valid["K"] = np.array(k_valid, dtype=bool, copy=True)
    valid["K"][valid["K"].sum(axis=1) < 2, :] = False
    z, parameters = {}, {}
    for name, values in raw.items():
        z[name], parameters[name] = row_zscore(values, valid[name])
    audio_names = ("whisper", "prosody", "K")
    channel_count = sum(valid[name].astype(np.int16) for name in audio_names)
    if (channel_count == 0).any():
        raise ValueError("audio pair has no available channel")
    per_seed_audio = sum(np.where(valid[name], z[name], 0) for name in audio_names) / channel_count
    aggregated = {"Audio": per_seed_audio.mean(axis=0),
                  "ASR_XLMR": z["xlmr"].mean(axis=0),
                  "ASR_TFIDF": z["tfidf"].mean(axis=0)}
    if not all(np.isfinite(value).all() for value in aggregated.values()):
        raise ValueError("nonfinite aggregate score")
    rankings = {name: np.argsort(-value, kind="stable") for name, value in aggregated.items()}
    return aggregated, rankings, {"z": z, "parameters": parameters, "valid": valid,
                                   "audio_channel_count": channel_count, "per_seed_audio": per_seed_audio}


def return_mask(ranking, budget, n):
    ranking = np.asarray(ranking)
    if budget < 0 or budget > n or ranking.shape != (n,) or set(ranking.tolist()) != set(range(n)):
        raise ValueError("invalid budget or incomplete ranking")
    mask = np.zeros(n, dtype=bool)
    mask[ranking[:budget]] = True
    return mask


def selected_summary(mask, y, weights):
    mask, y, weights = np.asarray(mask, bool), np.asarray(y), np.asarray(weights, float)
    known = y >= 0
    positive = y == 1
    count = int(mask.sum())
    unknown = int((mask & ~known).sum())
    tp = int((mask & positive).sum())
    total = int(positive.sum())
    complete = bool(known.all())
    selected_weight = float(weights[mask].sum())
    positive_weight = float(weights[positive].sum())
    tp_weight = float(weights[mask & positive].sum())
    return {"returned": count, "known_positive_returned": tp, "unknown_returned": unknown,
            "known_positive_reference": total, "fully_observed_reference": complete,
            "precision": tp / count if complete and count else None,
            "coverage": tp / total if complete and total else None,
            "returned_weight": selected_weight, "known_positive_weight": positive_weight,
            "known_positive_returned_weight": tp_weight,
            "weighted_precision": tp_weight / selected_weight if complete and selected_weight else None,
            "weighted_coverage": tp_weight / positive_weight if complete and positive_weight else None}


def comparison_summary(symbolic, geometric, y, weights):
    a, b = selected_summary(symbolic, y, weights), selected_summary(geometric, y, weights)
    positives = np.asarray(y) == 1
    cells = {"symbolic_only": int((positives & symbolic & ~geometric).sum()),
             "similarity_only": int((positives & ~symbolic & geometric).sum()),
             "both": int((positives & symbolic & geometric).sum()),
             "neither": int((positives & ~symbolic & ~geometric).sum())}
    delta = {}
    for key in ("precision", "coverage", "weighted_precision", "weighted_coverage"):
        delta[key] = b[key] - a[key] if a[key] is not None and b[key] is not None else None
    return {"symbolic": a, "similarity": b, "known_positive_overlap": cells,
            "similarity_minus_symbolic": delta}


def completion_bounds(y, weights, masks, pairs):
    """Enumerate shared unknown completions, including paired ratios per completion."""
    y, weights = np.asarray(y), np.asarray(weights, float)
    unknown = np.flatnonzero(y < 0)
    if len(unknown) > 13:
        raise ValueError("completion enumeration exceeds the fixed budget")
    bits = ((np.arange(1 << len(unknown))[:, None] >> np.arange(len(unknown))) & 1).astype(np.int8)
    completed = np.broadcast_to(y, (len(bits), len(y))).copy()
    completed[:, unknown] = bits
    values = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, bool)
        tp = completed[:, mask].sum(axis=1).astype(float)
        total = completed.sum(axis=1).astype(float)
        weighted_tp = completed[:, mask] @ weights[mask]
        weighted_total = completed @ weights
        values[name] = {}
        for key, numerator, denominator in (
            ("precision", tp, np.full(len(bits), mask.sum(), dtype=float)),
            ("coverage", tp, total),
            ("weighted_precision", weighted_tp, np.full(len(bits), weights[mask].sum())),
            ("weighted_coverage", weighted_tp, weighted_total),
        ):
            values[name][key] = np.divide(numerator, denominator,
                                         out=np.full(len(bits), np.nan), where=denominator > 0)

    def bounds(value):
        valid = np.isfinite(value)
        return {"lower": float(value[valid].min()) if valid.any() else None,
                "upper": float(value[valid].max()) if valid.any() else None,
                "defined_completions": int(valid.sum()), "undefined_completions": int((~valid).sum())}

    result = {"completions": len(bits), "unknown": len(unknown),
              "individual": {name: {key: bounds(value) for key, value in metrics.items()}
                             for name, metrics in values.items()}, "paired": {}}
    for first, second in pairs:
        result["paired"][first + "_minus_" + second] = {
            key: bounds(value - values[second][key]) for key, value in values[first].items()}
    return result, bits
