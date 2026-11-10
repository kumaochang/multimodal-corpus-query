"""Pure synthetic-testable F1 helpers. No file loading or automatic execution."""

import numpy as np


def fuse_normalized(blocks):
    """Concatenate already normalized blocks, including final normalization for one block."""
    if not blocks or len({len(x) for x in blocks}) != 1:
        raise ValueError("nonempty aligned blocks required")
    x = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite normalized block")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def candidate_mask(i, source, clip, day, pool):
    source, clip, day = map(np.asarray, (source, clip, day))
    mask = np.arange(len(source)) != i
    if pool == "X":
        return mask & (source != source[i])
    if pool not in ("W1", "W2", "W3"):
        raise ValueError("unknown pool")
    mask &= source == source[i]
    if pool == "W2":
        mask &= clip != clip[i]
    if pool == "W3":
        mask &= day != day[i]
    return mask


def query_metrics(scores, candidates, labels, query):
    """Stable candidate order is supplied explicitly; zero-relevant AP remains NaN."""
    candidates, labels = np.asarray(candidates), np.asarray(labels)
    scores = np.asarray(scores)
    if scores.shape != candidates.shape or not np.isfinite(scores).all():
        raise ValueError("finite aligned scores required")
    if len(set(candidates.tolist())) != len(candidates) or query in candidates:
        raise ValueError("unique self-excluded candidates required")
    rank = candidates[np.argsort(-scores, kind="stable")]
    if not len(rank):
        return rank, np.full(5, np.nan)
    rel = labels[rank] == labels[query]
    chance = float(rel.mean())
    ap = float((rel.cumsum() / np.arange(1, len(rel) + 1))[rel].mean()) if rel.any() else np.nan
    # P1, P3, AP, precision chance, lift1
    return rank, np.array([float(rel[0]), rel[:min(3, len(rel))].mean(), ap,
                           chance, float(rel[0]) - chance])


def day_weights(source, day, source_order, draws, seed):
    """One shared integer selection table per source, no new RNG calls per metric."""
    source, day = np.asarray(source), np.asarray(day)
    if set(source) != set(source_order) or len(set(source_order)) != len(source_order):
        raise ValueError("source order must name each observed source once")
    rng = np.random.Generator(np.random.PCG64(seed))
    keys, selections, blocks = [], {}, []
    for s in source_order:
        dates = sorted(set(day[source == s]))
        selected = rng.integers(0, len(dates), size=(draws, len(dates)))
        selections[s] = selected
        counts = np.zeros((draws, len(dates)), dtype=np.int64)
        np.add.at(counts, (np.arange(draws)[:, None], selected), 1)
        keys.extend((str(s), str(d)) for d in dates)
        blocks.append(counts)
    return keys, selections, np.concatenate(blocks, axis=1)


def means_by_day(values, query_keys, keys, multiplicities):
    """Finite metric denominators, pooled query weighting, all coordinates share days."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if len(query_keys) != len(values) or len(set(keys)) != len(keys):
        raise ValueError("aligned query keys and unique day keys required")
    pos = {key: i for i, key in enumerate(keys)}
    idx = np.array([pos[tuple(k)] for k in query_keys])
    sums = np.zeros((len(keys), values.shape[1]))
    counts = np.zeros_like(sums)
    valid = np.isfinite(values)
    np.add.at(sums, idx, np.where(valid, values, 0))
    np.add.at(counts, idx, valid.astype(float))
    num, den = multiplicities @ sums, multiplicities @ counts
    boot = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    den0 = counts.sum(axis=0)
    observed = np.divide(sums.sum(axis=0), den0, out=np.full(values.shape[1], np.nan), where=den0 > 0)
    return observed, boot, den0, den


def intervals(observed, draws, min_joint_fraction=0.99):
    """Pointwise finite-draw percentiles and whole-family max-deviation intervals."""
    observed, draws = np.asarray(observed, float), np.asarray(draws, float)
    if draws.ndim != 2 or draws.shape[1] != len(observed) or not len(draws):
        raise ValueError("nonempty aligned draws required")
    finite = np.isfinite(draws)
    pointwise = [np.quantile(draws[finite[:, j], j], [.025, .975], method="linear").tolist()
                 if finite[:, j].any() else None for j in range(draws.shape[1])]
    joint = finite.all(axis=1)
    reason, band, critical = None, None, None
    sd = np.std(draws[joint], axis=0, ddof=1) if joint.sum() > 1 else np.full(len(observed), np.nan)
    if not np.isfinite(observed).all():
        reason = "undefined_observed_coordinate"
    elif joint.mean() < min_joint_fraction:
        reason = "insufficient_joint_finite_draws"
    elif not np.isfinite(sd).all() or np.any(sd <= 0):
        reason = "undefined_or_zero_SD_coordinate"
    else:
        maximum = np.max(np.abs((draws[joint] - observed) / sd), axis=1)
        critical = float(np.quantile(maximum, .95, method="linear"))
        band = np.column_stack((observed - critical * sd, observed + critical * sd)).tolist()
    return {"pointwise": pointwise, "invalid_draws": (~finite).sum(axis=0).tolist(),
            "joint_finite": int(joint.sum()), "SD_ddof1": sd.tolist(),
            "simultaneous": band, "critical_value": critical, "suppressed_reason": reason}
