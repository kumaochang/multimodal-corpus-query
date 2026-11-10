"""CPU-only, post-result Ch. 5 metric completion and interaction coverage inventory.

Does not run interactions, inference models, extraction, diagnostics, or pipeline mains.
Usage: python scripts/ch5_metric_supplement.py --input-root ROOT --trees-root TREES
Outputs: tracked aggregates under --report-dir, ignored numerical cache under --cache-dir.
Use --coverage-only before ranking. Reuse hash-locked caches on subsequent invocations.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_ch5", REPO / "pipeline/analysis/ch5_retrieval.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
GROUPS = ["all", "Explaining", "remainder", "Reporting", "Celebrating", "Urging", "Comforting"]
METRICS = {"p1": "hit", "p3": "p3", "map": "ap", "chance": "chance", "lift1": "lift1"}
SEED, B = 20261020, 10000
EXPECTED = {
    "features_full/features.npz": "80ab8584dd41383226e0986055480ead9cac774b9e3b5928d4f2bf45d80d1704",
    "features_full/tfidf_sub.npz": "37166887f01dbe39e61b3ffde9ed0f746ba76f9e18d5f55c0eb2506ef6c71047",
    "features_full/tfidf_tr.npz": "2583b3af57b9bbaad01a5bf2056f3355335daa3b14affcf276a74bcffa43f029",
    "label_analysis/units.tsv": "0c5f9fd8c80bfbde1f6034b88d37145cd1b76c7684851bae0af6b0e92fdcf554",
    "unit_table/units_k.npz": "d443a7566edd113a9ab6356797761de3a48bd427fffbc11d2f6c63451df38222",
    "ch5/ch5_retrieval.json": "bd2d785d8b05fc96565df47ccb867795560b19e87650c7c5659379af5db9d4e2",
    "ch5/ch5_sensitivity.json": "b39dba230d0d14f6c6e3a9d4d7edd0698c0563bb9dbfb09e22676f2eefe6c72f",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(p, rows):
    if not rows:
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def group_mask(labs, group):
    if group == "all":
        return np.ones(len(labs), bool)
    return labs != "Explaining" if group == "remainder" else labs == group


def load(root):
    """Keep only allowed arrays and TSV columns; never inspect/emit per-item content."""
    hashes = {f: sha(root / f) for f in EXPECTED}
    assert hashes == EXPECTED, "Frozen input hash mismatch"
    z = np.load(root / "features_full/features.npz", allow_pickle=False)
    ids = z["ids"]
    assert len(np.unique(ids)) == len(ids)
    cols = ["unit_id", "lang", "day", "clip", "haiku", "kept", "gap"]
    with (root / "label_analysis/units.tsv").open(encoding="utf-8") as f:
        rows = [{c: r[c] for c in cols} for r in csv.DictReader(f, delimiter="\t")]
    pos = {r["unit_id"]: r for r in rows}
    assert len(pos) == len(rows) == len(ids) and set(ids) == set(pos)
    U = {c: np.array([pos[u][c] for u in ids]) for c in cols[1:]}
    U["kept"] = U["kept"] == "True"
    U["gap"] = U["gap"] == "True"
    assert all(cr.LANG[l] == c for l, c in zip(z["lang"], U["lang"]))
    assert set(U["haiku"]) <= set(cr.LABELS)
    idx = {"lab": np.array([cr.LABELS.index(x) for x in U["haiku"]]),
           **{c: np.unique(U[c], return_inverse=True)[1] for c in ("clip", "day", "lang")}}
    W = z["whisper"].astype(np.float32)
    M = z["mfcc86"].astype(np.float64)
    sd = M.std(0)
    M = np.where(sd > 0, (M - M.mean(0)) / np.where(sd > 0, sd, 1), 0.0)
    P = z["prosody"].astype(np.float64)
    Pz = np.zeros_like(P)
    for l in sorted(set(U["lang"])):
        m = U["lang"] == l
        x = P[m]
        mu = np.nanmean(x, 0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        x = np.where(np.isnan(x), mu, x)
        sd = x.std(0)
        Pz[m] = np.where(sd > 0, (x - x.mean(0)) / np.where(sd > 0, sd, 1), 0.0)
    XS, XT = z["xlmr_sub"].astype(np.float32), z["xlmr_tr"].astype(np.float32)
    TS = sparse.load_npz(root / "features_full/tfidf_sub.npz")
    TT = sparse.load_npz(root / "features_full/tfidf_tr.npz")
    sub, tr = np.abs(XS).sum(1) > 0, np.abs(XT).sum(1) > 0
    k = np.load(root / "unit_table/units_k.npz", allow_pickle=False)
    ki = {u: j for j, u in enumerate(k["ids"])}
    assert len(ki) == len(k["ids"]) and set(ids) <= set(ki)
    j = np.array([ki[u] for u in ids])
    kin = k["within_shot"][j].astype(bool) & (k["k_missing"][j] == 0)
    K = np.zeros((len(ids), 14), np.float32)
    K[kin] = k["K"][j[kin]]
    full = np.ones(len(ids), bool)
    reps = {"whisper": (W, full, True), "mfcc86": (M, full, True),
            "prosody": (Pz, full, True), "xlmr_sub": (XS, sub, True),
            "tfidf_sub": (TS, sub, True), "xlmr_tr": (XT, tr, False),
            "tfidf_tr": (TT, tr, False), "kin14": (K, kin, True)}
    assert not tr[U["lang"] == "tsu"].any()
    main = json.loads((root / "ch5/ch5_retrieval.json").read_text(encoding="utf-8"))
    sens = json.loads((root / "ch5/ch5_sensitivity.json").read_text(encoding="utf-8"))
    assert list(main["analyses"]) and "descriptive" in next(iter(main["analyses"].values()))["W1"]
    for n, (_, ok, _) in reps.items():
        assert main["defined"][n] == {"all": int(ok.sum()), "kept": int((ok & U["kept"]).sum())}
    assert int((sub & kin & U["kept"]).sum()) == main["common_subset"]["kept"]
    return ids, U, idx, reps, main, sens, hashes


def describe(U, units, group, language=None):
    mask = group_mask(U["haiku"][units], group)
    if language is not None:
        mask &= U["lang"][units] == language
    q = units[mask]
    return {"queries": len(q), "days": len(set(U["day"][q])),
            "day_clusters": len(set(zip(U["lang"][q], U["day"][q]))),
            "clips": len(set(U["clip"][q]))}


def allowed(q, cand, mode, idx):
    m = cand != q
    if mode == "W2":
        m &= idx["clip"][cand] != idx["clip"][q]
    elif mode == "W3":
        m &= idx["day"][cand] != idx["day"][q]
    return cand[m]


def coverage(U, idx, reps, report):
    masks = {"A_acoustic": reps["whisper"][1] & reps["prosody"][1],
             "B_subtitle": reps["xlmr_sub"][1] & reps["tfidf_sub"][1],
             "C_visual_acoustic": reps["kin14"][1] & reps["whisper"][1],
             "six_common": reps["kin14"][1] & reps["xlmr_sub"][1],
             "transcription": reps["xlmr_tr"][1] & reps["tfidf_tr"][1]}
    rows, pairs = [], []
    for name, mask in masks.items():
        units = np.flatnonzero(mask & U["kept"])
        for mode in (["W1", "W2", "W3"] if name == "transcription" else ["W1", "W2", "W3", "X"]):
            counts, nrel = [], []
            for q in units:
                cand = units[(U["lang"][units] != U["lang"][q]) if mode == "X" else
                             (U["lang"][units] == U["lang"][q])]
                c = allowed(q, cand, mode, idx)
                counts.append(len(c))
                nrel.append(int((idx["lab"][c] == idx["lab"][q]).sum()))
            counts, nrel = np.array(counts), np.array(nrel)
            for lang in ["pooled"] + sorted(set(U["lang"][units])):
                for group in GROUPS:
                    m = group_mask(U["haiku"][units], group)
                    if lang != "pooled":
                        m &= U["lang"][units] == lang
                    scored = m & (counts > 0)
                    d = describe(U, units[scored], "all")
                    rows.append({"comparison_set": name, "pool": mode, "query_language": lang,
                                 "query_group": group, **d, "selected_queries": int(m.sum()),
                                 "candidate_min": int(counts[m].min()) if m.any() else None,
                                 "candidate_max": int(counts[m].max()) if m.any() else None,
                                 "no_candidate": int((m & (counts == 0)).sum()),
                                 "no_relevant": int((scored & (nrel == 0)).sum()),
                                 "pooled_threshold_met": d["days"] >= 15 and d["clips"] >= 25,
                                 "status": "descriptive_per_language" if lang != "pooled" else
                                 "coverage_only_no_interaction_run"})
        if name == "transcription":
            continue
        for src in sorted(set(U["lang"][units])):
            for dst in sorted(set(U["lang"][units])):
                if src == dst:
                    continue
                q = units[U["lang"][units] == src]
                c = units[U["lang"][units] == dst]
                for group in GROUPS:
                    m = group_mask(U["haiku"][q], group)
                    rel = np.array([(idx["lab"][c] == idx["lab"][i]).sum() for i in q[m]])
                    pairs.append({"comparison_set": name, "source": src, "target": dst,
                                  "query_group": group, **describe(U, q[m], "all"),
                                  "target_units": len(c),
                                  "relevant_min": int(rel.min()) if len(rel) else None,
                                  "relevant_max": int(rel.max()) if len(rel) else None,
                                  "queries_no_relevant": int((rel == 0).sum()),
                                  "status": "descriptive_coverage_only_not_X_or_s62_effect"})
    save_csv(report / "coverage.csv", rows)
    save_csv(report / "ordered_pair_coverage.csv", pairs)
    return rows


def durations(ids, U, trees):
    # Only the existing numeric boundary parser; no labeler/model module is imported.
    sp = importlib.util.spec_from_file_location("frozen_boundaries", REPO / "pipeline/stages/utterances.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    values, hashes = {}, {}
    for clip in sorted(set(U["clip"])):
        path = trees / f"{clip}.aligntree.TextGrid"
        hashes[path.name] = sha(path)
        for uid, start, end in mod.load_utterances(path, clip):
            assert uid not in values
            values[uid] = end - start
    assert set(ids) <= set(values)
    return np.array([values[u] for u in ids]), hashes


def tasks(U, reps, main, sens, d):
    common = reps["kin14"][1] & reps["xlmr_sub"][1]
    tasks = []
    for key, block in main["analyses"].items():
        rep, ver, sub = key.split("|")
        mask = reps[rep][1].copy()
        if ver == "kept":
            mask &= U["kept"]
        if sub == "common":
            mask &= common
        for pool, old in block.items():
            tasks.append(("main", rep, ver, sub, pool, mask, old))
    band = (d >= 29.9) & (d <= 30.1)
    assert int((band & U["kept"]).sum()) == sens["2_am8_item2"]["total_kept"]
    for l in sorted(set(U["lang"])):
        assert int((band & (U["lang"] == l)).sum()) == sens["2_am8_item2"]["count_all"][l]
        assert int((band & U["kept"] & (U["lang"] == l)).sum()) == sens["2_am8_item2"]["count_kept"][l]
    for scope, block, extra in [
        ("centered", sens["1_am4_b2_centered_xlmr"], np.ones(len(d), bool)),
        ("no_duration_band", sens["2_am8_item2"]["results"], ~band),
        ("no_input_gap", sens["3_no_input_gap"]["results"], ~U["gap"]),
    ]:
        if not isinstance(block, dict):
            continue
        for rep, pools in block.items():
            for pool, old in pools.items():
                tasks.append((scope, rep, "kept", "own", pool, reps[rep][1] & U["kept"] & extra, old))
    return tasks


def score(X, selected, pool, U, idx):
    rt = cr.Retriever(X, idx["lab"], idx["clip"], idx["day"], idx["lang"])
    qs, parts = [], []
    for lang in sorted(set(U["lang"])):
        q = selected[U["lang"][selected] == lang]
        if not len(q):
            continue
        c = selected[(U["lang"][selected] != lang) if pool == "X" else
                     (U["lang"][selected] == lang)]
        # Preserve an empty-candidate query explicitly, even if the old wrapper skipped it.
        r = rt.run(q, c, pool)
        qs.append(q)
        parts.append({k: r[k] for k in ("hit", "chance", "p3", "ap", "n_cand")})
    q = np.concatenate(qs)
    r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    r["lift1"] = r["hit"] - r["chance"]
    return q, r


def mean(x):
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def reconcile(q, r, U, old, key, differences, checks):
    def check(field, new, previous, digits=None):
        if digits is not None and new is not None:
            new = round(new, digits)
        checks.append(1)
        if new != previous:
            differences.append({"condition": key, "field": field, "reconstructed": new, "old": previous})
    valid = np.isfinite(r["hit"])
    check("queries", int(valid.sum()), old["queries"])
    check("mean_candidates", mean(r["n_cand"]), old["mean_candidates"], 1)
    desc = old.get("descriptive", old.get("hit_chance", {}))
    lifts = old.get("lift_at_1", old.get("lift", {}))
    for group in GROUPS[1:]:
        g = group_mask(U["haiku"][q], group)
        if group in desc:
            dd = desc[group]
            points = {"hit": mean(r["hit"][g]), "chance": mean(r["chance"][g]),
                      "p3": mean(r["p3"][g]), "p3_chance": mean(r["chance"][g]), "map": mean(r["ap"][g])}
            for f, v in points.items():
                if f in dd:
                    check(f"{group}.{f}", v, dd[f], 4)
            coverage_d = describe(U, q[g & valid], "all")
            for f in ("queries", "days", "clips"):
                if f in dd:
                    check(f"{group}.{f}", coverage_d[f], dd[f])
            if "tested_alone" in dd:
                check(f"{group}.tested_alone", coverage_d["days"] >= 15 and coverage_d["clips"] >= 25, dd["tested_alone"])
            if "ap_queries_dropped_no_relevant" in dd:
                check(f"{group}.ap_dropped", int((g & valid & ~np.isfinite(r["ap"])).sum()), dd["ap_queries_dropped_no_relevant"])
        if group in lifts:
            check(f"{group}.lift1", mean(r["lift1"][g]), lifts[group]["mean"], 4)
            check(f"{group}.n", int((g & valid).sum()), lifts[group]["n"])
    if "Explaining_minus_remainder" in lifts:
        diff = mean(r["lift1"][U["haiku"][q] == "Explaining"]) - mean(r["lift1"][U["haiku"][q] != "Explaining"])
        check("Explaining_minus_remainder", diff, lifts["Explaining_minus_remainder"]["diff"], 4)


def cluster_weights(ql, qd, seed=SEED, b=B):
    """Match original universe: each condition's selected query days, within language."""
    clusters = sorted(set(zip(ql, qd)))
    pos = {c: i for i, c in enumerate(clusters)}
    di = np.array([pos[c] for c in zip(ql, qd)])
    weights = np.zeros((b, len(clusters)), dtype=float)
    rng = np.random.default_rng(seed)
    for lang in sorted(set(ql)):
        cols = np.array([j for j, c in enumerate(clusters) if c[0] == lang])
        draws = rng.integers(0, len(cols), (b, len(cols)))
        rr = np.repeat(np.arange(b), len(cols))
        np.add.at(weights, (rr, cols[draws.ravel()]), 1)
    return weights, di


def ratio_stats(vals, mask, di, weights):
    valid = mask & np.isfinite(vals)
    S = np.bincount(di[valid], weights=vals[valid], minlength=weights.shape[1])
    C = np.bincount(di[valid], minlength=weights.shape[1])
    den = weights @ C
    boot = np.divide(weights @ S, den, out=np.full(len(weights), np.nan), where=den > 0)
    finite = boot[np.isfinite(boot)]
    return {"n": int(C.sum()), "estimate": float(S.sum() / C.sum()) if C.sum() else None,
            "ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
            "undefined_resamples": int((~np.isfinite(boot)).sum()),
            "support_day_clusters": int((C > 0).sum())}


def aggregate(q, r, U, task, weights, di):
    scope, rep, ver, subset, pool, _, old = task
    rows = []
    for group in GROUPS:
        g = group_mask(U["haiku"][q], group)
        valid = g & np.isfinite(r["hit"])
        d = describe(U, q[valid], "all")
        threshold = d["days"] >= 15 and d["clips"] >= 25
        lifts = old.get("lift_at_1", old.get("lift", {}))
        c = r["n_cand"][valid]
        row = {"scope": scope, "representation": rep, "pool": pool, "version": ver,
               "subset": subset, "query_group": group,
               "language_scope": ",".join(sorted(set(U["lang"][q[valid]]))), **d,
               "selected_queries": int(g.sum()), "no_candidate": int((g & ~np.isfinite(r["hit"])).sum()),
               "ap_excluded_no_relevant": int((valid & ~np.isfinite(r["ap"])).sum()),
               "candidate_min": int(c.min()) if len(c) else None,
               "candidate_max": int(c.max()) if len(c) else None,
               "candidate_mean": mean(c), "candidate_rule": {"W1": "same_language_exclude_self",
                   "W2": "same_language_exclude_clip", "W3": "same_language_exclude_day",
                   "X": "other_languages_union"}[pool],
               "coverage_threshold_met": threshold,
               "status": "descriptive_supplement_no_category_test" if group in cr.LABELS else
                         "descriptive_metric_completion",
               "existing_lift_ci95": lifts.get(group, {}).get("ci95"),
               "metrics": {m: ratio_stats(r[k], g, di, weights) for m, k in METRICS.items()}}
        rows.append(row)
    return rows


def writing_tables(rows, report):
    base = [r for r in rows if r["query_group"] in GROUPS[:3]]
    for title, dest, subset in [
        ("Primary kept/own", "PRIMARY_TABLE.md", [r for r in base if r["scope"] == "main" and r["version"] == "kept" and r["subset"] == "own"]),
        ("Common/all and existing sensitivities", "SENSITIVITY_TABLE.md", [r for r in base if not (r["scope"] == "main" and r["version"] == "kept" and r["subset"] == "own")]),
    ]:
        lines = [f"# {title}", "", "Descriptive pointwise conditional day-cluster 95% intervals; 10,000 draws.",
                 "Days/clips/queries and P1/P3/AP denominators appear for every row. Coverage and candidate ranges are in metrics.csv.",
                 "Chance applies to P@1/P@3, not MAP. E = Explaining, R = pooled remainder, A = all queries.",
                 "All query languages are pooled; no per-language inference or representation superiority test.",
                 "Transcription X/common and Face: not applicable in this retrieval suite.", "",
                 "| Condition | Pool | Group | Days/clips/q | n P1/P3/AP | P@1 [95% CI] | P@3 [95% CI] | MAP [95% CI] | Chance | lift@1 |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        def fmt(s):
            if s["estimate"] is None:
                return "undefined"
            lo, hi = s["ci95"]
            return f'{s["estimate"]:.3f} [{lo:.3f}, {hi:.3f}]'
        for r in subset:
            m = r["metrics"]
            cond = f'{r["scope"]}/{r["representation"]}/{r["version"]}/{r["subset"]}'
            g = {"all": "A", "Explaining": "E", "remainder": "R"}[r["query_group"]]
            n = "/".join(str(m[k]["n"]) for k in ("p1", "p3", "map"))
            lines.append(f'| {cond} | {r["pool"]} | {g} | {r["days"]}/{r["clips"]}/{r["queries"]} | {n} | {fmt(m["p1"])} | {fmt(m["p3"])} | {fmt(m["map"])} | {m["chance"]["estimate"]:.3f} | {m["lift1"]["estimate"]:.3f} |')
        (report / dest).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--trees-root", type=Path)
    ap.add_argument("--report-dir", type=Path, default=REPO / "analysis/study3/ch5_exploratory")
    ap.add_argument("--cache-dir", type=Path, default=REPO / "data/study3_stats/ch5_exploratory")
    ap.add_argument("--coverage-only", action="store_true")
    a = ap.parse_args()
    start = time.perf_counter()
    a.report_dir.mkdir(parents=True, exist_ok=True)
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    plan = a.report_dir / "PLAN.md"
    assert plan.exists(), "Save analysis plan before computation"
    ids, U, idx, reps, original, sensitivities, hashes = load(a.input_root)
    rows = coverage(U, idx, reps, a.report_dir)
    save_json(a.report_dir / "inventory.json", {"input_root": str(a.input_root.resolve()),
        "inputs_sha256": hashes, "main_blocks": len(original["analyses"]),
        "main_pool_cells": sum(map(len, original["analyses"].values())),
        "query_cache_present_in_existing_json": False, "common": original["common_subset"],
        "plan_sha256": sha(plan), "coverage_rows": len(rows), "interaction_effects_executed": False})
    if a.coverage_only:
        print(json.dumps({"coverage_complete": len(rows), "seconds": round(time.perf_counter() - start, 2)}))
        return
    assert a.trees_root is not None
    d, tree_hashes = durations(ids, U, a.trees_root)
    all_tasks = tasks(U, reps, original, sensitivities, d)
    hash_files = [Path(__file__), REPO / "pipeline/analysis/ch5_retrieval.py",
                  REPO / "pipeline/stages/utterances.py", plan]
    fingerprint = {"inputs": hashes, "boundaries": tree_hashes,
                   "code_and_plan": {str(p.relative_to(REPO)): sha(p) for p in hash_files},
                   "numpy": np.__version__, "scipy": scipy.__version__}
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    manifest_path = a.cache_dir / "cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["signature"] == signature, "Cache provenance changed; do not silently reuse or replace"
    else:
        manifest = {"signature": signature, "fingerprint": fingerprint, "cells": {}}
        save_json(manifest_path, manifest)
    scores, differences, checks = [], [], []
    normalized = {}
    reconstructed, reused = 0, 0
    for task in all_tasks:
        scope, rep, ver, sub, pool, mask, old = task
        key = "|".join((scope, rep, ver, sub, pool))
        filename = hashlib.sha256(key.encode()).hexdigest()[:20] + ".npz"
        cp = a.cache_dir / filename
        if key in manifest["cells"]:
            assert sha(cp) == manifest["cells"][key]["sha256"]
            z = np.load(cp, allow_pickle=False)
            q, r = z["q"], {k: z[k] for k in ["hit", "chance", "p3", "ap", "n_cand", "lift1"]}
            reused += 1
        else:
            nk = (rep, scope == "centered")
            if nk not in normalized:
                X, ok, _ = reps[rep]
                if scope == "centered":
                    Y = np.zeros_like(X, dtype=np.float64)
                    for l in sorted(set(U["lang"])):
                        m = (U["lang"] == l) & ok
                        Y[m] = X[m] - X[m].mean(0, keepdims=True)
                    X = Y.astype(np.float32)
                normalized[nk] = cr.l2rows(X)
            q, r = score(normalized[nk], np.flatnonzero(mask), pool, U, idx)
            np.savez_compressed(cp, q=q, **r)
            manifest["cells"][key] = {"file": filename, "sha256": sha(cp)}
            save_json(manifest_path, manifest)
            reconstructed += 1
        assert set(q) == set(np.flatnonzero(mask)) and len(q) == len(set(q))
        reconcile(q, r, U, old, key, differences, checks)
        scores.append((q, r, task))
        if len(scores) % 20 == 0:
            print(json.dumps({"scored_cells": len(scores), "differences": len(differences)}), flush=True)
    save_json(a.report_dir / "reconciliation.json", {"checks": len(checks), "differences": differences,
        "new_without_old_comparator": ["all-query group", "all new metric CIs", "sensitivity P@3/MAP"],
        "cells": len(scores), "all_points_checked_before_new_intervals": True})
    assert not differences, "Point mismatch: diagnose before calculating intervals"
    out, weight_cache = [], {}
    for q, r, task in scores:
        wc_key = tuple(sorted(set(zip(U["lang"][q], U["day"][q]))))
        if wc_key not in weight_cache:
            weight_cache[wc_key] = cluster_weights(U["lang"][q], U["day"][q])[0]
        # Cluster-column order depends only on the sorted universe; q order can differ.
        pos = {c: i for i, c in enumerate(wc_key)}
        di = np.array([pos[c] for c in zip(U["lang"][q], U["day"][q])])
        out.extend(aggregate(q, r, U, task, weight_cache[wc_key], di))
    save_json(a.report_dir / "metrics.json", {"seed": SEED, "resamples": B, "interval": "pointwise_percentile_conditional_fixed_candidates_rankings",
        "chance_applies_to": ["P@1", "P@3"], "map_null": None, "rows": out})
    flat = []
    for row in out:
        rr = {k: v for k, v in row.items() if k not in ("metrics", "existing_lift_ci95")}
        rr["existing_lift_ci95"] = json.dumps(row["existing_lift_ci95"])
        for m, value in row["metrics"].items():
            for k, v in value.items():
                if k == "ci95":
                    rr[m + "_ci_low"], rr[m + "_ci_high"] = v if v else (None, None)
                else:
                    rr[m + "_" + k] = v
        flat.append(rr)
    save_csv(a.report_dir / "metrics.csv", flat)
    writing_tables(out, a.report_dir)
    save_json(a.report_dir / "provenance.json", {"signature": signature, "fingerprint": fingerprint,
        "seed": SEED, "resamples": B, "bit_generator": "PCG64", "python": platform.python_version(),
        "platform": platform.platform(), "input_root": str(a.input_root.resolve()),
        "trees_root": str(a.trees_root.resolve()), "cache_dir": str(a.cache_dir.resolve()),
        "reconstructed_pool_cells": reconstructed, "reused_pool_cells": reused,
        "total_pool_cells": len(all_tasks), "rows": len(out), "old_point_checks": len(checks),
        "elapsed_seconds": round(time.perf_counter() - start, 3), "interaction_effects_executed": False,
        "output_hashes": {p.name: sha(p) for p in a.report_dir.iterdir() if p.suffix in (".csv", ".json", ".md") and p.name != "provenance.json"}})
    print(json.dumps({"complete_rows": len(out), "old_point_checks": len(checks), "reconstructed": reconstructed,
                      "reused": reused, "seconds": round(time.perf_counter() - start, 2)}))


if __name__ == "__main__":
    main()
