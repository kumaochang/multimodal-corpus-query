"""Coverage only for SPEECH_TEXT_ADDENDUM.md. No ranker/model imports or score calculation.

python scripts/ch5_speech_text_coverage.py --input-root RESTORED_STUDY3_RUNS
Only aligned eligibility, identification and language-judgment fields are used.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LABELS = ["Comforting", "Celebrating", "Explaining", "Urging", "Reporting"]
GROUPS = ["all", "Explaining", "remainder", "Reporting", "Celebrating", "Urging", "Comforting"]
SOURCES = ["ami", "pwn", "tao", "tay", "trv", "tsu"]
COVERED = SOURCES[:-1]
STATUS = ["M", "I", "not_chinese_rule_b_excluded", "other_chinese", "mixed", "no_speech",
          "unknown_explicit", "missing_or_none", "uncovered_answer"]
LANG = {"Amis": "ami", "Paiwan": "pwn", "Yami": "tao", "Atayal": "tay", "Truku": "trv", "Tsou": "tsu"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def speech_status(answer, mf):
    if answer == "mandarin":
        return "M"
    if answer == "not_chinese":
        return "not_chinese_rule_b_excluded" if mf == "mandarin" else "I"
    if answer in ("other_chinese", "mixed", "no_speech"):
        return answer
    if answer == "unknown":
        return "unknown_explicit"
    if answer in (None, "", "none"):
        return "missing_or_none"
    return "uncovered_answer"


def group_mask(label, group):
    if group == "all":
        return np.ones(len(label), bool)
    return label != "Explaining" if group == "remainder" else label == group


def support(U, q):
    return {"queries": len(q), "days": len(set(zip(U["lang"][q], U["day"][q]))),
            "clips": len(set(U["clip"][q]))}


def signature(index):
    return hashlib.sha256(np.asarray(index, dtype="<i8").tobytes()).hexdigest()


def candidate_counts(U, q, bank, pool):
    """N_query × 2 speech groups × 5 original labels, without similarities or rankings."""
    totals = np.zeros((2, 5), np.int64)
    by_l = defaultdict(lambda: np.zeros((2, 5), np.int64))
    by_clip = defaultdict(lambda: np.zeros((2, 5), np.int64))
    by_day = defaultdict(lambda: np.zeros((2, 5), np.int64))
    in_bank = np.zeros(len(U["lang"]), bool)
    in_bank[bank] = True
    for i in bank:
        g, k = U["gidx"][i], U["kidx"][i]
        assert g in (0, 1)
        totals[g, k] += 1
        by_l[U["lang"][i]][g, k] += 1
        by_clip[U["clip"][i]][g, k] += 1
        by_day[(U["lang"][i], U["day"][i])][g, k] += 1
    out = np.zeros((len(q), 2, 5), np.int64)
    for row, i in enumerate(q):
        l = U["lang"][i]
        if pool == "X":
            out[row] = totals - by_l[l]
        elif pool == "W2":
            out[row] = by_l[l] - by_clip[U["clip"][i]]
        elif pool == "W3":
            out[row] = by_l[l] - by_day[(l, U["day"][i])]
        else:
            assert pool == "W1"
            out[row] = by_l[l]
            if in_bank[i]:
                out[row, U["gidx"][i], U["kidx"][i]] -= 1
    assert (out >= 0).all()
    return out


def count_stats(U, q, C, mask):
    n = C.sum((1, 2))
    rel = C[np.arange(len(q)), :, U["kidx"][q]].sum(1)
    scored = mask & (n > 0)
    ap = scored & (rel > 0)
    d = support(U, q[mask])
    sd = support(U, q[scored])
    def bounds(x):
        a = x[mask]
        return (int(a.min()), float(a.mean()), int(a.max())) if len(a) else (None, None, None)
    lo, avg, hi = bounds(n)
    ml, ma, mh = bounds(C[:, 0].sum(1))
    il, ia, ih = bounds(C[:, 1].sum(1))
    rl, ra, rh = bounds(rel)
    return {**d, "scored_queries": sd["queries"], "scored_days": sd["days"], "scored_clips": sd["clips"],
            "no_candidate": int((mask & (n == 0)).sum()), "ap_defined_queries": int(ap.sum()),
            "no_relevant_scored": int((scored & (rel == 0)).sum()),
            "candidate_min": lo, "candidate_mean": avg, "candidate_max": hi,
            "candidate_M_min": ml, "candidate_M_mean": ma, "candidate_M_max": mh,
            "candidate_I_min": il, "candidate_I_mean": ia, "candidate_I_max": ih,
            "relevant_min": rl, "relevant_mean": ra, "relevant_max": rh,
            "pooled_15day25clip_support": sd["days"] >= 15 and sd["clips"] >= 25,
            "sparse_diagnostic": sd["queries"] < 5 or sd["days"] < 3,
            "status": "coverage_only_no_new_inference"}


def load(root, report):
    old = json.loads((report / "inventory.json").read_text())
    hashes = {f: sha(root / f) for f in old["inputs_sha256"]}
    assert hashes == old["inputs_sha256"], "Frozen input changed"
    z = np.load(root / "features_full/features.npz", allow_pickle=False)
    ids = z["ids"]
    assert len(ids) == len(np.unique(ids))
    columns = ["unit_id", "lang", "day", "clip", "haiku", "kept", "lid_answer", "mf_class"]
    with (root / "label_analysis/units.tsv").open(encoding="utf-8") as f:
        rows = [{c: r[c] for c in columns} for r in csv.DictReader(f, delimiter="\t")]
    pos = {r["unit_id"]: r for r in rows}
    assert len(pos) == len(rows) == len(ids) and set(pos) == set(ids)
    U = {c: np.array([pos[u][c] for u in ids]) for c in columns[1:]}
    U["kept"] = U["kept"] == "True"
    assert all(LANG[l] == c for l, c in zip(z["lang"], U["lang"]))
    assert set(U["haiku"]) <= set(LABELS) and set(U["lang"]) == set(SOURCES)
    assert np.array_equal(U["kept"], np.isin(U["lid_answer"], ["not_chinese", "other_chinese"]) & (U["mf_class"] != "mandarin"))
    U["speech"] = np.array([speech_status(a, m) for a, m in zip(U["lid_answer"], U["mf_class"])])
    U["gidx"] = np.array([0 if g == "M" else 1 if g == "I" else -1 for g in U["speech"]])
    U["kidx"] = np.array([LABELS.index(k) for k in U["haiku"]])
    # Assert old W3's day key cannot accidentally combine different communities.
    day_sources = defaultdict(set)
    for l, d in zip(U["lang"], U["day"]):
        day_sources[d].add(l)
    assert max(map(len, day_sources.values())) == 1
    sub = np.abs(z["xlmr_sub"]).sum(1) > 0
    tr = np.abs(z["xlmr_tr"]).sum(1) > 0
    assert not tr[U["lang"] == "tsu"].any()
    covered = np.isin(U["lang"], COVERED)
    k = np.load(root / "unit_table/units_k.npz", allow_pickle=False)
    kp = {u: j for j, u in enumerate(k["ids"])}
    assert len(kp) == len(k["ids"]) and set(ids) <= set(kp)
    j = np.array([kp[u] for u in ids])
    kin = k["within_shot"][j].astype(bool) & (k["k_missing"][j] == 0)
    sparse_info = {}
    for source, ok in [("sub", sub), ("tr", tr)]:
        t = np.load(root / f"features_full/tfidf_{source}.npz", allow_pickle=False)
        assert t["shape"][0] == len(ids)
        nonzero = np.diff(t["indptr"]) > 0
        assert nonzero[ok].all(), "Defined text row lacks saved TF-IDF support"
        sparse_info[source] = {"shape": t["shape"].tolist(), "defined_rows": int(ok.sum()),
                               "defined_zero_sparse_rows": int((ok & ~nonzero).sum())}
    families = {"acoustic": np.ones(len(ids), bool), "subtitle": sub,
                "visual_acoustic": kin, "six_common": sub & kin, "transcription": tr & covered}
    paired = covered & U["kept"] & sub & tr
    return U, families, paired, sub, tr, hashes, sparse_info


def status_tables(U, masks, report):
    joint, daily = [], []
    for name, available in masks.items():
        source_levels = COVERED if name in ("transcription", "P5_kept_text_pair") else SOURCES
        for source in ["pooled"] + source_levels:
            sm = available if source == "pooled" else available & (U["lang"] == source)
            for status in STATUS:
                for group in ["all"] + LABELS:
                    m = sm & (U["speech"] == status) & group_mask(U["haiku"], group)
                    q = np.flatnonzero(m)
                    joint.append({"set": name, "source_community": source, "speech_status": status,
                                  "query_group": group, **support(U, q),
                                  "saved_kept_queries": int(U["kept"][q].sum()),
                                  "mf_missing_queries": int((U["mf_class"][q] == "missing").sum()),
                                  "sparse_diagnostic": len(q) < 5 or support(U, q)["days"] < 3})
    for source, day in sorted(set(zip(U["lang"], U["day"]))):
        sd = (U["lang"] == source) & (U["day"] == day)
        for status in STATUS:
            for group in ["all"] + LABELS:
                q = np.flatnonzero(sd & (U["speech"] == status) & group_mask(U["haiku"], group))
                daily.append({"source_community": source, "day": day, "speech_status": status,
                              "query_group": group, "queries": len(q), "clips": len(set(U["clip"][q]))})
    save_csv(report / "speech_text_status_coverage.csv", joint)
    save_csv(report / "speech_text_day_coverage.csv", daily)
    return joint, daily


def speech_candidates(U, families, report):
    rows, overlaps, sets = [], [], {}
    binary = U["gidx"] >= 0
    for family, available in families.items():
        q = np.flatnonzero(available & binary)
        sources = COVERED if family == "transcription" else SOURCES
        sets[family] = {"available": support(U, np.flatnonzero(available)),
                        "query_M": support(U, q[U["gidx"][q] == 0]),
                        "query_I": support(U, q[U["gidx"][q] == 1]),
                        "query_position_sha256": signature(q), "banks": {}}
        for policy, bank_mask in [("union_MI", binary), ("I_only", U["gidx"] == 1), ("M_only", U["gidx"] == 0)]:
            bank = np.flatnonzero(available & bank_mask)
            sets[family]["banks"][policy] = {**support(U, bank), "position_sha256": signature(bank)}
            for pool in (["W1", "W2", "W3"] if family == "transcription" else ["W1", "W2", "W3", "X"]):
                C = candidate_counts(U, q, bank, pool)
                for speech in ("M", "I"):
                    for source in ["pooled"] + sources:
                        sm = (U["speech"][q] == speech)
                        if source != "pooled":
                            sm &= U["lang"][q] == source
                        for group in GROUPS:
                            m = sm & group_mask(U["haiku"][q], group)
                            rows.append({"set": family, "candidate_policy": policy, "pool": pool,
                                         "query_speech_group": speech, "source_community": source,
                                         "query_group": group, **count_stats(U, q, C, m)})
                n = C.sum((1, 2))
                cells = []
                for source in sources:
                    for label in LABELS:
                        base = (U["lang"][q] == source) & (U["haiku"][q] == label) & (n > 0)
                        a = q[base & (U["gidx"][q] == 0)]
                        b = q[base & (U["gidx"][q] == 1)]
                        ad, bd = set(U["day"][a]), set(U["day"][b])
                        cells.append({"set": family, "candidate_policy": policy, "pool": pool,
                            "source_community": source, "model_class": label,
                            **{"M_" + k: v for k, v in support(U, a).items()},
                            **{"I_" + k: v for k, v in support(U, b).items()},
                            "shared_query_days": len(ad & bd), "M_only_query_days": len(ad - bd),
                            "I_only_query_days": len(bd - ad), "overlap_min_count": min(len(a), len(b)),
                            "overlap_weight": None, "status": "support_only_not_local_inference"})
                total = sum(c["overlap_min_count"] for c in cells)
                for cell in cells:
                    cell["overlap_weight"] = cell["overlap_min_count"] / total if total else None
                if total:
                    assert abs(sum(c["overlap_weight"] for c in cells) - 1) < 1e-12
                overlaps.extend(cells)
    save_csv(report / "speech_text_candidate_coverage.csv", rows)
    save_csv(report / "speech_text_overlap_weights.csv", overlaps)
    return rows, overlaps, sets


def text_tables(U, paired, sub, tr, report):
    availability, rows = [], []
    patterns = {"both_valid": sub & tr, "ASR_only": tr & ~sub,
                "subtitle_only": sub & ~tr, "neither_valid": ~sub & ~tr}
    for version in ("all", "kept"):
        version_mask = np.ones(len(paired), bool) if version == "all" else U["kept"]
        for source in ["pooled_five"] + SOURCES:
            sm = np.isin(U["lang"], COVERED) if source == "pooled_five" else U["lang"] == source
            for pattern, pattern_mask in patterns.items():
                for group in ["all"] + LABELS:
                    q = np.flatnonzero(version_mask & sm & pattern_mask & group_mask(U["haiku"], group))
                    availability.append({"version": version, "source_community": source, "availability": pattern,
                        "query_group": group, **support(U, q),
                        "status": "Tsou_transcription_not_applicable" if source == "tsu" else "coverage_only"})
    q = np.flatnonzero(paired)
    assert not (U["lang"][q] == "tsu").any()
    # Use one neutral histogram channel for ALL P5 targets; no speech grouping is applied.
    # Suppress both channel-split fields below. Actual speech statuses stay unchanged in U.
    V = dict(U)
    V["gidx"] = np.zeros(len(U["lang"]), np.int64)
    for pool in ("W1", "W2", "W3"):
        C = candidate_counts(V, q, q, pool)
        for source in ["pooled_five"] + COVERED:
            sm = np.ones(len(q), bool) if source == "pooled_five" else U["lang"][q] == source
            for group in GROUPS:
                m = sm & group_mask(U["haiku"][q], group)
                result = count_stats(V, q, C, m)
                # Avoid presenting other_chinese as Indigenous speech in text-pair candidates.
                for key in list(result):
                    if key.startswith("candidate_M_") or key.startswith("candidate_I_"):
                        del result[key]
                rows.append({"set": "P5_kept_text_pair", "pool": pool, "source_community": source,
                    "query_group": group, "paired_methods": "ASR_vs_subtitle_XLMR;ASR_vs_subtitle_TFIDF",
                    **result})
    save_csv(report / "speech_text_text_availability.csv", availability)
    save_csv(report / "speech_text_paired_coverage.csv", rows)
    return availability, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=REPO / "analysis/study3/ch5_exploratory")
    args = parser.parse_args()
    start = time.perf_counter()
    report = args.report_dir
    assert (report / "SPEECH_TEXT_ADDENDUM.md").exists()
    old_prov = json.loads((report / "completion_provenance.json").read_text())
    prior_files = {f: sha(report / f) for f in old_prov["output_sha256"]}
    assert prior_files == old_prov["output_sha256"], "Earlier outputs changed; diagnose before counting"
    U, families, paired, sub, tr, hashes, sparse = load(args.input_root, report)
    masks = {"core": np.ones(len(paired), bool), **families, "P5_kept_text_pair": paired}
    joint, daily = status_tables(U, masks, report)
    candidate_rows, overlap_rows, sets = speech_candidates(U, families, report)
    availability, text_rows = text_tables(U, paired, sub, tr, report)
    summary = {"core": support(U, np.arange(len(paired))),
        "statuses": {g: support(U, np.flatnonzero(U["speech"] == g)) for g in STATUS},
        "kept": support(U, np.flatnonzero(U["kept"])),
        "other_chinese_in_kept": int((U["kept"] & (U["speech"] == "other_chinese")).sum()),
        "sets": sets, "P5_kept_text_pair": support(U, np.flatnonzero(paired)),
        "P5_by_source": {l: support(U, np.flatnonzero(paired & (U["lang"] == l))) for l in COVERED},
        "P5_speech_status": {g: int((paired & (U["speech"] == g)).sum()) for g in STATUS},
        "P5_query_position_sha256": signature(np.flatnonzero(paired)), "sparse_matrix_checks": sparse,
        "table_rows": {"status": len(joint), "daily": len(daily), "candidate": len(candidate_rows),
                       "overlap_weights": len(overlap_rows), "text_availability": len(availability),
                       "text_paired": len(text_rows)}, "new_scores_or_interactions": 0}
    save_json(report / "speech_text_summary.json", summary)
    assert {f: sha(report / f) for f in prior_files} == prior_files
    assert all(sha(args.input_root / f) == h for f, h in hashes.items())
    produced = sorted(p for p in report.glob("speech_text_*") if p.name != "speech_text_provenance.json")
    save_json(report / "speech_text_provenance.json", {"input_root": str(args.input_root.resolve()),
        "input_sha256": hashes, "prior_output_sha256_unchanged": prior_files,
        "plan_sha256": sha(report / "SPEECH_TEXT_ADDENDUM.md"), "script_sha256": sha(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__,
        "feature_arrays_read": ["ids", "lang", "xlmr_sub", "xlmr_tr"],
        "units_k_arrays_read": ["ids", "within_shot", "k_missing"],
        "sparse_arrays_read": ["shape", "indptr"],
        "units_columns_read": ["unit_id", "lang", "day", "clip", "haiku", "kept", "lid_answer", "mf_class"],
        "new_rankings": 0, "new_intervals": 0, "model_fits": 0, "API_calls": 0,
        "output_sha256": {p.name: sha(p) for p in produced}, "elapsed_seconds": round(time.perf_counter() - start, 3)})
    print(json.dumps({"core_units": summary["core"]["queries"],
                      "status_counts": {k: v["queries"] for k, v in summary["statuses"].items()},
                      "P5": summary["P5_kept_text_pair"], "rows": summary["table_rows"]}))


if __name__ == "__main__":
    main()
