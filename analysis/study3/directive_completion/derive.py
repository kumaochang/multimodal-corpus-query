"""Root-gated, deterministic completion of saved directive evidence.

No model, ranking, human-answer intake, inference, or network execution.
Raw strings and item-level records belong exclusively in controlled data.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from datetime import datetime, timezone

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from lxml import etree as E
from threadpoolctl import threadpool_info, threadpool_limits

INVENTORY_SHA = "a53c9712997933a8ba968bff6a49d48ee7f0f41569bf00aca41bb07db4fac991"
GATE_SHA = "c10f05eee0315a2250629eba557d47b8e6c5373a5af4933e4b675ec919026437"
ROUTES = ("Audio", "ASR_XLMR", "ASR_TFIDF")
OUTCOMES = {1: "affirmative", 0: "negative", -1: "cannot_tell"}
BASE = "data/study3_runs/"
CACHE = "data/study3_stats/symbolic_similarity/local_run/run/"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [(line, json.loads(text)) for line, text in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1) if text.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def unique(rows, key):
    out = {}
    for row in rows:
        require(row[key] not in out, "duplicate ID in approved input")
        out[row[key]] = row
    return out


def frozen_functions(root, stop):
    """Compile only exact pure function nodes, not module imports/main bodies."""
    scope = {"re": re, "STOP": stop, "LEX": re.compile(r"^(ari|ma[jy]a)$")}
    locations = []
    for path, names in [("pipeline/analysis/form_candidates.py", {"norm", "form_of"}),
                        ("pipeline/analysis/item5/feat_text.py", {"form_vec"})]:
        tree = ast.parse((root / path).read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        require({n.name for n in nodes} == names, "frozen pure function missing")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), scope)
        locations += [{"path": path, "name": n.name, "first_line": n.lineno,
                       "last_line": n.end_lineno} for n in nodes]
    return scope, locations


def tokens(text, norm, stop):
    if text is None:
        return []
    require(isinstance(text, str), "non-string target text")
    spans = list(re.finditer(r"\S+", text))
    require([m.group() for m in spans] == text.split(), "span/token mismatch")
    return [{"token_index_1based": i, "start_char": m.start(), "end_char": m.end(),
             "raw": m.group(), "normalized": norm(m.group()),
             "eligible_ui_before_marker_priority": len(norm(m.group())) >= 2 and
             norm(m.group())[-1] in "ui" and norm(m.group()) not in stop,
             "marker": bool(re.fullmatch(r"ari|ma[jy]a", norm(m.group()))),
             "raw_utf8_sha256": sha(m.group().encode("utf-8"))}
            for i, m in enumerate(spans, 1)]


def fields(element, tag):
    return [{"attributes": dict(c.attrib), "text": c.text} for c in element.findall(tag)]


def standard(element):
    forms = [c for c in element.findall("FORM") if c.get("kindOf") == "standard"]
    require(len(forms) == 1, "not exactly one published standard FORM")
    return forms[0].text


def profile(word):
    # IDs, locations and word spelling are deliberately outside the gloss profile.
    value = [fields(word, "TRANSL"),
             [[fields(m, "FORM"), fields(m, "TRANSL")] for m in word.findall("M")]]
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha(encoded.encode("utf-8")), value


def resource_index(root, manifests, keys, norm):
    book, ntu = collections.defaultdict(list), collections.defaultdict(list)
    schema = collections.Counter()
    for resource, files in manifests.items():
        for path in files:  # frozen inventory order, no source rewriting
            doc = E.parse(str(root / path))
            parent = doc.getroot()
            sentences = list(parent.iter("S"))
            for si, s in enumerate(sentences, 1):
                schema[resource + "_S"] += 1
                if resource == "One_Hundred_Paiwan_Texts":
                    for wi, w in enumerate(s.findall("W"), 1):
                        schema["book_W"] += 1
                        for m in w.findall("M"):
                            schema["book_M"] += 1
                            if not any((c.text or "").strip() for c in m.findall("TRANSL")):
                                schema["book_M_missing_gloss"] += 1
                        raw = standard(w)
                        key = norm(raw or "")
                        if key not in keys:
                            continue
                        pid, pv = profile(w)
                        book[key].append({
                            "resource": resource, "path": path, "source": files[path],
                            "root_attributes": dict(parent.attrib), "S_index_1based": si,
                            "S_id": s.get("id"), "W_index_1based": wi, "W_id": w.get("id"),
                            "word_FORM": fields(w, "FORM"), "word_PHON": fields(w, "PHON"),
                            "word_TRANSL": fields(w, "TRANSL"),
                            "sentence_FORM": fields(s, "FORM"),
                            "sentence_TRANSL": fields(s, "TRANSL"),
                            "morphemes": [{"M_index_1based": mi, "M_id": m.get("id"),
                                           "FORM": fields(m, "FORM"), "PHON": fields(m, "PHON"),
                                           "TRANSL": fields(m, "TRANSL")}
                                          for mi, m in enumerate(w.findall("M"), 1)],
                            "normalized_surface": key, "match_basis": "frozen_normalized_W_standard_FORM",
                            "standard_FORM_utf8_sha256": sha((raw or "").encode("utf-8")),
                            "exact_profile_sha256": pid, "exact_profile": pv,
                            "target_morpheme_identity": "unknown"})
                else:
                    require(not list(s.iter("W")) and not list(s.iter("M")) and
                            not list(s.iter("TRANSL")), "NTU layer differs from gate")
                    raw = standard(s)
                    for t in tokens(raw, norm, set()):
                        key = t["normalized"]
                        if key in keys:
                            ntu[key].append({
                                "resource": resource, "path": path, "source": files[path],
                                "root_attributes": dict(parent.attrib), "S_index_1based": si,
                                "S_id": s.get("id"), "sentence_FORM": fields(s, "FORM"),
                                "sentence_PHON": fields(s, "PHON"),
                                "AUDIO_attributes": [dict(x.attrib) for x in s.findall("AUDIO")],
                                "token": t, "match_basis": "frozen_normalized_standard_S_FORM_whitespace_token",
                                "gloss_status": "layer_unavailable", "target_morpheme_identity": "unknown"})
    records = []
    for key in sorted(keys):  # index serialization only, never target/candidate order
        profiles = list(dict.fromkeys(r["exact_profile_sha256"] for r in book[key]))
        raw_forms = list(dict.fromkeys(
            [next(x["text"] for x in r["word_FORM"] if x["attributes"].get("kindOf") == "standard")
             for r in book[key]] + [r["token"]["raw"] for r in ntu[key]]))
        records.append({"normalized_surface": key, "book_records": book[key],
                        "book_profile_hashes": profiles,
                        "book_status": "no_attestation" if not profiles else
                        "single_recorded_profile" if len(profiles) == 1 else "multiple_recorded_profiles",
                        "NTU_records": ntu[key], "NTU_FORM_attested": bool(ntu[key]),
                        "NTU_gloss_status": "layer_unavailable",
                        "raw_source_variants": raw_forms,
                        "different_raw_forms_share_normalized_key": len(raw_forms) > 1,
                        "target_morpheme_identity": "unknown"})
    return records, dict(schema)


def summarize(rows, strata):
    """Fixed full outcome cross-tab; unique-unit margins are allowed to overlap."""
    statuses = ("no_attestation", "single_recorded_profile", "multiple_recorded_profiles")
    output = []
    for ending in ("all", "u", "i"):
        for group in ("all", *strata):
            for status in statuses:
                for ntu in (False, True):
                    for y, outcome in OUTCOMES.items():
                        rr = [r for r in rows if (ending == "all" or r["ending"] == ending) and
                              (group == "all" or r["saved_stratum"] == group) and
                              r["book_status"] == status and r["NTU_FORM_attested"] == ntu and
                              r["reference_code"] == y]
                        output.append({"ending": ending, "original_stratum": group,
                                       "book_status": status, "NTU_FORM_attested": ntu,
                                       "reference": outcome, "tokens": len(rr),
                                       "unique_utterances": len({r["utt_id"] for r in rr})})
    return output


def rank_cost(rank, y, masks):
    require(len(rank) == len(y) and set(map(int, rank)) == set(range(len(y))), "rank incomplete")
    require(all(int(v) in (0, 1) for v in y), "unknown in main curve")
    rows = []
    selected = set()
    for k, index in enumerate(rank, 1):
        selected.add(int(index))
        row = {"K": k, "similarity_returned": k,
               "similarity_known_positive": sum(int(y[j]) == 1 for j in selected)}
        for name, mask in masks.items():
            union = selected | set(map(int, np.flatnonzero(mask)))
            row[name + "_union_returned"] = len(union)
            row[name + "_union_known_positive"] = sum(int(y[j]) == 1 for j in union)
        rows.append(row)
    return rows


def memberships(z):
    idx = z["main_indices"]
    result = []
    for pos, whole in enumerate(idx):
        if int(z["y"][whole]) != 1:
            continue
        mm = []
        ranks = {route: 1 + list(map(int, z["main_ranks_" + route])).index(pos) for route in ROUTES}
        for base in ("Phone", "Forms"):
            K = int(z["main_mask_" + base].sum())
            for route in ROUTES:
                s = bool(z["main_mask_" + base][pos])
                g = bool(z["main_mask_" + route + "_at_" + base][pos])
                mm.append({"baseline": base, "route": route, "K": K, "symbolic": s,
                           "similarity": g, "partition": "both" if s and g else
                           "symbolic_only" if s else "similarity_only" if g else "neither",
                           "first_similarity_rank": ranks[route]})
        result.append({"utt_id": str(z["ids"][whole]), "candidate_position_1based": pos + 1,
                       "all6_memberships": mm})
    return result


def select_S(cases):
    reasons = collections.OrderedDict()
    def add(row, reason):
        reasons.setdefault(row["utt_id"], []).append(reason)
    for row in cases:
        mm = [r for r in row["all6_memberships"] if r["baseline"] == "Forms"]
        if not mm[0]["symbolic"] and any(r["similarity"] for r in mm):
            add(row, "all_Forms_outside_positive")
    for route in ROUTES:
        candidates = [r for r in cases if any(m["baseline"] == "Forms" and m["route"] == route and
                      m["partition"] == "symbolic_only" for m in r["all6_memberships"])]
        if candidates:
            add(candidates[0], "first_Forms_symbolic_only_" + route)
    for route in ROUTES:
        candidates = [r for r in cases if r["utt_id"] not in reasons and any(
            m["baseline"] == "Phone" and m["route"] == route and m["partition"] == "similarity_only"
            for m in r["all6_memberships"])]
        if candidates:
            add(candidates[0], "first_uncovered_Phone_similarity_only_" + route)
            controls = [r for r in cases if any(m["baseline"] == "Phone" and m["route"] == route and
                        m["partition"] == "symbolic_only" for m in r["all6_memberships"])]
            if controls:
                add(controls[0], "first_Phone_symbolic_only_counterpart_" + route)
    # Actual final display order stays original C99, not reason insertion order.
    return [{"utt_id": r["utt_id"], "reasons": reasons[r["utt_id"]]} for r in cases if r["utt_id"] in reasons]


def select_form_contrasts(ledger):
    result = []
    for ending in ("u", "i"):
        first = collections.OrderedDict()
        for row in ledger:
            if row["ending"] == ending:
                cell = first.setdefault(row["normalized"], {})
                cell.setdefault(row["reference_code"], row)
        both = next((key for key, yy in first.items() if 1 in yy and 0 in yy), None)
        result.append({"ending": ending, "status": "observed_exact_normalized_surface_contrast" if both else
                       "no_exact_surface_with_both_outcomes", "normalized_surface": both,
                       "selected": [] if both is None else [
                           {"utt_id": first[both][y]["utt_id"], "token_id": first[both][y]["token_id"],
                            "reference_code": y} for y in (1, 0)],
                       "target_morpheme_identity": "unknown"})
    return result


def execute(root, inventory_path, gate_path, output):
    started = time.perf_counter()
    require(pin(gate_path)["sha256"] == GATE_SHA, "root gate changed")
    require(pin(inventory_path)["sha256"] == INVENTORY_SHA, "inventory changed")
    inv, gate = load(inventory_path), load(gate_path)
    require(gate["inventory_sha256"] == INVENTORY_SHA, "gate/inventory mismatch")
    require(not output.exists(), "controlled output already exists; refuse overwrite")
    require(output.is_relative_to((root / "data/study3_stats/directive_completion").resolve()),
            "raw outputs must stay in dedicated controlled data")
    pins = dict(inv["inputs"])
    for files in inv["resource_XML_manifests"].values():
        pins.update(files)
    for path, expected in pins.items():
        require(pin(root / path) == expected, "source changed: " + path)
    require(pin(root / inv["baseline"]["formal_pdf"]["path"])["sha256"] ==
            inv["baseline"]["formal_pdf"]["sha256"], "formal PDF changed")
    output.mkdir(parents=True)
    (output / "inventory.frozen.json").write_bytes(inventory_path.read_bytes())
    (output / "execution_gate.json").write_bytes(gate_path.read_bytes())
    dump(output / "inputs.json", pins)
    stop = set(load(root / "pipeline/analysis/paiwan_suffix_stoplist.json")["stoplist"])
    fn, function_sources = frozen_functions(root, stop)
    items_all = load(root / (BASE + "annotation/items.json"))["items"]
    first = [r for r in items_all if r["batch"] in (1, 2)]
    item_map = unique(first, "utt_id")
    require(len(first) == 240, "wrong first-round size")
    textrows = jsonl(root / (BASE + "item5/texts.jsonl"))
    texts = unique([r for _, r in textrows], "utt_id")
    textlines = {r["utt_id"]: line for line, r in textrows}
    require(set(texts) == set(item_map), "text IDs differ")
    all_tokens = {}
    with np.load(root / (CACHE + "numeric_cache.npz"), allow_pickle=False) as z:
        ids = list(map(str, z["ids"]))
        require(ids == [r["utt_id"] for r in first], "frozen first-round order differs")
        with np.load(root / (BASE + "item5/text.npz"), allow_pickle=False) as ft:
            require(list(map(str, ft["ids"])) == ids, "form feature order differs")
            original_vectors = ft["form"]
        for i, row in enumerate(first):
            text = texts[row["utt_id"]]["transcript"]
            require(fn["form_of"](text) == row["form"], "frozen form class differs")
            require(np.array_equal(fn["form_vec"](text), original_vectors[i]), "form indicator differs")
            all_tokens[row["utt_id"]] = tokens(text, fn["norm"], stop)
        target_keys = {t["normalized"] for tt in all_tokens.values() for t in tt
                       if t["eligible_ui_before_marker_priority"]}
        resources, schema = resource_index(root, inv["resource_XML_manifests"], target_keys, fn["norm"])
        lookup = {r["normalized_surface"]: r for r in resources}
        units, ledger = [], []
        for i, item in enumerate(first):
            uid = item["utt_id"]
            tt = all_tokens[uid]
            matches = [t for t in tt if t["eligible_ui_before_marker_priority"]]
            units.append({"item_order_1based": i + 1, "metadata": item,
                          "reference_code": int(z["y"][i]), "reference_source": CACHE + "numeric_cache.npz::y",
                          "saved_form_vector": original_vectors[i].tolist(), "tokens": tt,
                          "transcript_status": "missing" if texts[uid]["transcript"] is None else
                          "empty" if not texts[uid]["transcript"].strip() else "present",
                          "prepriority_ui": bool(matches), "marker_overlap": item["form"] == "marker" and bool(matches)})
            for t in matches:
                source = lookup[t["normalized"]]
                ledger.append({**t, "token_id": uid + ":t" + str(t["token_index_1based"]),
                               "utt_id": uid, "item_order_1based": i + 1,
                               "saved_form": item["form"], "saved_stratum": item["stratum"],
                               "reference_code": int(z["y"][i]), "ending": t["normalized"][-1],
                               "book_status": source["book_status"],
                               "book_profile_hashes": source["book_profile_hashes"],
                               "NTU_FORM_attested": source["NTU_FORM_attested"],
                               "raw_target_in_source_variants": t["raw"] in source["raw_source_variants"],
                               "source_has_different_raw_same_normalized_key": any(
                                   raw != t["raw"] for raw in source["raw_source_variants"]),
                               "source_raw_variant_count": len(source["raw_source_variants"]),
                               "NTU_gloss_status": "layer_unavailable", "target_morpheme_identity": "unknown",
                               "source_text": {"path": BASE + "item5/texts.jsonl", "line": textlines[uid],
                                               "field": "transcript"}})
        coverage = summarize(ledger, ("marker", "suffix_cue", "suffix_nocue", "none_cue"))
        cases = memberships(z)
        chosen = select_S(cases)
        contrasts = select_form_contrasts(ledger)
        curves = []
        for route in ROUTES:
            rr = rank_cost(z["main_ranks_" + route], z["y"][z["main_indices"]],
                           {b: z["main_mask_" + b] for b in ("Phone", "Forms")})
            curves.extend({"route": route, **r} for r in rr)
        # Full ranks are copied, not reconstructed from scores.
        np.savez(output / "saved_rank_inputs.npz", ids=z["ids"], y=z["y"], days=z["days"], clips=z["clips"],
                 query_indices=z["query_indices"], main_indices=z["main_indices"],
                 **{k: z[k] for k in z.files if k.startswith("main_ranks_") or k.startswith("main_mask_")})
    require(schema["book_W"] == 24556 and schema["book_M"] == 36938 and
            schema["book_M_missing_gloss"] == 1, "resource schema differs")
    write_jsonl(output / "units.jsonl", units)
    write_jsonl(output / "token_ledger.jsonl", ledger)
    write_jsonl(output / "resource_attestations.jsonl", resources)
    dump(output / "coverage.json", coverage)
    dump(output / "positive_cases.json", cases)
    dump(output / "selection.json", {"S": chosen, "form_judgment": contrasts})
    # No new cue annotation. Preserve only already indexed rationale locators.
    layer_by_id = {r["utt_id"]: r for r in inv["S_case_inventory"]["all11_available_layer_locators"]}
    selected_ids = [r["utt_id"] for r in chosen]
    selected_ids += [r["utt_id"] for group in contrasts for r in group["selected"]]
    payload_ids = list(dict.fromkeys([r["utt_id"] for r in cases] + selected_ids))
    rationale_ids = set(payload_ids)
    rationales = {}
    # Exactly the named approved label path; no answer-directory scan.
    label_path = BASE + "labels/haiku_utt/labels.jsonl"
    for line, row in jsonl(root / label_path):
        if row.get("unit_id") in rationale_ids:
            rationales[row["unit_id"]] = {"path": label_path, "line": line,
                                          "rationale": row.get("rationale"),
                                          "evidence_role": "model_generated_explanation_not_independent_cue"}
    unit_map = {r["metadata"]["utt_id"]: r for r in units}
    payload = []
    for uid in payload_ids:
        item, text = item_map[uid], texts[uid]
        payload.append({"utt_id": uid, "metadata": item, "reference_code": unit_map[uid]["reference_code"],
                        "reference_source": CACHE + "numeric_cache.npz::y",
                        "transcript": text["transcript"], "subtitle_window": text["subtitle_window"],
                        "text_source": {"path": BASE + "item5/texts.jsonl", "line": textlines[uid]},
                        "text_field_sha256": {k: sha(text[k].encode("utf-8"))
                                              for k in ("transcript", "subtitle_window")},
                        "subtitle_definition": "Producer-order complete lines strictly overlapping [start_s-3,end_s+3]; not verified target translation",
                        "preceding_following_ASR": {"status": "not_saved_in_approved_text_cache", "text": None},
                        "all_token_positions": all_tokens[uid],
                        "resource_keys": list(dict.fromkeys(t["normalized"] for t in all_tokens[uid]
                                                            if t["eligible_ui_before_marker_priority"])),
                        "nontext_layer_locators": layer_by_id.get(uid, {}).get("already_saved_nontext_sources"),
                        "independent_nontext_cue_narrative": {"status": "unavailable", "text": None},
                        "existing_model_rationale": rationales.get(uid),
                        "target_morpheme_identity": "unknown"})
    write_jsonl(output / "writing_payload.jsonl", payload)
    with (output / "rank_cost.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    outcome_summary = []
    for y, label in OUTCOMES.items():
        u = [r for r in units if r["reference_code"] == y]
        eligible = [r for r in u if r["prepriority_ui"]]
        outcome_summary.append({"reference": label, "all_first_round": len(u),
                                "prepriority_ui_unique_utterances": len(eligible),
                                "ui_tokens": sum(len([t for t in r["tokens"] if t["eligible_ui_before_marker_priority"]]) for r in u),
                                "marker_ui_overlap_unique_utterances": sum(r["marker_overlap"] for r in u)})
    summary = {"status": "DERIVED_PENDING_INDEPENDENT_VALIDATION", "all_first_round": len(first),
               "reference_by_outcome": outcome_summary, "eligible_tokens": len(ledger),
               "eligible_unique_utterances": sum(r["prepriority_ui"] for r in units),
               "unique_normalized_forms": len(target_keys), "resource_schema": schema,
               "coverage": coverage, "selected_S_case_count": len(chosen),
               "selected_form_contrast_count": sum(len(r["selected"]) for r in contrasts),
               "all_S_positive_cases": len(cases), "all_S_membership_rows": 6 * len(cases),
               "curves_rows": len(curves), "curves_K": [1, 99],
               "fixed_anchors": [r for r in curves if r["K"] in (8, 90, 99)],
               "target_identity": "All news morpheme identities remain unknown",
               "weighting": "Unweighted descriptive token and unique-utterance counts; multi-status utterance cells overlap",
               "cost": "Unique returned items/first-appearance ranks only; no time or total preparation-cost estimate",
               "untouched_original_ABT": inv["registered_A_B_T_identity"],
               "inclusive_unknown_policy": "Original112/13unknown logical bounds retained in frozen S summary; no new curve or negative recoding",
               "cue_gap": "No independent cue narrative is saved in approved inputs; model rationale is not substitute evidence",
               "no_new_tests_intervals_rankings_or_human_annotation": True}
    dump(output / "summary.json", summary)
    for path, expected in pins.items():
        require(pin(root / path) == expected, "source changed during run: " + path)
    require(pin(root / inv["baseline"]["formal_pdf"]["path"])["sha256"] ==
            inv["baseline"]["formal_pdf"]["sha256"], "PDF changed during run")
    elapsed = time.perf_counter() - started
    require(elapsed <= 600, "soft wall-budget exceeded")
    peak = peak_memory()
    require(peak is None or peak < 4 * 1024 ** 3, "soft memory-budget exceeded")
    dump(output / "provenance.json", {
        "UTC": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "inventory_identity": INVENTORY_SHA, "gate_identity": GATE_SHA,
        "function_AST_sources": function_sources, "source_count": len(pins),
        "all_source_hashes_before_after_equal": True,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "numpy": np.__version__, "lxml": E.LXML_VERSION,
                        "platform": platform.platform(), "threadpools": threadpool_info(),
                        "thread_env": {k: os.environ[k] for k in os.environ if k.endswith("NUM_THREADS")}},
        "elapsed_seconds": elapsed, "peak_working_set_bytes": peak,
        "budgets": {"processes": 1, "threads": 1, "RAM_GiB": 4, "seconds": 600,
                    "enforcement": "soft post-run checks; no OS hard cap"},
        "program": pin(Path(__file__)),
        "outputs": {p.name: pin(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"status": summary["status"], "units": len(first), "tokens": len(ledger),
                      "curve_rows": len(curves), "elapsed_seconds": elapsed,
                      "peak_working_set_bytes": peak}))


def peak_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        obj = Memory()
        obj.cb = ctypes.sizeof(obj)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(obj), obj.cb),
                "could not measure Windows peak working set")
        return int(obj.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        execute(a.root.resolve(), a.inventory.resolve(), a.gate.resolve(), a.out.resolve())
