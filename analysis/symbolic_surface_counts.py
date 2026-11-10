"""Surface-form and translation-rule counts (exploratory, descriptive; 2026-10-15).

Run from the repository root:
    python analysis/symbolic_surface_counts.py <xml_dir> <output_dir>

Produces
  1. word-final counts for the -u (imp) and -i (hort) rows of Table 3.2;
  2. exploratory rule counts: sentence translation contains let's / let us,
     with apostrophes normalized and no match for we alone;
  3. counts of hort-i target word glosses containing pair or companion,
     including their overlap with the translation rule;
  4. rule_count_summary.json, with counts and matching source tokens.
The existing hort_i_sentences.tsv is not read or rewritten; it remains a
historical summary record, separate from the rule counts.

Definitions
  word phone string : word-level <PHON kindOf="original">, apostrophes
                      normalized, all non-word characters removed.
  imp target        : word containing an M whose gloss is `imp`.
  hort-i target     : word containing an M with original FORM `i` and gloss `hort`.
The translation rule is sentence-level, unvalidated, and exploratory. It does
not classify force or assign a participant reading to individual tokens.
"""
import collections, glob, json, os, re, sys
import xml.etree.ElementTree as ET

xml_dir, out_dir = sys.argv[1], sys.argv[2]
LET_US = re.compile(r"\blet's\b|\blet\s+us\b", re.I)

def norm(s):
    return (s or "").replace("’", "'").replace("ʼ", "'").replace("‘", "'").strip()

def orig(node, tag):
    for x in node.findall(tag):
        if x.get("kindOf") in (None, "original"):
            return norm(x.text)
    return ""

wf = collections.Counter()
rule_matches = {"hort_i": [], "imp_u": [], "pair_companion": []}
for f in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
    for s in ET.parse(f).getroot().iter("S"):
        tr = norm(s.findtext("TRANSL"))
        let_us = bool(LET_US.search(tr))
        for w in s.findall("W"):
            ph = re.sub(r"[^\w']", "", orig(w, "PHON"))
            ms = [(orig(m, "FORM"), orig(m, "TRANSL")) for m in w.findall("M")]
            is_imp = any(g == "imp" for _, g in ms)
            is_hort_i = any(f_ == "i" and g == "hort" for f_, g in ms)
            for v in "ui":
                if ph.endswith(v):
                    wf[v + "_wordfinal"] += 1
            if is_imp:
                wf["imp"] += 1; wf["imp_wordfinal_u"] += ph.endswith("u")
                wf["imp_let_us"] += let_us
                if let_us:
                    rule_matches["imp_u"].append({"sentence_id": s.get("id"),
                        "word": orig(w, "FORM"), "translation": tr})
            if is_hort_i:
                wf["hort"] += 1; wf["hort_wordfinal_i"] += ph.endswith("i")
                wf["hort_let_us"] += let_us
                gloss = orig(w, "TRANSL")
                pair_companion = "pair" in gloss or "companion" in gloss
                wf["hort_pair_companion"] += pair_companion
                wf["hort_pair_companion_let_us"] += pair_companion and let_us
                token = {"sentence_id": s.get("id"), "word": orig(w, "FORM"),
                         "word_gloss": gloss, "translation": tr,
                         "let_us_match": let_us}
                if let_us:
                    rule_matches["hort_i"].append(token)
                if pair_companion:
                    rule_matches["pair_companion"].append(token)

u_t, u_m = wf["imp_wordfinal_u"], wf["u_wordfinal"]
i_t, i_m = wf["hort_wordfinal_i"], wf["i_wordfinal"]
print(f"-u imp : {wf['imp']} targets; {u_t} word-final; {u_m} words ending in u; 1:{u_m/u_t:.1f} ({u_t/u_m:.1%})")
print(f"-i hort: {wf['hort']} targets; {i_t} word-final; {i_m} words ending in i; 1:{i_m/i_t:.1f} ({i_t/i_m:.1%})")
print(f"let's/let us rule (no we): hort-i {wf['hort_let_us']}/{wf['hort']}; imp-u {wf['imp_let_us']}/{wf['imp']}")
print(f"hort-i word gloss pair/companion: {wf['hort_pair_companion']} targets; {wf['hort_pair_companion_let_us']} match let's/let us")
summary = {
    "definition": "Original published sentence translation, apostrophes normalized; case-insensitive whole-word let's or let us; we alone is not matched. Counts are target word tokens, not unique sentences. Unvalidated, exploratory.",
    "hort_i": {"targets": wf["hort"], "let_us_matches": wf["hort_let_us"]},
    "imp_u": {"targets": wf["imp"], "let_us_matches": wf["imp_let_us"]},
    "hort_pair_companion": {"targets": wf["hort_pair_companion"],
                            "let_us_matches": wf["hort_pair_companion_let_us"]},
    "word_final": {"imp_u_targets": u_t, "u_words": u_m,
                   "hort_i_targets": i_t, "i_words": i_m},
    "matching_tokens": rule_matches,
}
with open(os.path.join(out_dir, "rule_count_summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print("Historical hort_i_sentences.tsv is outside this script's inputs and outputs.")
