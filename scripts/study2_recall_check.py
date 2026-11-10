"""Reproduce clause-form coverage sensitivities without changing the primary rule.

Requires spaCy 3.8.16, en_core_web_sm 3.8.0, lxml, and SciPy.
Run from the repository root:
    python scripts/study2_recall_check.py --output analysis/paiwan_texts/study2_recall_check_results.md
Only the published Paiwan text XML is read. No manual force labels are assigned.
"""

import argparse
import contextlib
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import sys

from lxml import etree
import numpy
import scipy
from scipy.stats import fisher_exact
import spacy

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "analysis/paiwan_texts/study2_rebuild.py"
XML = ROOT / "data/paiwan_texts/xml"
QUOTED = re.compile(r'["“]([^"”]+)["”]')


def normalize(text):
    return text.strip().lower().replace("’", "'")


def clean_start(text):
    # Identical initial punctuation stripping to the imported is_imp().
    return text.strip().lstrip("\"“‘'(").strip()


def expanded_starts(unit, maximum):
    """Retain the unit and each successive allowed comma-prefix removal.

    A word is a whitespace-delimited token. Each removed chunk is counted
    separately; a longer leading chunk stops removal for that unit.
    """
    text = clean_start(unit)
    yield text
    while "," in text:
        leading, remainder = text.split(",", 1)
        if len(leading.split()) > maximum:
            break
        text = clean_start(remainder)
        yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if spacy.__version__ != "3.8.16":
        raise RuntimeError("Requires spaCy 3.8.16")
    rule_hash = hashlib.sha256(RULE.read_bytes()).hexdigest()
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(RULE), str(XML), "sent"]
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            original = runpy.run_path(str(RULE))
    finally:
        sys.argv = previous_argv
    imported_summary = json.loads(captured.getvalue())
    nlp = original["nlp"]
    if nlp.meta["version"] != "3.8.0":
        raise RuntimeError("Requires en_core_web_sm 3.8.0")
    rows = original["rows"]
    # Memoize the original predicate, without changing its decisions.
    rule_globals = original["directive"].__globals__
    is_imp = lru_cache(maxsize=None)(original["is_imp"])
    rule_globals["is_imp"] = is_imp
    decisions = {}
    for variant in ("sent", "whole", "quotes"):
        rule_globals["variant"] = variant
        decisions[variant] = {
            sid: original["directive"](translation)
            for sid, translation, morphology in rows
        }

    @lru_cache(maxsize=None)
    def sensitivity_units(text):
        return tuple(sent.text for sent in nlp(text).sents) + tuple(
            QUOTED.findall(text)
        ) + tuple(re.split(r"[:;]", text))

    def sensitivity(text, maximum):
        if not text:
            return False
        if original["DEON"].search(text):
            return True
        return any(
            is_imp(start)
            for unit in sensitivity_units(text)
            for start in expanded_starts(unit, maximum)
        )

    for variant, maximum in (("voc2", 2), ("voc12", 12)):
        decisions[variant] = {
            sid: sensitivity(translation, maximum)
            for sid, translation, morphology in rows
        }

    lexical = {}
    xml_nodes = {}
    for path in sorted(XML.glob("*.xml")):
        for sentence in etree.parse(str(path)).getroot().iter("S"):
            sid = sentence.get("id")
            forms = [normalize(word.findtext("FORM[@kindOf='standard']") or "")
                     for word in sentence.findall("W")]
            glosses = [normalize(morph.findtext("TRANSL") or "")
                       for morph in sentence.iter("M")]
            maya = any(form.startswith("maya") for form in forms) or "don't" in glosses
            ari = "ari" in forms or "go.on" in glosses
            lexical[sid] = (maya, ari)
            xml_nodes[sid] = (path, sentence)

    totals = {}
    decompositions = {}
    transitions = {}
    for variant, labels in decisions.items():
        a = b = c = d = 0
        for sid, translation, morphology in rows:
            classified = labels[sid]
            if classified and morphology:
                a += 1
            elif classified:
                b += 1
            elif morphology:
                c += 1
            else:
                d += 1
        odds, p = fisher_exact([[a, b], [c, d]])
        totals[variant] = dict(a=a, b=b, c=c, d=d, directive=a+b,
                               share=b/(a+b), odds=float(odds), p=float(p))
        assert a+b+c+d == len(rows) and a+c == imported_summary["morph"]
        if variant in ("sent", "voc2", "voc12"):
            counts = dict(maya=0, ari=0, other=0, both=0)
            for sid, translation, morphology in rows:
                if labels[sid] and not morphology:
                    maya, ari = lexical[sid]
                    counts["maya" if maya else "ari" if ari else "other"] += 1
                    counts["both"] += int(maya and ari)
            assert counts["maya"]+counts["ari"]+counts["other"] == b
            decompositions[variant] = counts
        if variant in ("voc2", "voc12"):
            moved_suffix = []
            moved_neither = []
            for sid, translation, morphology in rows:
                assert not decisions["sent"][sid] or labels[sid]
                if not decisions["sent"][sid] and labels[sid]:
                    (moved_suffix if morphology else moved_neither).append(sid)
            assert a == totals["sent"]["a"] + len(moved_suffix)
            assert b == totals["sent"]["b"] + len(moved_neither)
            transitions[variant] = (moved_suffix, moved_neither)

    you_verb = []
    for sid, translation, morphology in rows:
        if morphology and not decisions["sent"][sid]:
            tokens = [token for token in nlp(clean_start(translation))
                      if not token.is_punct and not token.is_space]
            if len(tokens) >= 2 and tokens[0].lower_ == "you" and tokens[1].pos_ == "VERB":
                you_verb.append((sid, translation))

    lines = [
        "# Paiwan text-corpus study: clause-form coverage sensitivity", "",
        "## Provenance and operational rules", "",
        f"- Primary rule: `analysis/paiwan_texts/study2_rebuild.py`, SHA-256 `{rule_hash}`; imported unchanged.",
        f"- Environment: spaCy {spacy.__version__}, en_core_web_sm {nlp.meta['version']}, SciPy {scipy.__version__}, NumPy {numpy.__version__}.",
        "- Data: One Hundred Paiwan Texts, FormosanBank XML commit `52524b1f898ac5d919f23d9abd4852b36f8f9d31`, `data/paiwan_texts/xml/`.",
        f"- Shared totals: {len(rows):,} sentences; {imported_summary['morph']} with any morpheme glossed `imp` or `hort`.",
        "- `sent` remains primary. `whole` and `quotes` use the original functions with their original variant settings.",
        "- `voc2` and `voc12` were added after inspecting the cells. They are post hoc sensitivity analyses, not replacements or independent validation.",
        "- Both new variants test each spaCy sentence, each double-quoted span (the original straight/curly-double-quote regex), and each colon/semicolon-delimited part of the translation.",
        "- Each unit is tested unchanged and after repeatedly removing its first comma-delimited chunk while that chunk contains at most 2 (`voc2`) or 12 (`voc12`) whitespace-delimited words; each removed chunk is counted separately. Longer chunks stop removal for that unit. Initial punctuation is stripped exactly as in the imported rule.",
        "- The original `is_imp` predicate, including its question-mark exclusion, is unchanged. The original deontic regex is searched over the full translation before testing units, unchanged.",
        "- Classification is a proxy from published English clause forms. No manual force labels, precision estimate, or independently measured recall is introduced.",
        "", "## Five-variant comparison", "",
        "Cells: a = classified directive + morphology; b = classified directive without morphology; c = morphology without classification; d = neither. Share = b/(a+b). Odds ratio and Fisher p use [[a,b],[c,d]].", "",
        "| Variant | a | b | c | d | Directive classifications | Share without suffix | Odds ratio | Fisher p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, result in totals.items():
        r = result
        lines.append(f"| {variant} | {r['a']} | {r['b']} | {r['c']} | {r['d']} | {r['directive']} | {100*r['share']:.1f}% | {r['odds']:.1f} | {r['p']:.6e} |")
    lines += ["", "## Movement from the primary sent cells", "",
              "| Variant | From suffix-only (sent c) | From neither (sent d) | Lost sent-positive items |",
              "|---|---:|---:|---:|"]
    for variant, (suffix, neither) in transitions.items():
        lines.append(f"| {variant} | {len(suffix)} | {len(neither)} | 0 |")
    lines += ["", "## Lexical decomposition of the directive-without-suffix cell", "",
              "Matching uses standard word-level FORM and published morpheme TRANSL, lowercased and stripped; curly apostrophes in glosses are normalized to ASCII. Maya: any word form beginning `maya` or any morpheme gloss exactly `don't`. Ari: a word form exactly `ari` or a morpheme gloss exactly `go.on`. Maya has priority if both match; the both column audits that overlap. Other is the unmatched remainder and has no assigned linguistic subtype.", "",
              "| Variant | Maya | Ari (without maya) | Other | Total | Both lexical tests |",
              "|---|---:|---:|---:|---:|---:|"]
    for variant, counts in decompositions.items():
        c = counts
        lines.append(f"| {variant} | {c['maya']} | {c['ari']} | {c['other']} | {totals[variant]['b']} | {c['both']} |")
    lines += ["", "## Source checks for the retained examples", "",
              f"Of the {totals['sent']['c']} primary suffix-only sentences, {len(you_verb)} begin with `You` followed by a verb: first non-punctuation token lowercased `you`, second token spaCy POS `VERB`, after the original initial-punctuation stripping. This is a reproducible clause-form count, not manual force classification.", ""]
    lines += [f"- {sid}: {translation}" for sid, translation in you_verb]
    for sid in ("001S44", "001S59", "001S9"):
        path, sentence = xml_nodes[sid]
        lines.append(f"- Retained example {sid}: `{path.relative_to(ROOT).as_posix()}`:{sentence.sourceline}; published layers remain unchanged.")
    expected = {"voc2": (69, 34, 186, 98), "voc12": (86, 48, 203, 112)}
    lines += ["", "## Comparison with the author's quick check", "",
              "| Variant | Quick-check moved c / d; resulting a / b | Reproduced moved c / d; resulting a / b | Match |",
              "|---|---|---|---|"]
    for variant, expected_values in expected.items():
        actual = (len(transitions[variant][0]), len(transitions[variant][1]),
                  totals[variant]["a"], totals[variant]["b"])
        lines.append(f"| {variant} | {' / '.join(map(str,expected_values))} | {' / '.join(map(str,actual))} | {actual == expected_values} |")
    actual_sent = tuple(decompositions["sent"][key] for key in ("maya", "ari", "other"))
    lines.append(f"\nPrimary lexical quick check: 16 / 10 / 38; reproduced: {' / '.join(map(str,actual_sent))}; match: {actual_sent == (16, 10, 38)}.")
    lines += ["", "## Interpretation", "",
              "Counts depend on the rule's coverage of English directive clause forms. False-positive no-suffix classifications would increase b; missed no-suffix classifications would decrease b, while missed suffixed classifications would decrease a. The net error direction of b/(a+b) is not known. The share is reported for its stability across these operationalizations, not as an upper bound or an independently established force rate.",
              "", "## Reproduction", "",
              "`python scripts/study2_recall_check.py --output analysis/paiwan_texts/study2_recall_check_results.md`", ""]
    assert hashlib.sha256(RULE.read_bytes()).hexdigest() == rule_hash
    report = "\n".join(lines)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
