"""Rebuild of Study 2's 2x2 table from the documented rule (project doc
claude/study2-beyond-the-tag-set.md §2), 2026-10-14. Reproduction attempt; the original
script was lost. Usage: python3 study2_rebuild.py <xml_dir> <variant>"""
import sys, glob, re, json
from lxml import etree
import spacy
from scipy.stats import fisher_exact
nlp = spacy.load("en_core_web_sm")
xml_dir, variant = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "sent")
rows = []
for f in sorted(glob.glob(xml_dir + "/**/*.xml", recursive=True)):
    root = etree.parse(f).getroot()
    for s in root.iter("S"):
        tr = [t for t in s.findall("TRANSL") if t.get("{http://www.w3.org/XML/1998/namespace}lang") == "eng"]
        transl = (tr[0].text or "").strip() if tr else ""
        glosses = [ (m.findtext("TRANSL") or "").strip().lower() for m in s.iter("M") ]
        morph = any(g in ("imp", "hort") for g in glosses)
        rows.append((s.get("id"), transl, morph))
NEG = re.compile(r"^(don['’]t|do not|never)\b", re.I)
HORT = re.compile(r"^(let['’]s|let us)\b", re.I)
DEON = re.compile(r"\byou (should|must|['’]d better|had better)\b", re.I)
def clause_starts(text):
    if variant == "whole":
        return [text]
    doc = nlp(text)
    out = [sent.text for sent in doc.sents]
    if variant == "quotes":
        out += re.findall(r"[\"“]([^\"”]+)[\"”]", text)
    return out
def is_imp(span):
    t = span.strip().lstrip("\"“‘'(").strip()
    if not t or t.rstrip().endswith("?"):
        return False
    if NEG.match(t) or HORT.match(t):
        return True
    doc = nlp(t)
    toks = [x for x in doc if not x.is_punct and not x.is_space]
    if not toks:
        return False
    first = toks[0]
    if first.tag_ == "VB" and not any(c.dep_ in ("nsubj", "nsubjpass") for c in first.children):
        return True
    return False
def directive(text):
    if not text:
        return False
    if DEON.search(text):
        return True
    return any(is_imp(c) for c in clause_starts(text))
a=b=c=d=0
for sid, tr, morph in rows:
    dirv = directive(tr)
    if dirv and morph: a+=1
    elif dirv and not morph: b+=1
    elif morph: c+=1
    else: d+=1
OR, p = fisher_exact([[a,b],[c,d]])
print(json.dumps({"variant":variant,"sentences":len(rows),"morph":a+c,"directive":a+b,"overlap":a,"dir_no_morph":b,"morph_no_dir":c,"neither":d,"odds_ratio":round(OR,1),"p":p,"share_unmarked":round(b/(a+b),3) if a+b else None}))
