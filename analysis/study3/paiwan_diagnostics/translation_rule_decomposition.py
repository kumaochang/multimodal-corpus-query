"""Replay the unchanged English rule, reconcile, then decompose its positive pools.

Requires the already installed historical parser. No installation, Fisher test,
refitting, manual force assignment or modification of the original scripts.
"""
import ast
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys
from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA_ROOT = ROOT
OUT = None
RULE = ROOT/'analysis/paiwan_texts/study2_rebuild.py'
RECALL = ROOT/'scripts/study2_recall_check.py'
XML = DATA_ROOT/'data/paiwan_texts/xml'
LANG = '{http://www.w3.org/XML/1998/namespace}lang'
EXPECTED = {'sent': (117,64,132,2608), 'voc2': (186,98,63,2574), 'voc12': (203,112,46,2560)}
EXPECTED_LEX = {'sent': (16,10,38), 'voc2': (22,24,52), 'voc12': (25,25,62)}
PINS = {}
OPENED = {}
SOURCE_RECORD = json.loads((HERE / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))



def input_key(path):
    path = Path(path).resolve()
    for base in (DATA_ROOT, ROOT):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            pass
    raise ValueError(f"Input is outside the repository and input roots: {path}")


def checked_read(path):
    path = Path(path).resolve()
    key = input_key(path)
    data = path.read_bytes()
    record = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    expected = SOURCE_RECORD['scientific_inputs'].get(key, SOURCE_RECORD['public_code_inputs'].get(key))
    if expected is not None and record != expected:
        raise ValueError(f"Input identity mismatch: {key}")
    if key in PINS and PINS[key] != record:
        raise ValueError(f"Input changed during execution: {key}")
    PINS[key] = record
    OPENED[key] = path
    return data


def require_inputs(prefixes):
    required = [key for key in SOURCE_RECORD['scientific_inputs'] if key.startswith(prefixes)]
    missing = [key for key in required if not (DATA_ROOT / key).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} recorded inputs are missing under {DATA_ROOT}; first: {missing[0]}")


def read(path):
    return checked_read(path)


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def layers(node,tag):
    return [{'attributes':dict(x.attrib),'text':x.text} for x in node.findall(tag)]


def norm(s):
    return (s or '').strip().lower().replace('’',"'")


def load_rows():
    manifest={line.split(maxsplit=1)[1].lstrip('*'):line.split()[0]
              for line in read(DATA_ROOT/'data/paiwan_texts/SHA256SUMS.txt').decode().splitlines() if line.strip()}
    rows=[]
    for path in sorted(XML.glob('*.xml')):
        data=read(path);assert PINS[input_key(path)]['sha256']==manifest[path.name]
        for ordinal,s in enumerate(E.fromstring(data).iter('S'),1):
            tr=[t for t in s.findall('TRANSL') if t.get(LANG)=='eng']
            text=(tr[0].text or '').strip() if tr else ''
            glosses=[(m.findtext('TRANSL') or '').strip().lower() for m in s.iter('M')]
            forms=[norm(w.findtext("FORM[@kindOf='standard']") or '') for w in s.findall('W')]
            maya=any(f.startswith('maya') for f in forms) or "don't" in [norm(g) for g in glosses]
            ari='ari' in forms or 'go.on' in [norm(g) for g in glosses]
            rows.append({'id':s.get('id'),'file':input_key(path),'S_ordinal':ordinal,
                'line':s.sourceline,'english':text,'morph':any(g in ('imp','hort') for g in glosses),
                'maya':maya,'ari':ari,'node':s})
    assert len(rows)==2921 and len({r['id'] for r in rows})==2921
    assert sum(r['morph'] for r in rows)==249
    return rows


def mechanical_patterns(r):
    records=[]
    for i,w in enumerate(r['node'].findall('W'),1):
        f=w.findtext("FORM[@kindOf='standard']")
        t=(f or '').strip().lower()
        flags={'ends_av':t.endswith('av'),'ends_an':t.endswith('an'),
               'starts_si_ends_an':t.startswith('si') and t.endswith('an') and len(t)>4}
        if any(flags.values()):
            records.append({'W_ordinal':i,'W_id':w.get('id'),'FORM':layers(w,'FORM'),
                'pattern_flags':flags,'word_TRANSL':layers(w,'TRANSL'),
                'morphemes':[{'attributes':dict(m.attrib),'FORM':layers(m,'FORM'),
                             'TRANSL':layers(m,'TRANSL')} for m in w.findall('M')]})
    return records


def exact_payload(r):
    s=r['node']
    return {'id':r['id'],'file':r['file'],'line':r['line'],'S_ordinal':r['S_ordinal'],
        'FORM':layers(s,'FORM'),'PHON':layers(s,'PHON'),'TRANSL':layers(s,'TRANSL'),
        'words':[{'attributes':dict(w.attrib),'FORM':layers(w,'FORM'),'PHON':layers(w,'PHON'),
                  'TRANSL':layers(w,'TRANSL'),'morphemes':[{'attributes':dict(m.attrib),
                  'FORM':layers(m,'FORM'),'PHON':layers(m,'PHON'),'TRANSL':layers(m,'TRANSL')}
                  for m in w.findall('M')]} for w in s.findall('W')]}


def main():
    rows=load_rows()
    source=read(RULE).decode();read(RECALL)
    decision=json.loads(read(ROOT/'analysis/study3/clm_jev_preparation/DECISIONS.json'))
    assert PINS[RULE.relative_to(ROOT).as_posix()]['sha256']==decision['study2_rule_sha256']
    versions={}
    for package in ('spacy','en_core_web_sm'):
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:versions[package]=None
    if versions != {'spacy':'3.8.16','en_core_web_sm':'3.8.0'}:
        raise RuntimeError('Reproduction requires spacy==3.8.16 and en_core_web_sm==3.8.0; saved results are unchanged.')
    import spacy
    pipeline=spacy.load('en_core_web_sm')
    @lru_cache(maxsize=None)
    def nlp(t):return pipeline(t)
    scope={'re':re,'nlp':nlp,'variant':'sent'}
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('clause_starts','is_imp','directive')
           or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('NEG','HORT','DEON') for t in n.targets)]
    assert len(nodes)==6
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(RULE),'exec'),scope)
    aux=ast.parse(read(RECALL).decode());auxnodes=[n for n in aux.body if isinstance(n,ast.FunctionDef)
        and n.name in ('clean_start','expanded_starts')]
    exec(compile(ast.Module(body=auxnodes,type_ignores=[]),str(RECALL),'exec'),scope)
    is_imp=lru_cache(maxsize=None)(scope['is_imp'])
    quoted=re.compile(r'["“]([^"”]+)["”]')
    @lru_cache(maxsize=None)
    def variants(t):
        sent=tuple(s.text for s in nlp(t).sents)
        units=sent+tuple(quoted.findall(t))+tuple(re.split(r'[:;]',t))
        return {'sent':sent,'voc2':tuple(x for u in units for x in scope['expanded_starts'](u,2)),
                'voc12':tuple(x for u in units for x in scope['expanded_starts'](u,12))}
    def triggers(t,v):
        flags={'bare_VB':False,'negative':False,'hortative':False,'deontic':bool(scope['DEON'].search(t))}
        for span in variants(t)[v] if t else ():
            x=scope['clean_start'](span)
            if not x or x.rstrip().endswith('?'):continue
            neg=bool(scope['NEG'].match(x));hort=bool(scope['HORT'].match(x))
            flags['negative'] |= neg;flags['hortative'] |= hort
            # Original is_imp's VB branch, not a new lexical verb list.
            if not neg and not hort:
                flags['bare_VB'] |= bool(is_imp(x))
        return flags
    checks=[]
    assert scope['is_imp']('Go home!') is True
    assert scope['is_imp']("Don't go?") is False
    assert list(scope['expanded_starts']('a b c, Go home!',2)) == ['a b c, Go home!']
    assert list(scope['expanded_starts']('a b c, Go home!',12)) == ['a b c, Go home!','Go home!']
    test=triggers("You should go; Don't stay!",'voc2')
    assert test['deontic'] and test['negative']
    checks += ['original VB predicate and question exclusion', 'comma-prefix 2/12 distinction',
               'nonexclusive deontic and negative triggers']
    output={};identities={};traces={};pattern_rows=[];labelsets={}
    for v in ('sent','voc2','voc12'):
        labels={};hits={}
        for r in rows:
            t=r['english'];hits[r['id']]=triggers(t,v)
            if v=='sent':
                scope['variant']='sent';classified=scope['directive'](t)
            else:
                classified=bool(t) and (bool(scope['DEON'].search(t)) or any(is_imp(x) for x in variants(t)[v]))
            assert classified==any(hits[r['id']].values())
            labels[r['id']]=classified
        cells=tuple(sum((labels[r['id']],r['morph'])==pair for r in rows)
                    for pair in ((True,True),(True,False),(False,True),(False,False)))
        assert cells==EXPECTED[v], (v,cells)
        labelsets[v]=labels
        pool=[r for r in rows if labels[r['id']] and not r['morph']]
        lexical=tuple(sum((r['maya'] if k=='maya' else not r['maya'] and r['ari'] if k=='ari'
                           else not r['maya'] and not r['ari']) for r in pool) for k in ('maya','ari','residue'))
        assert lexical==EXPECTED_LEX[v],(v,lexical)
        residue=[r for r in pool if not r['maya'] and not r['ari']]
        identities[v]={'no_suffix':[r['id'] for r in pool],'residue':[r['id'] for r in residue]}
        def summarize(rr):
            counts=Counter(k for r in rr for k,b in hits[r['id']].items() if b)
            signatures=Counter('+'.join(k for k,b in hits[r['id']].items() if b) for r in rr)
            cross={key:{name:sum(hits[r['id']][key] and bool(r[name]) for r in rr)
                        for name in ('maya','ari')} for key in ('bare_VB','negative','hortative','deontic')}
            return {'n':len(rr),'trigger_counts_nonexclusive':counts,'trigger_signature_counts':signatures,
                    'trigger_by_lexical_test':cross,'multiple_trigger_sentences':sum(sum(hits[r['id']].values())>1 for r in rr)}
        output[v]={'cells':cells,'lexical_maya_ari_residue':lexical,'no_suffix':summarize(pool),
                   'residue':summarize(residue)}
        traces[v]={r['id']:hits[r['id']] for r in pool}
        for r in residue:
            for w in mechanical_patterns(r):pattern_rows.append(dict(variant=v,id=r['id'],**w))
    transitions={}
    for v,expected in [('voc2',(69,34)),('voc12',(86,48))]:
        assert all(not labelsets['sent'][r['id']] or labelsets[v][r['id']] for r in rows)
        moved=[r for r in rows if labelsets[v][r['id']] and not labelsets['sent'][r['id']]]
        counts=(sum(r['morph'] for r in moved),sum(not r['morph'] for r in moved))
        assert counts==expected
        transitions[v]={'from_sent_morphology_only':counts[0],'from_sent_neither':counts[1],
            'trigger_counts_nonexclusive':Counter(k for r in moved for k,b in triggers(r['english'],v).items() if b)}
    pattern_summary=[]
    for v in ('sent','voc2','voc12'):
        for pattern in ('ends_av','ends_an','starts_si_ends_an'):
            records=[r for r in pattern_rows if r['variant']==v and r['pattern_flags'][pattern]]
            profiles=Counter(json.dumps([t['text'] for m in r['morphemes'] for t in m['TRANSL']],
                                       ensure_ascii=False) for r in records)
            exact_m=Counter(k for r in records for m in r['morphemes'] for k in ('av','an','si')
                            if any((f['text'] or '').strip().lower()==k for f in m['FORM']))
            exact_gloss=Counter(k+'|'+(g['text'] or '') for r in records for m in r['morphemes']
                for k in ('av','an','si') if any((f['text'] or '').strip().lower()==k for f in m['FORM'])
                for g in m['TRANSL'])
            pattern_summary.append({'variant':v,'condition':pattern,'sentences':len({r['id'] for r in records}),
                'words':len(records),'published_morpheme_gloss_profiles':profiles,
                'published_M_FORM_literal_node_counts':exact_m,
                'words_with_literal_M_FORM':{k:sum(any(any((f['text'] or '').strip().lower()==k
                    for f in m['FORM']) for m in w['morphemes']) for w in records) for k in ('av','an','si')},
                'published_literal_M_FORM_and_gloss':exact_gloss})
    # Illustration selection is by stated rule condition and source order, not effect size.
    selected=[];selection_roles={}
    def take(role,predicate):
        match=next((r for r in rows if predicate(r)),None)
        if match is not None:
            if match['id'] not in selected:selected.append(match['id'])
            selection_roles.setdefault(match['id'],[]).append(role)
    primary=set(identities['sent']['residue'])
    for trigger in ('bare_VB','negative','hortative','deontic'):
        take('first_primary_'+trigger,lambda r:r['id'] in primary and traces['sent'][r['id']][trigger])
    for v,previous in [('voc2','sent'),('voc12','voc2')]:
        new=set(identities[v]['residue'])-set(identities[previous]['residue'])
        take('first_new_'+v,lambda r:r['id'] in new)
    take('first_primary_an_with_lf_gloss',lambda r:r['id'] in primary and any(
        w['pattern_flags']['ends_an'] and any((g['text'] or '')=='lf'
            for m in w['morphemes'] for g in m['TRANSL']) for w in mechanical_patterns(r)))
    take('first_primary_an_without_literal_an_M',lambda r:r['id'] in primary and any(
        w['pattern_flags']['ends_an'] and not any((f['text'] or '').strip().lower()=='an'
            for m in w['morphemes'] for f in m['FORM']) for w in mechanical_patterns(r)))
    take('first_primary_si_an',lambda r:r['id'] in primary and any(
        w['pattern_flags']['starts_si_ends_an'] for w in mechanical_patterns(r)))
    all_ids=set(x for d in identities.values() for x in d['residue'])
    selected += [r['id'] for r in rows if r['id'] in all_ids and r['id'] not in selected][:10-len(selected)]
    selected=[r['id'] for r in rows if r['id'] in set(selected)]
    examples=[dict(exact_payload(r),selection_roles=selection_roles.get(sid,['source_order_fill']),
                   variants=[v for v in identities if r['id'] in identities[v]['residue']],
                   triggers={v:traces[v][r['id']] for v in identities if r['id'] in traces[v]},
                   spelling_patterns=mechanical_patterns(r)) for sid in selected for r in rows if r['id']==sid]
    for name,pin in PINS.items():
        b=OPENED[name].read_bytes();assert hashlib.sha256(b).hexdigest()==pin['sha256']
    write('translation_rule_decomposition.json',{'status':'reconciled','variants':output,'identities':identities,'triggers':traces,
        'all_unit_decisions':labelsets,'transitions_from_sent':transitions,'spelling_pattern_summary':pattern_summary,
        'synthetic_checks':checks,
        'residue_word_spelling_records':pattern_rows,
        'example_selection':'First source-ordered witness per primary trigger, variant addition, an/lf, an without literal an M, si...an; deduplicate and fill in source order to ten.',
        'parser_replay':{'spacy':versions['spacy'],
        'model':versions['en_core_web_sm'],'training_or_tuning':False}})
    location_keys = ('id', 'file', 'line', 'S_ordinal', 'selection_roles', 'variants', 'triggers')
    write('translation_rule_example_locations.json', [{k: r[k] for k in location_keys} for r in examples])
    write('execution_provenance.json', {'inputs': PINS, 'parser': versions, 'training_or_tuning': False})
    print('Original three-variant cells and lexical totals reconcile; no Fisher tests calculated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT,
                        help='Root containing data/paiwan_texts/xml and SHA256SUMS.txt.')
    parser.add_argument('--out', required=True, type=Path, help='New output directory; must not already exist.')
    args = parser.parse_args()
    DATA_ROOT = args.input_root.resolve()
    XML = DATA_ROOT / 'data/paiwan_texts/xml'
    require_inputs(('data/paiwan_texts/',))
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    main()
