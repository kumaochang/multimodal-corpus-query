"""Fixed existing-data diagnostics. No fitting, ranking, bootstrap or answer CSV intake."""
import ast
import argparse
from datetime import date
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time
from lxml import etree as E
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA_ROOT = ROOT
OUT = None
RUNS = DATA_ROOT / 'data/study3_runs'
P1 = DATA_ROOT / 'data/study3_stats/prosody_exploration/P1_augmented_local'
BASE = DATA_ROOT / 'data/study3_stats/prosody_exploration/P1_baseline_local'
CANDIDATES = ('adalah', 'dan', 'yang', 'jangan', 'masih', 'dalam', 'atau', 'ataupun', 'jarang')
R3 = 'news-Paiwan-20250621-01_u0043'
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


def pin(path):
    return checked_read(path)


def js(path):
    return json.loads(pin(path).decode('utf-8-sig'))


def jl(path):
    return [json.loads(l) for l in pin(path).decode('utf-8-sig').splitlines() if l.strip()]


def write(name, obj):
    (OUT / name).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8')


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def original_functions(path, names, scope=None):
    t = ast.parse(pin(path).decode('utf-8-sig'))
    nodes = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    out = {} if scope is None else dict(scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), out)
    return out


def unique(rows, key):
    out = {r[key]: r for r in rows}
    assert len(out) == len(rows), 'Duplicate ' + key
    return out


def np_fields(path, fields):
    pin(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in fields}


def csv_write(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ratio(n, d):
    return n / d if d else None


def lexical_sources(norm):
    book, ntu = defaultdict(set), defaultdict(set)
    schemas = {}
    for label, folder, word_level in [('book', DATA_ROOT / 'data/paiwan_texts/xml', True),
                                      ('ntu', DATA_ROOT / 'data/ntu_paiwan_asr/xml', False)]:
        exact = set()
        counts = Counter()
        for p in sorted(folder.glob('*.xml')):
            root = E.fromstring(pin(p))
            counts['files'] += 1
            for s in root.iter('S'):
                counts['sentences'] += 1
                for element in s.findall('W') if word_level else [s]:
                    forms = element.findall("FORM[@kindOf='standard']")
                    counts['missing_standard_FORM'] += not forms
                    counts['multiple_standard_FORM'] += len(forms) > 1
                    for f in forms:
                        if not (f.text or '').strip():
                            counts['empty_standard_FORM'] += 1
                        vals = [f.text] if word_level else (f.text or '').split()
                        for raw in vals:
                            if raw:
                                exact.add(raw)
                                counts['FORM_token_occurrences'] += 1
        target = book if word_level else ntu
        for raw in exact:
            k = norm(raw)
            if k:
                target[k].add(raw)
        schemas[label] = {'counts': counts, 'exact_types': len(exact),
                          'normalized_types': len(target),
                          'normalization_collision_keys': sum(len(v) > 1 for v in target.values()),
                          'exact_forms_in_collisions': sum(len(v) for v in target.values() if len(v) > 1)}
        schemas[label + '_exact'] = exact
    return book, ntu, schemas


def asr_coverage():
    norm = original_functions(ROOT / 'pipeline/analysis/form_candidates.py', ['norm'], {'re': re})['norm']
    book, ntu, sources = lexical_sources(norm)
    be, ne = sources.pop('book_exact'), sources.pop('ntu_exact')
    idx = unique(js(DATA_ROOT / 'data/study3_metadata/units.json'), 'id')
    pwn = {k: v for k, v in idx.items() if v['comm'] == 'pwn'}
    assert len(pwn) == 2463
    tree_scope = original_functions(ROOT / 'pipeline/stages/utterances.py', ['read_tier', 'load_utterances'],
        {'Path': Path, '_HEAD': re.compile(r'"IntervalTier"\n"([^"]*)"\n([-\d.]+)\n([-\d.]+)\n(\d+)\n')})
    pool = unique(jl(RUNS / 'annotation/pool.jsonl'), 'utt_id')
    records, source_checks = [], {}
    for part, folder in [('core', RUNS), ('ext84', RUNS / 'ext84')]:
        lid = unique(jl(folder / 'lid_gemini/lid.jsonl'), 'utt_id')
        parts, missing_trees = [], []
        tree_n = 0
        for p in sorted((folder / 'transcripts').glob('*Paiwan*.jsonl')):
            rows = jl(p)
            clip = p.stem
            tree_root = 'work' if part == 'core' else 'work84'
            tree = DATA_ROOT / f'{tree_root}/data/aligned_trees' / (clip + '.aligntree.TextGrid')
            if tree.exists():
                pin(tree)
                intervals = {u: (a, b) for u, a, b in tree_scope['load_utterances'](tree, clip)}
                assert set(intervals) == {r['utt_id'] for r in rows}
                for r in rows:
                    assert abs(intervals[r['utt_id']][0] - r['start_s']) <= 1e-6
                    assert abs(intervals[r['utt_id']][1] - r['end_s']) <= 1e-6
                tree_n += len(rows)
            else:
                missing_trees.append(clip)
            for r in rows:
                uid = r['utt_id']
                if part == 'core':
                    assert uid in pwn and r['text'] == pwn[uid]['asr']
                    assert r['lang'] == pwn[uid]['asr_lang']
                    assert abs(r['start_s'] - pwn[uid]['start_s']) <= 1e-6
                    assert abs(r['end_s'] - pwn[uid]['end_s']) <= 1e-6
                    assert lid.get(uid, {}).get('answer') == pwn[uid]['lid']
                parts.append(dict(r, part=part, clip=clip, lid=lid.get(uid, {}).get('answer', 'missing'),
                                  accepted_pool=uid in pool))
        unique(parts, 'utt_id')
        source_checks[part] = {'utterances': len(parts), 'clips': len({r['clip'] for r in parts}),
            'days': len({r['clip'].split('-')[2] for r in parts}),
            'nonempty_asr': sum(bool((r['text'] or '').strip()) for r in parts),
            'language_hint_counts': Counter(r['lang'] for r in parts),
            'lid_missing': sum(r['lid'] == 'missing' for r in parts),
            'aligned_tree_verified_utterances': tree_n, 'missing_aligned_tree_clips': len(missing_trees),
            'missing_aligned_tree_clip_ids_sha256': hashlib.sha256('\n'.join(missing_trees).encode()).hexdigest()}
        records += parts
    byid = unique(records, 'utt_id')
    assert {r['utt_id'] for r in records if r['part'] == 'core'} == set(pwn)
    assert set(pool) <= set(byid), 'Accepted candidate pool ASR coverage incomplete'
    for uid, p in pool.items():
        assert abs(p['start_s'] - byid[uid]['start_s']) <= 1e-6
        assert abs(p['end_s'] - byid[uid]['end_s']) <= 1e-6
    summary = []
    candidate_rows = []
    candidate_summary = []
    for universe in ('core', 'ext84', 'all_available', 'accepted_pool'):
        subset = [r for r in records if (r['part'] == universe or universe == 'all_available'
                  or universe == 'accepted_pool' and r['accepted_pool'])]
        for group in ['all_machine_groups'] + sorted({r['lid'] for r in subset}):
            rr = subset if group == 'all_machine_groups' else [r for r in subset if r['lid'] == group]
            tokens = [t for r in rr for t in (r['text'] or '').split()]
            normtokens = [norm(t) for t in tokens]
            variants = defaultdict(set)
            for raw, normalized in zip(tokens, normtokens):
                if normalized:
                    variants[normalized].add(raw)
            base = dict(universe=universe, machine_group=group, utterances=len(rr),
                clips=len({r['clip'] for r in rr}), days=len({r['clip'].split('-')[2] for r in rr}),
                empty_asr=sum(not (r['text'] or '').strip() for r in rr),
                total_raw_tokens=len(tokens), empty_after_normalization=sum(not t for t in normtokens),
                raw_type_collisions=sum(len(v)>1 for v in variants.values()))
            for mode, ts, bset, nset in [('exact', tokens, be, ne),
                                        ('frozen_normalization', [t for t in normtokens if t], book, ntu)]:
                for unit, values in [('tokens', ts), ('types', sorted(set(ts)))]:
                    bins = Counter(('both' if t in bset and t in nset else 'book_only' if t in bset
                                    else 'ntu_only' if t in nset else 'neither') for t in values)
                    b = bins['both'] + bins['book_only']; n = bins['both'] + bins['ntu_only']; union = len(values)-bins['neither']
                    summary.append(dict(base, matching=mode, unit=unit, denominator=len(values),
                         book=b, ntu=n, union=union, both=bins['both'], book_only=bins['book_only'],
                         ntu_only=bins['ntu_only'], neither=bins['neither'],
                         book_share=ratio(b,len(values)), ntu_share=ratio(n,len(values)), union_share=ratio(union,len(values))))
            c = Counter(normtokens)
            any_tokens=sum(c[t] for t in CANDIDATES)
            any_utterances=sum(any(norm(t) in CANDIDATES for t in (r['text'] or '').split()) for r in rr)
            candidate_summary.append(dict(universe=universe,machine_group=group,utterances=len(rr),
                raw_tokens=len(tokens),normalized_nonempty_tokens=sum(bool(t) for t in normtokens),
                candidate_tokens=any_tokens,candidate_utterances=any_utterances,
                normalized_token_share=ratio(any_tokens,sum(bool(t) for t in normtokens)),
                utterance_share=ratio(any_utterances,len(rr))))
            for token in CANDIDATES:
                candidate_rows.append(dict(universe=universe, machine_group=group, spelling=token,
                    tokens=c[token], utterances=sum(token in [norm(t) for t in (r['text'] or '').split()] for r in rr),
                    in_book=token in book, in_ntu=token in ntu))
    r3 = byid[R3]
    reference = js(HERE / 'reference_case.json')
    assert reference['source_id'] == R3
    assert hashlib.sha256(r3['text'].encode('utf-8')).hexdigest() == reference['asr_text_sha256_utf8']
    row = {'source_id': R3, 'existing_case_table_exact_match': True, 'part': r3['part'],
           'machine_lid': r3['lid'], 'asr_language_hint': r3['lang'],
           'word_tokens': len(r3['text'].split()),
           'candidate_spelling_counts': {t: [norm(x) for x in r3['text'].split()].count(t) for t in CANDIDATES},
           'actual_spoken_language': 'not_determined'}
    out = {'sources': sources, 'coverage': source_checks, 'core_index_ids_asr_times_hints_equal': True,
           'accepted_pool': {'n': len(pool), 'all_ids_have_asr': True,
                             'core':sum(uid in pwn for uid in pool), 'ext84':sum(uid not in pwn for uid in pool)},
           'matching': summary, 'fixed_candidate_spellings': CANDIDATES, 'candidate_overlap': candidate_rows,
           'candidate_spelling_summary':candidate_summary,
           'R3': row}
    write('asr_lexical_support.json', out)
    csv_write('asr_lexical_support.csv', summary)
    csv_write('asr_candidate_spellings.csv', candidate_rows)
    return out


def weighted_metrics(y, s, w):
    p, n = w[y == 1].sum(), w[y == 0].sum()
    if p == 0 or n == 0:
        return None, None
    order = np.argsort(s, kind='stable')
    ss, yy, ww = s[order], y[order], w[order]
    start = np.r_[0, 1 + np.flatnonzero(ss[1:] != ss[:-1])]
    wp, wn = np.add.reduceat(ww*(yy == 1), start), np.add.reduceat(ww*(yy == 0), start)
    auc = float((wp*(np.cumsum(wn)-wn+.5*wn)).sum()/(p*n))
    wp, wn = wp[::-1], wn[::-1]
    ap = float(((wp/p)*(np.cumsum(wp)/np.cumsum(wp+wn))).sum())
    return auc, ap


def pairs(y, s, w, days):
    ip, inn = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    weights = w[ip,None]*w[None,inn]
    win = (s[ip,None] > s[None,inn]).astype(float)+.5*(s[ip,None] == s[None,inn])
    same = days[ip,None] == days[None,inn]
    total = float(weights.sum())
    rows = []
    masks = [('within_day',same),('between_days',~same)]
    for name, mask in masks:
        mass = float(weights[mask].sum()); num = float((weights*win)[mask].sum())
        rows.append(dict(pair_group=name, pairs=int(mask.sum()), pair_weight=mass,
             share_of_all_pair_weight=ratio(mass,total), concordance_weight=num,
             conditional_AUC=ratio(num,mass), contribution_to_pooled_AUC=ratio(num,total),
             tie_pair_weight=float(weights[(win==.5)&mask].sum())))
    return rows, weights, win, ip, inn


def synthetic():
    y=np.array([1,1,0,0]);s=np.array([.8,.5,.5,.1]);w=np.array([2.,1.,1.,3.]);d=np.array(['a','b','a','b'])
    a,ap=weighted_metrics(y,s,w);assert abs(a-23/24)<1e-15 and abs(ap-11/12)<1e-15
    rows,*_=pairs(y,s,w,d);assert abs(sum(r['contribution_to_pooled_AUC'] for r in rows)-a)<1e-15
    assert weighted_metrics(np.array([0]),np.array([.5]),np.array([1.])) == (None,None)
    assert abs(weighted_metrics(y,np.ones(4),w)[0]-.5)<1e-15
    return ['hand-calculated weighted AUC 23/24 and AP 11/12 with exact tie',
            'within/between pair contributions sum to pooled AUC', 'single-class undefined', 'all tied AUC 0.5']


def saved_predictions():
    fields=('ids','y','w','days','clips','local_T','T_plus_P','T_plus_W')
    z=np_fields(P1/'cache.npz',fields)
    b=np_fields(BASE/'cache.npz',('ids','y','w','days','clips','T','local_T'))
    inventory=js(ROOT/'analysis/study3/prosody_preparation/INVENTORY.json')
    original_path=DATA_ROOT/inventory['saved_oof_source']
    registered=np_fields(original_path,('ids','y','w','days','T'))
    assert PINS[input_key(original_path)]==inventory['sources'][inventory['saved_oof_source']]
    for k in ('ids','y','w','days'):
        np.testing.assert_array_equal(z[k],registered[k])
    np.testing.assert_array_equal(b['T'],registered['T'])
    for k in ('ids','y','w','days','clips','local_T'):
        np.testing.assert_array_equal(z[k],b[k])
    manifest=js(P1/'MANIFEST.json')
    for name in ('cache.npz','RESULTS.json'):
        pin(P1/name);assert PINS[input_key(P1/name)]==manifest[name]
    saved=js(P1/'RESULTS.json');y,w,days=z['y'],z['w'],z['days'];dates=sorted(set(days))
    assert len(y)==192 and y.sum()==20 and len(dates)==12 and np.isfinite(w).all() and (w>0).all()
    overall,folds,pairgroups,paircells=[],[],[],[]
    for model in ('local_T','T_plus_P','T_plus_W'):
        scores=z[model];assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
        auc,ap=weighted_metrics(y,scores,w)
        for metric,value in [('AUC',auc),('AP',ap)]:
            reference=next(r['estimate'] for r in saved['models'] if r['model']==model and r['metric']==metric)
            assert abs(value-reference)<1e-14
        overall.append(dict(model=model,n=len(y),positives=int(y.sum()),days=len(dates),AUC=auc,AP=ap,
                            weighted_positive_share=float(w[y==1].sum()/w.sum())))
        pr,mass,win,ip,inn=pairs(y,scores,w,days)
        assert abs(sum(r['contribution_to_pooled_AUC'] for r in pr)-auc)<1e-14
        pairgroups += [dict(model=model,**r) for r in pr]
        for dp in dates:
            for dn in dates:
                mask=(days[ip,None]==dp)&(days[None,inn]==dn)
                weight=float(mass[mask].sum());num=float((mass*win)[mask].sum())
                paircells.append(dict(model=model,positive_day=dp,negative_day=dn,pairs=int(mask.sum()),
                    pair_weight=weight,conditional_AUC=ratio(num,weight),
                    pooled_AUC_contribution=ratio(num,float(mass.sum()))))
        for i,day in enumerate(dates):
            te=days==day;tr=~te;yy=y[te];ww=w[te];sc=scores[te]
            path=(BASE if model=='local_T' else P1/model)/f'fold_{i:02d}.npz'
            ff=np_fields(path,('train','test'))
            np.testing.assert_array_equal(ff['test'],np.flatnonzero(te));np.testing.assert_array_equal(ff['train'],np.flatnonzero(tr))
            if model!='local_T':
                logged=saved['folds'][model][i]
                assert logged['day']==day and logged['training_positive']==int(y[tr].sum())
                assert PINS[input_key(path)]==manifest[f'{model}/fold_{i:02d}.npz']
            a,aa=weighted_metrics(yy,sc,ww)
            neg_mean=float(np.average(sc[yy==0],weights=ww[yy==0])) if (yy==0).any() else None
            pos_mean=float(np.average(sc[yy==1],weights=ww[yy==1])) if (yy==1).any() else None
            folds.append(dict(model=model,held_out_day=day,n=int(te.sum()),positives=int(yy.sum()),
                negatives=int((yy==0).sum()),test_positive_share=float(yy.mean()),
                train_positive_share=float(y[tr].mean()),
                train_weighted_positive_share=float(w[tr&(y==1)].sum()/w[tr].sum()),
                test_weighted_positive_share=float(ww[yy==1].sum()/ww.sum()),
                score_mean=float(sc.mean()),score_weighted_mean=float(np.average(sc,weights=ww)),
                score_median=float(np.median(sc)),score_min=float(sc.min()),score_max=float(sc.max()),
                positive_weighted_score_mean=pos_mean,negative_weighted_score_mean=neg_mean,
                within_fold_AUC=a,within_fold_AP=aa,rank_estimable=a is not None))
    associations=[]
    for model in ('local_T','T_plus_P','T_plus_W'):
        rr=[r for r in folds if r['model']==model]
        for predictor in ('train_positive_share','train_weighted_positive_share','test_positive_share'):
            x=np.array([r[predictor] for r in rr]);v=np.array([r['score_mean'] for r in rr])
            corr=float(np.corrcoef(x,v)[0,1]) if x.std()>0 and v.std()>0 else None
            associations.append(dict(model=model,day_level_predictor=predictor,response='unweighted held-out score mean',
                                      days=len(rr),Pearson_r=corr,inference='none'))
    result={'support':{'n':192,'positives':20,'negatives':172,'days':12,'both_class_days':sum((y[days==d].min()!=y[days==d].max()) for d in dates)},
            'adopted_P1_point_estimates_reconciled':True,
            'registered_primary_IDs_y_weights_days_exact':True,
            'registered_vs_local_T_max_abs_score_difference':float(np.max(np.abs(registered['T']-z['local_T']))),
            'models':overall,'folds':folds,
            'pair_decomposition':pairgroups,'day_score_associations':associations,
            'original_reference_or_scores_modified':False,'new_fits_bootstrap_p_values_or_intervals':False}
    write('prediction_diagnostics.json',result);csv_write('prediction_folds.csv',folds)
    csv_write('prediction_pair_cells.csv',paircells)
    return result


def main():
    started=time.monotonic()
    checks=synthetic()
    results = {}
    if TASK in ("asr", "all"):
        results["asr"] = asr_coverage()
    if TASK in ("predictions", "all"):
        results["predictions"] = saved_predictions()
    changed=[]
    for name,rec in PINS.items():
        b=OPENED[name].read_bytes()
        if len(b)!=rec['bytes'] or hashlib.sha256(b).hexdigest()!=rec['sha256']:
            changed.append(name)
    assert not changed, changed
    write('execution_provenance.json',{'date':date.today().isoformat(),'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':TASK,
        'inputs':PINS,'protected_sources_unchanged':True,'synthetic_checks':checks,
        'seconds':time.monotonic()-started,'raw_answer_CSVs_read':False,
        'models_API_ranking_bootstrap_executed':False})
    print(json.dumps({'completed': list(results), 'output': str(OUT)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT,
                        help='Root containing the recorded scientific input paths.')
    parser.add_argument('--out', type=Path, help='New output directory; must not already exist.')
    parser.add_argument('--task', choices=['asr', 'predictions', 'all'], default='all')
    parser.add_argument('--self-test', action='store_true', help='Run only hand-calculated numerical fixtures.')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({'checks': synthetic(), 'real_data_read': False}))
        sys.exit(0)
    if args.out is None:
        parser.error('--out is required unless --self-test is selected')
    DATA_ROOT = args.input_root.resolve()
    RUNS = DATA_ROOT / 'data/study3_runs'
    P1 = DATA_ROOT / 'data/study3_stats/prosody_exploration/P1_augmented_local'
    BASE = DATA_ROOT / 'data/study3_stats/prosody_exploration/P1_baseline_local'
    prefixes = []
    if args.task in ('asr', 'all'):
        prefixes += ['data/paiwan_texts/', 'data/ntu_paiwan_asr/',
                     'data/study3_metadata/', 'data/study3_runs/',
                     'work/data/aligned_trees/', 'work84/data/aligned_trees/']
    if args.task in ('predictions', 'all'):
        prefixes += ['data/study3_stats/prosody_exploration/', 'data/study3_stats/registered/']
    require_inputs(tuple(prefixes))
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    TASK = args.task
    main()
