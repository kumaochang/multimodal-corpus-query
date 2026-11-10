"""Adopted post-result exploration. Frozen numeric inputs only; no upstream models.

Run --stage preflight before --stage rank and --stage statistics. Paths are parameterized.
All individual arrays remain in ignored data; reports contain aggregates only.
"""
import os
for _v in ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse
from threadpoolctl import threadpool_info
import ch5_metric_supplement as base
import ch5_speech_text_coverage as cov

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / 'analysis/study3/ch5_exploratory'
OUT = REPORT / 'execution_20261020'
CACHE = REPO / 'data/study3_stats/ch5_exploratory/execution_20261020'
SOURCES = cov.SOURCES
POOLS = ['W1', 'W2', 'W3', 'X']
SIX = ['whisper', 'mfcc86', 'prosody', 'xlmr_sub', 'tfidf_sub', 'kin14']
FAMILY = {'acoustic': SIX[:3], 'subtitle': SIX[3:5],
          'visual_acoustic': ['kin14', 'whisper'], 'six_common': SIX,
          'transcription': ['xlmr_tr', 'tfidf_tr']}
METRICS = {'p1': 'hit', 'p3': 'p3', 'map': 'ap', 'chance_p1_p3': 'chance', 'lift1': 'lift1'}
SEED, B = 20261024, 10000
sha, save_json, save_csv = base.sha, base.save_json, base.save_csv


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def load(root):
    ids, U, idx, reps, main, sens, hashes = base.load(root)
    V, fam, p5, _, _, hh, _ = cov.load(root, REPORT)
    assert hashes == hh
    for k in ('lang', 'day', 'clip', 'kept', 'haiku'):
        assert np.array_equal(U[k], V[k])
    U.update({k: V[k] for k in ('speech', 'gidx', 'kidx')})
    return ids, U, idx, reps, fam, p5, hashes


def tasks(U, fam, p5):
    result = []
    binary = np.isin(U['speech'], ['M', 'I'])
    for name, rr in FAMILY.items():
        q = np.flatnonzero(fam[name] & binary)
        for bank in ('union_MI', 'M_only', 'I_only'):
            c = q if bank == 'union_MI' else q[U['speech'][q] == bank[0]]
            for pool in POOLS[:3] if name == 'transcription' else POOLS:
                for rep in rr:
                    result.append(dict(analysis='A', subset=name, bank=bank, pool=pool,
                                       representation=rep, q=q, c=c))
    p = np.flatnonzero(p5)
    assert len(p) == 2079 and cov.support(U, p) == dict(queries=2079, days=60, clips=154)
    for pool in POOLS[:3]:
        for rep in ['xlmr_sub', 'tfidf_sub', 'xlmr_tr', 'tfidf_tr']:
            result.append(dict(analysis='B', subset='P5', bank='P5', pool=pool,
                               representation=rep, q=p, c=p))
    c = np.flatnonzero(fam['six_common'] & U['kept'])
    assert len(c) == 384
    for pool in POOLS:
        for rep in SIX:
            result.append(dict(analysis='H0', subset='kept_common', bank='kept_common',
                               pool=pool, representation=rep, q=c, c=c))
    return result


def key(t):
    return '|'.join(t[k] for k in ('analysis', 'subset', 'bank', 'pool', 'representation'))


def order(t, U):
    return np.concatenate([t['q'][U['lang'][t['q']] == l] for l in SOURCES])


def banks_by_source(t, U):
    return {l: t['c'][(U['lang'][t['c']] != l) if t['pool'] == 'X' else
                       (U['lang'][t['c']] == l)] for l in SOURCES
            if np.any(U['lang'][t['q']] == l)}


def preflight(root):
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    _, U, _, _, fam, p5, hashes = load(root)
    tt = tasks(U, fam, p5)
    day_keys = sorted(set(zip(U['lang'], U['day'])))
    assert len(day_keys) == 72
    decisions = []
    with (REPORT / 'speech_text_overlap_weights.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            d = dict(row)
            keep = all(int(row[g + '_days']) >= 6 and int(row[g + '_queries']) >= 10 for g in ['M', 'I'])
            d['restricted_included'] = keep
            decisions.append(d)
    for name in FAMILY:
        for bank in ('union_MI', 'M_only', 'I_only'):
            for pool in POOLS[:3] if name == 'transcription' else POOLS:
                rows = [r for r in decisions if (r['set'], r['candidate_policy'], r['pool']) == (name, bank, pool)]
                total = sum(float(r['overlap_min_count']) for r in rows if r['restricted_included'])
                for r in rows:
                    r['restricted_weight'] = float(r['overlap_min_count']) / total if r['restricted_included'] and total else 0.
                    r['excluded_original_weight_mass'] = sum(float(x['overlap_weight']) for x in rows if not x['restricted_included'])
    save_csv(OUT / 'adopted_overlap_weights.csv', decisions)
    preserved = {str(p.relative_to(REPO)): sha(p) for p in REPORT.glob('*') if p.is_file()}
    scripts = [Path(__file__), REPO/'scripts/ch5_metric_supplement.py',
               REPO/'scripts/ch5_speech_text_coverage.py', REPO/'pipeline/analysis/ch5_retrieval.py']
    artifact = {'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'decision_sha256': sha(REPORT/'EXECUTION_20261020.md'),
                'inputs': hashes, 'preserved': preserved,
                'code_sha256': {str(p.relative_to(REPO)): sha(p) for p in scripts},
                'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                'platform': platform.platform(), 'threadpools': threadpool_info(),
                'environment': {k: os.environ[k] for k in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
                'seed': SEED, 'resamples': B, 'day_keys': day_keys,
                'dtype': 'float32 cosine; float64 accumulation/statistics',
                'layout': 'dense C-contiguous normalized full arrays; sparse original l2rows CSR',
                'matmul': 'NumPy MKL Q @ C.T; scipy sparse @ then toarray',
                'chunk': 512, 'tie': 'argsort(-score,kind=stable); no rounding',
                'tasks': [{'key': key(t), 'ordered_query_signature': cov.signature(order(t,U)),
                           'candidate_signatures': {l:cov.signature(c) for l,c in banks_by_source(t,U).items()}}
                          for t in tt]}
    path = OUT/'preflight.json'
    if path.exists():
        old = read_json(path)
        assert old['decision_sha256'] == artifact['decision_sha256']
        assert old['inputs'] == hashes and old['tasks'] == artifact['tasks']
        print('Preflight already fixed; not overwritten', flush=True)
    else:
        save_json(path, artifact)
        print('Saved pre-effect decision/input/code/set hashes:', len(tt), 'conditions', flush=True)


def rank(root):
    started = time.perf_counter()
    pf = read_json(OUT/'preflight.json')
    assert pf['decision_sha256'] == sha(REPORT/'EXECUTION_20261020.md')
    ids, U, idx, reps, fam, p5, hashes = load(root)
    assert hashes == pf['inputs']
    assert np.__version__ == '1.26.4' and scipy.__version__ == '1.15.2'
    for lib in threadpool_info():
        assert lib['num_threads'] == 1
    normalized = {n: base.cr.l2rows(x[0]) for n,x in reps.items()}
    olddir = CACHE.parent
    old = read_json(olddir/'cache_manifest.json')
    assert old['fingerprint']['inputs'] == hashes
    assert old['fingerprint']['numpy'] == np.__version__
    assert old['fingerprint']['scipy'] == scipy.__version__
    for fn in ['scripts/ch5_metric_supplement.py','pipeline/analysis/ch5_retrieval.py']:
        codehash=next(v for k,v in old['fingerprint']['code_and_plan'].items() if k.replace('\\','/')==fn)
        assert codehash == sha(REPO/fn)
    manifest_path = CACHE/'ranking_manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {'preflight_sha256':sha(OUT/'preflight.json'),'cells':{}}
    assert manifest['preflight_sha256'] == sha(OUT/'preflight.json')
    proofs = []
    for t in tasks(U, fam, p5):
        name = key(t)
        if name in manifest['cells']:
            assert sha(CACHE/manifest['cells'][name]['file']) == manifest['cells'][name]['sha256']
            continue
        q = order(t, U)
        reuse = t['analysis'] == 'H0' or (t['analysis'] == 'B' and t['representation'].endswith('_sub'))
        proof = None
        if reuse:
            sub = 'common' if t['analysis'] == 'H0' else 'own'
            oldkey = f"main|{t['representation']}|kept|{sub}|{t['pool']}"
            meta = old['cells'][oldkey]
            assert sha(olddir/meta['file']) == meta['sha256']
            z = np.load(olddir/meta['file'], allow_pickle=False)
            oldq = z['q']
            if t['analysis'] == 'H0':
                assert np.array_equal(q, oldq)
            else:
                expected = np.flatnonzero(reps[t['representation']][1] & U['kept'])
                assert np.array_equal(np.sort(oldq), expected)
                assert np.array_equal(t['c'], expected[np.isin(U['lang'][expected], cov.COVERED)])
            take = np.flatnonzero(np.isin(oldq, q))
            assert np.array_equal(oldq[take], q)
            # Same complete per-source Q and C; no new chunk boundary within a source.
            for l, c in banks_by_source(t,U).items():
                oc = oldq[(U['lang'][oldq] != l) if t['pool']=='X' else (U['lang'][oldq]==l)]
                assert np.array_equal(c, np.sort(oc))
                assert np.array_equal(q[U['lang'][q]==l], oldq[U['lang'][oldq]==l])
            r = {k:z[k][take] for k in ['hit','chance','p3','ap','n_cand','lift1']}
            proof = {'old_key':oldkey,'old_sha256':meta['sha256'],
                     'ordered_per_source_queries_candidates_identical':True,
                     'same_preprocess_code':True, 'local_MKL_version_only':True}
        else:
            rt = base.cr.Retriever(normalized[t['representation']], idx['lab'],idx['clip'],idx['day'],idx['lang'])
            parts = []
            for l,c in banks_by_source(t,U).items():
                qq = q[U['lang'][q]==l]
                z = rt.run(qq,c,t['pool'])
                parts.append({k:z[k] for k in ['hit','chance','p3','ap','n_cand']})
            r = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            r['lift1'] = r['hit'] - r['chance']
        # Independent metadata-only verification of candidate and relevance denominators.
        nn = np.zeros(len(q),np.int64); nr = nn.copy()
        for l,c in banks_by_source(t,U).items():
            for j in np.flatnonzero(U['lang'][q]==l):
                cc = base.allowed(q[j],c,t['pool'],idx)
                nn[j] = len(cc); nr[j] = (idx['lab'][cc]==idx['lab'][q[j]]).sum()
        valid = nn>0
        assert np.array_equal(np.isfinite(r['hit']),valid)
        np.testing.assert_array_equal(r['n_cand'][valid],nn[valid])
        np.testing.assert_array_equal(r['chance'][valid],nr[valid]/nn[valid])
        assert np.array_equal(np.isfinite(r['ap']), valid & (nr>0))
        r['n_cand'] = nn
        r['n_rel'] = nr
        dest = hashlib.sha256(name.encode()).hexdigest()[:20]+'.npz'
        np.savez_compressed(CACHE/dest,q=q,**r)
        manifest['cells'][name] = {'file':dest,'sha256':sha(CACHE/dest),
                                  'query_signature':cov.signature(q),
                                  'candidate_signatures':{l:cov.signature(c) for l,c in banks_by_source(t,U).items()},
                                  'source':'reused_local_cache' if reuse else 'new_shared_set_ranking',
                                  'reuse_proof':proof}
        save_json(manifest_path,manifest)
        print(len(manifest['cells']),name,'reused' if reuse else 'ranked',flush=True)
    manifest['elapsed_rank_stage_seconds'] = time.perf_counter()-started
    manifest['normalized_layouts'] = {n: {'dtype':str(x.dtype),'sparse':sparse.issparse(x),
        'C_contiguous': bool(x.flags.c_contiguous) if not sparse.issparse(x) else None,
        'format':x.format if sparse.issparse(x) else 'dense'} for n,x in normalized.items()}
    manifest['code_sha256_at_run'] = sha(__file__)
    save_json(manifest_path,manifest)
    save_json(OUT/'ranking_manifest.json',manifest)


def weights(U):
    keys = [tuple(x) for x in read_json(OUT/'preflight.json')['day_keys']]
    pos = {k:j for j,k in enumerate(keys)}
    di = np.array([pos[k] for k in zip(U['lang'],U['day'])])
    rng = np.random.Generator(np.random.PCG64(SEED))
    w = np.zeros((B,len(keys)),float)
    for l in SOURCES:
        cc = np.array([j for j,k in enumerate(keys) if k[0]==l])
        ix = rng.integers(0,len(cc),(B,len(cc)))
        np.add.at(w,(np.repeat(np.arange(B),len(cc)),cc[ix.ravel()]),1)
    assert np.all(w.sum(1)==72)
    np.savez_compressed(CACHE/'bootstrap_weights.npz',weights=w)
    return w,di


def ratio(v,mask,days,w):
    valid = mask & np.isfinite(v)
    s = np.bincount(days[valid],weights=v[valid],minlength=w.shape[1])
    c = np.bincount(days[valid],minlength=w.shape[1])
    den = w@c
    return (float(s.sum()/c.sum()) if c.sum() else np.nan,
            np.divide(w@s,den,out=np.full(len(w),np.nan),where=den>0),int(c.sum()))


def interval(est,boot,n):
    b = boot[np.isfinite(boot)]
    return clean({'estimate':est,'n':n,'ci95_pointwise':np.percentile(b,[2.5,97.5]).tolist() if len(b) else None,
                  'undefined_resamples':len(boot)-len(b)})


def fetch(manifest,analysis,subset,bank,pool,rep):
    k = '|'.join([analysis,subset,bank,pool,rep])
    c = manifest['cells'][k]
    assert sha(CACHE/c['file'])==c['sha256']
    z = np.load(CACHE/c['file'],allow_pickle=False)
    return {k:z[k] for k in z.files}


def statistics(root):
    started=time.perf_counter()
    loaded_source=Path(__file__).read_bytes()
    (CACHE/'statistics_loaded_source.py').write_bytes(loaded_source)
    save_json(OUT/'statistics_provenance.json',dict(recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        code_sha256=hashlib.sha256(loaded_source).hexdigest(),
        preflight_sha256=sha(OUT/'preflight.json'),
        ranking_manifest_sha256=sha(CACHE/'ranking_manifest.json'),
        seed=SEED,resamples=B,numeric_algorithm='unchanged adopted ratio-of-day-sums and joint max-standardized family'))
    _,U,_,_,fam,p5,_=load(root)
    manifest=read_json(CACHE/'ranking_manifest.json')
    w,di=weights(U)
    metrics=[]; paired=[]; coordinates=[]; draws=[]; weightrows=[]
    with (OUT/'adopted_overlap_weights.csv').open(encoding='utf-8') as f:
        weightrows=list(csv.DictReader(f))
    # All pooled query groups; source panels retain all-query descriptive means.
    for t in tasks(U,fam,p5):
        z=fetch(manifest,t['analysis'],t['subset'],t['bank'],t['pool'],t['representation'])
        q=z['q']; days=di[q]
        gg = [(g,'pooled','all',cov.group_mask(U['haiku'][q],g)) for g in cov.GROUPS]
        if t['analysis']=='A':
            gg += [(g,'pooled',s,cov.group_mask(U['haiku'][q],g)&(U['speech'][q]==s)) for s in ['M','I'] for g in cov.GROUPS]
        gg += [('all',l,'all',U['lang'][q]==l) for l in SOURCES if np.any(U['lang'][q]==l)]
        for group,lang,speech,mask in gg:
            valid=mask&np.isfinite(z['hit']); support=cov.support(U,q[valid])
            status='descriptive_per_source' if lang!='pooled' else ('descriptive_sparse_category' if group in cov.LABELS and (support['days']<15 or support['clips']<25) else 'pooled_exploratory')
            row={k:t[k] for k in ['analysis','subset','bank','pool','representation']}
            row.update(version='adopted_local_MKL_20261020',query_group=group,query_language=lang,
                       language_scope=','.join(sorted(set(U['lang'][q[mask]]))),speech_group=speech,
                       **support,selected_queries=int(mask.sum()),no_candidate=int((mask&(z['n_cand']==0)).sum()),
                       no_relevant=int((valid&(z['n_rel']==0)).sum()),
                       candidate_min=int(z['n_cand'][mask].min()) if mask.any() else None,
                       candidate_max=int(z['n_cand'][mask].max()) if mask.any() else None,status=status)
            for name,col in METRICS.items():
                est,boot,n=ratio(z[col],mask,days,w)
                row[name]=interval(est,boot,n)
            metrics.append(row)
        print('summarized',key(t),flush=True)
    # B metric-specific direct paired means and all fixed class/source descriptions.
    for pool in POOLS[:3]:
        for method in ['xlmr','tfidf']:
            a=fetch(manifest,'B','P5','P5',pool,method+'_tr')
            b=fetch(manifest,'B','P5','P5',pool,method+'_sub')
            q=a['q']; assert np.array_equal(q,b['q'])
            for k in ['chance','n_cand','n_rel']:
                np.testing.assert_array_equal(a[k],b[k])
            assert np.array_equal(np.isfinite(a['ap']),np.isfinite(b['ap']))
            np.testing.assert_allclose(a['lift1']-b['lift1'],a['hit']-b['hit'],atol=2e-16,rtol=0)
            gr=[(g,l,cov.group_mask(U['haiku'][q],g)&((U['lang'][q]==l) if l!='pooled' else True)) for l in ['pooled']+cov.COVERED for g in cov.GROUPS]
            lb={}
            for g,l,mask in gr:
                for metric,col in METRICS.items():
                    v=a[col]-b[col]
                    est,boot,n=ratio(v,mask,di[q],w)
                    support=cov.support(U,q[mask&np.isfinite(v)])
                    row=dict(analysis='B',method=method,pool=pool,contrast='ASR_minus_subtitle',
                             query_group=g,query_language=l,metric=metric,**support,**interval(est,boot,n),
                             status='descriptive_per_source' if l!='pooled' else 'pooled_exploratory')
                    if g in cov.LABELS and (support['days']<15 or support['clips']<25):
                        row['status']='descriptive_sparse_category'
                    paired.append(row)
                    if l=='pooled' and metric=='lift1': lb[g]=(est,boot,n,support)
            for g in ['all','Explaining_minus_Reporting']:
                if g=='all': est,boot,n,sp=lb['all']; eligible=True
                else:
                    aa,bb=lb['Explaining'],lb['Reporting']; est=aa[0]-bb[0]; boot=aa[1]-bb[1]; n=aa[2]+bb[2]
                    eligible=all(x[3]['days']>=15 and x[3]['clips']>=25 for x in [aa,bb])
                coordinates.append(dict(family='B',coordinate=f'B|{method}|{pool}|{g}',
                    eligible=eligible,**interval(est,boot,n)))
                draws.append(boot)
    # A restricted fixed-weight DIDs, including full-overlap/resource descriptions.
    for subset,ar,br in [('acoustic','whisper','prosody'),('subtitle','xlmr_sub','tfidf_sub'),('visual_acoustic','kin14','whisper')]:
        for bank in ['union_MI','M_only','I_only']:
            for pool in POOLS:
                a=fetch(manifest,'A',subset,bank,pool,ar); b=fetch(manifest,'A',subset,bank,pool,br)
                q=a['q']; np.testing.assert_array_equal(q,b['q'])
                for k in ['chance','n_cand','n_rel']: np.testing.assert_array_equal(a[k],b[k])
                wr=[r for r in weightrows if (r['set'],r['candidate_policy'],r['pool'])==(subset,bank,pool)]
                for target in ['restricted','full_positive']:
                    ww='restricted_weight' if target=='restricted' else 'overlap_weight'
                    cells=[r for r in wr if float(r[ww])>0]
                    for metric,col in METRICS.items():
                        estimate=0.; boot=np.zeros(B); n=0
                        for r in cells:
                            cell=(U['lang'][q]==r['source_community'])&(U['haiku'][q]==r['model_class'])
                            aa=ratio(a[col]-b[col],cell&(U['speech'][q]=='M'),di[q],w)
                            bb=ratio(a[col]-b[col],cell&(U['speech'][q]=='I'),di[q],w)
                            estimate+=float(r[ww])*(aa[0]-bb[0]); boot+=float(r[ww])*(aa[1]-bb[1]); n+=aa[2]+bb[2]
                        if not cells: estimate=np.nan; boot[:]=np.nan
                        row=dict(analysis='A',subset=subset,bank=bank,pool=pool,contrast=ar+'_minus_'+br,
                                 query_group='M_minus_I',metric=metric,target=target,cells=len(cells),
                                 excluded_original_mass=float(wr[0]['excluded_original_weight_mass']),
                                 status='adopted_exploratory' if target=='restricted' and bank=='union_MI' else 'descriptive_sensitivity',
                                 **interval(estimate,boot,n))
                        paired.append(row)
                        if target=='restricted' and bank=='union_MI' and metric=='lift1':
                            coordinates.append(dict(family='A',coordinate=f'A|{ar}-{br}|{pool}',eligible=bool(cells),**interval(estimate,boot,n)))
                            draws.append(boot)
    assert len(coordinates)==24
    D=np.column_stack(draws)
    joint=np.isfinite(D).all(1)
    stable=joint.mean()>=.99
    sd=np.std(D[joint],axis=0,ddof=1) if joint.sum()>1 else np.full(24,np.nan)
    center=np.array([np.nan if r['estimate'] is None else r['estimate'] for r in coordinates])
    passed=bool(stable and np.isfinite(center).all() and np.all(np.isfinite(sd)&(sd>0)) and all(r['eligible'] for r in coordinates))
    crit=float(np.quantile(np.max(np.abs((D[joint]-center)/sd),axis=1),.95)) if passed else None
    for j,r in enumerate(coordinates):
        r.update(bootstrap_sd=clean(sd[j]),simultaneous95_24=[float(center[j]-crit*sd[j]),float(center[j]+crit*sd[j])] if passed else None,
                 status='simultaneous_family_completed' if passed else 'family_not_estimable_as_specified')
    save_json(OUT/'metrics.json',clean(metrics)); save_json(OUT/'paired.json',clean(paired))
    save_json(OUT/'joint_family.json',clean(dict(seed=SEED,resamples=B,coordinates=coordinates,
        joint_valid=int(joint.sum()),joint_undefined=int((~joint).sum()),joint_valid_fraction=float(joint.mean()),
        minimum_fraction=.99,formal_family_completed=passed,critical_max_standardized_deviation=crit,
        centering='observed point estimate; bootstrap SD on complete jointly finite draws; ddof=1',
        conditional='fixed candidates and rankings; valid-resample conditioning disclosed',
        elapsed_statistics_seconds=time.perf_counter()-started)))
    np.savez_compressed(CACHE/'joint_family_draws.npz',draws=D,valid=joint)
    for file,rows in [('metrics.csv',metrics),('paired.csv',paired),('joint_family.csv',coordinates)]:
        columns=list(dict.fromkeys(k for r in rows for k in r))
        save_csv(OUT/file,[{k:json.dumps(r.get(k),ensure_ascii=False,allow_nan=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in columns} for r in clean(rows)])
    print('Saved',len(metrics),'metric rows,',len(paired),'paired rows; joint valid',joint.sum(),'formal',passed,flush=True)


def main():
    a=argparse.ArgumentParser(); a.add_argument('--input-root',type=Path,required=True)
    a.add_argument('--stage',choices=['preflight','rank','statistics'],required=True)
    args=a.parse_args(); globals()[args.stage](args.input_root)


if __name__=='__main__': main()
