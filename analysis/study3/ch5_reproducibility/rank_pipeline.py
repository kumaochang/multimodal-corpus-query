"""Two clean frozen-feature pipelines, preserving production formulas and isolated I/O."""
import os
for v in ['MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[v]='1'
import argparse,csv,hashlib,json,sys,time,resource
from pathlib import Path
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info
from common import *
sys.path.insert(0,str(CORE/'scripts'))
import ch5_metric_supplement as base
import ch5_exploratory_execute as ex
import ch5_exploratory_hierarchy as hierarchy

def inputs():
    ids,U,idx,reps,main,sens,hashes=base.load(INPUT)
    V,fam,p5,_,_,hh,_=ex.cov.load(INPUT,REF)
    assert hashes==hh
    U.update({k:V[k] for k in ['speech','gidx','kidx']})
    duration,dh=base.durations(ids,U,INPUT/'boundaries/textgrids')
    old=base.tasks(U,reps,main,sens,duration)
    new=ex.tasks(U,fam,p5)
    assert len(old)==175 and len(new)==210
    return ids,U,idx,reps,old,new,hashes

def old_descriptor(t,U):
    scope,rep,ver,subset,pool,mask,_=t
    units=np.flatnonzero(mask);q=np.concatenate([units[U['lang'][units]==l] for l in ex.SOURCES])
    banks={l:units[(U['lang'][units]!=l) if pool=='X' else (U['lang'][units]==l)] for l in ex.SOURCES if np.any(U['lang'][q]==l)}
    return '|'.join(t[:5]),rep,scope=='centered',pool,q,banks

def descriptors(old,new,U):
    return [('old',*old_descriptor(t,U)) for t in old]+[
        ('new',ex.key(t),t['representation'],False,t['pool'],ex.order(t,U),ex.banks_by_source(t,U)) for t in new]

def prepare():
    ids,U,idx,reps,old,new,hashes=inputs();rows=[]
    for suite,key,rep,center,pool,q,banks in descriptors(old,new,U):
        rows.append({'suite':suite,'condition':key,'representation':rep,'centered':center,'pool':pool,
                     'queries':len(q),'ordered_query_sha256':arrhash(q),
                     'candidate_sha256':{l:arrhash(c) for l,c in banks.items()},'chunk':512})
    dump(REPORT/'CONDITIONS.json',{'conditions':rows,'deduplication':'none; all 385 separately recomputed','inputs':hashes,
         'loaded_source_sha256':{str(m.__file__):sha(m.__file__) for m in [base,ex,hierarchy]},
         'original_helper_sha256':sha(base.cr.__file__)})
    print('FROZEN CONDITIONS',len(rows),flush=True)

def score(X,U,idx,q,banks,pool,full=False):
    keys=['hit','chance','p3','ap','n_cand','n_rel'];out={k:[] for k in keys}
    topidx=[];tops=[];fingerprints=[];maskprints=[];scoreprints=[];full_idx=[];full_scores=[];offsets=[0]
    for l,c in banks.items():
        qq=q[U['lang'][q]==l];cn=X[c]
        for start in range(0,len(qq),512):
            qs=qq[start:start+512];S=base.cr.sims(X[qs],cn)
            assert S.dtype==np.float32
            for row,i in enumerate(qs):
                keep=c!=i
                if pool=='W2':keep &= idx['clip'][c]!=idx['clip'][i]
                elif pool=='W3':keep &= idx['day'][c]!=idx['day'][i]
                ci=np.flatnonzero(keep);candidates=c[ci]
                order=ci[np.argsort(-S[row,ci],kind='stable')]
                ranked=c[order];scores=S[row,order];n=len(order)
                rel=idx['lab'][ranked]==idx['lab'][i];nr=int(rel.sum())
                out['hit'].append(float(rel[0]) if n else np.nan)
                out['chance'].append(nr/n if n else np.nan)
                out['p3'].append(float(rel[:min(3,n)].mean()) if n else np.nan)
                out['ap'].append(float((np.cumsum(rel)/np.arange(1,n+1))[rel].mean()) if nr else np.nan)
                out['n_cand'].append(n);out['n_rel'].append(nr)
                ti=np.full(3,-1,np.int64);ts=np.full(3,np.nan,np.float32)
                ti[:min(3,n)]=ranked[:3];ts[:min(3,n)]=scores[:3]
                topidx.append(ti);tops.append(ts);fingerprints.append(arrhash(ranked));scoreprints.append(arrhash(scores));maskprints.append(arrhash(candidates))
                if full:full_idx.append(ranked);full_scores.append(scores);offsets.append(offsets[-1]+n)
    r={k:np.array(v,np.int64 if k in ['n_cand','n_rel'] else float) for k,v in out.items()}
    r['lift1']=r['hit']-r['chance']
    proof={'top3_indices':np.array(topidx),'top3_scores':np.array(tops),'rank_sha256':np.array(fingerprints),
           'score_sha256':np.array(scoreprints),'post_exclusion_sha256':np.array(maskprints),'ap_defined':np.isfinite(r['ap'])}
    if full:proof.update(full_rank=np.concatenate(full_idx),full_scores=np.concatenate(full_scores),full_offsets=np.array(offsets))
    return r,proof

def independent(q,r,proof,U,idx,pool,banks):
    # Independent scalar/Python stable sort of retained scores; no production rank/metric call.
    for j,i in enumerate(q):
        expected=set(int(c) for c in banks[U['lang'][i]] if c!=i and
            (pool!='W3' or U['day'][c]!=U['day'][i]))
        a,b=proof['full_offsets'][j:j+2];ri=proof['full_rank'][a:b];ss=proof['full_scores'][a:b]
        assert set(map(int,ri))==expected
        # Stable tie convention reconstructed from original candidate order, independently.
        values=dict(zip(map(int,ri),map(float,ss)))
        ordered=sorted(expected,key=lambda c:(-values[c],c))
        assert ordered==ri.tolist()
        relevant=[idx['lab'][c]==idx['lab'][i] for c in ordered];nr=sum(relevant);n=len(ordered)
        assert r['n_cand'][j]==n and r['n_rel'][j]==nr
        assert r['chance'][j]==nr/n and r['hit'][j]==int(relevant[0])
        assert r['p3'][j]==sum(relevant[:min(3,n)])/min(3,n)
        if nr:
            seen=0;terms=[]
            for position,yes in enumerate(relevant,1):
                seen+=int(yes)
                if yes:terms.append(seen/position)
            assert abs(sum(terms)/nr-r['ap'][j])<=1e-12
        else:assert np.isnan(r['ap'][j])

def run(number):
    assert (REPORT/'CONDITIONS.json').exists()
    dest=DATA/f'run{number}';dest.mkdir(exist_ok=False)
    cache=dest/'execution';cache.mkdir();out=dest/'execution_reports';out.mkdir()
    ids,U,idx,reps,old,new,hashes=inputs()
    ex.REPORT=REF;ex.OUT=out;ex.CACHE=cache
    # Keep read-only source root for hashing/definition lookup; all writes are overridden.
    spec=read(REPORT/'CONDITIONS.json');assert spec['inputs']==hashes
    norm={};manifest={'cells':{}};old_manifest={'cells':{}};oldrows=[];weights={};indchecks=[]
    desc=descriptors(old,new,U);start=time.perf_counter()
    for ii,(suite,key,rep,center,pool,q,banks) in enumerate(desc):
        planned=spec['conditions'][ii]
        assert arrhash(q)==planned['ordered_query_sha256']
        assert {l:arrhash(c) for l,c in banks.items()}==planned['candidate_sha256']
        nk=(rep,center)
        if nk not in norm:
            X,ok,_=reps[rep]
            if center:
                Y=np.zeros_like(X,dtype=np.float64)
                for l in ex.SOURCES:
                    m=(U['lang']==l)&ok;Y[m]=X[m]-X[m].mean(0,keepdims=True)
                X=Y.astype(np.float32)
            norm[nk]=base.cr.l2rows(X)
        full=suite=='old' and key in ['main|xlmr_sub|kept|own|W3','main|kin14|kept|own|X','main|whisper|kept|own|W3']
        r,proof=score(norm[nk],U,idx,q,banks,pool,full)
        filename=hashlib.sha256(key.encode()).hexdigest()[:20]+'.npz'
        folder=dest/'old' if suite=='old' else cache;folder.mkdir(exist_ok=True)
        path=folder/filename
        np.savez_compressed(path,q=q,**r,**proof)
        if full:
            independent(q,r,proof,U,idx,pool,banks);indchecks.append(key)
            if rep!='whisper':np.savez_compressed(folder/(filename[:-4]+'_normalized.npz'),normalized=norm[nk],ids=ids)
        item={'file':filename,'sha256':sha(path),'query_sha256':arrhash(q),'candidate_sha256':{l:arrhash(c) for l,c in banks.items()},
              'rank_fingerprints_sha256':arrhash(proof['rank_sha256']),'score_fingerprints_sha256':arrhash(proof['score_sha256'])}
        if suite=='new':manifest['cells'][key]=item
        else:
            old_manifest['cells'][key]=item
            task=old[ii];universe=tuple(sorted(set(zip(U['lang'][q],U['day'][q]))))
            if universe not in weights:
                weights[universe]=base.cluster_weights(U['lang'][q],U['day'][q])[0]
                np.savez_compressed(folder/('weights_'+hashlib.sha256(str(universe).encode()).hexdigest()[:16]+'.npz'),weights=weights[universe],day_keys=np.array(universe))
            dp={c:j for j,c in enumerate(universe)};di=np.array([dp[c] for c in zip(U['lang'][q],U['day'][q])])
            oldrows.extend(base.aggregate(q,r,U,task,weights[universe],di))
        print('RANKED',number,ii+1,len(desc),key,flush=True)
    dump(dest/'old_manifest.json',old_manifest);dump(cache/'ranking_manifest.json',manifest)
    dump(dest/'old_metrics.json',{'seed':20261020,'rows':oldrows})
    # Adopted preflight day order and fixed overlap weights are metadata/spec inputs.
    pf=read(REF/'execution_20261020/preflight.json')
    dump(out/'preflight.json',{'day_keys':pf['day_keys'],'seed':20261024})
    import shutil
    shutil.copyfile(REF/'execution_20261020/adopted_overlap_weights.csv',out/'adopted_overlap_weights.csv')
    ex.statistics(INPUT)  # Existing statistics formulas, mapped to NEW outputs/cache.
    for name in ATTEMPTS:
        model,pool=name.split('_');d,f=hierarchy.build(model,pool,U,manifest)
        arrays={k:np.asarray(v) for k,v in d.items()}
        arrays.update(all_q=f['qi'],all_chance=f['chance'],all_hit=f['y'],all_design=f['X'],all_day=f['day'],all_query=f['query'],all_slopes=f['slopes'])
        np.savez_compressed(cache/(name+'_model_data.npz'),**arrays)
    dump(dest/'run_complete.json',{'run':number,'conditions':len(desc),'elapsed_seconds':time.perf_counter()-start,
        'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'independent_checks':indchecks,
        'threadpools':threadpool_info(),'runner_sha256':sha(__file__),'run_spec_sha256':sha(REPORT/'RUN_SPEC.md')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',type=int,choices=[1,2])
    a=p.parse_args();prepare() if a.prepare else run(a.run)
