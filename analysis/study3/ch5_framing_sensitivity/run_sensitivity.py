"""Bounded post-result framing sensitivity. Immutable ULFR inputs; no feature inference.

Run from any directory with --root. Output directory must not exist. Numerical
payloads contain IDs and integer classification codes, never source utterance text.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

LANGS = ['ami', 'pwn', 'tao', 'tay', 'trv', 'tsu']
F_NAMES = ['bbox_h','bbox_area','face_extent','pose_extent','fallback_frac','ppf_mean','frac_1p','frac_3plus']
LABELS = ['Comforting','Celebrating','Explaining','Urging','Reporting']
SEED, B = 2026102801, 10000

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def dump(p, obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def normalize(x):
    x = np.asarray(x,dtype=np.float32)
    n = np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,n,out=np.zeros_like(x),where=n>0)

def metric(rank, label):
    """One full ordered candidate list. No-relevant AP is undefined."""
    if len(rank)==0: return np.array([np.nan]*5)
    rel = np.asarray(rank,dtype=bool); nr=int(rel.sum()); chance=nr/len(rel)
    ap=float((np.cumsum(rel)/np.arange(1,len(rel)+1))[rel].mean()) if nr else np.nan
    return np.array([float(rel[0]),chance,rel[:min(3,len(rel))].mean(),ap,float(rel[0])-chance])

def from_ranks(ranks, labels):
    return np.array([metric(labels[row[row>=0]]==labels[i],labels[i]) for i,row in enumerate(ranks)])

def transform(K,F,kind):
    if kind=='original': return K.copy(), {}
    if kind=='center':
        mu=K.mean(0);return K-mu, {'K_mean':mu,'beta':mu[None,:]}
    fill=np.zeros(F.shape[1]) if kind=='zero' else np.nanmedian(F,axis=0)
    ff=np.where(np.isnan(F),fill,F);mean=ff.mean(0);sd=ff.std(0)
    z=(ff-mean)/(sd+1e-9);zm=z.mean(0);zc=z-zm
    _,s,vt=np.linalg.svd(zc,full_matrices=False)
    pc=zc@vt[:3].T;design=np.column_stack([np.ones(len(K)),pc])
    beta,_,rank,_=np.linalg.lstsq(design,K,rcond=None);resid=K-design@beta
    tss=((K-K.mean(0))**2).sum(0);rss=(resid**2).sum(0)
    r2=np.divide(tss-rss,tss,out=np.zeros_like(tss),where=tss>0)
    assert np.max(np.abs(design.T@resid))<1e-8
    return resid,dict(fill=fill,F_mean=mean,F_sd=sd,z_mean=zm,vt=vt,singular_values=s,
                     pc=pc,beta=beta,design=design,design_rank=np.array(rank),
                     explained_variance=s*s/(s*s).sum(),R2=r2)

def ranked(E,active,groups,bank_orders):
    n=len(E);ranks=np.full((n,n),-1,np.int32);scores=np.full((n,n),np.nan,np.float32)
    mask=np.zeros((n,n),bool);norm=normalize(E)
    for l,q in groups.items():
        qs=q[active[q]];cand=bank_orders[l][active[bank_orders[l]]]
        if not len(qs) or not len(cand):continue
        for s in range(0,len(qs),512):
            qq=qs[s:s+512];S=norm[qq]@norm[cand].T
            for j,i in enumerate(qq):
                # Banks are other-source; self exclusion is nevertheless explicit.
                ii=np.flatnonzero(cand!=i);cc=cand[ii];ss=S[j,ii]
                order=np.argsort(-ss,kind='stable');ranks[i,:len(cc)]=cc[order]
                scores[i,cc]=ss;mask[i,cc]=True
    return ranks,scores,mask,norm

def estimate(v,W,select):
    valid=select & np.isfinite(v)
    den=W[:,valid].sum(1);num=W[:,valid]@v[valid]
    draws=np.divide(num,den,out=np.full(len(W),np.nan),where=den>0)
    vals=draws[np.isfinite(draws)]
    point=float(v[valid].mean()) if valid.any() else None
    return dict(point=point,ci95=np.percentile(vals,[2.5,97.5]).tolist() if len(vals) else None,
                n=int(valid.sum()),invalid_draws=int(len(W)-len(vals))),draws

def main(a):
    root=Path(a.root).resolve();out=Path(a.out).resolve();assert not out.exists(),out
    own=root/'analysis/study3/ch5_framing_sensitivity';assert (own/'PLAN.md').is_file()
    out.mkdir(parents=True)
    sys.path.insert(0,str(root/'analysis/study3/ulfr_local'))
    from ulfr_reader import Release,load_path_map
    reader=Release(path_map=load_path_map(a.path_map or root/'analysis/study3/ulfr_local/LOCAL_PATH_MAP.json'))
    inputs={}
    def pin(p,role,expected=None):
        p=Path(p);h=sha(p)
        if expected:assert h==expected,(p,h,expected)
        inputs[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h,'role':role}
        return p
    # Capture before any new score, transformation or interval computation.
    protected={}
    paths = [root/p for p in ['pipeline/analysis/ch5_retrieval.py','pipeline/analysis/ch5_sensitivity.py',
        'pipeline/analysis/s62_lift.py','pipeline/analysis/unit_table.py','pipeline/analysis/item5/feat_kin.py',
        'analysis/study3/ulfr_local/ulfr_reader.py','analysis/study3/ulfr_local/LOCAL_PATH_MAP.json']]
    for p in paths:
        if p.is_file() and (p.suffix.lower() in {'.tex','.bib','.cls','.sty','.pdf','.png','.jpg','.jpeg','.py','.json','.md'}):
            protected[str(p.relative_to(root))]=sha(p)
    dump(out/'protected_before.json',protected)
    w=reader.writer/'analysis/study3/formal_release'
    for name in ['RELEASE_MANIFEST.json','fr_rank_registered.py','fr_common.py']:
        pin(w/name,'formal scientific metadata/program')
    for name in ['label_analysis/units.tsv','unit_table/units_k.npz']:
        pin(reader.restore/name,'frozen input',reader.release_manifest['frozen_inputs'][name])
    for name,(_,h) in reader.release_manifest['original_programs'].items():
        pin(root/f'pipeline/analysis/{name}.py','original program',h)
    for name in ['PLAN.md','run_sensitivity.py']:pin(own/name,'new specification/program')
    primary_p=reader._numeric_file('registered/runA/json/ch5_retrieval.json','primary aggregate')
    original=json.loads(primary_p.read_text())['analyses']['kin14|kept|own']['X']
    for p in ['registered/runA/json/ch5_sensitivity.json','registered/runA/json/s62_results.json','registered/runA/json/s62_by_label.json']:
        reader._numeric_file(p,'existing aggregate context')
    conds=[f'ch5_retrieval|analysis|kin14|kept|own|X|{l}' for l in LANGS]
    blocks={l:reader.block(c) for l,c in zip(LANGS,conds)}
    rows=np.concatenate([b.query_rows for b in blocks.values()]);assert len(rows)==len(set(rows))==524
    loc={int(r):i for i,r in enumerate(rows)};u=reader.units
    labels=np.array([LABELS.index(v) for v in u['haiku'][rows]],np.int8)
    E=labels==LABELS.index('Explaining');lang=u['lang'][rows];day=u['day'][rows];clip=u['clip'][rows]
    groups={l:np.array([loc[int(v)] for v in b.query_rows]) for l,b in blocks.items()}
    banks={l:np.array([loc[int(v)] for v in b.bank_rows]) for l,b in blocks.items()}
    assert all(np.all(lang[banks[l]]!=l) for l in LANGS)
    k=np.load(reader.restore/'unit_table/units_k.npz',allow_pickle=False)
    kp={str(v):i for i,v in enumerate(k['ids'])};ki=np.array([kp[str(reader.ids[r])] for r in rows])
    K=k['K'][ki];F=k['F'][ki];assert K.shape==(524,14) and np.isfinite(K).all()
    assert k['within_shot'][ki].all() and not k['k_missing'][ki].any()
    assert np.all(u['kept'][rows]=='True')
    cc=np.isfinite(F).all(1);assert not np.isinf(F).any()
    np.savez_compressed(out/'population.npz',unit_ids=reader.ids[rows],global_rows=rows,label_codes=labels,
        lang=lang,day=day,clip=clip,K=K,F=F,complete_case=cc,k_indices=ki,
        n_records=k['n_records'][ki],n_tight=k['n_tight'][ki],within_shot=k['within_shot'][ki],
        k_missing=k['k_missing'][ki],shot_index=k['shot_index'][ki],
        **{f'bank_{l}':banks[l] for l in LANGS},**{f'queries_{l}':groups[l] for l in LANGS})
    # Read formal complete rankings and independently reconstruct per-query metrics.
    n=len(rows);rr=np.full((n,n),-1,np.int32);ss=np.full((n,n),np.nan,np.float32);mm=np.zeros((n,n),bool)
    saved_metric_checks=[]
    for l,c in zip(LANGS,conds):
        b=blocks[l];p=reader._numeric_file('rankings/'+reader.condition(c)['perquery_file'],'original per-query metrics')
        pq=np.load(p);assert np.array_equal(pq['query_rows'],b.query_rows)
        for j,global_q in enumerate(b.query_rows):
            qi=loc[int(global_q)];start,end=b.offsets[j:j+2];rank=np.array([loc[int(v)] for v in b.rank_rows[start:end]])
            assert set(rank)==set(banks[l])
            rr[qi,:len(rank)]=rank;ss[qi,rank]=b.rank_scores[start:end];mm[qi,rank]=True
            got=metric(labels[rank]==labels[qi],labels[qi])
            expected=np.array([pq['hit'][j],pq['chance'][j],pq['p3'][j],pq['ap'][j],pq['hit'][j]-pq['chance'][j]])
            assert np.allclose(got,expected,atol=0,rtol=0,equal_nan=True)
        saved_metric_checks.append({'source':l,'queries':len(b.query_rows),'entries':int(b.offsets[-1]),'exact':True})
    metrics={};cache={};trans={};eligibility={}
    def save_variant(name,r,s,m,active,values=None,norm=None):
        v=from_ranks(r,labels);cache[name]=(r,s,m);metrics[name]=v;eligibility[name]=active.copy()
        np.savez_compressed(out/f'{name}.npz',query_active=active,eligible_candidates=m,
            ranked_candidates=r,scores_by_candidate=s,metrics=v,
            **({} if values is None else {'transformed_K':values,'normalized_K':norm}))
    save_variant('release_original',rr,ss,mm,np.ones(n,bool))
    variants=[('local_original','original',False),('center_only','center',False),('pc3_zero','zero',False),
              ('pc3_median','median',False),('cc_original','original',True),('cc_center','center',True),('cc_pc3','zero',True)]
    for name,kind,complete in variants:
        active=cc.copy() if complete else np.ones(n,bool)
        values,params=transform(K[active],F[active],kind);x=np.zeros_like(K);x[active]=values
        r,s,m,norm=ranked(x,active,groups,banks);save_variant(name,r,s,m,active,x,norm)
        if params:np.savez_compressed(out/f'{name}_transform.npz',active_indices=np.flatnonzero(active),**params)
        trans[name]={key:(val.tolist() if hasattr(val,'tolist') else val) for key,val in params.items() if key in ['design_rank','explained_variance','R2']}
    assert all(np.array_equal(cache[v][2],mm) for v in ['local_original','center_only','pc3_zero','pc3_median'])
    bridge={'changed_top1_all':int((cache['local_original'][0][:,0]!=rr[:,0]).sum()),
        'changed_top1_Explaining':int((cache['local_original'][0][E,0]!=rr[E,0]).sum()),
        'changed_hit_Explaining':int((metrics['local_original'][E,0]!=metrics['release_original'][E,0]).sum()),
        'different_full_rank_positions':int((cache['local_original'][0]!=rr).sum()),
        'max_abs_cosine_difference':float(np.nanmax(np.abs(cache['local_original'][1]-ss)))}
    # One shared source-day stream on full original universe, reused for every variant.
    rng=np.random.default_rng(SEED);W=np.zeros((B,n),np.uint8);draw_arrays={};day_keys=[]
    for l in LANGS:
        days=sorted(set(day[lang==l]));idx=rng.integers(0,len(days),size=(B,len(days)))
        draw_arrays['indices_'+l]=idx;draw_arrays['days_'+l]=np.array(days)
        cnt=np.zeros((B,len(days)),np.uint8)
        for j in range(len(days)):cnt[:,j]=(idx==j).sum(1)
        for j,d in enumerate(days):W[:,(lang==l)&(day==d)]=cnt[:,j,None]
        day_keys.extend([(l,d) for d in days])
    np.savez_compressed(out/'bootstrap_weights.npz',query_weights=W,seed=SEED,**draw_arrays)
    results={};draws={};names=['p1','chance','p3','map','lift1']
    for v,M in metrics.items():
        results[v]={}
        for j,key in enumerate(names):
            results[v][key],draws[v+'__'+key]=estimate(M[:,j],W,E)
        active=np.isfinite(M[:,0]);g=active&E
        results[v]['support']={'universe':int(active.sum()),'queries':int(g.sum()),
            'source_days':len(set(zip(lang[g],day[g]))),'clips':len(set(clip[g])),
            'candidate_range':[int(cache[v][2][g].sum(1).min()),int(cache[v][2][g].sum(1).max())],
            'no_candidates':int((E&eligibility[v]&~active).sum()),
            'excluded_qualification_queries':int((E&~eligibility[v]).sum()),
            'ap_no_relevant':int((g&~np.isfinite(M[:,3])).sum())}
        results[v]['neighbor_changes_vs_release']=int((cache[v][0][g,0]!=rr[g,0]).sum())
    pairs=[('pc3_zero','release_original'),('local_original','release_original'),('center_only','local_original'),
           ('pc3_zero','local_original'),('pc3_zero','center_only'),('pc3_median','pc3_zero'),
           ('cc_original','local_original'),('cc_center','cc_original'),('cc_pc3','cc_original'),('cc_pc3','cc_center')]
    contrasts={}
    for a1,b1 in pairs:
        contrasts[a1+'--'+b1]={}
        for key in names:
            ds=draws[a1+'__'+key]-draws[b1+'__'+key];valid=ds[np.isfinite(ds)]
            contrasts[a1+'--'+b1][key]={'point':results[a1][key]['point']-results[b1][key]['point'],
                'ci95':np.percentile(valid,[2.5,97.5]).tolist(),'invalid_draws':B-len(valid)}
            draws[a1+'--'+b1+'__'+key]=ds
    # Original-neighbor proximity in the same pooled framing PC space.
    t=np.load(out/'pc3_zero_transform.npz');pc=t['pc'];D=((pc[:,None,:]-pc[None,:,:])**2).sum(2)
    near=np.array([D[i,rr[i,0]]-D[i,mm[i]].mean() for i in range(n)])
    same=np.array([D[i,rr[i,0]]-D[i,mm[i]&(labels==labels[i])].mean() if np.any(mm[i]&(labels==labels[i])) else np.nan for i in range(n)])
    prox,draws['framing_proximity_all_candidates']=estimate(near,W,E)
    prox_same,draws['framing_proximity_same_label']=estimate(same,W,E)
    np.savez_compressed(out/'framing_proximity.npz',top_minus_bank_pc_distance=near,top_minus_same_label_pc_distance=same)
    np.savez_compressed(out/'bootstrap_draws.npz',**draws)
    bysource=[]
    for l in LANGS:
        g=(lang==l)&E
        bysource.append({'source':l,'universe':int((lang==l).sum()),'queries':int(g.sum()),
            'days':len(set(day[g])),'clips':len(set(clip[g])), 'candidate_count':len(banks[l]),
            'missing_F_universe':int(((lang==l)&~cc).sum()),'missing_F_Explaining':int((g&~cc).sum())})
    checks=dict(saved_metric_checks=saved_metric_checks,paired_candidate_masks_equal=True,
        no_invalid_K=True,within_shot_all=True,tight_missing_zero=True,
        original_precision_agrees=all(round(float(metrics['release_original'][E,j].mean()),4)==original['descriptive']['Explaining'][key]
        for j,key in [(0,'hit'),(1,'chance'),(2,'p3'),(3,'map')]))
    assert checks['original_precision_agrees']
    for v in metrics:
        z=np.load(out/f'{v}.npz');re=from_ranks(z['ranked_candidates'],labels)
        assert np.array_equal(re,z['metrics'],equal_nan=True)
    for p,h in protected.items():assert sha(root/p)==h,p
    for path,info in reader.read_log.items():pin(root/path,info['role'],info['sha256'])
    dump(out/'results.json',dict(release=reader.map['release'],seed=SEED,resamples=B,
        original_saved=original['lift_at_1']['Explaining'],results=results,contrasts=contrasts,
        framing_proximity=prox,framing_proximity_same_label=prox_same,transformations=trans,
        numerical_bridge=bridge,coverage=bysource,checks=checks,
        missing_F=np.isnan(F).sum(0).tolist(),framing_names=F_NAMES,
        code_legend=LABELS,metric_columns=names,day_strata=len(day_keys)))
    import scipy
    dump(out/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        numpy=np.__version__,scipy=scipy.__version__,cpu=platform.processor(),logical_cpus=os.cpu_count(),
        threadpools=threadpool_info(),env={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']},
        git_head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip()))
    dump(out/'inputs.json',inputs)
    output_files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()}
    dump(out/'manifest.json',dict(schema='ch5-framing-sensitivity/1',plan_sha256=sha(own/'PLAN.md'),
        inputs=inputs,outputs=output_files,protected_unchanged=True,
        rank_schema='Rows follow population.unit_ids. ranked_candidates indexes that order, padded -1; scores_by_candidate and eligible_candidates are full unit-order matrices; no excluded score is a candidate.'))
    print(json.dumps({'out':str(out),'variants':len(metrics),'queries_Explaining':int(E.sum()),'verified':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--path-map');p.add_argument('--out',required=True)
    with threadpool_limits(limits=1):main(p.parse_args())
