"""Numerical H0/HA/HB adapter, prefit audit and prior predictive checks.

No extraction, classifier, encoder or text access. Outcome fitting requires CmdStanPy
and an existing CmdStan compiler. Missing runtime is a failed fit, never posterior.
"""
import os
for v in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'): os.environ[v]='1'
import argparse
import importlib.util
import itertools
import json
import platform
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import expit
import ch5_exploratory_execute as ex


def helmert(n):
    a=np.zeros((n,n-1))
    for j in range(1,n):
        a[:j,j-1]=1/np.sqrt(j*(j+1)); a[j,j-1]=-j/np.sqrt(j*(j+1))
    np.testing.assert_allclose(a.T@a,np.eye(n-1),atol=2e-15)
    np.testing.assert_allclose(a.sum(0),0,atol=2e-15)
    return a


def design(factors,terms):
    bases={k:helmert(n)[index] for k,(n,index) in factors.items()}
    blocks=[]; family=[]
    for f,term in enumerate(terms):
        x=np.ones((len(next(iter(bases.values()))),1))
        for name in term:
            x=(x[:,:,None]*bases[name][:,None,:]).reshape(len(x),-1)
        blocks.append(x); family.extend([f+1]*x.shape[1])
    return np.column_stack(blocks),np.array(family),bases


def build(model,pool,U,manifest):
    if model=='H0':
        route=[ex.fetch(manifest,'H0','kept_common','kept_common',pool,r) for r in ex.SIX]
    elif model=='HA':
        route=[ex.fetch(manifest,'A','six_common','union_MI',pool,r) for r in ex.SIX]
    else:
        route=[ex.fetch(manifest,'B','P5','P5',pool,r) for r in ['xlmr_sub','tfidf_sub','xlmr_tr','tfidf_tr']]
    q=route[0]['q']; nr=len(route)
    for z in route:
        for k in ['q','chance','n_cand','n_rel']:
            np.testing.assert_array_equal(z[k],route[0][k])
    # Query-major repeated measurements, frozen source/representation level order.
    qi=np.repeat(q,nr); rr=np.tile(np.arange(nr),len(q))
    y=np.column_stack([z['hit'] for z in route]).ravel()
    chance=np.repeat(route[0]['chance'],nr)
    factors={'L':(6 if model!='HB' else 5,np.array([ex.SOURCES.index(x) for x in U['lang'][qi]])),
             'K':(5,U['kidx'][qi])}
    if model!='HB':
        factors={'R':(6,rr),**factors}
        terms=['R','L','K','RL','RK','LK','RLK']
        if model=='HA':
            factors['G']=(2,U['gidx'][qi]); terms+=['G','LG','KG','RG','RKG']
    else:
        # Source levels subtitle=0, ASR=1; method XLM-R=0, TF-IDF=1.
        factors={'S':(2,rr//2),'M':(2,rr%2),**factors}
        terms=['S','M','L','K','SL','SK','LK','SM','SLK']
    X,family,bases=design(factors,terms)
    assert X.shape[1]==dict(H0=179,HA=214,HB=51)[model]
    slopes=bases['S' if model=='HB' else 'R']
    daykeys=sorted(set(zip(U['lang'][q],U['day'][q])))
    dp={k:j for j,k in enumerate(daykeys)}
    dd=np.repeat([dp[x] for x in zip(U['lang'][q],U['day'][q])],nr)
    qindex=np.repeat(np.arange(len(q)),nr)
    scored=np.isfinite(chance)&np.isfinite(y)
    structural=scored&((chance==0)|(chance==1))
    np.testing.assert_array_equal(y[structural],chance[structural])
    interior=scored&~structural
    data={'N':int(interior.sum()),'J':X.shape[1],'D':len(daykeys),'Q':len(q),'H':slopes.shape[1],
          'F':len(terms),'interaction_order':[len(t) for t in terms],
          'family':family,'X':X[interior],'slopes':slopes[interior],
          'chance_logit':np.log(chance[interior]/(1-chance[interior])),
          'day':dd[interior]+1,'query':qindex[interior]+1,'hit':y[interior].astype(int),
          'likelihood_weight':np.ones(interior.sum()),'prior_scale':1.}
    return data,dict(q=q,qi=qi,rr=rr,y=y,chance=chance,structural=structural,interior=interior,
                     X=X,slopes=slopes,day=dd,query=qindex,terms=terms,factors=factors)


def prior_predictive(data,full,seed,scale):
    rng=np.random.Generator(np.random.PCG64(seed))
    X=full['X']; unique,inv=np.unique(X,axis=0,return_inverse=True)
    n=256
    tau=np.abs(rng.normal(size=(n,data['F'])))*np.array([.75 if o==1 else .5 if o==2 else .25 for o in data['interaction_order']])
    for j,o in enumerate(data['interaction_order']):
        if o==1: tau[:,j]=.75
    beta=rng.normal(size=(n,data['J']))*tau[:,data['family']-1]*scale
    eta=(unique@beta.T)[inv].T+rng.normal(0,1.5,(n,1))
    sd=np.abs(rng.normal(size=(n,1))); sq=np.abs(rng.normal(size=(n,1)))
    eta+= (rng.normal(size=(n,data['D']))*sd)[:,full['day']]
    eta+= (rng.normal(size=(n,data['Q']))*sq)[:,full['query']]
    ss=np.abs(rng.normal(0,.5,(n,data['H'])))
    zs=rng.normal(size=(n,data['D'],data['H']))*ss[:,None,:]
    eta+=np.einsum('bnh,nh->bn',zs[:,full['day']],full['slopes'])
    c=full['chance']; interior=full['interior']; structural=full['structural']
    pi=np.full_like(eta,np.nan)
    pi[:,interior]=expit(eta[:,interior]+np.log(c[interior]/(1-c[interior])))
    pi[:,structural]=c[structural]
    yy=(rng.random(pi.shape)<pi).astype(float); yy[:,~np.isfinite(c)]=np.nan
    return dict(prior_scale=scale,seed=seed,draws=n,
                probability_quantiles=np.nanquantile(pi,[.01,.5,.99]).tolist(),
                probability_extreme_fraction=float(np.nanmean((pi<.01)|(pi>.99))),
                pooled_expected_lift_quantiles=np.nanquantile(np.nanmean(pi-c,axis=1),[.025,.5,.975]).tolist(),
                pooled_replicated_p1_quantiles=np.nanquantile(np.nanmean(yy,axis=1),[.025,.5,.975]).tolist(),
                status='prior_predictive_only_not_posterior',
                assessment='broad prior permits near-deterministic retrieval; requires outcome calibration and fixed prior-scale sensitivities, no prior selected by effects')


def prefit(root):
    start=time.perf_counter()
    _,U,_,_,_,_,_=ex.load(root)
    manifest=ex.read_json(ex.CACHE/'ranking_manifest.json')
    audits=[]; cells=[]
    for model in ['H0','HA','HB']:
        for pn,pool in enumerate(ex.POOLS[:3] if model=='HB' else ex.POOLS,1):
            data,f=build(model,pool,U,manifest)
            seed=dict(H0=202610250,HA=202610260,HB=202610270)[model]+pn
            # Rank on unique metadata rows gives exactly the same column rank, cheaply.
            rank_all=int(np.linalg.matrix_rank(np.unique(np.column_stack([np.ones(len(f['X'])),f['X']]),axis=0)))
            xi=f['X'][f['interior']]
            rank_like=int(np.linalg.matrix_rank(np.unique(np.column_stack([np.ones(len(xi)),xi]),axis=0)))
            priors=[prior_predictive(data,f,seed+1000,s) for s in [.5,1.,2.]]
            name=model+'_'+pool
            # Numeric cache, not a printed individual classification or ID listing.
            np.savez_compressed(ex.CACHE/(name+'_model_data.npz'),**{k:np.asarray(v) for k,v in data.items()},
                                all_q=f['qi'],all_chance=f['chance'],all_hit=f['y'],all_design=f['X'],
                                all_day=f['day'],all_query=f['query'],all_slopes=f['slopes'])
            group_keys=np.column_stack([v[1] for v in f['factors'].values()])
            for vals in np.unique(group_keys,axis=0):
                mask=(group_keys==vals).all(1); valid=mask&np.isfinite(f['y']); support=ex.cov.support(U,np.unique(f['qi'][valid]))
                row=dict(model=model,pool=pool,**{k:int(v) for k,v in zip(f['factors'],vals)},
                         **support,rows=int(valid.sum()),likelihood_rows=int((mask&f['interior']).sum()),
                         hit_mean=float(f['y'][valid].mean()) if valid.any() else None,
                         lift_mean=float((f['y']-f['chance'])[valid].mean()) if valid.any() else None,
                         complete_separation_observed=bool(valid.any() and len(np.unique(f['y'][valid]))==1),
                         status='observed_descriptive_O7_not_local_inference')
                cells.append(row)
            audits.append(dict(model=model,pool=pool,seed=seed,formula='+'.join(f['terms']),
                               contrasts=data['J'],saturated_coordinates=data['J']+1,
                               metadata_rank=rank_all,likelihood_rank=rank_like,
                               queries=len(f['q']),days=data['D'],clips=ex.cov.support(U,f['q'])['clips'],
                               repeated_rows=len(f['y']),interior_rows=data['N'],
                               structural_zero_rows=int((f['chance']==0).sum()),structural_one_rows=int((f['chance']==1).sum()),
                               no_candidate_rows=int((~np.isfinite(f['chance'])).sum()),
                               priors=priors,model_data_sha256=ex.sha(ex.CACHE/(name+'_model_data.npz')),
                               fit_status='not_fitted_toolchain_check_pending'))
            print('Prefit',name,'rank',rank_like,'/',data['J']+1,flush=True)
    ex.save_json(ex.OUT/'hierarchy_prefit.json',ex.clean(dict(models=audits,elapsed_seconds=time.perf_counter()-start,
        code_sha256=ex.sha(__file__),stan_sha256=ex.sha(Path(__file__).with_suffix('.stan')))))
    ex.save_json(ex.OUT/'hierarchy_observed_surface.json',ex.clean(cells))
    names=list(dict.fromkeys(k for r in cells for k in r))
    ex.save_csv(ex.OUT/'hierarchy_observed_surface.csv',[{k:r.get(k) for k in names} for r in cells])


def fit():
    """Execute/resume real fixed-cache fits in the dedicated statistics environment.

    Historical zero-chain status/freeze files are never overwritten. The runner
    validates complete chains and data hashes, or starts genuine four-chain fits.
    """
    import subprocess
    interpreter=ex.REPO/'data/ch5_stan/env/Scripts/python.exe'
    if not interpreter.exists():
        print('Missing isolated statistics environment; see posterior_20261020/RUN_SPEC.md',flush=True)
        return 2
    return subprocess.call([str(interpreter),str(ex.REPO/'scripts/ch5_posterior_resume.py')])


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--input-root',type=Path,required=True)
    a.add_argument('--stage',choices=['prefit','fit'],required=True); args=a.parse_args()
    if args.stage=='prefit': prefit(args.input_root)
    else: sys.exit(fit())
