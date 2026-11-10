"""Real CmdStan continuation using ONLY the eleven hash-frozen model-data caches.

Use an isolated CmdStanPy environment. Never imports a retrieval ranker.
"""
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','STAN_NUM_THREADS']:os.environ[_k]='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'data/ch5_stan'
os.environ.setdefault('CMDSTAN', str(RUNTIME/'cmdstan-2.37.0'))
# Configure compiler executables in the invoking environment.
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from scipy.special import expit
from scipy.stats import norm
import psutil
import cmdstanpy
import arviz as az

REPORT=ROOT/'analysis/study3/ch5_exploratory'
OUT=REPORT/'posterior_20261020'
OLD=ROOT/'data/study3_stats/ch5_exploratory/execution_20261020'
WORK=ROOT/'data/study3_stats/ch5_exploratory/posterior_20261020'
PREFIT=json.loads((REPORT/'execution_20261020/hierarchy_prefit.json').read_text())
MODELS={x['model']+'_'+x['pool']:x for x in PREFIT['models']}
SOURCES=['ami','pwn','tao','tay','trv','tsu']
LABELS=['Comforting','Celebrating','Explaining','Urging','Reporting']
VARIANTS=['base','prior_half','prior_double','flexible_baseline','add_clip','no_day_slopes']
SOURCE='scripts/ch5_posterior.stan'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):x=x.item()
    return None if isinstance(x,float) and not np.isfinite(x) else x


def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def loadjson(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def summary_directory(name,variant,attempt):
    # Keep detailed sensitivity surfaces/PPC out of Git; publish compact
    # cross-variant comparisons and base surfaces in the analysis report.
    root=OUT if variant=='base' else WORK
    return root/'summaries'/f'{name}_{variant}_attempt_{attempt}'


def metadata():
    root=ROOT/'data/study3_runs'
    expected=loadjson(REPORT/'inventory.json')['inputs_sha256']
    for key in ['features_full/features.npz','label_analysis/units.tsv']:
        assert sha(root/key)==expected[key],'Frozen identification metadata hash mismatch: '+key
    ids=np.load(root/'features_full/features.npz',allow_pickle=False)['ids']
    with (root/'label_analysis/units.tsv').open(encoding='utf-8') as f:
        rows={r['unit_id']:{k:r[k] for k in ['lang','day','clip','haiku','lid_answer','mf_class']} for r in csv.DictReader(f,delimiter='\t')}
    return {k:np.array([rows[u][k] for u in ids]) for k in ['lang','day','clip','haiku','lid_answer','mf_class']}


def frozen(name):
    path=OLD/(name+'_model_data.npz')
    assert sha(path)==MODELS[name]['model_data_sha256'],name+' model-data hash mismatch'
    z=np.load(path,allow_pickle=False)
    return {k:z[k] for k in z.files}


def adapter(name,variant='base',U=None):
    z=frozen(name);U=metadata() if U is None else U
    n=int(z['N']);q=z['all_q'];chance=z['all_chance'];hit=z['all_hit']
    inside=np.isfinite(chance)&(chance>0)&(chance<1)
    structural=np.isfinite(chance)&~inside
    assert inside.sum()==n
    np.testing.assert_array_equal(hit[structural],chance[structural])
    np.testing.assert_array_equal(z['X'],z['all_design'][inside])
    np.testing.assert_array_equal(z['hit'],hit[inside])
    np.testing.assert_array_equal(z['query'],z['all_query'][inside]+1)
    np.testing.assert_array_equal(z['day'],z['all_day'][inside]+1)
    np.testing.assert_allclose(z['chance_logit'],np.log(chance[inside]/(1-chance[inside])),rtol=0,atol=0)
    cells,ci=np.unique(z['all_design'],axis=0,return_inverse=True)
    np.testing.assert_array_equal(cells[ci],z['all_design'])
    H=0 if variant=='no_day_slopes' else int(z['H'])
    slope=z['all_slopes'][:,:H]
    ds,si=np.unique(np.column_stack([z['all_day']+1,slope]),axis=0,return_inverse=True)
    np.testing.assert_array_equal(ds[si,1:],slope)
    np.testing.assert_array_equal(ds[si,0],z['all_day']+1)
    # Clip indices are nested in source/day, verified before adding a clip effect.
    clips=U['clip'][q]
    unique_clips,clip_index=np.unique(clips,return_inverse=True)
    for clip in unique_clips:
        assert len(set(zip(U['lang'][q][clips==clip],U['day'][q][clips==clip])))==1
    CL=len(unique_clips) if variant=='add_clip' else 0
    x=z['chance_logit'];xc=x-x.mean();s=max(float(xc.std()),1e-12)
    linear=xc/s;quad=linear**2;quad-=quad.mean();qs=max(float(quad.std()),1e-12);quad/=qs
    weight=np.ones(n)
    if variant=='day_holdout':weight[z['day']==1]=0
    if variant=='day_weighted':
        rng=np.random.Generator(np.random.PCG64(202610281))
        keys=[tuple(k) for k in loadjson(REPORT/'execution_20261020/preflight.json')['day_keys']]
        multiplicity=np.zeros(len(keys),int)
        for l in SOURCES:
            ix=np.array([i for i,k in enumerate(keys) if k[0]==l]);chosen=rng.choice(ix,size=len(ix),replace=True)
            np.add.at(multiplicity,chosen,1)
        day_weights=dict(zip(keys,multiplicity))
        weight=np.array([day_weights[k] for k in zip(U['lang'][q[inside]],U['day'][q[inside]])])
    d=dict(N=n,J=int(z['J']),C=len(cells),D=int(z['D']),Q=int(z['Q']),H=H,DS=len(ds),F=int(z['F']),CL=CL,
        interaction_order=z['interaction_order'],family=z['family'],X_cell=cells,cell=ci[inside]+1,
        slope_design=ds[:,1:],slope_day=ds[:,0].astype(int),slope_index=si[inside]+1,
        chance_logit=z['chance_logit'],baseline_basis=np.column_stack([linear,quad]),
        day=z['day'],query=z['query'],clip=clip_index[inside]+1 if CL else np.zeros(n,int),hit=z['hit'],
        likelihood_weight=weight,prior_scale=.5 if variant=='prior_half' else 2. if variant=='prior_double' else 1.,
        flexible_baseline=int(variant=='flexible_baseline'))
    if os.environ.get('CH5_STAN_IMPLEMENTATION') in ['native_events','whitened_events','whitened_vectorized']:
        events,ei=np.unique(np.column_stack([d['cell'],d['day'],d['slope_index'],d['clip'],
            d['chance_logit'],d['baseline_basis']]),axis=0,return_inverse=True)
        np.testing.assert_array_equal(events[ei,0],d['cell'])
        np.testing.assert_array_equal(events[ei,4],d['chance_logit'])
        d.update(NE=len(events),event=ei+1,event_cell=events[:,0].astype(int),event_day=events[:,1].astype(int),
            event_slope=events[:,2].astype(int),event_clip=events[:,3].astype(int),
            event_offset=events[:,4],event_basis=events[:,5:])
    info=dict(inside=inside,structural=structural,cell=ci,slope_index=si,clip_index=clip_index,
        offset_center=float(x.mean()),offset_sd=s,quad_center=float(np.mean(linear**2)),quad_sd=qs)
    if os.environ.get('CH5_STAN_IMPLEMENTATION') in ['whitened_events','whitened_vectorized']:
        from ch5_whiten import prepare
        d=prepare(d,z,info)
    return d,z,info,U


def preservation(create=False):
    path=OUT/'preservation_before.json'
    if create and not path.exists():
        files=list(REPORT.glob('*'))+list((REPORT/'execution_20261020').glob('*'))
        cache=OLD.parent
        files+=list(cache.glob('*.npz'))+[cache/'cache_manifest.json']+list(OLD.glob('*.npz'))
        inputs=loadjson(REPORT/'inventory.json')['inputs_sha256']
        files += [ROOT/'data/study3_runs'/f for f in inputs]
        files += [ROOT/'scripts/ch5_exploratory_hierarchy.py',ROOT/'scripts/ch5_exploratory_hierarchy.stan',ROOT/'scripts/ch5_metric_supplement.py']
        dump(path,{str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()})
    expected=loadjson(path)
    mismatch=[];authorized=[]
    for f,h in expected.items():
        if sha(ROOT/f)!=h:
            archive=WORK/'historical_ch5_exploratory_hierarchy.py'
            if f.replace('\\','/')=='scripts/ch5_exploratory_hierarchy.py' and archive.exists() and sha(archive)==h:
                authorized.append(dict(path=f,before_sha256=h,current_sha256=sha(ROOT/f),
                    archived_original_sha256=sha(archive),reason='author requested replacement of fit stub by genuine isolated-runner dispatcher'))
            elif f.replace('\\','/')=='analysis/study3/ch5_exploratory/PLAN_INDEX.md':
                original=WORK/'historical_PLAN_INDEX.md'
                if original.exists() and sha(original)==h and (ROOT/f).read_bytes().startswith(original.read_bytes()):
                    authorized.append(dict(path=f,before_sha256=h,current_sha256=sha(ROOT/f),
                        original_prefix_preserved=True,reason='append-only dated continuation index; historical content unchanged'))
                else:mismatch.append(f)
            else:mismatch.append(f)
    return dict(checked_files=len(expected),mismatches=mismatch,authorized_source_edits=authorized)


def audit():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    preservation(create=True)
    U=metadata();rows=[]
    for name in MODELS:
        d,z,info,_=adapter(name,U=U)
        # Nonzero fixed and random coefficients check original vs compressed design.
        rng=np.random.Generator(np.random.PCG64(MODELS[name]['seed']+50000))
        beta=rng.normal(size=d['J']);ud=rng.normal(size=d['D']);uq=rng.normal(size=d['Q'])
        bs=rng.normal(size=(d['D'],d['H']))
        eta0=z['chance_logit']+.2+z['X']@beta+ud[z['day']-1]+uq[z['query']-1]+np.einsum('nh,nh->n',z['slopes'],bs[z['day']-1])
        eta1=d['chance_logit']+.2+(d['X_cell']@beta)[d['cell']-1]+ud[d['day']-1]+uq[d['query']-1]+np.einsum('nh,nh->n',d['slope_design'],bs[d['slope_day']-1])[d['slope_index']-1]
        np.testing.assert_allclose(eta0,eta1,rtol=1e-14,atol=1e-13)
        ll0=np.sum(np.where(z['hit'], -np.logaddexp(0,-eta0),-np.logaddexp(0,eta0)))
        ll1=np.sum(d['likelihood_weight']*(d['hit']*eta1-np.logaddexp(0,eta1)))
        np.testing.assert_allclose(ll0,ll1,rtol=1e-14,atol=1e-10)
        rows.append(dict(model=name,N=d['N'],J=d['J'],cells=d['C'],D=d['D'],Q=d['Q'],
            input_sha256=sha(OLD/(name+'_model_data.npz')),max_eta_error=float(np.max(abs(eta0-eta1))),log_likelihood_error=float(ll0-ll1),
            structural_rows=int(info['structural'].sum()),indices_offsets_design_verified=True))
    dump(OUT/'adapter_audit.json',dict(models=rows,no_ranking=True,code_sha256=sha(__file__),stan_sha256=sha(ROOT/SOURCE)))
    # Preserve historical source bytes before turning its fit stub into a dispatcher.
    for f in ['scripts/ch5_exploratory_hierarchy.py','scripts/ch5_exploratory_hierarchy.stan']:
        source=ROOT/f;archive=WORK/('historical_'+source.name)
        if not archive.exists():shutil.copyfile(source,archive)
        expected=loadjson(OUT/'preservation_before.json')
        old_hash=next(h for key,h in expected.items() if key.replace('\\','/')==f)
        assert sha(archive)==old_hash,'Historical source archive must never be overwritten'
    freeze=dict(recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        source_sha256={f:sha(ROOT/f) for f in [SOURCE,'scripts/ch5_posterior.py','scripts/ch5_install_stan.py']},
        old_code_freeze_preserved=sha(REPORT/'execution_20261020/hierarchy_code_freeze.json'),
        model_data_sha256={k:v['model_data_sha256'] for k,v in MODELS.items()},
        run_spec_sha256=sha(OUT/'RUN_SPEC.md'))
    target=OUT/'continuation_freeze.json'
    dump(OUT/'continuation_reaudit.json' if target.exists() else target,freeze)
    print('Audited all eleven frozen model data; no ranks computed',flush=True)


def compiled():
    cmdstanpy.set_cmdstan_path(str(RUNTIME/'cmdstan-2.37.0'))
    implementation=os.environ.get('CH5_STAN_IMPLEMENTATION','original')
    white=implementation in ['whitened_events','whitened_vectorized']
    native=implementation in ['native_bernoulli','native_events'] or white
    p=WORK/('compiled_whitened_vectorized' if implementation=='whitened_vectorized' else 'compiled_whitened_events' if white else 'compiled_native_events' if implementation=='native_events' else 'compiled_native' if native else 'compiled');p.mkdir(parents=True,exist_ok=True)
    source=p/'ch5_posterior.stan'
    program=(ROOT/SOURCE).read_text(encoding='utf-8')
    if native and not white:
        original='target+=dot_product(likelihood_weight,y .* eta-log1p_exp(eta));'
        assert program.count(original)==1
        program=program.replace(original,'if (min(likelihood_weight)==1 && max(likelihood_weight)==1)\n    target+=bernoulli_logit_lpmf(hit | eta);\n  else '+original)
    if implementation=='native_events':
        needle='  int<lower=1> N;'
        extra='''  int<lower=1> NE;
  array[N] int<lower=1,upper=NE> event;
  array[NE] int<lower=1> event_cell;
  array[NE] int<lower=1> event_day;
  array[NE] int<lower=1> event_slope;
  array[NE] int<lower=0> event_clip;
  vector[NE] event_offset;
  matrix[NE,2] event_basis;
'''
        program=program.replace(needle,needle+'\n'+extra)
        old='''  vector[N] eta=chance_logit+rep_vector(alpha,N)+cell_eta[cell]
    +sigma_day*z_day[day]+sigma_query*z_query[query]+slope_eta[slope_index];
  if (CL>0) eta+=sigma_clip*z_clip[clip];
  if (flexible_baseline==1) eta+=baseline_basis*baseline_delta;'''
        new='''  vector[D] day_mu=sigma_day*z_day;
  vector[Q] query_mu=sigma_query*z_query;
  vector[NE] event_eta=event_offset+rep_vector(alpha,NE)+cell_eta[event_cell]
    +day_mu[event_day]+slope_eta[event_slope];
  vector[N] eta;
  if (CL>0) event_eta+=sigma_clip*z_clip[event_clip];
  if (flexible_baseline==1) event_eta+=event_basis*baseline_delta;
  eta=event_eta[event]+query_mu[query];'''
        assert program.count(old)==1
        program=program.replace(old,new)
    if white:program=(ROOT/'scripts/ch5_posterior_whitened.stan').read_text(encoding='utf-8')
    if implementation=='whitened_vectorized':
        extra='''  array[Q] int<lower=1,upper=Q> query_coordinate;
  vector[Q] householder_csr_values;
  array[Q] int<lower=1,upper=Q> householder_csr_columns;
  array[G+1] int<lower=1,upper=Q+1> householder_csr_rows;
'''
        program=program.replace('  vector[Q] householder_v;',extra+'  vector[Q] householder_v;')
        begin=program.index('  for (q in 1:Q) {')
        end=program.index('  for (j in 1:J)',begin)
        program=program[:begin]+'''  query_temp=append_row(query_mean,z_query_deviation)[query_coordinate];
  projection=csr_matrix_times_vector(G,Q,householder_csr_values,
    householder_csr_columns,householder_csr_rows,query_temp);
  z_query=query_temp-2*householder_v .* projection[query_group];
'''+program[end:]
    if not source.exists() or source.read_text(encoding='utf-8')!=program:source.write_text(program,encoding='utf-8')
    return cmdstanpy.CmdStanModel(stan_file=str(source),cpp_options={'O':3},stanc_options={'O1':True})


def validate_stan():
    model=compiled();d,_,_,_=adapter('H0_W1')
    params=dict(alpha=.1,z_beta=np.zeros(d['J']),tau=np.full(d['F'],.25),z_day=np.zeros(d['D']),
        z_query=np.zeros(d['Q']),z_slope=np.zeros((d['D'],d['H'])),sigma_day=.5,sigma_query=.5,
        sigma_slope=np.full(d['H'],.5),z_clip=np.empty(0),sigma_clip=.5,baseline_delta=np.zeros(2))
    # Log-probability validation includes every ancillary proper prior, no Jacobian.
    eta=d['chance_logit']+.1
    expected=float(np.sum(d['hit']*eta-np.logaddexp(0,eta)))+float(norm.logpdf(.1,0,1.5))
    expected+=sum(float(norm.logpdf(v).sum()) for v in [params['z_beta'],params['z_day'],params['z_query'],params['z_slope'],params['z_clip']])
    expected+=sum(float(norm.logpdf(t,0,.25 if o==3 else .5)) for t,o in zip(params['tau'],d['interaction_order']))
    expected+=float(norm.logpdf(.5,0,1))*3+float(norm.logpdf(np.full(d['H'],.5),0,.5).sum())
    expected+=float(norm.logpdf(0,0,.5))+float(norm.logpdf(0,0,.25))
    if os.environ.get('CH5_STAN_IMPLEMENTATION') in ['whitened_events','whitened_vectorized']:
        from ch5_whiten import encode
        params,_=encode(params,d)
    actual=model.log_prob(params=clean(params),data=clean(d),jacobian=False,sig_figs=18)
    got=float(actual['lp__'].iloc[0])
    # CmdStan computes propto=true, omitting additive data-independent constants.
    # Validate the likelihood by zeroing only likelihood weights in the same model.
    no_like=dict(d);no_like['likelihood_weight']=np.zeros(d['N'])
    prior=model.log_prob(params=clean(params),data=clean(no_like),jacobian=False,sig_figs=18)
    got_lik=got-float(prior['lp__'].iloc[0])
    expected_lik=float(np.sum(d['hit']*eta-np.logaddexp(0,eta)))
    np.testing.assert_allclose(got_lik,expected_lik,rtol=0,atol=2e-9)
    assert np.isfinite(actual.to_numpy()).all()
    native=p_implementation=os.environ.get('CH5_STAN_IMPLEMENTATION','original')
    native=p_implementation in ['native_bernoulli','native_events','whitened_events','whitened_vectorized']
    dump(OUT/('stan_validation_native.json' if native else 'stan_validation.json'),dict(compiled=True,stan_loglik=got_lik,python_loglik=expected_lik,
        absolute_error=abs(got_lik-expected_lik),finite_logprob_gradient=True,
        binary_sha256=sha(model.exe_file),source_sha256=sha(ROOT/SOURCE),generated_source_sha256=sha(model.stan_file),
        implementation=p_implementation,
        compiler_options={'O':3},stanc_options={'O1':True},not_a_posterior_fit=True))
    print('Stan adapter likelihood and gradient validated',flush=True)


class MemoryMonitor:
    def __init__(self):self.peak=0;self.stop=threading.Event()
    def __enter__(self):
        def watch():
            p=psutil.Process()
            while not self.stop.wait(.25):
                total=0
                for c in [p]+p.children(recursive=True):
                    try:total+=c.memory_info().rss
                    except psutil.Error:pass
                self.peak=max(self.peak,total)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()


def job_dir(name,variant,attempt=1):return WORK/'fits'/name/variant/f'attempt_{attempt}'


def sample(name,variant='base',attempt=1,numerical_restart=False):
    d,z,info,U=adapter(name,variant)
    repair=attempt>1 and not numerical_restart
    variant_index=VARIANTS.index(variant) if variant in VARIANTS else 6 if variant=='day_holdout' else 7 if variant=='day_weighted' else 8
    settings=dict(chains=4,parallel_chains=4,threads_per_chain=1,iter_warmup=4000 if repair else 2000,
        iter_sampling=4000 if repair else 2000,adapt_delta=.99 if repair else .95,max_treedepth=14 if repair else 12,
        seed=MODELS[name]['seed']+10000*variant_index,sig_figs=16,chain_ids=[1,2,3,4])
    p=job_dir(name,variant,attempt);p.mkdir(parents=True,exist_ok=True)
    implementation=os.environ.get('CH5_STAN_IMPLEMENTATION','original')
    tuning=None
    if variant!='base' and implementation=='native_events' and (not (p/'sampling.json').exists() or (p/'initialization_provenance.json').exists()):
        from ch5_warm_start import variant_initialization
        options,tuning=variant_initialization(name,variant,d,p)
        settings.update(options)
    target=dict(model_data_sha256=MODELS[name]['model_data_sha256'],stan_sha256=sha(ROOT/SOURCE),settings=settings,variant=variant)
    if tuning:target['initialization_provenance_sha256']=sha(p/'initialization_provenance.json')
    native=implementation in ['native_bernoulli','native_events','whitened_events','whitened_vectorized']
    if native:target['equivalent_implementation']=implementation
    if implementation in ['whitened_events','whitened_vectorized']:target['implementation_source_sha256']=sha(ROOT/'scripts/ch5_posterior_whitened.stan')
    if numerical_restart:target['numerical_restart_preserving_initial_iterations']=True
    fingerprint=hashlib.sha256(json.dumps(target,sort_keys=True).encode()).hexdigest()
    done=p/'sampling.json'
    if done.exists():
        old=loadjson(done)
        assert old['fingerprint']==fingerprint,'Refusing to resume changed fit target/settings'
        if old['status']=='sampled' and all(Path(f).exists() and sha(f)==h for f,h in old['csv_sha256'].items()):
            assert loadjson(p/'data.json')==clean(d),'Refusing to reuse changed converted model data'
            assert sha(WORK/('compiled_whitened_vectorized' if implementation=='whitened_vectorized' else 'compiled_whitened_events' if implementation=='whitened_events' else 'compiled_native_events' if implementation=='native_events' else 'compiled_native' if native else 'compiled')/'ch5_posterior.exe')==old['exe_sha256'],'Sampler binary changed'
            dump(p/'resume_verification.json',dict(converted_data_identical=True,Stan_source_identical=True,
                sampler_binary_identical=True,complete_chain_hashes_identical=True,
                current_adapter_sha256=sha(__file__),original_adapter_sha256=sha(p/'loaded_runner.py'),
                note='Runner bookkeeping/PATH may differ; exact converted data, Stan, binary and all chain settings must match.'))
            print('Reusing completed four-chain fit',name,variant,attempt,flush=True);return old
    if any(p.glob('*.csv')) and not done.exists():
        raise RuntimeError('Partial chains found; preserve and use a new attempt directory: '+str(p))
    dump(p/'data.json',d)
    dump(p/'target.json',dict(**target,fingerprint=fingerprint,adapter_sha256=sha(__file__)))
    shutil.copyfile(__file__,p/'loaded_runner.py')
    model=compiled();started=time.time()
    dump(p/'running.json',dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),model=name,variant=variant,settings=settings))
    try:
        with MemoryMonitor() as memory:
            fit=model.sample(data=str(p/'data.json'),output_dir=str(p),show_progress=False,show_console=False,
                             refresh=100,save_warmup=False,**settings)
        assert fit.chains==4 and fit.num_draws_sampling==settings['iter_sampling']
        record=dict(status='sampled',model=name,variant=variant,attempt=attempt,fingerprint=fingerprint,
            settings=settings,elapsed_seconds=time.time()-started,peak_RSS_process_tree_bytes=memory.peak,
            csv_files=fit.runset.csv_files,csv_sha256={f:sha(f) for f in fit.runset.csv_files},
            chains=fit.chains,retained_per_chain=fit.num_draws_sampling,
            data_sha256=sha(p/'data.json'),exe_sha256=sha(model.exe_file))
        dump(done,record);print('SAMPLED',name,variant,attempt,round(record['elapsed_seconds'],1),'s',flush=True)
        return record
    except Exception as err:
        dump(p/'failed.json',dict(status='failed',model=name,variant=variant,attempt=attempt,error=str(err),elapsed_seconds=time.time()-started))
        raise


def diagnose(name,variant='base',attempt=1):
    p=job_dir(name,variant,attempt);record=loadjson(p/'sampling.json')
    fit=cmdstanpy.from_csv(record['csv_files'])
    methods=fit.method_variables()
    # CmdStan summary uses rank-normalized split Rhat and bulk/tail ESS.
    table=fit.summary(percentiles=(2.5,50,97.5),sig_figs=12)
    table.to_csv(p/'parameter_diagnostics.csv')
    cols=list(table.columns)
    rcol=next(c for c in cols if c.lower().replace('_','')=='rhat')
    bulk=next(c for c in cols if c.lower().replace('_','')=='essbulk')
    tail=next(c for c in cols if c.lower().replace('_','')=='esstail')
    keep=table.index!='lp__';tt=table.loc[keep]
    bfmi=az.bfmi(methods['energy__'].T).tolist()
    result=dict(model=name,variant=variant,attempt=attempt,
        divergences=int(methods['divergent__'].sum()),
        treedepth_saturated=int((methods['treedepth__']>=record['settings']['max_treedepth']).sum()),
        max_Rhat=float(tt[rcol].max()),min_bulk_ESS=float(tt[bulk].min()),min_tail_ESS=float(tt[tail].min()),
        parameters_Rhat_over_1_01=int((tt[rcol]>1.01).sum()),parameters_bulk_ESS_under_400=int((tt[bulk]<400).sum()),
        parameters_tail_ESS_under_400=int((tt[tail]<400).sum()),E_BFMI_per_chain=bfmi,
        mean_acceptance_per_chain=methods['accept_stat__'].mean(0).tolist(),
        median_stepsize_per_chain=np.median(methods['stepsize__'],axis=0).tolist(),
        median_leapfrog_per_chain=np.median(methods['n_leapfrog__'],axis=0).tolist(),
        elapsed_seconds=record['elapsed_seconds'],peak_RSS_process_tree_bytes=record['peak_RSS_process_tree_bytes'])
    result['parameters_pass']=bool(result['divergences']==0 and result['max_Rhat']<=1.01 and result['min_bulk_ESS']>=400 and result['min_tail_ESS']>=400 and min(bfmi)>.3 and result['treedepth_saturated']==0)
    dump(p/'diagnostics.json',result)
    dump(OUT/'diagnostics'/f'{name}_{variant}_attempt_{attempt}.json',result)
    print('DIAGNOSTICS',name,variant,'pass',result['parameters_pass'],'Rhat',result['max_Rhat'],'ESS',result['min_bulk_ESS'],'div',result['divergences'],flush=True)
    return result


def verify_completed(name,variant,attempt):
    """Read-only exact verification; never starts sampling or compiles a binary."""
    folder=job_dir(name,variant,attempt);record=loadjson(folder/'sampling.json')
    target=loadjson(folder/'target.json');implementation=target.get('equivalent_implementation','original')
    previous=os.environ.get('CH5_STAN_IMPLEMENTATION')
    try:
        os.environ['CH5_STAN_IMPLEMENTATION']=implementation
        d,_,_,_=adapter(name,variant)
    finally:
        if previous is None:os.environ.pop('CH5_STAN_IMPLEMENTATION',None)
        else:os.environ['CH5_STAN_IMPLEMENTATION']=previous
    assert target['model_data_sha256']==MODELS[name]['model_data_sha256']
    assert target['stan_sha256']==sha(ROOT/SOURCE)
    if implementation in ['whitened_events','whitened_vectorized']:
        assert target['implementation_source_sha256']==sha(ROOT/'scripts/ch5_posterior_whitened.stan')
    compiled_folder={'original':'compiled','native_bernoulli':'compiled_native','native_events':'compiled_native_events','whitened_events':'compiled_whitened_events','whitened_vectorized':'compiled_whitened_vectorized'}[implementation]
    assert sha(WORK/compiled_folder/'ch5_posterior.exe')==record['exe_sha256']
    assert sha(folder/'data.json')==record['data_sha256']
    assert clean(d)==loadjson(folder/'data.json')
    assert record['settings']==target['settings'] and record['chains']==4
    assert sha(folder/'loaded_runner.py')==target['adapter_sha256']
    if 'initialization_provenance_sha256' in target:
        assert sha(folder/'initialization_provenance.json')==target['initialization_provenance_sha256']
        initialization=loadjson(folder/'initialization_provenance.json')
        assert all(sha(path)==h for path,h in initialization['files_sha256'].items())
    if 'proposal_tuning_sha256' in target:
        tuning=WORK/'warm_start'/name
        assert sha(tuning/'validation.json')==target['proposal_tuning_sha256']
        for i,row in enumerate(loadjson(tuning/'validation.json')['chains'],1):
            assert sha(tuning/f'chain_{i}_init.json')==row['init_sha256']
            assert sha(tuning/f'chain_{i}_metric.json')==row['metric_sha256']
            assert sha(row['original_csv'])==row['original_csv_sha256']
    assert all(sha(csv)==h for csv,h in record['csv_sha256'].items())
    result=dict(model=name,variant=variant,attempt=attempt,verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        frozen_data_converted_data_binary_original_loaded_source_settings_and_CSV_hashes_verified=True)
    dump(folder/'read_only_resume_verification.json',result)
    return result


def run_one(name,variant,attempt=1,numerical_restart=False):
    sample(name,variant,attempt,numerical_restart);return diagnose(name,variant,attempt)


def run_all(variants=None,jobs=4,names=None,attempt=1,numerical_restart=False):
    variants=['base'] if variants is None else variants;names=list(MODELS) if names is None else names
    # Worker processes isolate memory monitoring and all child sampler processes.
    commands=[(n,v) for n in names for v in variants]
    def worker(n,v):
        p=WORK/'worker_logs';p.mkdir(exist_ok=True)
        logfile=p/f'{n}_{v}_attempt_{attempt}.log'
        with logfile.open('w',encoding='utf-8') as log:
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            command=[sys.executable,__file__,'one','--model',n,'--variant',v,'--attempt',str(attempt)]
            if numerical_restart:command+=['--numerical-restart']
            ret=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        path=job_dir(n,v,attempt)/'diagnostics.json'
        good=path.exists() and loadjson(path).get('parameters_pass',False)
        return dict(model=n,variant=v,attempt=attempt,exit_code=ret.returncode if ret.returncode else 0 if good else 2,
                    parameters_pass=good,log=str(logfile))
    results=[]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        todo=[pool.submit(worker,n,v) for n,v in commands]
        for item in as_completed(todo):
            result=item.result();results.append(result);dump(OUT/'batch_status.json',results)
            print('BATCH',result,flush=True)
    return 0 if all(r['exit_code']==0 for r in results) else 2


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['audit','compile','benchmark','one','batch','verify'])
    a.add_argument('--model',choices=list(MODELS),default='H0_W1');a.add_argument('--variant',choices=VARIANTS+['day_holdout','day_weighted','native_benchmark'],default='base')
    a.add_argument('--attempt',type=int,default=1);a.add_argument('--variants',nargs='+',default=['base'])
    a.add_argument('--numerical-restart',action='store_true')
    a.add_argument('--jobs',type=int,default=4);a.add_argument('--models',nargs='+',default=list(MODELS));args=a.parse_args()
    if args.stage=='audit':audit()
    elif args.stage=='compile':validate_stan()
    elif args.stage=='benchmark':return 0 if run_one('H0_W1','base')['parameters_pass'] else 2
    elif args.stage=='one':return 0 if run_one(args.model,args.variant,args.attempt,args.numerical_restart)['parameters_pass'] else 2
    elif args.stage=='batch':return run_all(args.variants,args.jobs,args.models,args.attempt,args.numerical_restart)
    elif args.stage=='verify':
        r=preservation();dump(OUT/'preservation_after.json',r);assert not r['mismatches'];print(r)
    return 0


if __name__=='__main__':sys.exit(main())
