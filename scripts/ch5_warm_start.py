"""Verified native-coordinate tuning reuse; fresh full posterior chains only.

No old partial draws are combined. Initialization/metric reuse changes a proposal,
never the likelihood, priors, population or retained-draw requirement.
"""
import argparse
import shutil
import ch5_posterior as p
import numpy as np


def parameter_shapes(d):
    return dict(alpha=(),z_beta=(d['J'],),tau=(d['F'],),z_day=(d['D'],),z_query=(d['Q'],),
        z_slope=(d['D'],d['H']),sigma_day=(),sigma_query=(),sigma_slope=(d['H'],),
        z_clip=(d['CL'],),sigma_clip=(),baseline_delta=(2,))


def parameter_columns(d):
    columns=[]
    for key,shape in parameter_shapes(d).items():
        if not shape:columns.append(key)
        elif len(shape)==1:columns.extend(key+'.'+str(i+1) for i in range(shape[0]))
        else:columns.extend(key+'.'+str(i+1)+'.'+str(j+1) for j in range(shape[1]) for i in range(shape[0]))
    return columns


def extract(path,d):
    """Accept only complete finite states and a fully written positive metric."""
    header=None;last=None;step=None;metric=None;next_metric=False;complete_rows=0
    for line in path.open():
        if line.startswith('#'):
            if 'Step size =' in line:step=float(line.split('=')[1])
            if next_metric:
                metric=np.array([float(x) for x in line.lstrip('# ').strip().split(',')]);next_metric=False
            if 'Diagonal elements of inverse mass matrix' in line:next_metric=True
            continue
        if not line.strip():continue
        values=line.strip().split(',')
        if header is None:header=values;continue
        if line.endswith('\n') and len(values)==len(header):
            last=values;complete_rows+=1
    if last is None or metric is None or step is None:return None
    last=np.array([float(x) for x in last]);assert np.isfinite(last).all()
    assert np.isfinite(metric).all() and (metric>0).all() and step>0
    assert len(metric)==len(parameter_columns(d))
    row=dict(zip(header,last));params={}
    for key,shape in parameter_shapes(d).items():
        if not shape:params[key]=row[key]
        elif len(shape)==1:params[key]=np.array([row[key+'.'+str(i+1)] for i in range(shape[0])])
        else:params[key]=np.array([[row[key+'.'+str(i+1)+'.'+str(j+1)] for j in range(shape[1])] for i in range(shape[0])])
    return dict(params=params,metric=metric,step=step,lp=float(row['lp__']),
        complete_rows=complete_rows,source=str(path),source_sha256=p.sha(path))


def project_state(state,base_data,new_data):
    """Proposal-only matching by original parameter name, never new data values."""
    old=state['params'];params={}
    for key,shape in parameter_shapes(new_data).items():
        if not shape:params[key]=old[key]
        else:
            value=np.zeros(shape);previous=np.asarray(old[key])
            common=tuple(slice(0,min(a,b)) for a,b in zip(shape,previous.shape))
            value[common]=previous[common];params[key]=value
    params['z_beta']=params['z_beta']/new_data['prior_scale']
    params['baseline_delta']=np.zeros(2)
    if new_data['CL']:params['sigma_clip']=.5
    byname=dict(zip(parameter_columns(base_data),state['metric']))
    metric=np.array([byname.get(k,1.) for k in parameter_columns(new_data)])
    assert len(metric)==sum(int(np.prod(s)) if s else 1 for s in parameter_shapes(new_data).values())
    assert np.isfinite(metric).all() and (metric>0).all()
    return params,metric


def variant_initialization(name,variant,d,folder):
    """Full ordinary adaptation is retained for all sensitivity fits."""
    record=folder/'initialization_provenance.json'
    if record.exists():
        saved=p.loadjson(record)
        assert all(p.sha(path)==h for path,h in saved['files_sha256'].items())
        return saved['options'],saved
    candidates=[]
    for f in (p.WORK/'fits'/name/'base').glob('attempt_*/sampling.json'):
        target=p.loadjson(f.parent/'target.json');diag=f.parent/'diagnostics.json'
        if target.get('equivalent_implementation','original') not in ['original','native_events','native_bernoulli']:continue
        if diag.exists():
            fit=p.loadjson(f)
            if fit['chains']==4 and fit['retained_per_chain']>0:
                candidates.append((p.loadjson(diag)['parameters_pass'],fit['attempt'],f))
    if not candidates:return {},None
    base_pass,_,source=max(candidates);base=p.loadjson(source);base_data=p.loadjson(source.parent/'data.json')
    assert p.loadjson(source.parent/'target.json')['model_data_sha256']==p.MODELS[name]['model_data_sha256']
    options=dict(metric='diag_e',inits=[],inv_metric=[],step_size=[]);files={};origins=[]
    for i,path in enumerate(base['csv_files'],1):
        assert p.sha(path)==base['csv_sha256'][path]
        state=extract(p.Path(path),base_data);assert state is not None
        params,metric=project_state(state,base_data,d)
        initial=folder/f'initialization_chain_{i}.json';mass=folder/f'initial_metric_chain_{i}.json'
        p.dump(initial,params);p.dump(mass,dict(inv_metric=metric));files[str(initial)]=p.sha(initial);files[str(mass)]=p.sha(mass)
        options['inits'].append(str(initial));options['inv_metric'].append(str(mass));options['step_size'].append(state['step'])
        origins.append(dict(chain=i,base_csv=path,base_csv_sha256=state['source_sha256']))
    saved=dict(options=options,files_sha256=files,model=name,variant=variant,
        base_fit_parameters_pass=base_pass,
        base_sampling_fingerprint=base['fingerprint'],sources=origins,
        rule='Proposal initialization from a complete, diagnosed native base, preferring a passing fit. A flagged complete fit may initialize a proposal, but supplies no retained draws or inferential qualification. Match parameter names; new clip coordinates and baseline start at zero; rescale beta raw coordinates to preserve initial linear effects. Full 2000/4000 adaptive warmup and all retained draws remain unchanged. This changes no prior, design, likelihood, candidate bank, target or reporting diagnostic threshold.')
    p.dump(record,saved);return options,saved


def prepare(name,source_attempt=2):
    p.os.environ['CH5_STAN_IMPLEMENTATION']='native_events'
    d,_,_,_=p.adapter(name);source=p.job_dir(name,'base',source_attempt)
    assert p.clean(d)==p.loadjson(source/'data.json')
    assert p.loadjson(source/'target.json')['model_data_sha256']==p.MODELS[name]['model_data_sha256']
    model=p.compiled()
    assert p.sha(model.exe_file)==p.loadjson(p.OUT/'nonzero_validation_native_events.json')['binary_sha256']
    states=[]
    for chain in range(1,5):
        paths=list(source.glob('*_'+str(chain)+'.csv'));assert len(paths)==1
        states.append(extract(paths[0],d))
    assert any(s is not None for s in states)
    donor=next(i for i,s in enumerate(states) if s is not None)
    out=p.WORK/'warm_start'/name;out.mkdir(parents=True,exist_ok=True);records=[]
    for i,state in enumerate(states):
        own=state is not None;s=state if own else states[donor]
        value=model.log_prob(params=p.clean(s['params']),data=str(source/'data.json'),jacobian=True,sig_figs=18)
        error=abs(float(value['lp__'].iloc[0])-s['lp']);assert error<2e-8,(name,i,error)
        p.dump(out/f'chain_{i+1}_init.json',s['params']);p.dump(out/f'chain_{i+1}_metric.json',dict(inv_metric=s['metric']))
        records.append(dict(original_chain=i+1,new_chain_id=101+i,own_tuning=own,
            donor_chain=i+1 if own else donor+1,complete_original_warmup=2000,
            partial_draws_reused_in_posterior=0,step_size=s['step'],state_logprob_error=error,
            original_csv=s['source'],original_csv_sha256=s['source_sha256'],
            init_sha256=p.sha(out/f'chain_{i+1}_init.json'),metric_sha256=p.sha(out/f'chain_{i+1}_metric.json')))
    record=dict(model=name,source_attempt=source_attempt,model_data_sha256=p.MODELS[name]['model_data_sha256'],
        native_binary_sha256=p.sha(model.exe_file),chains=records,
        method='Fresh 4-chain 2000 fixed-metric warmup + 2000 retained, original model seed, disjoint chain IDs 101:104. No partial draws are retained. A missing state uses the lowest-ID complete donor from the same model, never a best-effect selection.')
    p.dump(out/'validation.json',record);p.dump(p.OUT/'warm_start_validation'/f'{name}.json',record)
    print('Verified saved native tuning/state',name,'own states',sum(s is not None for s in states),flush=True)
    return record


def fit(name,attempt=4,repair=False):
    p.os.environ['CH5_STAN_IMPLEMENTATION']='native_events'
    source=p.WORK/'warm_start'/name;proof=p.loadjson(source/'validation.json')
    folder=p.job_dir(name,'base',attempt);folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'sampling.json').exists() and not list(folder.glob('*.csv'))
    d,_,_,_=p.adapter(name);model=p.compiled()
    assert proof['native_binary_sha256']==p.sha(model.exe_file)
    inits=[str(source/f'chain_{i}_init.json') for i in range(1,5)]
    metric=[str(source/f'chain_{i}_metric.json') for i in range(1,5)]
    assert all(p.sha(path)==r['init_sha256'] for path,r in zip(inits,proof['chains']))
    assert all(p.sha(path)==r['metric_sha256'] for path,r in zip(metric,proof['chains']))
    settings=dict(chains=4,parallel_chains=4,threads_per_chain=1,iter_warmup=4000 if repair else 2000,iter_sampling=4000 if repair else 2000,
        seed=p.MODELS[name]['seed'],chain_ids=[201,202,203,204] if repair else [101,102,103,104],adapt_engaged=repair,
        metric='diag_e',inv_metric=metric,step_size=[r['step_size'] for r in proof['chains']],max_treedepth=14 if repair else 12,sig_figs=16)
    if repair:settings['adapt_delta']=.99
    target=dict(model_data_sha256=p.MODELS[name]['model_data_sha256'],stan_sha256=p.sha(p.ROOT/p.SOURCE),
        equivalent_implementation='native_events',variant='base',settings=settings,
        proposal_tuning_sha256=p.sha(source/'validation.json'),adapter_sha256=p.sha(__file__))
    fingerprint=p.hashlib.sha256(p.json.dumps(target,sort_keys=True).encode()).hexdigest()
    p.dump(folder/'data.json',d);p.dump(folder/'target.json',dict(target,fingerprint=fingerprint))
    shutil.copyfile(__file__,folder/'loaded_runner.py')
    started=p.time.time();p.dump(folder/'running.json',dict(model=name,variant='base',pid=p.os.getpid(),
        started_at=p.time.strftime('%Y-%m-%dT%H:%M:%S%z'),settings=settings))
    try:
        with p.MemoryMonitor() as memory:
            result=model.sample(data=str(folder/'data.json'),output_dir=str(folder),inits=inits,
                show_progress=False,show_console=False,refresh=100,save_warmup=False,**settings)
        assert result.chains==4 and result.num_draws_sampling==settings['iter_sampling']
        p.dump(folder/'sampling.json',dict(status='sampled',model=name,variant='base',attempt=attempt,
            fingerprint=fingerprint,settings=settings,chains=4,retained_per_chain=settings['iter_sampling'],
            elapsed_seconds=p.time.time()-started,peak_RSS_process_tree_bytes=memory.peak,
            csv_files=result.runset.csv_files,csv_sha256={f:p.sha(f) for f in result.runset.csv_files},
            data_sha256=p.sha(folder/'data.json'),exe_sha256=p.sha(model.exe_file),
            proposal_tuning_validation=proof,old_partial_draws_included=0))
        return p.diagnose(name,'base',attempt)
    except Exception as e:
        p.dump(folder/'failed.json',dict(error=str(e),elapsed_seconds=p.time.time()-started));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit'])
    parser.add_argument('--model',choices=['HA_W1','HA_W2','HA_W3','HA_X'],required=True)
    parser.add_argument('--attempt',type=int,default=4);parser.add_argument('--repair',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.model)
    else:return 0 if fit(args.model,args.attempt,args.repair)['parameters_pass'] else 2


if __name__=='__main__':raise SystemExit(main())
