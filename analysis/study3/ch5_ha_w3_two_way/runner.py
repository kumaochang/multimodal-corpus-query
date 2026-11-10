"""Five adopted logical jobs, server execution only; local preflight is read-only.

No retrieval, API, consultant files or implicit environment installation.
"""
import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from design import HERE,JOBS,SEEDS,load,adapt,target_vectors,sha,write,clean,require,log_likelihood

ROOT=HERE.parents[2]
EXPECTED={'python':'3.12.0','numpy':'2.2.6','scipy':'1.15.3'}


def now():return datetime.now(timezone.utc).isoformat()


def verify_code():
    freeze=json.loads((HERE/'CODE_FREEZE.json').read_text(encoding='utf-8'))
    for name,x in freeze['files'].items():require(sha(HERE/name)==x['sha256'],'Code/PLAN freeze differs: '+name)
    return freeze


def destination(value):
    p=Path(value).resolve();allowed=[(HERE/'work').resolve(),(ROOT/'data/study3_stats/ch5_ha_w3_two_way').resolve()]
    require(any(p.is_relative_to(a) and p!=a for a in allowed),'Output must be a new subdirectory of the independent candidate work root')
    return p


def preflight(formal,frozen):
    z,u,overlap,inputs=load(formal,frozen);jobs={}
    for job in JOBS:
        data,info=adapt(z,u,job);T,keys=target_vectors(z,u,overlap)
        active=info['inside']&(info['day_weights'][z['all_day']]>0)
        jobs[job]={'N':data['N'],'J':data['J'],'F':data['F'],'Q':data['Q'],'D':data['D'],'H':data['H'],
                   'fixed_coordinates':95,'likelihood_rank':info['likelihood_rank'],'nullity':info['nullity'],
                   'positive_weight_likelihood_rank':info['positive_weight_likelihood_rank'],
                   'positive_weight_rows':int(np.sum(data['weight']>0)),'day_keys':info['day_keys'],
                   'training_queries':len(np.unique(z['all_query'][active])),
                   'training_days':len(np.unique(z['all_day'][active])),
                   'training_clips':len(set(u['clip'][active])),
                   'observed_fixed_cells':len(np.unique(info['X'],axis=0)),
                   'training_fixed_cells':len(np.unique(info['X'][active],axis=0)),
                   'day_weights':info['day_weights'],'heldout_days_0':info['heldout_days_0'],'seed':SEEDS[job],
                   'column_mapping':info['column_mapping'],'prior_scale':data['prior_scale']}
    old=json.loads(Path(inputs['base_contrasts']['path']).read_text(encoding='utf-8'))
    oldkeys={(r['prediction'],r['contrast'],r['query_group'],r['target']) for r in old if r['model']=='HA_W3'}
    newkeys={(p,*key) for p in ('conditional','population_average') for key in keys}
    require(oldkeys==newkeys and len(newkeys)==36,'Original 36 summary target keys differ')
    return dict(status='passed',scope='design_only_not_sampling',inputs=inputs,jobs=jobs,contrast_keys=sorted(newkeys),
                source_contract_sha256=sha(HERE/'SOURCE_CONTRACT.json'),candidate_bank_recomputed=False)


def environment(cmdstan,execution=False):
    result={'python':platform.python_version(),'platform':platform.platform(),'packages':{}}
    for key in ('numpy','scipy','cmdstanpy','arviz','psutil'):
        try:result['packages'][key]=importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:result['packages'][key]=None
    result['cmdstan']=str(Path(cmdstan).resolve()) if cmdstan else None
    result['thread_settings']={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
    if execution:
        require(platform.system()=='Linux','Real computation is assigned to Linux, not local Windows')
        require(result['python']==EXPECTED['python'] and all(result['packages'][k]==v for k,v in EXPECTED.items() if k!='python'),
                'Existing formal posterior environment differs; report instead of installing/upgrading')
        require(all(result['packages'][k] for k in ('cmdstanpy','arviz','psutil')),'Required existing dependencies absent')
        require(all(v=='1' for v in result['thread_settings'].values()),'Set process-local BLAS/OMP threads to 1')
        import cmdstanpy,psutil
        require(cmdstan and Path(cmdstan).is_dir(),'Explicit existing CmdStan directory required')
        cmdstanpy.set_cmdstan_path(str(Path(cmdstan).resolve()))
        require(cmdstanpy.cmdstan_version()==(2,37),'Expected existing CmdStan 2.37; do not substitute')
        result['cmdstan_version']=list(cmdstanpy.cmdstan_version())
        result['available_RAM_bytes']=psutil.virtual_memory().available;result['logical_CPUs']=os.cpu_count()
        result['total_RAM_bytes']=psutil.virtual_memory().total;result['physical_CPUs']=psutil.cpu_count(logical=False)
        cpuinfo=Path('/proc/cpuinfo')
        if cpuinfo.is_file():result['CPU_model']=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')),None)
        require(result['available_RAM_bytes']>=16*2**30 and result['logical_CPUs']>=4,
                f"Insufficient resources: available RAM {result['available_RAM_bytes']} bytes, logical CPUs {result['logical_CPUs']}; planned minimum 16 GiB / 4")
        for tool in ('g++','make'):
            output=subprocess.check_output([tool,'--version'],text=True,stderr=subprocess.STDOUT)
            result[tool]=output.splitlines()[0]
        result['compiler_executable_sha256']={tool:sha(Path(shutil.which(tool)).resolve()) for tool in ('g++','make')}
        from threadpoolctl import threadpool_info
        result['native_threadpools']=threadpool_info()
        for pool in result['native_threadpools']:
            if pool.get('filepath'):pool['binary_sha256']=sha(pool['filepath'])
    return result


class Memory:
    def __enter__(self):
        import psutil
        self.stop=threading.Event();self.peak=0
        def watch():
            p=psutil.Process()
            while not self.stop.wait(.25):
                total=0
                for process in [p]+p.children(recursive=True):
                    try:total+=process.memory_info().rss
                    except psutil.Error:pass
                self.peak=max(self.peak,total)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()


@contextlib.contextmanager
def lock(root):
    root.mkdir(parents=True,exist_ok=True);path=root/'execution.lock'
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise ValueError('Existing lock: investigate owner, do not remove automatically')
    try:os.write(fd,str(os.getpid()).encode());yield
    finally:os.close(fd);path.unlink()


def compiled(out):
    import cmdstanpy
    folder=out/'compiled';folder.mkdir(exist_ok=True);source=folder/'model.stan'
    if source.exists():require(sha(source)==sha(HERE/'model.stan'),'Preserve changed compiled source')
    else:shutil.copyfile(HERE/'model.stan',source)
    return cmdstanpy.CmdStanModel(stan_file=str(source),cpp_options={'O':3},stanc_options={'O1':True})


def validate_stan(model,data):
    # Nonzero coefficients, source/day/query/slopes and unequal weights make
    # wrong dimensions, zero-based indices or omitted random effects detectable.
    par=dict(alpha=.17,z_beta=np.linspace(-.2,.3,data['J']),tau=np.linspace(.21,.37,6),
        z_query=np.linspace(-.4,.2,data['Q']),z_day=np.linspace(.3,-.2,data['D']),
        z_slope=np.arange(data['D']*5).reshape(data['D'],5)/max(1,data['D']*5)-.3,
        sigma_query=.4,sigma_day=.3,sigma_slope=np.linspace(.1,.3,5))
    cases=[]
    for scale in (.5,1.,2.):
        d=dict(data,prior_scale=scale);zero=dict(d,weight=np.zeros(data['N']))
        a=model.log_prob(params=clean(par),data=clean(d),jacobian=False,sig_figs=18)
        b=model.log_prob(params=clean(par),data=clean(zero),jacobian=False,sig_figs=18)
        actual=float(a['lp__'].iloc[0]-b['lp__'].iloc[0]);expected=log_likelihood(d,par)
        require(abs(actual-expected)<1e-7 and np.isfinite(a.to_numpy()).all(),'Compiled adapter likelihood/gradient mismatch')
        cases.append(dict(scale=scale,Stan_log_likelihood=actual,Python_log_likelihood=expected))
    return dict(status='passed',nonzero_parameters=True,cases=cases,finite_log_density_gradient=True,
        stan_source_sha256=sha(HERE/'model.stan'),binary_sha256=sha(model.exe_file),
        compiler_options={'O':3},stanc_options={'O1':True},actual_posterior_sampling=False)


def diagnose(fit,folder,settings):
    import arviz as az
    table=fit.summary(percentiles=(2.5,50,97.5),sig_figs=12);table.to_csv(folder/'parameter_diagnostics.csv')
    names={x.lower().replace('_',''):x for x in table.columns};t=table.loc[table.index!='lp__']
    r=t[names['rhat']];b=t[names['essbulk']];tail=t[names['esstail']];m=fit.method_variables()
    bfmi=az.bfmi(m['energy__'].T).tolist()
    result=dict(max_Rhat=float(r.max()),min_bulk_ESS=float(b.min()),min_tail_ESS=float(tail.min()),
        all_parameter_diagnostics_finite=bool(np.isfinite(np.column_stack([r,b,tail])).all()),
        divergences=int(m['divergent__'].sum()),treedepth_saturated=int((m['treedepth__']>=settings['max_treedepth']).sum()),
        E_BFMI_per_chain=bfmi,acceptance_mean_per_chain=m['accept_stat__'].mean(0),
        stepsize_median_per_chain=np.median(m['stepsize__'],axis=0),leapfrog_median_per_chain=np.median(m['n_leapfrog__'],axis=0))
    result['pass']=bool(result['all_parameter_diagnostics_finite'] and r.max()<=1.01 and b.min()>=400 and tail.min()>=400
        and result['divergences']==0 and result['treedepth_saturated']==0 and np.isfinite(bfmi).all() and min(bfmi)>.3)
    write(folder/'diagnostics.json',result);return result


def execute(args):
    require(args.execute_server,'Real execution disabled; preflight does not authorize sampling')
    freeze=verify_code();env=environment(args.cmdstan,True)
    root=destination(args.out_root)
    require(not root.is_relative_to(Path(args.formal_root).resolve()) and not root.is_relative_to(Path(args.frozen_root).resolve()),'Output overlaps protected inputs')
    require(args.job in JOBS,'One of five adopted jobs required')
    require(args.quadrature==40,'Initial sampling reports use fixed 40/80 check; refine only in a separate reporting attempt')
    require(args.attempt>=1 and args.warmup>=2000 and args.draws>=2000,'Do not reduce initial chains/iterations')
    require((args.attempt==1 and not args.reason and args.warmup==2000 and args.draws==2000 and args.adapt_delta==.95 and args.treedepth==12)
            or (args.attempt>1 and bool(args.reason) and .95<=args.adapt_delta<1 and args.treedepth>=12),'Repair needs new attempt and technical reason; initial settings fixed')
    with lock(root):
        code=root/'CODE_FREEZE.json'
        if code.exists():require(sha(code)==sha(HERE/'CODE_FREEZE.json'),'Candidate run belongs to another code freeze')
        else:shutil.copyfile(HERE/'CODE_FREEZE.json',code)
        stable={k:v for k,v in env.items() if k not in ('available_RAM_bytes',)}
        env_path=root/'environment_contract.json'
        if env_path.exists():require(json.loads(env_path.read_text(encoding='utf-8'))==clean(stable),'Toolchain/environment changed between jobs; investigate')
        else:write(env_path,stable)
        pre=preflight(args.formal_root,args.frozen_root)
        frozen=root/'preflight.json'
        if frozen.exists():require(json.loads(frozen.read_text(encoding='utf-8'))==clean(pre),'Frozen inputs/design changed')
        else:write(frozen,pre)
        # Require completed base checkpoint before any later logical job.
        for previous in JOBS[:JOBS.index(args.job)]:
            records=list((root/previous).glob('attempt_*/completed.json'))+list((root/previous).glob('attempt_*/report_*/completed.json'))
            require(any(json.loads(p.read_text(encoding='utf-8'))['pass'] for p in records),'Earlier logical job not accepted: '+previous)
        folder=root/args.job/f'attempt_{args.attempt}'
        require(not folder.exists(),'Existing attempt preserved; verify with audit, use a justified new attempt for repair')
        folder.mkdir(parents=True)
        write(folder/'authorization.json',dict(basis='Fixed two-way model specification, 2026-10-30',
            jobs=JOBS,consultant_results_required=False,code_freeze_sha256=sha(HERE/'CODE_FREEZE.json')))
        write(folder/'environment.json',env);write(folder/'reason.json',dict(attempt=args.attempt,technical_reason=args.reason))
        z,u,overlap,inputs=load(args.formal_root,args.frozen_root);data,info=adapt(z,u,args.job)
        write(folder/'data.json',data);write(folder/'design.json',dict(info,inputs=inputs))
        phase_start=time.monotonic()
        try:
            with Memory() as preparation_memory:
                model=compiled(root);write(folder/'compiled_validation.json',validate_stan(model,data))
                if args.job=='base':
                    from report import prior_predictive
                    if (root/'prior_predictive.json').exists():shutil.copyfile(root/'prior_predictive.json',folder/'prior_predictive.json')
                    else:
                        prior_predictive(data,z,u,info,folder/'prior_predictive.json')
                        shutil.copyfile(folder/'prior_predictive.json',root/'prior_predictive.json')
            write(folder/'pre_sampling_resources.json',dict(seconds=time.monotonic()-phase_start,peak_RSS_process_tree_bytes=preparation_memory.peak,
                phases='compile, log-density validation, prior predictive or reuse'))
        except Exception as error:
            write(folder/'failed.json',dict(status='compile_validation_or_prior_failed',error_type=type(error).__name__,message=str(error),UTC=now()))
            raise
        settings=dict(chains=4,parallel_chains=4,threads_per_chain=1,chain_ids=[1,2,3,4],iter_warmup=args.warmup,
            iter_sampling=args.draws,adapt_delta=args.adapt_delta,max_treedepth=args.treedepth,seed=SEEDS[args.job],sig_figs=18)
        write(folder/'settings.json',settings);started=time.monotonic()
        write(folder/'started.json',dict(UTC=now(),job=args.job,pid=os.getpid(),code_freeze_sha256=sha(HERE/'CODE_FREEZE.json')))
        try:
            with Memory() as memory:
                (folder/'chains').mkdir()
                fit=model.sample(data=str(folder/'data.json'),output_dir=str(folder/'chains'),show_progress=False,
                    show_console=False,save_warmup=False,refresh=100,**settings)
            sampling_seconds=time.monotonic()-started
            require(fit.chains==4 and fit.num_draws_sampling==args.draws,'Incomplete four-chain result')
            write(folder/'sampling.json',dict(chains=4,retained_per_chain=args.draws,warmup_per_chain=args.warmup,
                seconds=sampling_seconds,peak_RSS_process_tree_bytes=memory.peak,exe_sha256=sha(model.exe_file),
                data_sha256=sha(folder/'data.json'),code_freeze_sha256=sha(HERE/'CODE_FREEZE.json'),
                csv_files=[dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size) for p in fit.runset.csv_files]))
            from report import summarize
            begin=time.monotonic()
            with Memory() as memory:
                diagnostics=diagnose(fit,folder,settings)
                summary=summarize(fit,data,z,u,info,overlap,folder,args.job,SEEDS[args.job])
            okay=diagnostics['pass'] and summary['reported_diagnostics_pass']
            # Verify original source bytes after each full job, no repinning.
            load(args.formal_root,args.frozen_root)
            write(folder/'completed.json',dict(status='completed' if okay else 'diagnostic_repair_required',
                **{'pass':okay},job=args.job,attempt=args.attempt,summary=summary,postprocess_seconds=time.monotonic()-begin,
                postprocess_peak_RSS_process_tree_bytes=memory.peak,source_hashes_after_unchanged=True,
                full_job_seconds_since_compilation_start=time.monotonic()-phase_start,
                file_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
            require(okay,'Technical diagnostics require repair; do not select attempt by effect direction')
        except Exception as error:
            write(folder/'failed.json',dict(status='failed_or_diagnostic_repair_required',error_type=type(error).__name__,
                message=str(error),elapsed_seconds=time.monotonic()-started,UTC=now(),prior_attempts_preserved=True))
            raise


def report_existing(args):
    require(args.execute_server and args.reason and args.report_attempt>=2,'Reporting continuation needs server flag, technical reason and report attempt >=2')
    verify_code();env=environment(args.cmdstan,True);root=destination(args.out_root)
    with lock(root):
        require(sha(root/'CODE_FREEZE.json')==sha(HERE/'CODE_FREEZE.json'),'Code freeze changed; do not relabel existing chains')
        require(json.loads((root/'environment_contract.json').read_text(encoding='utf-8'))==clean({k:v for k,v in env.items() if k!='available_RAM_bytes'}),
                'Reporting environment changed; investigate before reusing chains')
        source=root/args.job/f'attempt_{args.attempt}';record=json.loads((source/'sampling.json').read_text(encoding='utf-8'))
        require(record['code_freeze_sha256']==sha(HERE/'CODE_FREEZE.json'),'Sampler code freeze changed')
        for x in record['csv_files']:require(sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes'],'Raw chain mismatch')
        z,u,overlap,inputs=load(args.formal_root,args.frozen_root);data,info=adapt(z,u,args.job)
        require(sha(source/'data.json')==record['data_sha256'] and json.loads((source/'data.json').read_text(encoding='utf-8'))==clean(data),'Original sampled data differ')
        target=source/f'report_{args.report_attempt}';require(not target.exists(),'Existing reporting attempt preserved');target.mkdir()
        for name in ('environment.json','compiled_validation.json','pre_sampling_resources.json','settings.json','sampling.json','authorization.json','prior_predictive.json'):
            if (source/name).is_file():shutil.copyfile(source/name,target/name)
        write(target/'report_environment.json',env);write(target/'reason.json',dict(technical_reason=args.reason,sampler_attempt=args.attempt,report_attempt=args.report_attempt))
        write(target/'reuse.json',dict(source=str(source),raw_chains_reused=True,new_MCMC=False,source_sampling_sha256=sha(source/'sampling.json')))
        import cmdstanpy
        fit=cmdstanpy.from_csv([x['path'] for x in record['csv_files']]);settings=json.loads((source/'settings.json').read_text(encoding='utf-8'))
        require(fit.chains==4 and fit.num_draws_sampling==record['retained_per_chain'],'Incomplete retained chains')
        begin=time.monotonic()
        try:
            from report import summarize
            with Memory() as memory:
                diagnostics=diagnose(fit,target,settings)
                summary=summarize(fit,data,z,u,info,overlap,target,args.job,SEEDS[args.job],args.quadrature)
            okay=diagnostics['pass'] and summary['reported_diagnostics_pass']
            write(target/'completed.json',dict(status='completed' if okay else 'diagnostic_repair_required',**{'pass':okay},
                job=args.job,attempt=args.attempt,summary=summary,postprocess_seconds=time.monotonic()-begin,
                postprocess_peak_RSS_process_tree_bytes=memory.peak,file_sha256={p.name:sha(p) for p in target.iterdir() if p.is_file()}))
            require(okay,'Technical report/diagnostic repair still required')
        except Exception as error:
            write(target/'failed.json',dict(status='reporting_failed_preserved',error_type=type(error).__name__,message=str(error),UTC=now()));raise


def audit(root):
    root=destination(root);records=[];selected={}
    for job in JOBS:
        attempts=[]
        sampler_paths=sorted((root/job).glob('attempt_*'),key=lambda p:int(p.name.split('_')[-1]))
        paths=[]
        for source in sampler_paths:
            paths.append(source);paths.extend(sorted(source.glob('report_*'),key=lambda p:int(p.name.split('_')[-1])))
        for path in paths:
            completion=path/'completed.json'
            row={'job':job,'attempt_or_report':path.name,'path':str(path),'state':'incomplete'}
            if completion.exists():
                obj=json.loads(completion.read_text(encoding='utf-8'));row['state']=obj['status'];row['pass']=obj['pass']
                for name,digest in obj['file_sha256'].items():require(sha(path/name)==digest,'Completed output changed: '+name)
                sample=json.loads((path/'sampling.json').read_text(encoding='utf-8'))
                require(sample['code_freeze_sha256']==sha(root/'CODE_FREEZE.json'),'Attempt freeze differs')
                for x in sample['csv_files']:require(Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256'],'Raw chains changed')
                row['chains']=sample['chains'];row['draws_per_chain']=sample['retained_per_chain']
                if obj['pass']:selected[job]=path
            elif (path/'failed.json').exists():row['state']='failed_preserved'
            attempts.append(row)
        records.extend(attempts)
    return {'status':'all_five_completed' if len(selected)==5 else 'not_complete','attempts':records,
            'selected_by_technical_pass_then_latest_attempt':{k:str(v) for k,v in selected.items()},'selection_by_effect_direction':False}


def package(root):
    root=destination(root);a=audit(root);require(a['status']=='all_five_completed','Do not package partial five-job completion')
    out=root/'summary';require(not out.exists(),'Existing package preserved');out.mkdir()
    allowed=['environment.json','report_environment.json','reuse.json','compiled_validation.json','pre_sampling_resources.json','settings.json','sampling.json','diagnostics.json',
             'parameter_diagnostics.csv','summaries.json','contrasts.json','PPC.json','benchmark.json','support.json',
             'completed.json','reason.json','authorization.json','prior_predictive.json']
    for job,source in a['selected_by_technical_pass_then_latest_attempt'].items():
        target=out/job;target.mkdir()
        for name in allowed:
            if (Path(source)/name).is_file():shutil.copyfile(Path(source)/name,target/name)
    for i,row in enumerate(a['attempts']):
        target=out/'all_attempt_records'/str(i);target.mkdir(parents=True)
        for name in allowed+['failed.json','started.json']:
            source=Path(row['path'])/name
            if source.is_file():shutil.copyfile(source,target/name)
    comparison=[]
    for job in JOBS:
        contrasts=json.loads((out/job/'contrasts.json').read_text(encoding='utf-8'))
        require(len(contrasts)==36,'Incomplete contrasts in package')
        comparison.extend(dict(job=job,**x) for x in contrasts)
    pre=json.loads((root/'preflight.json').read_text(encoding='utf-8'));oldref=pre['inputs']['base_contrasts']
    require(sha(oldref['path'])==oldref['sha256'],'Old formal base changed')
    old=json.loads(Path(oldref['path']).read_text(encoding='utf-8'))
    key=lambda x:tuple(x[k] for k in ('prediction','contrast','query_group','target'))
    old={key(x):x for x in old if x['model']=='HA_W3'}
    joined=[dict(job=x['job'],identity=list(key(x)),old_three_way_mean=old[key(x)]['mean'],old_three_way_credible95=old[key(x)]['credible95'],
        candidate_two_way_mean=x['mean'],candidate_two_way_credible95=x['credible95'],candidate_conditioning=x['conditioning'],
        interval_scope='Separate saved model CrIs; not a paired cross-fit difference interval') for x in comparison]
    write(out/'all_five_contrasts.json',comparison);write(out/'old_vs_candidate.json',joined);write(out/'attempt_inventory.json',a)
    shutil.copyfile(root/'prior_predictive.json',out/'prior_predictive.json')
    shutil.copyfile(root/'preflight.json',out/'preflight.json');shutil.copyfile(HERE/'CODE_FREEZE.json',out/'CODE_FREEZE.json')
    shutil.copyfile(root/'environment_contract.json',out/'environment_contract.json')
    files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'MANIFEST.json',dict(scope='HA_W3_two_way_only',files=files,original_68_fits_modified=False,raw_chains_in_package=False))
    return dict(status='packaged_full_summaries_not_raw_chains',path=str(out),manifest_sha256=sha(out/'MANIFEST.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['preflight','environment','run','report-only','audit','package'])
    parser.add_argument('--formal-root');parser.add_argument('--frozen-root');parser.add_argument('--out-root');parser.add_argument('--cmdstan')
    parser.add_argument('--execute-server',action='store_true');parser.add_argument('--job',choices=JOBS);parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--reason');parser.add_argument('--warmup',type=int,default=2000);parser.add_argument('--draws',type=int,default=2000)
    parser.add_argument('--adapt-delta',type=float,default=.95);parser.add_argument('--treedepth',type=int,default=12)
    parser.add_argument('--report-attempt',type=int,default=2)
    parser.add_argument('--quadrature',type=int,choices=[40,80,160],default=40)
    args=parser.parse_args()
    try:
        if args.action=='environment':result=environment(args.cmdstan,False)
        elif args.action=='preflight':
            verify_code();result=preflight(args.formal_root,args.frozen_root)
            if args.out_root:write(destination(args.out_root)/'preflight.json',result)
            result={'status':result['status'],'logical_jobs':5,'coordinates':95,'rank':90,'sampling':False}
        elif args.action=='run':execute(args);result={'status':'job_completed','job':args.job}
        elif args.action=='report-only':report_existing(args);result={'status':'report_completed_without_new_MCMC','job':args.job}
        elif args.action=='audit':result=audit(args.out_root)
        else:result=package(args.out_root)
        print(json.dumps(clean(result),ensure_ascii=False))
    except Exception as error:
        print(json.dumps(dict(status='stopped',error_type=type(error).__name__,reason=str(error)),ensure_ascii=False));sys.exit(2)
