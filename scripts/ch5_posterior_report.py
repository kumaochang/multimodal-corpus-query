"""Posterior-only summaries, model diagnostics and day-level PPC. No retrieval."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import ch5_posterior as p
import numpy as np
from scipy import sparse
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
import arviz as az
import cmdstanpy


def csv_save(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])


def credible(draws):
    return dict(mean=float(np.mean(draws)),sd=float(np.std(draws,ddof=1)),
        credible95=np.quantile(draws,[.025,.975]).tolist())


def population_fixed_predictor(xb,cell,alpha,offset,baseline_delta=None,baseline_basis=None):
    """Fixed predictor before integrating new random effects; offset/basis once."""
    base=xb[:,cell]+alpha[:,None]+offset
    if baseline_delta is not None:base+=baseline_delta@baseline_basis.T
    return base


def diagnostics(values,chains,ndraw):
    # values: chain-major merged draws x reporting coordinate.
    out=[{} for _ in range(values.shape[1])]
    varying=np.ptp(values,axis=0)>1e-14
    if varying.any():
        shaped=values[:,varying].reshape(chains,ndraw,-1)
        result=az.summary(az.from_dict(posterior={'v':shaped}),kind='diagnostics',round_to='none')
        for j,r in zip(np.flatnonzero(varying),result.to_dict('records')):
            out[j]=dict(Rhat=float(r['r_hat']),bulk_ESS=float(r['ess_bulk']),tail_ESS=float(r['ess_tail']),
                diagnostics_pass=bool(r['r_hat']<=1.01 and r['ess_bulk']>=400 and r['ess_tail']>=400))
    for j in np.flatnonzero(~varying):
        out[j]=dict(Rhat=None,bulk_ESS=None,tail_ESS=None,diagnostics_pass=True,deterministic=True)
    return out


def groups(name,z,info,U):
    q=z['all_q'];ci=info['cell'];C=int(ci.max()+1);nr=4 if name.startswith('HB') else 6
    routes=['subtitle_xlmr','subtitle_tfidf','ASR_xlmr','ASR_tfidf'] if nr==4 else ['whisper','mfcc86','prosody','xlmr_sub','tfidf_sub','kin14']
    counts=np.bincount(ci,minlength=C);first=np.array([np.flatnonzero(ci==i)[0] for i in range(C)])
    R=np.arange(len(q))%nr
    ids=[]
    for c,j in enumerate(first):
        speech='M' if U['lid_answer'][q[j]]=='mandarin' else 'I' if U['lid_answer'][q[j]]=='not_chinese' and U['mf_class'][q[j]]!='mandarin' else 'other_status'
        ids.append(dict(route=routes[R[j]],source=U['lang'][q[j]],model_class=U['haiku'][q[j]],speech_group=speech if name.startswith('HA') else 'not_a_factor'))
    # Each fixed-design row must have exactly one empirical identification cell.
    for c,j in enumerate(first):
        sel=ci==c
        assert np.unique(R[sel]).size==1
        assert np.unique(U['lang'][q[sel]]).size==1 and np.unique(U['haiku'][q[sel]]).size==1
    selections=[];descriptions=[]
    def add(scope,route,source='pooled',klass='all',speech='all',cell=None):
        sel=np.array([x['route']==route and (source=='pooled' or x['source']==source)
            and (speech=='all' or x['speech_group']==speech)
            and (klass=='all' or (x['model_class']!='Explaining' if klass=='remainder' else x['model_class']==klass)) for x in ids])
        if cell is not None:sel=np.arange(C)==cell
        if not sel.any():return
        rowmask=sel[ci];qq=np.unique(q[rowmask]);days=len(set(zip(U['lang'][qq],U['day'][qq])));clips=len(set(U['clip'][qq]))
        w=counts*sel;w=w/w.sum()
        status='pooled_exploratory_model_summary'
        if scope=='supported_cell' or source!='pooled':status='descriptive_local_O7'
        if klass in p.LABELS and (days<15 or clips<25):status='descriptive_sparse_O7'
        cc=z['all_chance'][rowmask]
        interior=int(np.sum(np.isfinite(cc)&(cc>0)&(cc<1)))
        selections.append(w);descriptions.append(dict(model=name,scope=scope,route=route,source=source,
            model_class=klass,speech_group=speech,queries=len(qq),days=days,clips=clips,status=status,
            likelihood_rows=interior,structural_zero_rows=int(np.sum(cc==0)),structural_one_rows=int(np.sum(cc==1)),
            no_candidate_rows=int(np.sum(~np.isfinite(cc))),
            likelihood_support='interior_observed_rows' if interior else 'structural_boundary_only_no_likelihood_information'))
    for c,x in enumerate(ids):add('supported_cell',x['route'],x['source'],x['model_class'],x['speech_group'],c)
    for r in routes:
        for k in ['all','remainder']+p.LABELS:add('pooled_margin',r,klass=k)
        for l in p.SOURCES[:5] if nr==4 else p.SOURCES:add('source_margin',r,source=l)
        if name.startswith('HA'):
            for g in ['M','I']:
                for k in ['all','remainder']+p.LABELS:add('speech_margin',r,klass=k,speech=g)
    # One query contributes once per route and once per source/day PPC group.
    row_sets=[];ppcmeta=[]
    for route in routes:
        for day in range(int(z['D'])):
            m=(R==routes.index(route))&(z['all_day']==day)
            if not m.any():continue
            qq=q[m]
            row_sets.append(m.astype(float)/m.sum())
            ppcmeta.append(dict(scope='source_day',route=route,source=U['lang'][qq][0],day=U['day'][qq][0],queries=len(qq)))
    Sc=sparse.csr_matrix((1/counts[ci],(ci,np.arange(len(ci)))),shape=(C,len(ci)))
    return np.array(selections),descriptions,Sc,sparse.csr_matrix(np.array(row_sets)),ppcmeta,ids


def summarize(name,variant='base',attempt=1):
    directory=p.job_dir(name,variant,attempt)
    loaded_source=Path(__file__).read_bytes()
    loaded_sha=p.hashlib.sha256(loaded_source).hexdigest()
    registry=p.WORK/'source_versions';registry.mkdir(exist_ok=True)
    for content in [loaded_source]+([(directory/'loaded_postprocessor.py').read_bytes()] if (directory/'loaded_postprocessor.py').exists() else []):
        digest=p.hashlib.sha256(content).hexdigest();archive=registry/(digest+'.py')
        if not archive.exists():archive.write_bytes(content)
        assert p.sha(archive)==digest
    record=p.loadjson(directory/'sampling.json')
    assert record['status']=='sampled' and record['chains']==4
    done=directory/'postprocessed.json'
    if done.exists():
        previous=p.loadjson(done)
        if previous.get('sampling_fingerprint')==record['fingerprint'] and previous.get('postprocessor_sha256')==loaded_sha:
            print('POSTPROCESS reuse',name,variant,attempt,flush=True);return previous
    fit=cmdstanpy.from_csv(record['csv_files']);theta=fit.stan_variables()
    (directory/'loaded_postprocessor.py').write_bytes(loaded_source)
    ndraw=record['retained_per_chain'];chains=record['chains'];M=ndraw*chains
    if 'z_block' in theta:
        from ch5_whiten import decode
        actual_data=p.loadjson(directory/'data.json')
        actual_data.update({k:np.asarray(v) for k,v in actual_data.items() if isinstance(v,list)})
        firstcol=next(c for c in fit.column_names if c.startswith('z_block.'))
        np.testing.assert_array_equal(theta['z_block'][:,0],fit.draws()[:,:,fit.column_names.index(firstcol)].T.ravel())
        theta=decode(theta,actual_data)
    else:
        rawalpha=fit.draws(inc_warmup=False)[:,:,fit.column_names.index('alpha')].T.ravel()
        np.testing.assert_array_equal(theta['alpha'],rawalpha)
    d,z,info,U=p.adapter(name,variant)
    W,desc,Sc,Sday,daydesc,cellids=groups(name,z,info,U)
    C=d['C'];nall=len(z['all_q']);ci=info['cell'];inside=info['inside'];chance=z['all_chance'];hit=z['all_hit']
    # All adopted families, including the rank-deficient directions, are retained.
    # Their RMS is a descriptive magnitude under a continuous prior, never a test
    # that the family exists. Observed support weights do not fill missing cells.
    terms=p.MODELS[name]['formula'].split('+');components=[];component_values=[]
    cell_weight=np.bincount(ci,minlength=C)/len(ci)
    for family,term in enumerate(terms,1):
        ix=np.flatnonzero(d['family']==family)
        value=theta['beta'][:,ix]@d['X_cell'][:,ix].T
        rms=np.sqrt((value**2)@cell_weight)
        component_values.append(rms)
        components.append(dict(model=name,variant=variant,attempt=attempt,term=term,
            coefficients=len(ix),observed_component_logit_RMS=credible(rms),
            interpretation='descriptive prior-conditional magnitude; positive RMS is not evidence of interaction; aliased families cannot be uniquely data-identified'))
    component_diag=diagnostics(np.column_stack(component_values),chains,ndraw)
    for row,diag in zip(components,component_diag):row.update(diag)
    original_beta_diag=diagnostics(theta['beta'],chains,ndraw)
    insideidx=np.flatnonzero(inside);offset=np.zeros(nall);offset[inside]=d['chance_logit']
    likelihood_weight=np.zeros(nall);likelihood_weight[inside]=d['likelihood_weight']
    active_day=np.bincount(z['all_day'],weights=likelihood_weight,minlength=int(z['D']))>0
    # Baseline and integration inputs collapse exact cell/offset duplicates.
    unique,uinv=np.unique(np.column_stack([ci[inside],offset[inside]]),axis=0,return_inverse=True)
    ucell=unique[:,0].astype(int);uoffset=unique[:,1]
    ux=(uoffset-info['offset_center'])/info['offset_sd']
    ubasis=np.column_stack([ux,(ux**2-info['quad_center'])/info['quad_sd']])
    first=np.array([np.flatnonzero(ci==j)[0] for j in range(C)])
    slope_per_cell=z['all_slopes'][first,:d['H']]
    gh,gw=hermgauss(40);gw=gw/np.sqrt(np.pi)
    gh80,gw80=hermgauss(80);gw80/=np.sqrt(np.pi)
    cond=np.empty((M,C));marg=np.empty_like(cond)
    rng=np.random.Generator(np.random.PCG64(record['settings']['seed']+80000))
    selected=np.linspace(0,M-1,512).astype(int);selected_set=set(selected)
    ppc=np.empty((512,W.shape[0]+Sday.shape[0]));rep_corr=[];obs_corr=None
    q=z['all_q'];nr=4 if name.startswith('HB') else 6
    ywide=hit.reshape(-1,nr)
    with np.errstate(invalid='ignore',divide='ignore'):obs_corr=np.corrcoef(ywide,rowvar=False)
    day_ll=np.empty((M,int(z['D'])));integration_error=0.;take_j=0
    day_counts=np.bincount(z['all_day'],minlength=int(z['D']))
    Sd=sparse.csr_matrix((np.ones(nall),(z['all_day'],np.arange(nall))),shape=(int(z['D']),nall))
    observed=np.concatenate([W@np.asarray(Sc@hit).ravel(),np.asarray(Sday@hit).ravel()])
    chance_cell=np.asarray(Sc@chance).ravel()
    for start in range(0,M,128):
        end=min(start+128,M);sl=slice(start,end);b=end-start
        xb=theta['beta'][sl]@d['X_cell'].T
        eta=xb[:,ci]+theta['alpha'][sl,None]
        eta+=theta['sigma_day'][sl,None]*theta['z_day'][sl][:,z['all_day']]
        eta+=theta['sigma_query'][sl,None]*theta['z_query'][sl][:,z['all_query']]
        if d['H']:
            bs=theta['z_slope'][sl]*theta['sigma_slope'][sl,None,:]
            ds=np.einsum('bdh,dh->bd',bs[:,d['slope_day']-1,:],d['slope_design'])
            eta+=ds[:,info['slope_index']]
        if d['CL']:eta+=theta['sigma_clip'][sl,None]*theta['z_clip'][sl][:,info['clip_index']]
        if d['flexible_baseline']:eta[:,inside]+=theta['baseline_delta'][sl]@d['baseline_basis'].T
        eta+=offset[None,:]
        pi=expit(eta);pi[:,info['structural']]=chance[info['structural']]
        pi[:,~np.isfinite(chance)]=np.nan
        cond[sl]=(Sc@pi.T).T
        variance=theta['sigma_day'][sl,None]**2+theta['sigma_query'][sl,None]**2+np.zeros((b,C))
        if d['H']:variance+=(theta['sigma_slope'][sl]**2)@(slope_per_cell**2).T
        if d['CL']:variance+=theta['sigma_clip'][sl,None]**2
        base=population_fixed_predictor(xb,ucell,theta['alpha'][sl],uoffset,
            theta['baseline_delta'][sl] if d['flexible_baseline'] else None,ubasis)
        scale=np.sqrt(2*variance[:,ucell])
        integrated=np.zeros_like(base)
        for node,weight in zip(gh,gw):integrated+=weight*expit(base+scale*node)
        pop=np.broadcast_to(chance,(b,nall)).copy();pop[:,inside]=integrated[:,uinv]
        marg[sl]=(Sc@pop.T).T
        if start==0:
            alt=np.zeros_like(base)
            for node,weight in zip(gh80,gw80):alt+=weight*expit(base+scale*node)
            integration_error=float(np.max(np.abs(alt-integrated)))
        ll=hit[None,:]*eta-np.logaddexp(0,eta)
        ll[:,info['structural']]=0;ll[:,~np.isfinite(chance)]=0
        ll*=likelihood_weight[None,:]
        day_ll[sl]=(Sd@ll.T).T
        for j in range(start,end):
            if j not in selected_set:continue
            pred=(rng.random(nall)<pi[j-start]).astype(float)
            ppc[take_j]=np.concatenate([W@np.asarray(Sc@pred).ravel(),np.asarray(Sday@pred).ravel()])
            with np.errstate(invalid='ignore',divide='ignore'):rep_corr.append(np.corrcoef(pred.reshape(-1,nr),rowvar=False))
            take_j+=1
    assert take_j==512
    allsummary=[];gchance=W@chance_cell
    for label,values in [('conditional',cond@W.T),('population_average',marg@W.T)]:
        ds=diagnostics(values,chains,ndraw)
        for j,base in enumerate(desc):
            r=dict(**base,variant=variant,attempt=attempt,prediction=label,
                probability=credible(values[:,j]),lift=credible(values[:,j]-gchance[j]),
                chance=float(gchance[j]),**ds[j])
            allsummary.append(r)
    # Predeclared interpretable representation/source differences, paired by draw.
    contrasts=[]
    def margin_index(route,klass='all',speech='all'):
        scope='speech_margin' if speech!='all' else 'pooled_margin'
        return next((j for j,r in enumerate(desc) if r['scope']==scope and r['route']==route and r['model_class']==klass and r['speech_group']==speech),None)
    pairs=[('ASR_xlmr','subtitle_xlmr'),('ASR_tfidf','subtitle_tfidf')] if name.startswith('HB') else [('whisper','prosody'),('xlmr_sub','tfidf_sub'),('kin14','whisper')]
    for label,cellvalues in [('conditional',cond),('population_average',marg)]:
        vv=cellvalues@W.T-gchance
        vals=[];meta=[]
        for a,b in pairs:
            for klass in ['all','Explaining','Reporting']:
                ia,ib=margin_index(a,klass),margin_index(b,klass)
                if ia is None or ib is None:continue
                vals.append(vv[:,ia]-vv[:,ib]);meta.append(dict(contrast=a+'-'+b,query_group=klass,target='empirical_query_weighted'))
            ae,be,ar,br=[margin_index(r,k) for r,k in [(a,'Explaining'),(b,'Explaining'),(a,'Reporting'),(b,'Reporting')]]
            if all(x is not None for x in [ae,be,ar,br]):
                vals.append(vv[:,ae]-vv[:,be]-vv[:,ar]+vv[:,br]);meta.append(dict(contrast=a+'-'+b,query_group='Explaining_minus_Reporting_DID',target='empirical_query_weighted'))
            if name.startswith('HA'):
                idx=[margin_index(r,'all',g) for r,g in [(a,'M'),(b,'M'),(a,'I'),(b,'I')]]
                vals.append(vv[:,idx[0]]-vv[:,idx[1]]-vv[:,idx[2]]+vv[:,idx[3]])
                meta.append(dict(contrast=a+'-'+b,query_group='M_minus_I_DID',target='unbalanced_empirical_groups_composition_sensitive'))
                # The common-set overlap weights were fixed before new ranking effects.
                with (p.REPORT/'execution_20261020/adopted_overlap_weights.csv').open(encoding='utf-8') as f:
                    overlap=[r for r in csv.DictReader(f) if r['set']=='six_common' and r['candidate_policy']=='union_MI' and r['pool']==name.split('_')[1] and float(r['restricted_weight'])>0]
                value=np.zeros(M)
                for cell in overlap:
                    idx=[]
                    for route,g in [(a,'M'),(b,'M'),(a,'I'),(b,'I')]:
                        idx.append(next(j for j,r in enumerate(cellids) if r['route']==route and r['speech_group']==g and r['source']==cell['source_community'] and r['model_class']==cell['model_class']))
                    value+=float(cell['restricted_weight'])*(cellvalues[:,idx[0]]-cellvalues[:,idx[1]]-cellvalues[:,idx[2]]+cellvalues[:,idx[3]])
                assert np.isclose(sum(float(c['restricted_weight']) for c in overlap),1)
                vals.append(value);meta.append(dict(contrast=a+'-'+b,query_group='M_minus_I_DID',target='six_common_restricted_overlap_four_cells_not_family_24_target'))
        if vals:
            v=np.column_stack(vals);diag=diagnostics(v,chains,ndraw)
            for j,m in enumerate(meta):
                route=m['contrast'].split('-',1)[0];group=m['query_group'];coverage={}
                keys=['Explaining','Reporting'] if group=='Explaining_minus_Reporting_DID' else [group] if group in ['all']+p.LABELS else []
                for k in keys:
                    row=desc[margin_index(route,k)]
                    coverage[k]={key:row[key] for key in ['queries','days','clips','status']}
                sparse_coverage=any(k in p.LABELS and (r['days']<15 or r['clips']<25) for k,r in coverage.items())
                contrasts.append(dict(model=name,variant=variant,attempt=attempt,prediction=label,**m,
                    **credible(v[:,j]),**diag[j],coverage=coverage,
                    coverage_status='descriptive_sparse_O7' if sparse_coverage else 'descriptive_model_summary_O7_local_class_limits_retained',
                    restricted_overlap_coverage_source='execution_20261020/adopted_overlap_weights.csv' if 'restricted_overlap' in m['target'] else None,
                    status='descriptive_model_credible_interval_not_simultaneous_CI'))
    ppcreport=[]
    for j,m in enumerate(desc+daydesc):
        interval=np.quantile(ppc[:,j],[.025,.975])
        ppcreport.append(dict(m,model=name,variant=variant,observed_hit=float(observed[j]),
            replicated_hit_mean=float(ppc[:,j].mean()),replicated95=interval.tolist(),
            observed_in_replicated95=bool(interval[0]<=observed[j]<=interval[1]),posterior_predictive_draws=512))
    corr=np.asarray(rep_corr);correlations=[]
    for a,b in itertools.combinations(range(nr),2):
        correlations.append(dict(route_a=a,route_b=b,observed=float(obs_corr[a,b]),
            replicated_mean=float(np.nanmean(corr[:,a,b])),replicated95=np.nanquantile(corr[:,a,b],[.025,.975]).tolist()))
    # Joint day likelihood diagnostic only; explicitly conditional on latent effects.
    shaped=day_ll[:,active_day].reshape(chains,ndraw,-1)
    data=az.from_dict(posterior={'alpha':theta['alpha'].reshape(chains,ndraw)},log_likelihood={'day':shaped})
    loo=az.loo(data,pointwise=True,var_name='day')
    pareto=np.full(int(z['D']),np.nan);elpd=pareto.copy()
    pareto[active_day]=loo.pareto_k.values;elpd[active_day]=loo.loo_i.values
    influence=[]
    cellcount=np.bincount(ci,minlength=C)
    for day in range(int(z['D'])):
        qq=np.unique(q[z['all_day']==day]);ix=np.flatnonzero(z['all_day']==day)[0]
        remaining=np.bincount(ci[z['all_day']!=day],minlength=C)
        # Deleted-day standardized target summaries, not refitted posteriors/CV.
        maxshift=0.
        for route in set(x['route'] for x in cellids):
            sel=np.array([x['route']==route for x in cellids]);w0=cellcount*sel;w1=remaining*sel
            w0=w0/w0.sum();w1=w1/w1.sum()
            shift=(marg@(w1-w0))
            maxshift=max(maxshift,float(abs(shift.mean())))
        influence.append(dict(model=name,variant=variant,source=U['lang'][q[ix]],day=U['day'][q[ix]],
            queries=len(qq),conditional_Pareto_k=float(pareto[day]),
            conditional_elpd_loo=float(elpd[day]),likelihood_day_active=bool(active_day[day]),
            deleted_day_target_max_absolute_probability_mean_shift=maxshift,
            limitation='conditional latent-effect PSIS is diagnostic only, not reliable new-day validation; deletion changes summary weights, not fitted outcomes'))
    np.savez_compressed(directory/'posterior_cell_draws.npz',conditional=cond,population_average=marg,
        group_weights=W,day_loglik=day_ll,ppc=ppc)
    target=p.summary_directory(name,variant,attempt);target.mkdir(parents=True,exist_ok=True)
    p.dump(target/'summaries.json',allsummary);p.dump(target/'contrasts.json',contrasts)
    p.dump(target/'interaction_components.json',components)
    p.dump(target/'identification.json',dict(model=name,variant=variant,
        saturated_coordinates=p.MODELS[name]['saturated_coordinates'],
        metadata_rank=p.MODELS[name]['metadata_rank'],likelihood_rank=p.MODELS[name]['likelihood_rank'],
        observed_factor_cells=len(cellids),unobserved_cells_excluded_from_empirical_averages=True,
        original_coefficient_diagnostics=original_beta_diag,
        original_coefficients_all_diagnostics_pass=all(r['diagnostics_pass'] for r in original_beta_diag),
        explanation='Proper priors yield a posterior despite rank deficiency. Sampling convergence does not identify aliased components or validate missing-cell extrapolation. No missing-cell predictions are presented as observed data.'))
    p.dump(target/'PPC.json',dict(calibration=ppcreport,paired_hit_correlations=correlations,
        fit_likelihood_weighting=variant if variant.startswith('day_') else 'all_interior_rows_weight_one',
        evaluation_weights='original fixed empirical query targets; holdout/weighted refit PPC is explicitly not independent full-corpus validation'))
    p.dump(target/'influential_days.json',influence)
    csv_save(target/'summaries.csv',[{**r,'probability_mean':r['probability']['mean'],'lift_mean':r['lift']['mean']} for r in allsummary])
    csv_save(target/'contrasts.csv',contrasts)
    result=dict(model=name,variant=variant,attempt=attempt,sampling_fingerprint=record['fingerprint'],
        chains=chains,retained_per_chain=ndraw,reported_summary_rows=len(allsummary),contrast_rows=len(contrasts),
        summaries_diagnostics_pass=all(r['diagnostics_pass'] for r in allsummary+contrasts+components+original_beta_diag),
        max_Rhat_reported=max(r['Rhat'] for r in allsummary+contrasts+components+original_beta_diag if r['Rhat'] is not None),
        min_bulk_ESS_reported=min(r['bulk_ESS'] for r in allsummary+contrasts+components+original_beta_diag if r['bulk_ESS'] is not None),
        min_tail_ESS_reported=min(r['tail_ESS'] for r in allsummary+contrasts+components+original_beta_diag if r['tail_ESS'] is not None),
        quadrature_max_40_vs_80_abs_probability_error_first128draws=integration_error,
        influential_days_conditional_k_over_0_7=sum(r['conditional_Pareto_k']>.7 for r in influence),
        posterior_predictive_calibration_rows=len(ppcreport),postprocessor_sha256=loaded_sha,
        caveat='credible intervals conditional on local fixed rankings/model/prior; O7 descriptive restrictions and unobserved-cell exclusions retained')
    p.dump(done,result);p.dump(p.OUT/'postprocess_status'/f'{name}_{variant}_attempt_{attempt}.json',result)
    p.dump(p.OUT/'summary_manifests'/f'{name}_{variant}_attempt_{attempt}.json',
        dict(model=name,variant=variant,attempt=attempt,absolute_path=str(target),
             sha256={f.name:p.sha(f) for f in target.iterdir() if f.is_file()}))
    print('POSTPROCESSED',name,variant,'Rhat',result['max_Rhat_reported'],'ESS',result['min_bulk_ESS_reported'],'quadrature',integration_error,flush=True)
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',choices=list(p.MODELS),required=True)
    a.add_argument('--variant',default='base');a.add_argument('--attempt',type=int,default=1);x=a.parse_args()
    summarize(x.model,x.variant,x.attempt)
