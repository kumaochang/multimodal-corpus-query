"""Server-only posterior/prior reporting. Imported locally for synthetic tests only."""
import itertools
from pathlib import Path
import numpy as np
from scipy.special import expit,xlogy,xlog1py
from numpy.polynomial.hermite import hermgauss
from design import (LABELS,SOURCES,ROUTES,PAIRS,target_vectors,require,write,clean)


def integrated(fixed,variance,points=40):
    x,w=hermgauss(points)
    return expit(np.asarray(fixed)[:,None]+np.sqrt(2*np.asarray(variance))[:,None]*x)@(w/np.sqrt(np.pi))


def prediction(z,info,params,points=40):
    """Full fixed query order; unsupported day effects never enter predictions."""
    X=info['X'];chance=z['all_chance'];inside=np.isfinite(chance)&(chance>0)&(chance<1)
    eta=np.zeros(len(chance));eta[inside]=np.log(chance[inside]/(1-chance[inside]))
    eta+=params['alpha']+X@params['beta']
    slope_var=(z['all_slopes']**2)@(params['sigma_slope']**2)
    variance=params['sigma_query']**2+params['sigma_day']**2+slope_var
    # Repeated fixed predictor/variance combinations are integrated only once.
    keys,index=np.unique(np.column_stack([eta[inside],variance[inside]]),axis=0,return_inverse=True)
    probability=integrated(keys[:,0],keys[:,1],points)
    pop=np.array(chance,copy=True);pop[inside]=probability[index]
    check=integrated(keys[:,0],keys[:,1],points*2)
    error=float(np.max(np.abs(check-probability))) if len(keys) else 0.
    cond=pop.copy();supported=inside&info['supported_day'][z['all_day']]
    q=z['all_query'][supported];d=z['all_day'][supported]
    # Index only supported effects. Tests deliberately poison excluded z entries.
    e=eta[supported]+params['sigma_query']*params['z_query'][q]+params['sigma_day']*params['z_day'][d]
    e+=np.sum(z['all_slopes'][supported]*params['z_slope'][d]*params['sigma_slope'],axis=1)
    cond[supported]=expit(e)
    return cond,pop,error


def replication(z,info,params,rng,new_latent=False):
    """Joint new effects are shared across all routes of a query/day."""
    chance=z['all_chance'];inside=np.isfinite(chance)&(chance>0)&(chance<1)
    q=z['all_query'];d=z['all_day'];supported=info['supported_day'][d]
    effects=dict(params)
    day=np.zeros(int(z['D']));slope=np.zeros((int(z['D']),int(z['H'])));query=np.zeros(int(z['Q']))
    fresh_days=np.ones(len(day),bool) if new_latent else ~info['supported_day']
    fresh_queries=np.ones(len(query),bool) if new_latent else np.array([not info['supported_day'][d[np.flatnonzero(q==i)[0]]] for i in range(len(query))])
    if not new_latent:
        day[~fresh_days]=params['z_day'][~fresh_days];slope[~fresh_days]=params['z_slope'][~fresh_days]
        query[~fresh_queries]=params['z_query'][~fresh_queries]
    day[fresh_days]=rng.normal(size=fresh_days.sum());slope[fresh_days]=rng.normal(size=(fresh_days.sum(),slope.shape[1]))
    query[fresh_queries]=rng.normal(size=fresh_queries.sum())
    p=np.array(chance,copy=True)
    eta=np.log(chance[inside]/(1-chance[inside]))+params['alpha']+info['X'][inside]@params['beta']
    eta+=params['sigma_day']*day[d[inside]]+params['sigma_query']*query[q[inside]]
    eta+=np.sum(z['all_slopes'][inside]*slope[d[inside]]*params['sigma_slope'],axis=1)
    p[inside]=expit(eta)
    result=np.full(len(p),np.nan);valid=np.isfinite(p);result[valid]=(rng.random(valid.sum())<p[valid]).astype(float)
    return result


def supports(z,u,mask,training=None):
    ids=np.unique(z['all_query'][mask]);q=z['all_q'][mask]
    return dict(queries=len(ids),days=len(set(zip(u['lang'][mask],u['day'][mask]))),
        clips=len(set(u['clip'][mask])),scored_rows=int(mask.sum()),
        interior_rows=int(np.sum(mask&np.isfinite(z['all_chance'])&(z['all_chance']>0)&(z['all_chance']<1))),
        training_likelihood_rows=int(np.sum(mask&training)) if training is not None else None)


def margin_groups(z,u,info):
    valid=np.isfinite(z['all_chance']);train=info['inside']&(info['day_weights'][z['all_day']]>0)
    masks=[];names=[]
    def add(mask,meta):
        absent=int(np.sum(mask&~valid));mask=mask&valid
        if not mask.any():return
        support=supports(z,u,mask,train)
        masks.append(mask);names.append(dict(meta,**support,no_candidate_rows=absent,
            structural_zero_rows=int(np.sum(mask&(z['all_chance']==0))),structural_one_rows=int(np.sum(mask&(z['all_chance']==1))),
            status='structural_only_deterministic' if not support['interior_rows'] else
                   'prior_only_observed_cell' if not support['training_likelihood_rows'] else
                   'descriptive_local_O7' if meta['scope']=='supported_cell' else 'pooled_descriptive_model_summary_O7_retained'))
    for route in range(6):
        for klass in ['all','remainder']+LABELS:
            k=np.ones(len(valid),bool) if klass=='all' else u['haiku']!='Explaining' if klass=='remainder' else u['haiku']==klass
            add((u['route']==route)&k,dict(scope='pooled_margin',route=ROUTES[route],model_class=klass,speech_group='all'))
        for speech in ('M','I'):
            for klass in ['all','remainder']+LABELS:
                k=np.ones(len(valid),bool) if klass=='all' else u['haiku']!='Explaining' if klass=='remainder' else u['haiku']==klass
                add((u['route']==route)&(u['speech']==speech)&k,dict(scope='speech_margin',route=ROUTES[route],model_class=klass,speech_group=speech))
        for source in SOURCES:
            add((u['route']==route)&(u['lang']==source),dict(scope='source_margin',route=ROUTES[route],source=source))
    occupied,first=np.unique(info['X'],axis=0,return_index=True)
    for i in first:
        mask=np.all(info['X']==info['X'][i],axis=1)
        add(mask,dict(scope='supported_cell',route=ROUTES[u['route'][i]],source=u['lang'][i],model_class=u['haiku'][i],speech_group=u['speech'][i]))
    return masks,names


def credible(values):
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),credible95=np.quantile(values,[.025,.975]).tolist())


def diagnostics(values,chains,draws):
    import arviz as az
    result=[dict(Rhat=None,bulk_ESS=None,tail_ESS=None,diagnostics_pass=True,deterministic=True) for _ in range(values.shape[1])]
    vary=np.ptp(values,axis=0)>1e-14
    if vary.any():
        table=az.summary(az.from_dict(posterior={'value':values[:,vary].reshape(chains,draws,-1)}),kind='diagnostics',round_to='none')
        for i,row in zip(np.flatnonzero(vary),table.to_dict('records')):
            result[i]=dict(Rhat=float(row['r_hat']),bulk_ESS=float(row['ess_bulk']),tail_ESS=float(row['ess_tail']),
                diagnostics_pass=bool(row['r_hat']<=1.01 and row['ess_bulk']>=400 and row['ess_tail']>=400))
    return result


def extract(fit,data):
    raw=fit.draws(concat_chains=False);require(raw.shape[:2]==(fit.num_draws_sampling,4),'Expected four full chains')
    flat=raw.transpose(1,0,2).reshape(-1,raw.shape[-1]);columns={c:i for i,c in enumerate(fit.column_names)}
    def var(name,shape=()):
        if not shape:return flat[:,columns[name]]
        ix=np.empty(shape,int)
        for ind in np.ndindex(shape):
            # Retained CmdStan 2.37 collector uses dot-form CSV names. Some
            # readers expose bracket form. Resolve exactly one explicit index.
            options=[name+'.'+'.'.join(str(i+1) for i in ind),name+'['+','.join(str(i+1) for i in ind)+']']
            found=[columns[x] for x in options if x in columns]
            require(len(found)==1,'Missing/ambiguous indexed CSV parameter: '+name)
            ix[ind]=found[0]
        return flat[:,ix]
    return {k:var(k,shape) for k,shape in [('alpha',()),('beta',(data['J'],)),('tau',(6,)),
        ('sigma_query',()),('sigma_day',()),('sigma_slope',(5,)),('z_query',(data['Q'],)),
        ('z_day',(data['D'],)),('z_slope',(data['D'],5))]}


def ppc_groups(z,u,info):
    valid=np.isfinite(z['all_chance']);masks,meta=margin_groups(z,u,info)
    def add(mask,**description):
        mask=mask&valid
        if mask.any():masks.append(mask);meta.append(description)
    add(valid,scope='pool',pool='W3')
    for source in SOURCES:add(u['lang']==source,scope='source',source=source)
    for klass in LABELS:add(u['haiku']==klass,scope='model_class',model_class=klass)
    for i,route in enumerate(ROUTES):add(u['route']==i,scope='representation',route=route)
    for day in np.unique(z['all_day']):
        for route in range(6):add((z['all_day']==day)&(u['route']==route),scope='source_day',day_index_0=int(day),route=ROUTES[route])
    return masks,meta


def summarize(fit,data,z,u,info,overlap,out,job,seed,quadrature=40):
    out=Path(out);arrays=extract(fit,data);M=len(arrays['alpha']);draws=fit.num_draws_sampling
    T,keys=target_vectors(z,u,overlap);masks,names=margin_groups(z,u,info)
    # Each target has constant weights within each retained fixed cell.
    cell=info['cell'];C=int(cell.max()+1);valid=np.isfinite(z['all_chance'])
    counts=np.maximum(np.bincount(cell,weights=valid.astype(int),minlength=C),1)
    A=np.array([mask/mask.sum() for mask in masks]);allW=np.vstack([A,T])
    W=np.array([np.bincount(cell,weights=w,minlength=C) for w in allW])
    for w,v in zip(allW,W):require(np.allclose(w,np.where(valid,v[cell]/counts[cell],0),atol=1e-14),'Target not homogeneous within fixed cell')
    cells={p:np.empty((M,C)) for p in ('conditional','population_average')};max_error=0.
    pmasks,pmeta=ppc_groups(z,u,info);selection=np.linspace(0,M-1,min(512,M)).astype(int);chosen=set(selection.tolist())
    ppc={p:[] for p in cells};correlations={p:[] for p in cells};rng=np.random.default_rng(seed+100)
    held=np.isin(z['all_day'],info['heldout_days_0'])&np.isfinite(z['all_chance'])
    benchmark=[];held_sum=np.zeros(int(held.sum()))
    for i in range(M):
        par={k:v[i] for k,v in arrays.items()};cond,pop,error=prediction(z,info,par,quadrature);max_error=max(max_error,error)
        for mode,p in [('conditional',cond),('population_average',pop)]:
            cells[mode][i]=np.bincount(cell,weights=np.nan_to_num(p),minlength=C)/counts
        if held.any():
            y=z['all_hit'][held];prob=pop[held]
            held_sum+=prob
            benchmark.append([np.mean((y-prob)**2),np.mean(xlogy(y,prob)+xlog1py(1-y,-prob))])
        if i in chosen:
            for mode,new in [('conditional',False),('population_average',True)]:
                rep=replication(z,info,par,rng,new_latent=new)
                ppc[mode].append([np.mean(rep[m]) for m in pmasks])
                with np.errstate(invalid='ignore',divide='ignore'):
                    byquery=rep.reshape(-1,6);complete=np.isfinite(byquery).all(axis=1)
                    correlations[mode].append(np.corrcoef(byquery[complete],rowvar=False))
    require(max_error<=1e-5,'Quadrature refinement mismatch; preserve attempt and report')
    summaries=[];contrasts=[]
    chance=np.bincount(cell,weights=np.nan_to_num(z['all_chance']),minlength=C)/counts
    for mode,values in cells.items():
        results=values@W.T;diag=diagnostics(results,4,draws)
        conditioning=('all_query_day_slopes_integrated_new' if mode=='population_average' else
                      'training_fitted_zero_weight_days_integrated_new' if np.any(~info['supported_day']) else 'observed_fitted_effects')
        for j,meta in enumerate(names):
            summaries.append(dict(meta,prediction=mode,conditioning=conditioning,probability=credible(results[:,j]),
                lift=credible(results[:,j]-W[j]@chance),chance=float(W[j]@chance),**diag[j]))
        for j,key in enumerate(keys,len(names)):
            contrasts.append(dict(contrast=key[0],query_group=key[1],target=key[2],prediction=mode,conditioning=conditioning,
                **credible(results[:,j]-W[j]@chance),**diag[j],status='descriptive_pointwise_model_CrI_O7_retained'))
    require(len(contrasts)==36,'Complete old HA target contract required')
    write(out/'summaries.json',summaries);write(out/'contrasts.json',contrasts)
    reports=[]
    for mode,values in ppc.items():
        a=np.array(values)
        for j,meta in enumerate(pmeta):reports.append(dict(meta,prediction=mode,observed=float(np.mean(z['all_hit'][pmasks[j]])),**credible(a[:,j])))
    with np.errstate(invalid='ignore',divide='ignore'):
        observed=z['all_hit'].reshape(-1,6);complete=np.isfinite(observed).all(axis=1)
        observed_corr=np.corrcoef(observed[complete],rowvar=False)
    write(out/'PPC.json',dict(calibration=reports,paired_route_order=ROUTES,observed_hit_correlation=observed_corr,
        paired_correlation_queries=int(complete.sum()),
        replicated_correlation={mode:{'mean':np.nanmean(a,axis=0),'credible95':np.nanquantile(a,[.025,.975],axis=0)} for mode,a in correlations.items()},
        semantics='population_average PPC uses joint fresh query/day/slopes, not independent marginal Bernoulli rows'))
    mean_held=held_sum/M if held.any() else None
    zero_mass_errors=int(np.sum(((mean_held==0)&(z['all_hit'][held]==1))|((mean_held==1)&(z['all_hit'][held]==0)))) if held.any() else 0
    write(out/'benchmark.json',dict(job=job,heldout_source_days=[info['day_keys'][i] for i in info['heldout_days_0']],
        heldout_support=supports(z,u,held) if held.any() else None,
        posterior_predictive_Brier=float(np.mean((z['all_hit'][held]-mean_held)**2)) if held.any() else None,
        posterior_predictive_mean_log_score=float(np.mean(xlogy(z['all_hit'][held],mean_held)+xlog1py(1-z['all_hit'][held],-mean_held))) if held.any() else None,
        log_score_status='zero_predictive_mass_error' if zero_mass_errors else 'finite' if held.any() else 'not_holdout',
        zero_predictive_mass_errors=zero_mass_errors,
        per_draw_Brier_distribution=credible(np.array(benchmark)[:,0]) if benchmark else None,
        per_draw_mean_log_score_distribution=credible(np.array(benchmark)[:,1]) if benchmark and np.isfinite(np.array(benchmark)[:,1]).all() else None,
        per_draw_nonfinite_log_score_count=int(np.sum(~np.isfinite(np.array(benchmark)[:,1]))) if benchmark else 0,
        score_interpretation='Marginal proper scores of posterior-mean predictions; per-draw distributions are not independent-day validation intervals',
        candidate_bank='unchanged fixed original W3; not new corpus validation',quadrature_max_abs_error=max_error,
        quadrature_points=quadrature,quadrature_check_points=quadrature*2,
        conditional_semantics='Supported training effects retained; zero-weight query/day/slopes integrated as new effects',
        zero_weight_source_days=[info['day_keys'][i] for i,w in enumerate(info['day_weights']) if w==0]))
    occupied={(u['lang'][i],u['haiku'][i],u['speech'][i]) for i in range(len(u['lang']))}
    write(out/'support.json',dict(fixed_coordinates=95,likelihood_rank=info['likelihood_rank'],nullity=info['nullity'],
        positive_weight_likelihood_rank=info['positive_weight_likelihood_rank'],
        no_candidate_rows=int(np.sum(~valid)),structural_zero_rows=int(np.sum(z['all_chance']==0)),structural_one_rows=int(np.sum(z['all_chance']==1)),
        unsupported_source_class_speech=[x for x in itertools.product(SOURCES,LABELS,['M','I']) if x not in occupied],
        observed_cells=[x for x in summaries if x['prediction']=='conditional' and x['scope']=='supported_cell'],
        prior_only_status_definition='No direct training likelihood in this observed cell; shared lower-order coefficients can still learn elsewhere, and unsupported random effects are integrated from their distribution',
        missing_cells_in_empirical_average=False))
    np.savez_compressed(out/'posterior_cell_draws.npz',**cells,group_weights=W,cell_chance=chance)
    return dict(contrast_rows=36,margin_rows=len(summaries),reported_diagnostics_pass=all(x['diagnostics_pass'] for x in summaries+contrasts),
                quadrature_max_abs_error=max_error)


def prior_predictive(data,z,u,info,out):
    rng=np.random.Generator(np.random.PCG64(2026103000));reports=[]
    for scale in (.5,1.,2.):
        values=[]
        for _ in range(256):
            tau=np.abs(rng.normal(0,.5,6));family=data['family']
            par={'alpha':rng.normal(0,1.5),'beta':rng.normal(size=94)*scale*np.array([.75 if f==0 else tau[f-1] for f in family]),
                 'sigma_query':abs(rng.normal()),'sigma_day':abs(rng.normal()),'sigma_slope':np.abs(rng.normal(0,.5,5)),
                 'z_query':rng.normal(size=data['Q']),'z_day':rng.normal(size=data['D']),'z_slope':rng.normal(size=(data['D'],5))}
            # No quadrature or fitted outcomes used for this prior simulation.
            chance=z['all_chance'];inside=info['inside'];p=chance.copy()
            eta=np.log(chance[inside]/(1-chance[inside]))+par['alpha']+info['X'][inside]@par['beta']
            eta+=par['sigma_query']*par['z_query'][z['all_query'][inside]]+par['sigma_day']*par['z_day'][z['all_day'][inside]]
            eta+=np.sum(z['all_slopes'][inside]*par['z_slope'][z['all_day'][inside]]*par['sigma_slope'],axis=1)
            p[inside]=expit(eta);values.append([np.mean(p),np.mean(p-chance),np.mean((p<.01)|(p>.99))])
        reports.append(dict(scale=scale,draws=256,mean_probability=credible(np.array(values)[:,0]),mean_lift=credible(np.array(values)[:,1]),extreme_share=credible(np.array(values)[:,2])))
    write(out,dict(status='prior_predictive_not_posterior',seed=2026103000,scales=reports,selection_rule='Fixed all scales; no outcome-dependent prior selection'))
