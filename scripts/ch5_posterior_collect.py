"""Collect actual posterior statuses and compare fixed sensitivities, no fitting."""
import argparse
import itertools
from pathlib import Path
import ch5_posterior as p
import numpy as np
from ch5_posterior_report import csv_save


def select(name,variant):
    rows=[]
    for folder in (p.WORK/'fits'/name/variant).glob('attempt_*'):
        if not (folder/'sampling.json').exists():continue
        sampling=p.loadjson(folder/'sampling.json')
        diag=p.loadjson(folder/'diagnostics.json') if (folder/'diagnostics.json').exists() else {}
        post=p.loadjson(folder/'postprocessed.json') if (folder/'postprocessed.json').exists() else {}
        good=diag.get('parameters_pass',False) and post.get('summaries_diagnostics_pass',False)
        rows.append((good,sampling['attempt'],folder,sampling,diag,post))
    if not rows:return None
    return max(rows,key=lambda x:(x[0],x[1]))


def summary_path(name,variant,attempt):return p.summary_directory(name,variant,attempt)


def summarize_all():
    statuses=[];base_tables=[];contrast_tables=[];sensitivity=[];prior_cells=[];benchmarks=[];component_sensitivity=[]
    for name in p.MODELS:
        for variant in p.VARIANTS+(['day_holdout','day_weighted'] if name=='H0_W1' else []):
            chosen=select(name,variant)
            if not chosen:
                folders=list((p.WORK/'fits'/name/variant).glob('attempt_*'))
                running=[d for d in folders if (d/'running.json').exists() and not any((d/f).exists() for f in ['failed.json','interrupted.json','sampling.json'])]
                live=[d for d in running if p.psutil.pid_exists(p.loadjson(d/'running.json')['pid'])]
                statuses.append(dict(model=name,variant=variant,status='running' if live else 'not_completed',chains=0,retained_per_chain=0,
                    running_chains=4 if live else 0,accepted_completed_chains=0));continue
            good,attempt,folder,sampling,diag,post=chosen
            status=dict(model=name,variant=variant,attempt=attempt,status='completed' if good else 'sampled_diagnostics_or_postprocessing_incomplete',
                chains=sampling['chains'],retained_per_chain=sampling['retained_per_chain'],
                warmup_per_chain=sampling['settings']['iter_warmup'],
                warmup_adaptation='adaptive' if sampling['settings'].get('adapt_engaged',True) else 'fixed_metric',
                chain_ids=sampling['settings']['chain_ids'],seed=sampling['settings']['seed'],
                sampling_seconds=sampling['elapsed_seconds'],sampling_peak_RSS_bytes=sampling['peak_RSS_process_tree_bytes'],
                max_Rhat_parameters=diag.get('max_Rhat'),min_bulk_ESS_parameters=diag.get('min_bulk_ESS'),min_tail_ESS_parameters=diag.get('min_tail_ESS'),
                divergences=diag.get('divergences'),treedepth_saturated=diag.get('treedepth_saturated'),
                min_E_BFMI=min(diag['E_BFMI_per_chain']) if 'E_BFMI_per_chain' in diag else None,
                max_Rhat_summaries=post.get('max_Rhat_reported'),min_bulk_ESS_summaries=post.get('min_bulk_ESS_reported'),
                min_tail_ESS_summaries=post.get('min_tail_ESS_reported'),
                quadrature_error=post.get('quadrature_max_40_vs_80_abs_probability_error_first128draws'))
            statuses.append(status)
            tablepath=summary_path(name,variant,attempt)
            if variant=='base' and (tablepath/'summaries.json').exists():
                for r in p.loadjson(tablepath/'summaries.json'):
                    if r['scope']=='pooled_margin' and r['model_class']=='all':base_tables.append(dict(r,fit_diagnostics_pass=good,fit_status=status['status']))
                contrast_tables += [dict(r,fit_diagnostics_pass=good,fit_status=status['status']) for r in p.loadjson(tablepath/'contrasts.json')]
    # Sensitivity comparisons retain original empirical targets. Cross-model mean
    # shifts are descriptive, not arbitrarily paired cross-model posterior intervals.
    for name in p.MODELS:
        selected_base=select(name,'base')
        if selected_base is None:continue
        base_good,a,_,_,_,_=selected_base
        basepath=summary_path(name,'base',a)/'summaries.json'
        if not basepath.exists():continue
        base=p.loadjson(basepath)
        component_path=summary_path(name,'base',a)/'interaction_components.json'
        base_components={r['term']:r for r in p.loadjson(component_path)} if component_path.exists() else {}
        key=lambda r:tuple(r[k] for k in ['scope','route','source','model_class','speech_group','prediction'])
        by={key(r):r for r in base}
        for variant in p.VARIANTS[1:]+(['day_holdout','day_weighted'] if name=='H0_W1' else []):
            chosen=select(name,variant)
            if chosen is None:continue
            good,attempt,_,_,_,_=chosen;src=summary_path(name,variant,attempt)/'summaries.json'
            if not src.exists():continue
            component_path=summary_path(name,variant,attempt)/'interaction_components.json'
            if component_path.exists():
                for row in p.loadjson(component_path):
                    if row['term'] not in base_components:continue
                    original=base_components[row['term']]['observed_component_logit_RMS'];changed=row['observed_component_logit_RMS']
                    component_sensitivity.append(dict(model=name,variant=variant,term=row['term'],
                        coefficients=row['coefficients'],base_RMS=original,variant_RMS=changed,
                        absolute_mean_shift_over_base_SD=abs(changed['mean']-original['mean'])/original['sd'] if original['sd']>1e-14 else None,
                        diagnostics_pass=bool(good and base_good),base_fit_diagnostics_pass=base_good,note='Descriptive aliased logit-component magnitude; positive RMS is not evidence of interaction. Cross-fit shifts are not paired posterior differences.'))
            for r in p.loadjson(src):
                old=by[key(r)];sd=old['lift']['sd'];delta=r['lift']['mean']-old['lift']['mean']
                row=dict(model=name,variant=variant,attempt=attempt,scope=r['scope'],route=r['route'],
                    source=r['source'],model_class=r['model_class'],speech_group=r['speech_group'],prediction=r['prediction'],
                    queries=r['queries'],days=r['days'],clips=r['clips'],base_lift_mean=old['lift']['mean'],
                    variant_lift_mean=r['lift']['mean'],mean_shift=delta,mean_shift_over_base_SD=abs(delta)/sd if sd>1e-14 else None,
                    base_credible95=old['lift']['credible95'],variant_credible95=r['lift']['credible95'],
                    diagnostics_pass=bool(good and base_good),base_fit_diagnostics_pass=base_good,status=r['status'])
                if r['scope']=='supported_cell' and variant in ['prior_half','prior_double']:prior_cells.append(row)
                if r['scope'] in ['pooled_margin','speech_margin'] and r['model_class'] in ['all','Explaining','remainder','Reporting']:
                    sensitivity.append(row)
                    if variant.startswith('day_'):benchmarks.append(row)
    p.dump(p.OUT/'model_status.json',statuses);csv_save(p.OUT/'model_status.csv',statuses)
    p.dump(p.OUT/'pooled_predictions.json',base_tables);csv_save(p.OUT/'pooled_predictions.csv',base_tables)
    p.dump(p.OUT/'base_contrasts.json',contrast_tables);csv_save(p.OUT/'base_contrasts.csv',contrast_tables)
    p.dump(p.OUT/'sensitivity_comparison.json',sensitivity);csv_save(p.OUT/'sensitivity_comparison.csv',sensitivity)
    p.dump(p.OUT/'prior_dependence_observed_cells.json',prior_cells);csv_save(p.OUT/'prior_dependence_observed_cells.csv',prior_cells)
    p.dump(p.OUT/'day_refit_comparison.json',benchmarks);csv_save(p.OUT/'day_refit_comparison.csv',benchmarks)
    p.dump(p.OUT/'interaction_component_sensitivity.json',component_sensitivity);csv_save(p.OUT/'interaction_component_sensitivity.csv',component_sensitivity)
    complete=sum(s['status']=='completed' for s in statuses)
    report=['# Posterior continuation: actual execution status','',
        '**This file reports current state. Only completed, diagnosed posterior fits count as completed.**','',
        f'Completed model/variant fits: {complete}/{len(statuses)}. Historical zero-chain files remain unchanged.',
        '', '| Model | Variant | Status | Chains × retained | Seconds | Peak sampling RSS MiB | max parameter Rhat | min bulk ESS | Divergences |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for s in statuses:
        report.append(f"| {s['model']} | {s['variant']} | {s['status']} | {s['chains']} × {s['retained_per_chain']} | {s.get('sampling_seconds',0):.1f} | {s.get('sampling_peak_RSS_bytes',0)/1048576:.1f} | {s.get('max_Rhat_parameters')} | {s.get('min_bulk_ESS_parameters')} | {s.get('divergences')} |")
    report+=['','Model-standardized probability and lift use fixed observed-query weights. Conditional observed random effects and population-average integration are separate. Missing cells are excluded; posterior intervals are model/prior-conditional credible intervals, not the earlier 24-family confidence bands. No P@3/MAP likelihood or reranking is introduced.','',
        'Source/class local summaries remain descriptive under O7. Convergence does not identify rank-deficient surfaces; positive continuous-prior variance parameters do not demonstrate interaction.','',
        'Day-level conditional latent-effect PSIS-LOO has explicit high-k limitations. Deleted-day target summaries are composition diagnostics, not refitted cross-validation. Check the statuses of the fixed whole-day holdout and shared-day weighted refit; only completed, diagnosed benchmarks supply actual refit comparisons and costs. No full reliable leave-day-out validation is claimed from conditional PSIS.','',
        'Detailed supported surfaces/PPC/influence are in summaries/. Raw chains and cell draws remain ignored. Runtime, source code, every attempt and interrupted HA files are preserved separately.']
    (p.OUT/'CURRENT_STATUS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Collected actual statuses',complete,'/',len(statuses),flush=True)
    return statuses


if __name__=='__main__':summarize_all()
