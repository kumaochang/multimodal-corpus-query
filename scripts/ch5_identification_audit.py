"""Design/support audit only, independent of posterior signs and no ranking."""
import ch5_posterior as p
import numpy as np
from ch5_posterior_report import csv_save


def fixed_rank(matrix):
    unique=np.unique(matrix,axis=0)
    singular=np.linalg.svd(unique,compute_uv=False)
    tolerance=max(unique.shape)*np.finfo(float).eps*singular[0]
    return int(np.sum(singular>tolerance)),float(tolerance),len(unique)


def main():
    p.os.environ['CH5_STAN_IMPLEMENTATION']='native_events'
    U=p.metadata();rows=[]
    for name in p.MODELS:
        z=p.frozen(name);base=np.column_stack([np.ones(int(z['N'])),z['X']])
        rank,tol,unique=fixed_rank(base)
        assert rank==p.MODELS[name]['likelihood_rank'],(name,rank,p.MODELS[name]['likelihood_rank'])
        for variant in p.VARIANTS+(['day_holdout','day_weighted'] if name=='H0_W1' else []):
            d,_,info,_=p.adapter(name,variant,U)
            active=d['likelihood_weight']>0
            matrix=base[active]
            if d['flexible_baseline']:matrix=np.column_stack([matrix,d['baseline_basis'][active]])
            r,t,u=fixed_rank(matrix) if d['flexible_baseline'] or not active.all() else (rank,tol,unique)
            grid=100 if name.startswith('HB') else 360 if name.startswith('HA') else 180
            rows.append(dict(model=name,variant=variant,input_sha256=p.MODELS[name]['model_data_sha256'],
                queries=int(z['Q']),source_day_clusters=int(z['D']),clips=p.MODELS[name]['clips'],
                potential_factor_grid_cells=grid,observed_factor_cells=d['C'],missing_factor_cells=grid-d['C'],
                metadata_factorial_rank=p.MODELS[name]['metadata_rank'],frozen_factorial_coordinates=int(z['J'])+1,
                active_fixed_coordinates=matrix.shape[1],active_fixed_rank=r,active_fixed_nullity=matrix.shape[1]-r,
                active_unique_fixed_design_rows=u,rank_tolerance=t,
                rows_with_positive_likelihood_weight=int(active.sum()),
                queries_with_positive_likelihood=int(len(np.unique(z['query'][active]))),
                source_days_with_positive_likelihood=int(len(np.unique(z['day'][active]))),
                structural_zero_rows=int(np.sum(z['all_chance']==0)),structural_one_rows=int(np.sum(z['all_chance']==1)),
                additional_clip_random_coordinates=d['CL'],day_slope_random_coordinates=d['D']*d['H'],
                inference_scope='fixed-part rank only; proper random-effect priors do not convert missing cells into data or establish reliable local class inference'))
    p.dump(p.OUT/'identification_audit.json',rows);csv_save(p.OUT/'identification_audit.csv',rows)
    text=['# Fixed-design identification and support','',
        'This audit uses frozen model data, fixed variant weights and design columns only. It does not use posterior signs. Factorial rank in per-fit identification.json refers to the frozen factorial block. This table additionally includes the two active baseline columns where applicable and removes zero-weight days for the two refit benchmarks. Clip/day/query effects are random, not additional fixed columns.',
        '', '| Model | Variant | Observed / potential factor cells | Active fixed rank / columns | Nullity | Positive-likelihood queries / all | Positive-likelihood source-days / all |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['model']} | {r['variant']} | {r['observed_factor_cells']} / {r['potential_factor_grid_cells']} | {r['active_fixed_rank']} / {r['active_fixed_coordinates']} | {r['active_fixed_nullity']} | {r['queries_with_positive_likelihood']} / {r['queries']} | {r['source_days_with_positive_likelihood']} / {r['source_day_clusters']} |")
    text+=['','Every fit remains conditional on its proper priors and fixed candidate/ranking resources. Observed cells and standardized empirical margins exclude missing factor cells. This exclusion does not remove prior dependence within aliased observed surfaces. No missing-cell extrapolation is presented as an empirical observation. Rank alone does not validate the likelihood, functional labels, random-effect assumptions or O7 qualification.']
    (p.OUT/'IDENTIFICATION.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Audited fixed-part rank/support for',len(rows),'adopted model/variant targets',flush=True)


if __name__=='__main__':main()
