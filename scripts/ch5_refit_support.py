"""Observed-target likelihood support for day refits."""
import ch5_posterior as p
import numpy as np
from ch5_posterior_report import groups

def refit_support(name,variant):
    """Separate fixed summary-target counts from actual positive-weight likelihood."""
    d,z,info,U=p.adapter(name,variant)
    W,metadata,_,day_matrix,day_metadata,_=groups(name,z,info,U)
    weight=np.zeros(len(z['all_q']));weight[info['inside']]=d['likelihood_weight']
    masks=[W[j,info['cell']]>0 for j in range(len(metadata))]
    masks += [day_matrix.getrow(j).toarray().ravel()>0 for j in range(len(day_metadata))]
    result=[]
    for meta,mask in zip(metadata+day_metadata,masks):
        target_q=np.unique(z['all_q'][mask]);assert len(target_q)==meta['queries']
        interior=mask&info['inside'];active=interior&(weight>0);qq=np.unique(z['all_q'][active])
        row={k:v for k,v in meta.items() if k not in ['likelihood_rows','likelihood_support']}
        row.update(model=name,variant=variant,original_target_interior_rows=int(interior.sum()),
            positive_likelihood_rows=int(active.sum()),positive_likelihood_weight_sum=float(weight[active].sum()),
            positive_likelihood_queries=len(qq),positive_likelihood_source_days=len(np.unique(z['all_day'][active])),
            positive_likelihood_clips=len(set(U['clip'][qq])),
            refit_support='positive_weight_interior_rows' if active.any() else 'all_original_interior_rows_zero_weight' if interior.any() else 'structural_only_no_likelihood_information',
            interpretation='Original observed target retained; zero-weight queries/days have no direct refit likelihood. Shared fixed effects may still borrow information elsewhere. This is not a new observed cell or full corpus holdout.')
        result.append(row)
    return result


