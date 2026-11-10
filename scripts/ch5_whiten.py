"""Exact Gaussian-block preconditioning from fixed design/chance, never outcomes.

Original Helmert coefficients and iid Gaussian random-effect priors are unchanged.
All directions, including the design null space, are retained. Query Householder
coordinates expose group means while preserving an identity covariance exactly.
"""
import os
import numpy as np
from scipy.special import expit


def slices(d):
    at=1+d['J'];r={'beta':slice(1,at)}
    for k,n in [('day',d['D']),('slope',d['D']*d['H']),('qmean',d['G']),('clip',d['CL']),('baseline',2)]:
        r[k]=slice(at,at+n);at+=n
    return r,at


def reflect(x,d):
    x=np.asarray(x);one=x.ndim==1
    if one:x=x[None,:]
    out=x.copy();group=np.asarray(d['query_group'])-1;v=np.asarray(d['householder_v'])
    for g in range(d['G']):
        q=np.flatnonzero(group==g);project=x[:,q]@v[q]
        out[:,q]-=2*project[:,None]*v[q]
    return out[0] if one else out


def block_design(d,sd_beta=None,sigma_day=1.,sigma_query=1.,sigma_slope=None,sigma_clip=1.):
    refs=np.where(d['interaction_order'][d['family']-1]==1,.75,np.where(d['interaction_order'][d['family']-1]==2,.5,.25))*d['prior_scale']
    sd_beta=refs if sd_beta is None else sd_beta
    sigma_slope=np.full(d['H'],.5) if sigma_slope is None else sigma_slope
    offsets,B=slices(d);A=np.zeros((d['NE'],B));r=np.arange(d['NE'])
    A[:,0]=1.5;A[:,offsets['beta']]=d['X_cell'][d['event_cell']-1]*sd_beta
    A[r,offsets['day'].start+d['event_day']-1]=sigma_day
    for h in range(d['H']):
        A[r,offsets['slope'].start+d['event_day']-1+d['D']*h]=sigma_slope[h]*d['slope_design'][d['event_slope']-1,h]
    qgroup=np.asarray(d['cell_query_group'])[d['event_cell']-1]
    n=np.asarray(d['query_group_size'])
    A[r,offsets['qmean'].start+qgroup]=sigma_query/np.sqrt(n[qgroup])
    if d['CL']:A[r,offsets['clip'].start+d['event_clip']-1]=sigma_clip
    if d['flexible_baseline']:A[:,offsets['baseline']]=d['event_basis']*np.array([.5,.25])
    return A


def prepare(d,z,info):
    first=np.unique(z['all_query'],return_index=True)[1]
    _,group=np.unique(z['all_design'][first],axis=0,return_inverse=True)
    G=int(group.max()+1);Q=d['Q'];sizes=np.bincount(group,minlength=G)
    v=np.zeros(Q);firsts=np.zeros(Q,bool)
    for g in range(G):
        q=np.flatnonzero(group==g);firsts[q[0]]=True
        if len(q)>1:
            u=np.full(len(q),-1/np.sqrt(len(q)));u[0]+=1;v[q]=u/np.linalg.norm(u)
    dev=np.zeros(Q,int);dev[~firsts]=np.arange(1,Q-G+1)
    cg=np.empty(d['C'],int)
    for c in range(d['C']):
        values=np.unique(group[z['all_query'][info['cell']==c]])
        assert len(values)==1
        cg[c]=values[0]
    d.update(G=G,query_group=group+1,query_group_size=sizes,query_deviation_index=dev,
        householder_v=v,cell_query_group=cg)
    # Exact sparse group projection for a vectorized equivalent Stan program.
    if os.environ.get('CH5_STAN_IMPLEMENTATION')=='whitened_vectorized':
        order=np.concatenate([np.flatnonzero(group==g) for g in range(G)])
        d.update(query_coordinate=np.where(dev>0,G+dev,group+1),
            householder_csr_values=v[order],householder_csr_columns=order+1,
            householder_csr_rows=np.r_[0,np.cumsum(sizes)]+1)
    A=block_design(d);d['B']=A.shape[1]
    c=expit(d['chance_logit'])
    weight=np.bincount(d['event']-1,weights=c*(1-c)*d['likelihood_weight'],minlength=d['NE'])
    precision=np.eye(d['B'])+A.T@(weight[:,None]*A)
    val,vec=np.linalg.eigh(precision)
    assert val.min()>.99
    T=vec/np.sqrt(val)
    whiten=T.T@precision@T
    error=float(np.max(np.abs(whiten-np.eye(d['B']))))
    assert error<2e-8,error
    d['block_transform']=T;d['transform_log_abs_det']=float(np.linalg.slogdet(T)[1])
    d['preconditioner_reference_condition']=float(val.max()/val.min())
    d['preconditioner_identity_error']=error
    return d


def encode(params,d):
    qtemp=reflect(params['z_query'],d);group=np.asarray(d['query_group'])-1;dev=np.asarray(d['query_deviation_index'])
    means=np.array([qtemp[np.flatnonzero((group==g)&(dev==0))[0]] for g in range(d['G'])])
    u=np.concatenate([[params['alpha']/1.5],params['z_beta'],params['z_day'],params['z_slope'].ravel(order='F'),
        means,params['z_clip'],params['baseline_delta']/np.array([.5,.25])])
    out={k:params[k] for k in ['tau','sigma_day','sigma_query','sigma_slope','sigma_clip']}
    out['z_block']=np.linalg.solve(d['block_transform'],u)
    out['z_query_deviation']=qtemp[dev>0]
    return out,u


def decode(theta,d):
    raw=theta['z_block']@np.asarray(d['block_transform']).T
    offsets,_=slices(d);M=len(raw)
    theta['alpha']=1.5*raw[:,0];theta['z_beta']=raw[:,offsets['beta']]
    theta['z_day']=raw[:,offsets['day']]
    theta['z_slope']=raw[:,offsets['slope']].reshape(M,d['H'],d['D']).transpose(0,2,1)
    theta['z_clip']=raw[:,offsets['clip']]
    theta['baseline_delta']=raw[:,offsets['baseline']]*np.array([.5,.25])
    means=raw[:,offsets['qmean']];dev=np.asarray(d['query_deviation_index']);group=np.asarray(d['query_group'])-1
    temp=np.empty((M,d['Q']));temp[:,dev==0]=means[:,group[dev==0]]
    temp[:,dev>0]=theta['z_query_deviation'][:,dev[dev>0]-1]
    theta['z_query']=reflect(temp,d)
    order=np.asarray(d['interaction_order'])[np.asarray(d['family'])-1]
    sd=np.where(order[None,:]==1,.75,theta['tau'][:,np.asarray(d['family'])-1])*d['prior_scale']
    theta['beta']=theta['z_beta']*sd
    return theta
