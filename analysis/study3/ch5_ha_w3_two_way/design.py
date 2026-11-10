"""Pinned HA W3 numerical adapter. No models, network, ranking or sampling."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TERMS = ['R','L','K','RL','RK','LK','RLK','G','LG','KG','RG','RKG']
KEEP = [t for t in TERMS if len(t)<=2]
SOURCES = ['ami','pwn','tao','tay','trv','tsu']
LABELS = ['Comforting','Celebrating','Explaining','Urging','Reporting']
ROUTES = ['whisper','mfcc86','prosody','xlmr_sub','tfidf_sub','kin14']
PAIRS = [(0,2),(3,4),(5,0)]
JOBS = ['base','prior_half','prior_double','day_weighted','day_holdout']
SEEDS = dict(zip(JOBS,range(2026103011,2026103016)))


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(clean(obj),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def helmert(n):
    a=np.zeros((n,n-1))
    for j in range(1,n):a[:j,j-1]=1/np.sqrt(j*(j+1));a[j,j-1]=-j/np.sqrt(j*(j+1))
    return a


def factorial(levels,terms):
    sizes={'R':6,'L':6,'K':5,'G':2};out=[]
    for term in terms:
        x=np.ones(1)
        for factor in term:x=np.kron(x,helmert(sizes[factor])[levels[factor]])
        out.extend(x)
    return np.array(out)


def column_map(old_family):
    old_family=np.asarray(old_family,int)
    require(set(old_family)==set(range(1,13)),'Original family set changed')
    selected=np.array([len(TERMS[f-1])<=2 for f in old_family])
    interactions=[t for t in KEEP if len(t)==2]
    new_family=np.array([0 if len(TERMS[f-1])==1 else interactions.index(TERMS[f-1])+1
                         for f in old_family[selected]])
    mapping=[{'old_column_0':int(j),'old_family_1':int(old_family[j]),
              'new_column_0':i,'term':TERMS[old_family[j]-1],
              'new_interaction_family_1_or_main_0':int(new_family[i])}
             for i,j in enumerate(np.flatnonzero(selected))]
    require(len(mapping)==94 and len(interactions)==6,'Wrong two-way J/F')
    return selected,new_family,mapping


def day_weights(keys,job):
    keys=[tuple(k) for k in keys]
    require(len(keys)==len(set(keys)),'Duplicate source-day keys')
    weights=np.ones(len(keys),int);held=[]
    if job=='day_weighted':
        weights[:]=0;rng=np.random.Generator(np.random.PCG64(2026103001))
        for source in SOURCES:
            ids=np.array([i for i,k in enumerate(keys) if k[0]==source],int)
            require(len(ids)>0,'Missing source in weighted plan')
            np.add.at(weights,rng.choice(ids,len(ids),replace=True),1)
    elif job=='day_holdout':
        for source in SOURCES:
            ids=[i for i,k in enumerate(keys) if k[0]==source]
            require(bool(ids),'Missing source in holdout plan')
            chosen=min(ids,key=lambda i:keys[i][1]);weights[chosen]=0;held.append(chosen)
    return weights,held


def load(formal_root,frozen_root):
    contract=json.loads((HERE/'SOURCE_CONTRACT.json').read_text(encoding='utf-8'))
    paths={}
    for role,x in contract['input_roles'].items():
        p=Path(formal_root if x['root']=='formal' else frozen_root)/x['relative_path']
        require(p.is_file() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],
                'Frozen input mismatch: '+role)
        paths[role]=p
    prefit=json.loads(paths['prefit'].read_text(encoding='utf-8'))
    selected=[x for x in prefit['models'] if x['model']=='HA' and x['pool']=='W3']
    require(len(selected)==1 and selected[0]['model_data_sha256']==contract['input_roles']['model']['sha256']
            and selected[0]['formula'].split('+')==TERMS,'Original formal prefit/model lineage differs')
    with np.load(paths['model'],allow_pickle=False) as archive:z={k:archive[k] for k in archive.files}
    with np.load(paths['ids'],allow_pickle=False) as archive:ids=archive['ids'].astype(str)
    require(len(np.unique(ids))==len(ids),'Duplicate frozen global IDs')
    needed=('lang','day','clip','haiku','lid_answer','mf_class')
    with paths['metadata'].open(encoding='utf-8',newline='') as stream:
        metadata={}
        for row in csv.DictReader(stream,delimiter='\t'):
            require(row['unit_id'] not in metadata,'Duplicate metadata ID')
            metadata[row['unit_id']]={k:row[k] for k in needed}
    q=z['all_q'].astype(int);require(q.min()>=0 and q.max()<len(ids),'Global query index out of bounds')
    u={k:np.array([metadata[ids[i]][k] for i in q]) for k in needed}
    u['unit_id']=ids[q];u['route']=np.arange(len(q))%6
    u['speech']=np.where(u['lid_answer']=='mandarin','M',
                        np.where((u['lid_answer']=='not_chinese')&(u['mf_class']!='mandarin'),'I','other'))
    require(set(u['speech'])=={'M','I'},'HA speech population changed')
    with paths['overlap'].open(encoding='utf-8',newline='') as stream:
        overlap=[x for x in csv.DictReader(stream) if x['set']=='six_common' and
                 x['candidate_policy']=='union_MI' and x['pool']=='W3' and float(x['restricted_weight'])>0]
    require(bool(overlap) and np.isclose(sum(float(x['restricted_weight']) for x in overlap),1),
            'Original restricted-overlap weights missing')
    return z,u,overlap,{k:{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in paths.items()}


def adapt(z,u,job='base',strict=True):
    require(job in JOBS,'Unknown logical job')
    chance=z['all_chance'];inside=np.isfinite(chance)&(chance>0)&(chance<1)
    structural=np.isfinite(chance)&~inside
    require(int(z['N'])==int(inside.sum()),'Frozen likelihood denominator differs')
    require(len(np.unique(z['all_q']))==int(z['Q']),'Different query indices duplicate a global unit')
    require(np.array_equal(z['all_hit'][structural],chance[structural]),'Structural hit differs from chance')
    for key,original in [('X',z['all_design'][inside]),('hit',z['all_hit'][inside]),
                          ('query',z['all_query'][inside]+1),('day',z['all_day'][inside]+1),
                          ('slopes',z['all_slopes'][inside])]:
        require(np.array_equal(z[key],original),'Frozen interior adapter differs: '+key)
    # Use the frozen offset verbatim. This independent arithmetic check permits
    # only library-level last-bit differences, never a newly computed offset.
    require(np.allclose(z['chance_logit'],np.log(chance[inside]/(1-chance[inside])),rtol=0,atol=1e-14),'Offset differs')
    require(np.all(z['likelihood_weight']==1),'Frozen base weights changed')
    selected,family,mapping=column_map(z['family']);X=z['all_design'][:,selected]
    cells,cell=np.unique(X,axis=0,return_inverse=True)
    require(np.array_equal(cells[cell],X),'Compressed fixed design not equivalent')
    D=int(z['D']);keys=[]
    for day in range(D):
        mask=z['all_day']==day
        pairs=set(zip(u['lang'][mask],u['day'][mask]));require(len(pairs)==1,'Day identity not source-nested')
        keys.append(next(iter(pairs)))
    weights,held=day_weights(keys,job)
    if strict:
        frozen_mapping=json.loads((HERE/'COLUMN_MAPPING.json').read_text(encoding='utf-8'))
        require(mapping==frozen_mapping['mapping'],'Retained column/family map differs from frozen adapter plan')
        day_plan=json.loads((HERE/'DAY_PLAN.json').read_text(encoding='utf-8'))
        require([list(k) for k in keys]==day_plan['source_day_order'],'Frozen day-plan order changed')
        if job=='day_weighted':require(weights.tolist()==day_plan['day_weighted_multiplicity'],'RNG day weights differ from frozen design plan')
        if job=='day_holdout':require(weights.tolist()==day_plan['holdout_multiplicity'] and held==day_plan['holdout_day_indices_0'],'Holdout plan differs')
    for query in np.unique(z['all_query']):
        m=z['all_query']==query
        require(m.sum()==6 and len(set(z['all_day'][m]))==1 and len(set(u['unit_id'][m]))==1,'Unpaired query identity')
    fixed=np.column_stack([np.ones(inside.sum()),X[inside]])
    rank=int(np.linalg.matrix_rank(np.unique(fixed,axis=0)))
    active_rank=int(np.linalg.matrix_rank(np.unique(fixed[weights[z['all_day'][inside]]>0],axis=0)))
    if strict:
        require((len(chance),int(inside.sum()),int(z['Q']),D,int(z['H']),rank)==(16518,16512,2753,72,5,90),
                'Adopted shape/rank changed')
        require(np.sum(chance==0)==6 and np.sum(chance==1)==0 and np.isfinite(chance).all(),'Structural population changed')
        require(all(sum(k[0]==s for k in keys)==12 for s in SOURCES),'Not twelve source-days each')
        # Independently reconstruct every occupied full factorial row, not only a subset.
        _,first=np.unique(z['all_design'],axis=0,return_index=True)
        for i in first:
            levels={'R':int(u['route'][i]),'L':SOURCES.index(u['lang'][i]),
                    'K':LABELS.index(u['haiku'][i]),'G':0 if u['speech'][i]=='M' else 1}
            require(np.allclose(factorial(levels,TERMS),z['all_design'][i],atol=2e-15,rtol=0),'Original Helmert mapping differs')
            require(np.allclose(factorial(levels,KEEP),X[i],atol=2e-15,rtol=0),'Retained column order differs')
    data={'N':int(inside.sum()),'J':94,'F':6,'C':len(cells),'D':D,'Q':int(z['Q']),'H':int(z['H']),
          'family':family,'X_cell':cells,'cell':cell[inside]+1,'day':z['day'],'query':z['query'],
          'slopes':z['slopes'],'chance_offset':z['chance_logit'],'hit':z['hit'].astype(int),
          'weight':weights[z['all_day'][inside]],'prior_scale':.5 if job=='prior_half' else 2. if job=='prior_double' else 1.}
    info={'inside':inside,'cell':cell,'X':X,'day_weights':weights,'heldout_days_0':held,
          'day_keys':keys,'column_mapping':mapping,'likelihood_rank':rank,'positive_weight_likelihood_rank':active_rank,
          'supported_day':weights>0,'fixed_coordinates':95,'nullity':95-rank}
    return data,info


def target_vectors(z,u,overlap):
    """Exact old 36-coordinate target definitions as linear probability margins."""
    n=len(z['all_q']);valid=np.isfinite(z['all_chance']);vectors=[];identities=[]
    def average(mask):
        m=mask&valid;require(m.any(),'Empty required target');return m.astype(float)/m.sum()
    def difference(a,b,mask):return average((u['route']==a)&mask)-average((u['route']==b)&mask)
    allrows=np.ones(n,bool)
    for a,b in PAIRS:
        pair=ROUTES[a]+'-'+ROUTES[b]
        for group in ('all','Explaining','Reporting'):
            vectors.append(difference(a,b,allrows if group=='all' else u['haiku']==group))
            identities.append((pair,group,'empirical_query_weighted'))
        vectors.append(difference(a,b,u['haiku']=='Explaining')-difference(a,b,u['haiku']=='Reporting'))
        identities.append((pair,'Explaining_minus_Reporting_DID','empirical_query_weighted'))
        vectors.append(difference(a,b,u['speech']=='M')-difference(a,b,u['speech']=='I'))
        identities.append((pair,'M_minus_I_DID','unbalanced_empirical_groups_composition_sensitive'))
        v=np.zeros(n)
        for row in overlap:
            mask=(u['lang']==row['source_community'])&(u['haiku']==row['model_class'])
            v+=float(row['restricted_weight'])*(difference(a,b,mask&(u['speech']=='M'))-difference(a,b,mask&(u['speech']=='I')))
        vectors.append(v);identities.append((pair,'M_minus_I_DID','six_common_restricted_overlap_four_cells_not_family_24_target'))
    require(len(identities)==18 and len(set(identities))==18,'Target contract changed')
    return np.array(vectors),identities


def eta(data,params):
    beta=np.asarray(params['z_beta'])*data['prior_scale']*np.array([.75 if f==0 else params['tau'][f-1] for f in data['family']])
    return (data['chance_offset']+params['alpha']+(data['X_cell']@beta)[data['cell']-1]
            +params['sigma_query']*np.asarray(params['z_query'])[data['query']-1]
            +params['sigma_day']*np.asarray(params['z_day'])[data['day']-1]
            +np.sum(data['slopes']*(np.asarray(params['z_slope'])*params['sigma_slope'])[data['day']-1],axis=1))


def log_likelihood(data,params):
    e=eta(data,params)
    return float(np.dot(data['weight'],data['hit']*e-np.logaddexp(0,e)))
