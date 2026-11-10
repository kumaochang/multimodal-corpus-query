"""In-memory registered-metric/report adapter. Used here only on synthetic data.
No file or network access. No fitting, threshold optimization or calibration fit.
"""
import math
from datetime import datetime,timedelta
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

PRIMARY=('marker','suffix_cue','suffix_nocue')
COMPS={'primary T+A+V - T':('T+A+V','T'),'T+A - T':('T+A','T'),'T+V - T':('T+V','T'),
       'T+K - T':('T+K','T'),'T+Fa - T':('T+Fa','T'),'T+V - F':('T+V','F'),
       'T+A+V - CLM_a':('T+A+V','CLM_a'),'T+A+V - CLM_b':('T+A+V','CLM_b')}
class NotReady(ValueError):pass

def unique_index(rows,key):
 out={}
 for r in rows:
  if not isinstance(r.get(key),str) or not r[key] or r[key] in out:raise NotReady('Duplicate or invalid identifiers')
  out[r[key]]=r
 return out

def instant(v):
 try:d=datetime.fromisoformat(v.replace('Z','+00:00'))
 except (ValueError,AttributeError):raise NotReady('Invalid timestamp')
 if d.tzinfo is None:raise NotReady('Timestamp requires timezone')
 return d

def require_authorization(receipt):
 if receipt.get('authorization')!='author-approved-registered-evaluation':raise NotReady('Formal execution not authorized')
 if receipt.get('scope_decision_adopted') is not True or receipt.get('minimum_positive_scope')!='all_batches_1_2':
  raise NotReady('Registered versus frozen-script minimum-positive scope must be resolved explicitly')

def completion_gate(items,answers,receipt):
 require_authorization(receipt)
 byitem=unique_index(items,'item_id');main=[r for r in items if r['batch'] in (1,2)]
 unique_index(main,'utt_id')
 if len(main)!=240 or len(byitem)!=280:raise NotReady('Incomplete frozen item design')
 expected={b:{r['item_id'] for r in items if r['batch']==b} for b in (1,2,3)}
 if [len(expected[b]) for b in (1,2,3)]!=[50,190,40]:raise NotReady('Invalid batch sizes')
 answer={};times={};freeze=instant(receipt['frozen_at'])
 if freeze!=instant('2026-10-17T11:29:13+08:00'):raise NotReady('Wrong frozen execution timestamp')
 for b in (1,2,3):
  if b not in answers:raise NotReady('A required batch is missing')
  idx=unique_index(answers[b],'item_id')
  if set(idx)!=expected[b]:raise NotReady('Missing, extra or wrong-batch answer IDs')
  times[b]=[]
  for k,row in idx.items():
   if str(row.get('answer')) not in ('1','2','3'):raise NotReady('Invalid judgment answer')
   t=instant(row['time']);times[b].append(t)
   if t<freeze:raise NotReady('Judgment precedes frozen features/scores')
   answer[k]=int(row['answer'])
 if sum(answer[k]==1 for k in expected[1])<3:raise NotReady('Batch 1 no-go: stop, no formal model evaluation')
 if min(times[2])<max(times[1]):raise NotReady('Batch 2 precedes batch 1 completion/go decision')
 b2end=instant(receipt['batch2_completed_at']);b3start=instant(receipt['batch3_started_at'])
 if receipt.get('completion_time_basis') not in ('author_confirmed','verified_csv_utc'):
  raise NotReady('Unverified completion-time basis')
 if b2end<max(times[2]) or b3start>min(times[3]) or b3start-b2end<timedelta(days=7):
  raise NotReady('Retest interval or completion chronology invalid')
 first={r['utt_id']:r for r in main};repeat=[r for r in items if r['batch']==3]
 unique_index(repeat,'utt_id')
 for r in repeat:
  if r['utt_id'] not in first or any(r[k]!=first[r['utt_id']][k] for k in ('day','clip','stratum','incl_prob')):
   raise NotReady('Repeat does not match frozen primary item')
 positives=sum(answer[r['item_id']]==1 for r in main)
 return {'answers':answer,'positives_all_batches':positives,'descriptive_only':positives<15,
         'answer3_count':sum(answer[r['item_id']]==3 for r in main),'batch3_n':40}

def prepare_primary(items,answer_map):
 main=[r for r in items if r['batch'] in (1,2)];out=[]
 for r in main:
  if r['item_id'] not in answer_map:raise NotReady('Missing judgment')
  if r['stratum'] in PRIMARY and answer_map[r['item_id']] in (1,2):
   p=float(r['incl_prob'])
   if not math.isfinite(p) or not 0<p<=1:raise NotReady('Invalid inclusion probability')
   out.append(dict(r,y=int(answer_map[r['item_id']]==1),weight=1/p))
 return out

def metrics(y,s,w):
 y=np.asarray(y);s=np.asarray(s,dtype=float);w=np.asarray(w,dtype=float)
 if not(len(y)==len(s)==len(w)):raise NotReady('Metric arrays misaligned')
 if not set(y.tolist())<={0,1}:raise NotReady('Binary metric labels required')
 if np.any(~np.isfinite(s)) or np.any((s<0)|(s>1)) or np.any(~np.isfinite(w)) or np.any(w<=0):
  raise NotReady('Illegal score or weight; never impute failure')
 status='empty' if not len(y) else 'single_class' if len(set(y))<2 else 'ok'
 return {'n':len(y),'positives':int(y.sum()),'weight_sum':float(w.sum()),'status':status,
         'auc':float(roc_auc_score(y,s,sample_weight=w)) if status=='ok' else None,
         'ap':float(average_precision_score(y,s,sample_weight=w)) if status=='ok' else None}

def paired_boot(y,s1,s0,w,days,rng,B=10000,descriptive=False):
 y=np.asarray(y);days=np.asarray(days);w=np.asarray(w);s1=np.asarray(s1);s0=np.asarray(s0)
 m1,m0=metrics(y,s1,w),metrics(y,s0,w)
 if m1['status']!='ok':return {'status':m1['status'],'delta':None,'ci95':None,'p_one_sided':None,'valid_resamples':0,'invalid_resamples':0,'requested_resamples':0},[]
 ud=np.array(sorted(set(days)));idx={d:np.where(days==d)[0] for d in ud};bs=[];sampled=[]
 for _ in range(B):
  # Exactly the frozen probe's choice/concatenate sequence. Both models share sel.
  draw=rng.choice(ud,size=len(ud),replace=True);sampled.append(draw.tolist())
  sel=np.concatenate([idx[d] for d in draw]);a=metrics(y[sel],s1[sel],w[sel])['auc'];b=metrics(y[sel],s0[sel],w[sel])['auc']
  if a is not None and b is not None:bs.append(a-b)
 return {'status':'ok' if bs else 'no_valid_resamples','delta':m1['auc']-m0['auc'],
         'ci95':np.percentile(bs,[2.5,97.5]).tolist() if bs else None,
         'p_one_sided':float((np.sum(np.array(bs)<=0)+1)/(len(bs)+1)) if bs and not descriptive else None,
         'valid_resamples':len(bs),'invalid_resamples':B-len(bs),'requested_resamples':B},sampled

def report(primary,scores,face_missing,descriptive,B=10000):
 unique_index(primary,'utt_id');y=np.array([r['y'] for r in primary]);w=np.array([r['weight'] for r in primary])
 days=np.array([r['day'] for r in primary]);strat=np.array([r['stratum'] for r in primary])
 if set(scores)!={'T','T+A','T+K','T+Fa','T+V','T+A+V','F','CLM_a','CLM_b'}:raise NotReady('Missing or unexpected model score columns')
 if len(face_missing)!=len(primary):raise NotReady('Face mask misaligned')
 if not set(np.asarray(face_missing).tolist())<={0,1}:raise NotReady('Illegal face missing flag')
 groups={'overall':np.ones(len(y),bool),**{'stratum='+st:strat==st for st in PRIMARY}}
 out={'descriptive_only':bool(descriptive),'metrics':{},'contrasts':{}};draws={}
 for scope,mask in groups.items():
  out['metrics'][scope]={k:metrics(y[mask],np.asarray(s)[mask],w[mask]) for k,s in scores.items()}
  support={'days':len(set(days[mask])),'clips':len({r['clip'] for r,keep in zip(primary,mask) if keep})}
  for row in out['metrics'][scope].values():row.update(support)
 rng=np.random.default_rng(20261017)
 def contrast(scope,mask,keys):
  for key in keys:
   a,b=COMPS[key];name=scope+': '+key
   out['contrasts'][name],draws[name]=paired_boot(y[mask],np.asarray(scores[a])[mask],np.asarray(scores[b])[mask],w[mask],days[mask],rng,B,descriptive)
   out['contrasts'][name].update({'n':int(mask.sum()),'positives':int(y[mask].sum()),
      'weight_sum':float(w[mask].sum()),'days':len(set(days[mask])),
      'clips':len({r['clip'] for r,keep in zip(primary,mask) if keep})})
 contrast('overall',groups['overall'],list(COMPS))
 contrast('suffix_only',strat!='marker',['primary T+A+V - T'])
 contrast('face_present',np.asarray(face_missing)==0,list(COMPS))
 # Existing observed-stratum ordering is retained; empty rows add no RNG calls.
 for st in sorted(set(strat)):
  contrast('stratum='+st,strat==st,['T+A+V - CLM_a','T+A+V - CLM_b'])
 for st in PRIMARY:
  if st not in set(strat):
   for key in ['T+A+V - CLM_a','T+A+V - CLM_b']:
    out['contrasts']['stratum='+st+': '+key]={'status':'empty','delta':None,'ci95':None,'p_one_sided':None,'valid_resamples':0,'invalid_resamples':0,'requested_resamples':0,'n':0,'positives':0,'weight_sum':0.,'days':0,'clips':0}
 return out,draws
