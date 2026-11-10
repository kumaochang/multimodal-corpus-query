"""Current registered CLM/probe entry; historical runner remains unchanged.
Only explicit authorization plus verified completion evidence can read answers.
"""
import argparse,csv,importlib.util,json,sys,warnings
from pathlib import Path
import numpy as np
import evaluation as e
from material_audit import check_scores
from contracts import (HOME,ROOT,Blocked,require,json_load,sha_file,canonical,ref,
 verified_path,artifact,authorize,create_json,safe_work,require_environment,decisions,verify_route_payloads)
from clm_inputs import verify

EXPECTED_ENV={'python':'3.10.22','numpy':'1.26.4','scipy':'1.13.1','scikit-learn':'1.5.2'}
MODELS=('T','T+A','T+K','T+Fa','T+V','T+A+V','F')

def checked_completion(receipt,mode='research'):
 authorize(receipt,'registered_clm_evaluation',mode)
 proof=artifact(receipt['completion'],'consultant_completion','batches_1_2_3',mode)
 require(proof.get('batch_counts')=={'1':50,'2':190,'3':40},'Incomplete completion evidence')
 require(proof.get('answer_files')==receipt.get('answer_files'),'Completion does not bind the same answer files')
 require(proof.get('items_sha256')==receipt.get('items_sha256'),'Completion belongs to different frozen items')
 require(proof.get('go_confirmed') is True,'Batch 1 go not attested')
 for k in ('batch2_completed_at','batch3_started_at','completion_time_basis'):
  require(proof.get(k)==receipt.get(k) and proof.get(k) is not None,'Completion time evidence differs')
 return proof

def completion(items,answers,receipt):
 # Adopted directly by the author on 2026-10-29. No new design-approval toggle.
 require(decisions()['minimum_positive_scope']=='all_batches_1_2','Adopted scope changed')
 old=dict(receipt,authorization='author-approved-registered-evaluation',scope_decision_adopted=True,
    minimum_positive_scope='all_batches_1_2',frozen_at='2026-10-17T11:29:13+08:00')
 return e.completion_gate(items,answers,old)

def compute(items,gate,inputs,out,mode='research',predictor=None,B=10000):
 require(mode=='synthetic' or (predictor is None and B==10000),'Research estimator/resampling cannot be replaced')
 out=Path(out);require(not out.exists(),'Preserve prior evaluation output')
 main=[r for r in items if r['batch'] in (1,2)];prim=e.prepare_primary(items,gate['answers']);ids=[r['utt_id'] for r in prim]
 require(len({r['day'] for r in prim})>=2,'Too few evaluable days; record not-estimable before modeling')
 base=inputs['item5/text.npz'].parent
 clm=[json.loads(l) for l in inputs['item5/clm.jsonl'].read_text(encoding='utf-8').splitlines()]
 texts=[json.loads(l) for l in inputs['item5/texts.jsonl'].read_text(encoding='utf-8').splitlines()]
 check_scores(clm,texts,[r['utt_id'] for r in main]);byid={r['utt_id']:r for r in clm}
 spec=importlib.util.spec_from_file_location('retained_registered_probe',inputs['pipeline/analysis/item5/probe.py'])
 legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 require(tuple(legacy.MODELS)==MODELS,'Frozen model family differs')
 y=np.array([r['y'] for r in prim]);w=np.array([r['weight'] for r in prim]);days=np.array([r['day'] for r in prim])
 S=legacy.load_sources(inputs['item5/text.npz'],inputs['item5/audio.npz'],inputs['item5/kin.npz'],inputs['item5/face.npz'],ids)
 scores={};folds={};warn={}
 for name,blocks in legacy.MODELS.items():
  src=[s for b in blocks for s in legacy.BLOCKS[b]]
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');scores[name],folds[name]=(predictor or legacy.oof_scores)(S,src,y,w,days)
  warn[name]=[r.category.__name__ for r in caught]
 scores['CLM_a']=np.array([byid[u]['clm_a'] for u in ids]);scores['CLM_b']=np.array([byid[u]['clm_b'] for u in ids])
 result,draws=e.report(prim,scores,S['face_missing'][:,0],gate['descriptive_only'],B=B)
 ans=gate['answers'];lab=[r for r in main if ans[r['item_id']] in (1,2)];descriptive={}
 for key in ('stratum','form'):
  descriptive[key]={}
  for value in sorted({r[key] for r in lab}):
   g=[r for r in lab if r[key]==value];ww=[1/r['incl_prob'] for r in g]
   descriptive[key][value]={'n':len(g),'positives':sum(ans[r['item_id']]==1 for r in g),
      'weighted_share':float(np.average([ans[r['item_id']]==1 for r in g],weights=ww))}
 first={r['utt_id']:r for r in main};repeat=[r for r in items if r['batch']==3]
 k=legacy.kappa([ans[r['item_id']] for r in repeat],[ans[first[r['utt_id']]['item_id']] for r in repeat])
 from threadpoolctl import threadpool_info
 from contracts import runtime
 result.update({'status':'registered_evaluation_recorded','mode':mode,'scope':'study3_s61_3',
   'completion_verified':True,'positives_all_batches':gate['positives_all_batches'],'primary_positives':int(y.sum()),
   'registration_minimum_positive_scope':'all_batches_1_2','answer3_excluded':gate['answer3_count'],
   'descriptive_denominators':descriptive,'kappa_n':len(repeat),'kappa_batch3':float(k) if np.isfinite(k) else None,
   'one_class_training_folds':folds,'execution':dict(runtime(),native_threadpools=threadpool_info(),
      bootstrap_seed=20261017,bootstrap_resamples=B,fit_warning_categories=warn,requires_fit_warning_review=any(warn.values())),
   'exclusions_by_stratum':{st:{'drawn':sum(r['stratum']==st for r in main),
     'cannot_tell':sum(r['stratum']==st and ans[r['item_id']]==3 for r in main),
     'binary_judgments':sum(r['stratum']==st and ans[r['item_id']] in (1,2) for r in main)} for st in sorted({r['stratum'] for r in main})}})
 out.mkdir(parents=True);create_json(out/'registered_metrics.json',result);create_json(out/'paired_day_draws.json',draws)
 np.savez_compressed(out/'oof_scores.npz',ids=np.array(ids),y=y,w=w,days=days,**scores)
 for outcome in ('judgment_primary','registered_secondary_models','clm_baselines','retest_kappa'):
  undefined=(outcome=='retest_kappa' and result['kappa_batch3'] is None) or (
      outcome=='judgment_primary' and result['contrasts']['overall: primary T+A+V - T']['delta'] is None) or (
      outcome=='clm_baselines' and all(result['metrics']['overall'][k]['auc'] is None for k in ('CLM_a','CLM_b')))
  create_json(out/(outcome+'.evidence.json'),{'artifact_kind':'registered_outcome','scope':'study3_s61_3',
      'mode':mode,'status':'completed','outcome':outcome,'result_state':'not_estimable_documented' if undefined else 'estimated',
      'reason':'Undefined statistic under the retained denominator/class support' if undefined else None,
      'support':{'primary_n':len(prim),'primary_positives':int(y.sum()),'repeat_n':len(repeat)},
      'registration_sha256':sha_file(ROOT/'PREREGISTRATION.md'),'material_audit_sha256':sha_file(HOME/'MATERIAL_AUDIT.json'),
      'payloads':[ref(out/'registered_metrics.json'),ref(out/'paired_day_draws.json'),ref(out/'oof_scores.npz')]})
 return result

def execute(receipt_path,enabled=False):
 require(enabled,'Registered execution disabled; no receipt or answers read')
 receipt=json_load(receipt_path);checked_completion(receipt)
 inputs=verify(json_load(verified_path(receipt['bindings'])))
 require(sha_file(inputs['annotation/items.json'])==receipt['items_sha256'],'Wrong sampling metadata')
 require_environment(EXPECTED_ENV)  # Before CSVs, fits and performance estimates.
 items=json_load(inputs['annotation/items.json'])['items'];answers={}
 require(set(receipt['answer_files'])=={'1','2','3'},'Missing batch files')
 for b,x in receipt['answer_files'].items():
  path=verified_path(x)
  with path.open(encoding='utf-8-sig',newline='') as f:answers[int(b)]=list(csv.DictReader(f))
 gate=completion(items,answers,receipt)
 target=safe_work(receipt['out']);compute(items,gate,inputs,target)
 create_json(target/'execution_provenance.json',{'receipt':ref(receipt_path),'completion':receipt['completion'],
    'authorization':receipt['authorization'],'inputs':{k:ref(p) for k,p in inputs.items()},
    'programs':{p.name:ref(p) for p in (HOME/'registered_entry.py',HOME/'evaluation.py',HOME/'contracts.py')}})
 print(json.dumps({'status':'registered_results_recorded','out':str(target)}))

def record_route(receipt_path,enabled=False):
 require(enabled,'Outcome recording disabled');r=json_load(receipt_path);authorize(r,'record_registered_route')
 support=verify_route_payloads(r['payloads'])
 obj={'artifact_kind':'registered_outcome','scope':'study3_s61_3','mode':'research','status':'completed',
   'outcome':'registered_route_comparison','result_state':'estimated' if support['arm_B_and_T_run'] else 'not_estimable_documented',
   'reason':None if support['arm_B_and_T_run'] else 'Registered no-seed branch: case table recorded; B/T not run',
   'support':support,'routes':['A','B','T'],'payloads':r['payloads'],
   'registration_sha256':sha_file(ROOT/'PREREGISTRATION.md'),'material_audit_sha256':sha_file(HOME/'MATERIAL_AUDIT.json')}
 create_json(safe_work(r['out']),obj)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt');p.add_argument('--execute-registered',action='store_true');p.add_argument('--check-environment',action='store_true');p.add_argument('--record-route',action='store_true');p.add_argument('--execute-record',action='store_true');a=p.parse_args()
 try:
  if a.check_environment:
   from contracts import runtime
   print(json.dumps({'observed':runtime(),'required':EXPECTED_ENV,'execution':False}))
  elif a.record_route:record_route(a.receipt,a.execute_record)
  else:execute(a.receipt,a.execute_registered)
 except (Blocked,e.NotReady) as ex:print(json.dumps({'status':'blocked','reason':str(ex)}));sys.exit(2)
