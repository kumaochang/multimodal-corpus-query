"""Future registered execution entry. Disabled without explicit receipt and flag.

This preparation task never invokes this entry on real judgments. All filenames
are explicit; partial answer directories are never discovered automatically.
"""
import argparse,csv,importlib.util,json,sys,warnings,platform
from pathlib import Path
import numpy as np
from evaluation import NotReady,require_authorization,completion_gate,prepare_primary,report,metrics
from material_audit import digest,check_scores
HOME=Path(__file__).resolve().parent
ROOT=HOME.parents[2]

def execute(a):
 if not a.execute_registered:raise NotReady('Preparation only: --execute-registered and approved receipt required')
 receipt=json.loads(Path(a.receipt).read_text(encoding='utf-8'))
 require_authorization(receipt)  # MUST precede any judgment file open.
 audit_path=HOME/'MATERIAL_AUDIT.json'
 if receipt.get('material_audit_sha256')!=digest(audit_path):raise NotReady('Material-audit version not approved')
 audit=json.loads(audit_path.read_text())
 for name,x in audit['protected_sources'].items():
  if digest(name)!=x['sha256']:raise NotReady('Pinned input/program changed; no fallback')
 frozen=[Path(p) for p,x in audit['protected_sources'].items() if x['role']=='frozen CLM/features']
 base=next(p.parent for p in frozen if p.name=='clm.jsonl')
 items_path=next(Path(p) for p,x in audit['protected_sources'].items() if x['role']=='sampling metadata, no judgments')
 items=json.loads(items_path.read_text(encoding='utf-8'))['items']
 files=receipt.get('answer_files',{})
 if set(files)!={'1','2','3'}:raise NotReady('Three explicit, hashed batch files required')
 answers={}
 for b,x in files.items():
  path=Path(x['path'])
  if not x.get('sha256') or digest(path)!=x['sha256']:raise NotReady('Judgment file not pinned')
  with path.open(encoding='utf-8-sig',newline='') as f:answers[int(b)]=list(csv.DictReader(f))
 gate=completion_gate(items,answers,receipt)
 main=[r for r in items if r['batch'] in (1,2)]
 clm=[json.loads(l) for l in (base/'clm.jsonl').read_text().splitlines() if l.strip()]
 texts=[json.loads(l) for l in (base/'texts.jsonl').read_text().splitlines() if l.strip()]
 check_scores(clm,texts,[r['utt_id'] for r in main]);clm_byid={r['utt_id']:r for r in clm}
 primary=prepare_primary(items,gate['answers']);ids=[r['utt_id'] for r in primary]
 target=Path(a.out).resolve()
 if not target.is_relative_to((HOME/'work').resolve()):raise NotReady('Formal numerical outputs must remain under ignored work/')
 if target.exists():raise NotReady('Preserve prior evaluation output; use a fresh directory')
 legacy_path=ROOT/'pipeline/analysis/item5/probe.py'
 spec=importlib.util.spec_from_file_location('frozen_item5_probe',legacy_path);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 y=np.array([r['y'] for r in primary]);w=np.array([r['weight'] for r in primary]);days=np.array([r['day'] for r in primary])
 if len(set(days))<2:raise NotReady('Too few eligible days for registered leave-one-day-out model; report not-estimable separately')
 S=legacy.load_sources(base/'text.npz',base/'audio.npz',base/'kin.npz',base/'face.npz',ids)
 scores={};oneclass={};fit_warnings={}
 for name,blocks in legacy.MODELS.items():
  src=[s for b in blocks for s in legacy.BLOCKS[b]]
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   scores[name],oneclass[name]=legacy.oof_scores(S,src,y,w,days)
  fit_warnings[name]=[x.category.__name__ for x in caught]
 scores['CLM_a']=np.array([clm_byid[u]['clm_a'] for u in ids]);scores['CLM_b']=np.array([clm_byid[u]['clm_b'] for u in ids])
 result,draws=report(primary,scores,S['face_missing'][:,0],gate['descriptive_only'])
 # Registered descriptive denominators and repeat reliability, not a new model.
 answer=gate['answers'];lab=[r for r in main if answer[r['item_id']] in (1,2)]
 desc={}
 for key in ('stratum','form'):
  desc[key]={}
  for value in sorted({r[key] for r in lab}):
   g=[r for r in lab if r[key]==value];ww=[1/float(r['incl_prob']) for r in g]
   desc[key][value]={'n':len(g),'positives':sum(answer[r['item_id']]==1 for r in g),
                     'weighted_share':float(np.average([answer[r['item_id']]==1 for r in g],weights=ww))}
 first={r['utt_id']:r for r in main};repeats=[r for r in items if r['batch']==3]
 ka=legacy.kappa([answer[r['item_id']] for r in repeats],[answer[first[r['utt_id']]['item_id']] for r in repeats])
 result.update({'status':'registered_evaluation_recorded','answer3_excluded':gate['answer3_count'],
                'positives_all_batches':gate['positives_all_batches'],'primary_positives':int(y.sum()),
                'descriptive_denominators':desc,'kappa_batch3':float(ka) if np.isfinite(ka) else None,
                'kappa_n':len(repeats),'one_class_training_folds':oneclass,
                'evaluation_days':len(set(days)),'frozen_probe_sha256':digest(legacy_path),
                'registration_minimum_positive_scope':'all_batches_1_2',
                'receipt_sha256':digest(a.receipt),'material_audit_sha256':digest(audit_path)})
 result['exclusions_by_stratum']={st:{'drawn':sum(r['stratum']==st for r in main),
    'cannot_tell':sum(r['stratum']==st and answer[r['item_id']]==3 for r in main),
    'binary_judgments':sum(r['stratum']==st and answer[r['item_id']] in (1,2) for r in main)}
    for st in sorted({r['stratum'] for r in main})}
 import scipy,sklearn
 from threadpoolctl import threadpool_info
 result['execution']={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,
    'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'native_threadpools':threadpool_info(),
    'fit_warning_categories':fit_warnings,'requires_fit_warning_review':any(fit_warnings.values()),
    'bootstrap_seed':20261017,'bootstrap_resamples':10000,
    'adapter_sha256':{p.name:digest(p) for p in [HOME/'formal_runner.py',HOME/'evaluation.py',HOME/'material_audit.py']}}
 target.mkdir(parents=True)
 (target/'registered_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 (target/'paired_day_draws.json').write_text(json.dumps(draws,sort_keys=True)+'\n')
 np.savez_compressed(target/'oof_scores.npz',ids=np.array(ids),y=y,w=w,days=days,**scores)
 print(json.dumps({'status':'completed','out':str(target),'no_raw_answers_printed':True}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute-registered',action='store_true');p.add_argument('--receipt');p.add_argument('--out');a=p.parse_args()
 try:execute(a)
 except (NotReady,KeyError,TypeError,FileNotFoundError) as e:
  print(json.dumps({'status':'blocked_not_evaluated','reason':str(e)}));sys.exit(2)
