"""Read frozen CLM payloads and item metadata only; never read judgment CSVs.
No model imports, no network, no performance estimation. Aggregate output only.
"""
import argparse,hashlib,json,math,subprocess
from collections import Counter
from pathlib import Path
import numpy as np

HOME=Path(__file__).resolve().parent
ROOT=HOME.parents[2]
QUESTION=('Is the speaker asking the listener (or a group that includes the speaker) to do '
          'something or not to do something, as in a command, request, urging, invitation, '
          'reminder or prohibition?')
OPTIONS=['yes, a directive','no, not a directive']
PAYLOADS=['audio.npz','face.npz','kin.npz','text.npz','clm.jsonl','texts.jsonl']

class CheckError(ValueError):pass
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ordered_hash(ids):return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
def unique(rows,key):
 ids=[r[key] for r in rows]
 if any(not isinstance(i,str) or not i for i in ids) or len(ids)!=len(set(ids)):
  raise CheckError('Duplicate, empty or invalid IDs')
 return ids
def check_scores(scores,texts,expected):
 sid=unique(scores,'utt_id');tid=unique(texts,'utt_id')
 if sid!=expected or tid!=expected:raise CheckError('Missing, extra or misordered IDs')
 empty=Counter()
 for s,t in zip(scores,texts):
  if set(s)!={'utt_id','clm_a','clm_b'}:raise CheckError('Unexpected score fields')
  for key,field in [('clm_a','subtitle_window'),('clm_b','transcript')]:
   v=s[key]
   if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:
    raise CheckError('Illegal CLM probability')
   if t[field] is not None and not isinstance(t[field],str):raise CheckError('Invalid text type')
   if not (t[field] or '').strip():
    empty[key]+=1
    if v!=0.5:raise CheckError('Frozen empty-input policy mismatch')
 return {'rows':len(scores),'empty_a':empty['clm_a'],'empty_b':empty['clm_b'],
         'ordered_ids_sha256':ordered_hash(expected),'finite_in_range':True}

def main(a):
 base=Path(a.item5_root).resolve();items_p=Path(a.items).resolve();out=Path(a.out).resolve()
 out.mkdir(parents=True,exist_ok=True)
 checksums={line.split()[-1].lstrip('*'):line.split()[0] for line in (base/'SHA256SUMS.txt').read_text().splitlines() if line.strip()}
 if set(checksums)!=set(PAYLOADS):raise CheckError('Unexpected frozen manifest membership')
 protected={}
 def pin(p,role,expected=None):
  h=digest(p)
  if expected and h!=expected:raise CheckError('Frozen input SHA-256 mismatch')
  protected[str(p)]={'bytes':p.stat().st_size,'sha256':h,'role':role}
 for name in PAYLOADS:pin(base/name,'frozen CLM/features',checksums[name])
 pin(base/'SHA256SUMS.txt','original manifest')
 pin(base/'clm.jsonl.provenance.json','original provenance','a19d42b9c60ffebe103cce162f5cc84ec17c67247069ad448350eac4562df4ce')
 pin(items_p,'sampling metadata, no judgments')
 env_path=base.parent/'env_clm.txt';pin(env_path,'historical CLM package inventory')
 packages={}
 for line in env_path.read_text().splitlines():
  if '==' in line:
   name,version=line.split('==',1)
   if name in {'contrastive-lm','numpy','torch','transformers','vllm'}:packages[name]=version
 for name in ['PREREGISTRATION.md','pipeline/analysis/item5/probe.py','pipeline/analysis/item5/clm_score.py',
              'pipeline/analysis/item5/feat_text.py','pipeline/analysis/draw_annotation.py','pipeline/analysis/make_annotation_kit.py']:
  pin(ROOT/name,'immutable registration/program')
 items=json.loads(items_p.read_text(encoding='utf-8'))['items'];unique(items,'item_id')
 main=[r for r in items if r['batch'] in [1,2]];ids=unique(main,'utt_id')
 if len(ids)!=240 or Counter(r['batch'] for r in items)!={1:50,2:190,3:40}:raise CheckError('Draw counts do not match fixed design')
 by_id={r['utt_id']:r for r in main}
 for r in items:
  if not 0<float(r['incl_prob'])<=1 or not math.isfinite(float(r['incl_prob'])):raise CheckError('Invalid inclusion probability')
  if not r['day'] or not r['clip']:raise CheckError('Missing cluster metadata')
  if r['batch']==3:
   m=by_id.get(r['utt_id'])
   if m is None or any(r[k]!=m[k] for k in ['day','clip','stratum','incl_prob']):raise CheckError('Repeat identity/metadata mismatch')
 unique([r for r in items if r['batch']==3],'utt_id')
 scores=[json.loads(l) for l in (base/'clm.jsonl').read_text().splitlines() if l.strip()]
 texts=[json.loads(l) for l in (base/'texts.jsonl').read_text().splitlines() if l.strip()]
 checks=check_scores(scores,texts,ids)
 shapes={}
 for name in ['text','audio','kin','face']:
  z=np.load(base/(name+'.npz'),allow_pickle=False);fid=z['ids'].astype(str).tolist()
  if len(fid)!=len(set(fid)) or fid!=ids:raise CheckError('Feature IDs are incomplete or misordered')
  shapes[name]={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
 prov=json.loads((base/'clm.jsonl.provenance.json').read_text())
 if prov['instructions']!=QUESTION or prov['options']!=OPTIONS or prov['items']!=240:raise CheckError('Frozen question/options mismatch')
 if any(prov[k]!=checks[k] for k in ('empty_a','empty_b')):raise CheckError('Provenance empty counts mismatch')
 if (prov['weights'],prov['backbone'])!=('Contrastive-LM/CLM-v0.1-8B','Qwen/Qwen3-8B'):raise CheckError('Frozen model mismatch')
 for p,x in protected.items():
  if digest(p)!=x['sha256']:raise CheckError('Source changed during read-only audit')
 summary={'status':'materials_ready_not_evaluated','clm_checks':checks,'provenance':prov,
  'counts':{'items':len(items),'unique_utterances':len(ids),'days':len(set(r['day'] for r in main)),
    'clips':len(set(r['clip'] for r in main)),'batch_counts':dict(Counter(r['batch'] for r in items)),
    'main_strata':dict(Counter(r['stratum'] for r in main))},'array_schemas':shapes,
  'all_payload_hashes_match':True,'all_id_orders_match':True,'protected_sources':protected,
  'question':QUESTION,'options':OPTIONS,'score_direction':'probability of first option',
  'read_boundaries':'No consultant answer files, raw per-item texts/scores printed, APIs or models accessed.',
  'historical_freeze_commit':'a85172b08fc1557e8cb681548bf560a8ab4dfc5f',
  'historical_clm_packages':packages,
  'head_at_audit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()}
 (out/'MATERIAL_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'payloads_verified':6,'rows':len(ids),'empty_a':checks['empty_a'],'empty_b':checks['empty_b'],'days':summary['counts']['days'],'all_id_orders_match':True}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--item5-root',required=True);p.add_argument('--items',required=True);p.add_argument('--out',required=True);main(p.parse_args())
