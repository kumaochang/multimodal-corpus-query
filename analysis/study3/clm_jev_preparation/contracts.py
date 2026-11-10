"""Shared, fail-closed execution evidence. No model, answer or network discovery."""
import hashlib,json,platform
from pathlib import Path
from datetime import datetime,timezone
from importlib import metadata

HOME=Path(__file__).resolve().parent
ROOT=HOME.parents[2]
MODEL='jev-1.14.0'

class Blocked(ValueError):pass

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def fingerprint(x):return sha_bytes(canonical(x))
def now():return datetime.now(timezone.utc).isoformat()
def require(ok,reason):
 if not ok:raise Blocked(reason)
def json_load(path):
 def pairs(xs):
  out={}
  for k,v in xs:
   require(k not in out,'Duplicate JSON key');out[k]=v
  return out
 def bad(_):raise Blocked('Nonfinite JSON constant')
 return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=bad)
def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha_file(path)}
def verified_path(x):
 require(isinstance(x,dict) and isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str) and len(x['sha256'])==64,'Missing file evidence/hash')
 p=Path(x['path']);require(p.is_file(),'Required evidence file missing')
 require(sha_file(p)==x['sha256'],'Evidence hash mismatch; no fallback')
 return p
def artifact(x,kind,scope,mode='research'):
 obj=json_load(verified_path(x))
 require(isinstance(obj,dict),'Evidence must be a JSON object')
 require(obj.get('artifact_kind')==kind and obj.get('scope')==scope,'Wrong evidence kind/scope')
 require(obj.get('mode')==mode,'Mock/synthetic evidence cannot authorize research')
 require(obj.get('status')=='completed','Evidence not complete')
 return obj
def authorize(receipt,operation,mode='research'):
 require(receipt.get('operation')==operation,'Wrong execution operation')
 request={k:v for k,v in receipt.items() if k!='authorization'}
 proof=artifact(receipt.get('authorization'),'execution_authorization',operation,mode)
 require(proof.get('approved') is True and proof.get('request_sha256')==fingerprint(request),'Authorization not bound to this execution request')
 require(bool(proof.get('authorization_basis')),'Actual author execution authorization must be recorded')
 try:stamp=datetime.fromisoformat(proof['approved_at'].replace('Z','+00:00'))
 except (KeyError,TypeError,ValueError):raise Blocked('Missing approval time')
 require(stamp.tzinfo is not None,'Approval time needs timezone')
 return proof
def safe_work(path,root=HOME/'work'):
 p=Path(path).resolve();r=Path(root).resolve()
 require(p.is_relative_to(r) and p!=r,'Outputs must stay in the specified controlled work root')
 return p
def create_json(path,obj):
 """Create new artifact exclusively, never replace a prior source/result."""
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(canonical(obj)+b'\n')
 return ref(p)
def runtime():
 out={'python':platform.python_version()}
 for p in ('numpy','scipy','scikit-learn','spacy','en_core_web_sm'):
  try:out[p]=metadata.version(p)
  except metadata.PackageNotFoundError:out[p]=None
 return out
def require_environment(expected,actual=None):
 actual=actual or runtime()
 require(all(actual.get(k)==v for k,v in expected.items()),'Execution dependencies differ from the fixed environment; do not upgrade silently')
 return actual

def decisions():return json_load(HOME/'DECISIONS.json')

def study2_gate(receipt,mode='research'):
 proof=authorize(receipt,'jev_study2',mode)
 require(receipt.get('model')==MODEL,'Pinned Jev version differs')
 require(receipt.get('input_condition') in ('paiwan_original','english_parallel'),'Unknown Study 2 input condition')
 if receipt['input_condition']=='english_parallel':
  require(proof.get('english_parallel_fixed_before_results') is True,'Optional English condition must be fixed before viewing Jev results')
 # No consultant or Study 3 field is required here.
 aligned=artifact(receipt.get('alignment'),'study2_alignment','study2_sent',mode)
 labels=artifact(receipt.get('labels'),'study2_rule_labels','study2_sent',mode)
 require(labels.get('alignment_sha256')==receipt['alignment']['sha256'],'Labels belong to another alignment')
 require(labels.get('primary_reconciled') is True,'Full primary summary has not reconciled')
 require(labels.get('rule_sha256')==receipt.get('rule_sha256'),'Rule version not bound')
 require(labels['rule_sha256']==decisions()['study2_rule_sha256'],'Rule differs from the adopted source version')
 if mode=='research':
  from study2_adapter import PRIMARY
  require(labels.get('full_primary')==PRIMARY and aligned.get('n_source_units')==PRIMARY['sentences'],'Not the reconciled full primary corpus')
 for x in (aligned['rows'],labels['rows']):verified_path(x)
 require(receipt.get('request_spec_sha256')==request_spec_sha(),'Request specification differs')
 return aligned,labels

OUTCOMES={'judgment_primary','registered_secondary_models','clm_baselines','retest_kappa','registered_route_comparison'}
def verify_route_payloads(payloads):
 require(len(payloads)>=2,'Route case table and setup decision are required')
 case=json_load(verified_path(payloads[0]));setup=json_load(verified_path(payloads[1]))
 require(isinstance(case.get('case_table'),list) and len(case['case_table'])==case.get('reference'),'Incomplete route case table')
 require(case.get('scoring_half_kept')==setup.get('by_half',{}).get('scoring',{}).get('kept'),'Route population differs')
 require(case['reference']==setup['by_half']['scoring'].get('reference'),'Route reference support differs')
 require(setup.get('decisions',{}).get('arm_B_and_T_run')==(setup.get('by_half',{}).get('seed',{}).get('reference',0)>0),'Route execution decision contradicts support')
 if not setup['decisions']['arm_B_and_T_run']:
  require(all(r.get('arm_B')=='not run (no seed)' and r.get('arm_T')=='not run (no seed)' for r in case['case_table']),'Missing documented no-seed branch')
 return {'seed_reference':setup['by_half']['seed']['reference'],'scoring_reference':case['reference'],
      'arm_B_and_T_run':setup['decisions']['arm_B_and_T_run']}

def verify_registered_outcome(x,name,mode='research'):
 obj=artifact(x,'registered_outcome','study3_s61_3',mode)
 require(obj.get('outcome')==name,'Wrong registered outcome')
 require(obj.get('registration_sha256')==sha_file(ROOT/'PREREGISTRATION.md'),'Registration version differs')
 require(obj.get('material_audit_sha256')==sha_file(HOME/'MATERIAL_AUDIT.json'),'Different frozen material population')
 require(obj.get('result_state') in ('estimated','not_estimable_documented'),'Unrecorded outcome state')
 require(isinstance(obj.get('payloads'),list) and bool(obj['payloads']),'No underlying result artifacts')
 for x in obj['payloads']:verified_path(x)
 if obj['result_state']=='not_estimable_documented':
  require(bool(obj.get('reason')) and bool(obj.get('support')),'Not-estimable outcome needs support and reason')
 if name=='registered_route_comparison':
  require(set(obj.get('routes',[]))=={'A','B','T'},'Incomplete registered route comparison')
  support=verify_route_payloads(obj['payloads'])
  require(obj.get('support')==support,'Route support not tied to actual results')
 else:
  data=json_load(verified_path(obj['payloads'][0]))
  require(data.get('mode')==mode,'Mock model output cannot be prerequisite evidence')
  require(data.get('status')=='registered_evaluation_recorded','Model evaluation output incomplete')
  require(data.get('registration_minimum_positive_scope')=='all_batches_1_2','Wrong minimum-positive scope')
  require(set(data.get('metrics',{}).get('overall',{}))=={'T','T+A','T+K','T+Fa','T+V','T+A+V','F','CLM_a','CLM_b'},'Registered model outputs missing')
  require(data.get('kappa_n')==40 and data.get('completion_verified') is True,'Missing verified retest/completion')
  require(not data.get('execution',{}).get('requires_fit_warning_review',True) or obj.get('fit_warning_reviewed') is True,'Unresolved fit warnings')
 return obj

def study3_gate(receipt,mode='research'):
 authorize(receipt,'jev_study3',mode)
 require(receipt.get('model')==MODEL and receipt.get('input_condition')=='paiwan_transcript','Wrong Study 3 request')
 require(set(receipt.get('registered_outcomes',{}))==OUTCOMES,'All registered outcomes must be provided')
 for name,x in receipt['registered_outcomes'].items():verify_registered_outcome(x,name,mode)
 assessment=artifact(receipt.get('study2_assessment'),'study2_assessment','study2_sent',mode)
 require(assessment.get('model')==MODEL and assessment.get('input_condition')=='paiwan_original','Same-version primary Study 2 assessment required')
 require(assessment.get('reference')=='published_english_sent_rule' and assessment.get('calibration_fit') is False,'Wrong assessment reference')
 require(assessment.get('n',0)>0 and bool(assessment.get('request_cache_refs')),'Assessment has no actual support/requests')
 require(assessment['n']==len(assessment['request_cache_refs']),'Assessment support differs from actual successful requests')
 require(assessment.get('brier') is not None and assessment.get('log_loss') is not None,'Probability assessment missing')
 aligned=artifact(assessment['alignment'],'study2_alignment','study2_sent',mode)
 labels=artifact(assessment['labels'],'study2_rule_labels','study2_sent',mode)
 require(labels.get('primary_reconciled') is True and labels.get('alignment_sha256')==assessment['alignment']['sha256'],'Unreconciled prior assessment')
 require(labels.get('rule_sha256')==decisions()['study2_rule_sha256'],'Different rule source in assessment')
 verified_path(aligned['rows']);verified_path(labels['rows'])
 batch=artifact(assessment['batch'],'jev_request_batch','study2_sent',mode)
 require(batch.get('input_condition')=='paiwan_original' and batch.get('model')==MODEL,'Wrong prior request batch')
 require(batch.get('n_requested')==aligned['n_eligible']==assessment.get('n_requested'),'Assessment not complete on the fixed eligible material')
 require(assessment.get('n_failed')+assessment['n']==assessment['n_requested'],'Unaccounted prior failures')
 require(set(x['sha256'] for x in assessment['request_cache_refs'])<=set(x['sha256'] for x in batch['records']),'Assessment requests not bound to batch')
 for x in batch['records']:verified_path(x)
 require(assessment.get('request_spec_sha256')==request_spec_sha(),'Assessment request version differs')
 for x in assessment['request_cache_refs']:
  r=json_load(verified_path(x));require(r.get('mode')==mode and r.get('status')=='success' and r.get('model')==MODEL,'Synthetic/unsuccessful assessment evidence')
  require(r.get('input_condition')=='paiwan_original' and r.get('scope')=='study2_sent','Wrong assessment request scope')
  verify_cache_bytes(r,Path(x['path']).parent)
 require(receipt.get('request_spec_sha256')==request_spec_sha(),'Request specification differs')
 return assessment

def request_spec_sha():
 from jev_mock_adapter import request
 # Empty state identifies the question/options/model/preprocessor, never data.
 return fingerprint({'request':request(''),'preprocessing':'strip_only_v1'})

def verify_cache_bytes(record,folder):
 """Check saved request and every raw response, including failed attempts."""
 require(record.get('request_sha256')==fingerprint(record['request']),'Cached request hash mismatch')
 if 'request_file' in record:
  q=Path(folder)/record['request_file'];require(q.parent.resolve()==Path(folder).resolve(),'Unsafe request path')
  require(q.is_file() and sha_file(q)==record['request_sha256'],'Raw request file hash mismatch')
 for a in record.get('attempts',[]):
  if 'response_file' in a:
   p=Path(folder)/a['response_file'];require(p.parent.resolve()==Path(folder).resolve(),'Unsafe response path')
   require(p.is_file() and sha_file(p)==a['response_sha256'],'Raw response hash mismatch')
 if record.get('status')=='success':
  from jev_mock_adapter import parse_response,request
  require(record['request']==request(record['request']['state']),'Cached request question/model changed')
  require(bool(record.get('attempts')) and record['attempts'][-1].get('http_status')==200,'No successful raw response')
  result=parse_response(json_load(Path(folder)/record['attempts'][-1]['response_file']))
  require(result['probability_directive']==record.get('score'),'Score differs from raw successful response')
 return True
