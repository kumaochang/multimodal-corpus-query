"""Portable bindings to the unchanged 4547f10 material audit; no repinning."""
import argparse,json,sys
from pathlib import Path,PurePosixPath
from contracts import HOME,ROOT,Blocked,require,json_load,sha_file,create_json,verified_path,safe_work

def contract():
 baseline=json_load(HOME/'DELIVERY_SHA256.json')['files']['MATERIAL_AUDIT.json']['sha256']
 require(sha_file(HOME/'MATERIAL_AUDIT.json')==baseline,'Historical material audit changed')
 audit=json_load(HOME/'MATERIAL_AUDIT.json');out={}
 for path,x in audit['protected_sources'].items():
  normalized=path.replace('\\','/');name=PurePosixPath(normalized).name;role=x['role']
  if role in ('frozen CLM/features','original manifest','original provenance'):key='item5/'+name
  elif role=='sampling metadata, no judgments':key='annotation/items.json'
  elif role=='historical CLM package inventory':key='env_clm.txt'
  elif name=='PREREGISTRATION.md':key=name
  else:
   require('/pipeline/' in normalized,'Unknown historical source role');key='pipeline/'+normalized.split('/pipeline/',1)[1]
  require(key not in out,'Ambiguous portable source key')
  out[key]={'sha256':x['sha256'],'bytes':x['bytes'],'historical_path':path,'role':role}
 return out

def verify(bindings):
 expected=contract();require(set(bindings)==set(expected),'Incomplete frozen input bindings')
 result={}
 for key,x in expected.items():
  p=Path(bindings[key]);require(p.is_file(),'Frozen input unavailable: '+key)
  require(sha_file(p)==x['sha256'] and p.stat().st_size==x['bytes'],'Frozen input mismatch: '+key)
  result[key]=p.resolve()
 return result

def make_bindings(item5,items,env,out):
 bindings={}
 for key in contract():
  if key.startswith('item5/'):p=Path(item5)/key.split('/',1)[1]
  elif key=='annotation/items.json':p=Path(items)
  elif key=='env_clm.txt':p=Path(env)
  else:p=ROOT/key
  bindings[key]=str(p.resolve())
 verify(bindings);create_json(safe_work(out),bindings)
 return bindings

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--item5-root',required=True);p.add_argument('--items',required=True);p.add_argument('--env-clm',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 try:
  b=make_bindings(a.item5_root,a.items,a.env_clm,a.out);print(json.dumps({'frozen_bindings_verified':len(b),'no_evaluation':True}))
 except Blocked as e:print(json.dumps({'status':'blocked','reason':str(e)}));sys.exit(2)
