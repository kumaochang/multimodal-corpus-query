"""Paths and immutable I/O for the independent reproduction (no analysis at import)."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
VERIFIED=ROOT
CORE=ROOT
REF=CORE/'analysis/study3/ch5_exploratory'
REFDATA=CORE/'data/study3_stats/ch5_exploratory'
_REPORT_ROOT=ROOT/'analysis/study3/ch5_reproducibility'
_DATA_ROOT=ROOT/'data/study3_stats/ch5_reproducibility'
REPORT=Path(os.environ.get('CH5_REPRO_REPORT',str(_REPORT_ROOT))).resolve()
DATA=Path(os.environ.get('CH5_REPRO_DATA',str(_DATA_ROOT))).resolve()
assert REPORT.is_relative_to(_REPORT_ROOT) and DATA.is_relative_to(_DATA_ROOT)
INPUT=ROOT/'data/study3_runs'
CMDSTAN=_DATA_ROOT/'toolchain/cmdstan-2.37.0'
ATTEMPTS={'H0_W3':1,'HA_W3':5,'HB_W3':1}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 import numpy as np
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n')
def arrhash(x):
 import numpy as np
 x=np.asarray(x);h=hashlib.sha256(json.dumps([x.dtype.str,list(x.shape)]).encode());h.update(x.tobytes(order='C'));return h.hexdigest()
def original_fit(name):return REFDATA/'posterior_20261020/fits'/name/'base'/f'attempt_{ATTEMPTS[name]}'
