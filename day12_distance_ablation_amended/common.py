"""Shared paths, immutable artifact verification, and original numerical settings."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib,json,sys,random
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'day11_day12'))
sys.path.insert(0,str(PROJECT/'day12_task2'))
from distance_core import ARMS,EPS,distance,rank_gap,standardized
from run_task2 import ReferenceDataset,posterior
import day11_rankall_pipeline as base

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def clean(o):
 if isinstance(o,dict):return {str(k):clean(v) for k,v in o.items()}
 if isinstance(o,(list,tuple)):return [clean(v) for v in o]
 if isinstance(o,np.ndarray):return clean(o.tolist())
 if isinstance(o,np.generic):return clean(o.item())
 if isinstance(o,Path):return str(o)
 return o
def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n');tmp.replace(path)
def csvread(path,**kw):return pd.read_csv(path,float_precision='round_trip',**kw)
def configure():
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the frozen primary arithmetic')
 return torch.device('cuda')
def reference(pole):
 folder=ROOT/'reference_stats'/pole
 with np.load(folder/'legacy_reference.npz') as l, np.load(folder/'aggregate_reference.npz') as a:
  return dict(legacy_mu=l['mu_ref'].copy(),legacy_variance=l['Sigma_ref'].copy(),registered_mu=a['mu_agg'].copy(),between_variance=a['v_between'].copy(),aggregate_variance_raw=a['v_aggregate_raw'].copy(),within_variance=a['v_within'].copy())
def refhashes(pole):return {n:sha(ROOT/'reference_stats'/pole/n) for n in ('manifest.json','legacy_reference.npz','between_reference.npz','aggregate_reference.npz','reference_posteriors.npz')}
def runpath(pole,arm,lam):return ROOT/'runs'/pole/arm/f'lambda_{lam}'/'seed_42'
def require(ok,msg):
 if not bool(ok):raise RuntimeError(msg)
def verify_frozen():
 lockpath=ROOT/'amendment_lock.json'
 require(sha(lockpath)==(ROOT/'amendment_lock.sha256').read_text().split()[0],'Amendment lock changed')
 lock=read(lockpath)
 for path,digest in lock['files'].items():require(sha(ROOT/path)==digest,f'Frozen artifact changed: {path}')
 for path,digest in lock['external_files'].items():require(sha(path)==digest,f'Frozen source changed: {path}')
 gate=read(ROOT/'preflight/gate.json')
 require(gate['passed'] and gate['state']=='PASS_AUTHORIZED_AMENDMENT','Amended gate does not permit training')
 require(sha(ROOT/'preflight/gate.json')==(ROOT/'preflight/gate.sha256').read_text().split()[0],'Gate hash changed')
 return lock
