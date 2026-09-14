"""Day 13 numerical rules and IO, independent of earlier experiment outputs."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
OUT=ROOT/'day12_blindspot_d0'
EPS=1e-8
ARMS={'H00':('hard',0.),'H10':('hard',.10),'H25':('hard',.25),'S00':('soft',0.),'S10':('soft',.10),'S25':('soft',.25)}
PRIORITY=['S10','H10','S00','H00','S25','H25']
CHECKPOINTS={'EA_init':PROJECT/'dataset_model2/checkpoints/wacv_ea_day7_seed42/epoch_0005.pth',
'EH_init':PROJECT/'checkpoints/hr_combined_ft1/best.pth',
'EH_ranked':PROJECT/'day11_day12/task12/day11b_pristine_rank/lambda_0.1/checkpoints/epoch_0020.pth',
'EA_rankall':PROJECT/'day11_day12/day11_rankall/lambda_0.1/checkpoints/epoch_0005.pth'}
EXPECTED={'EA_init':'c6880ae6bf66a32960dab5c84cedfc43c2a9f1b878f687ff4159c6643689f38c','EH_init':'4818fce7f2fbb2c24e93ae67ae3edc65aed0ff67a4461111dac5571517dbffa2','EH_ranked':'523c88c21c4f4227e925b4926488e5a7d3780e950af13e0cf86ec87247837037','EA_rankall':'a932964e58815535db592f4e399e187175db659cfff5000df4b1929bab8ce932'}
SPLITS=PROJECT/'dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv'
PAIRS=PROJECT/'day11_day12/pairs_severity_train.csv'
META=['dataset','image_id','ref_id','distortion_type','severity_or_level','split','distorted_path','ref_path']
PAIR_COLS=['dataset','ref_id','distortion_type','path_mild','path_severe','sev_mild','sev_severe','image_id_mild','image_id_severe']

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,Path):return str(v)
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def save_json(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)
def read_json(path):return json.loads(Path(path).read_text())
def read_csv(path,**kw):return pd.read_csv(path,float_precision='round_trip',**kw)
def remap(p):return p.replace('/data/projectwork/swati_mam/kadid10k','/data/projectwork/swati_mam/new model data/kadid10k')
def manifest():
 d=pd.read_csv(SPLITS,usecols=META,dtype=str,keep_default_na=False)
 d=d[d.dataset.isin(['KADID-10k','TID2013'])].copy()
 d['path']=d.distorted_path.map(remap);d['ref_path']=d.ref_path.map(remap)
 d['severity']=pd.to_numeric(d.severity_or_level,errors='raise');d['group_key']=d.dataset+'::'+d.ref_id
 return d.drop(columns=['distorted_path','severity_or_level'])
def verify_inputs():
 m=read_json(OUT/'input_manifest.json')
 assert sha(OUT/'input_manifest.json')==(OUT/'input_manifest.sha256').read_text().split()[0]
 for path,expected in m['source_hashes'].items():
  if sha(path)!=expected:raise RuntimeError('Changed locked input: '+path)
 return m

def construct_folds(d):
 rows=[]
 for ds,g in d.groupby('dataset',sort=True):
  keys=np.array(sorted(g.group_key.unique()));np.random.default_rng(20260910).shuffle(keys)
  rows.extend({'dataset':ds,'ref_id':str(k).split('::',1)[1],'group_key':str(k),'fold':i%5} for i,k in enumerate(keys))
 return pd.DataFrame(rows).sort_values(['dataset','group_key']).reset_index(drop=True)

def weight_values(gap,delta,alpha,kind):
 v=alpha*np.asarray(delta,float)-np.asarray(gap,float)
 temperature=max(float(np.median(np.abs(v-np.median(v)))),.05)
 raw=(v>=0).astype(float) if kind=='hard' else np.exp(-np.logaddexp(0,-v/temperature))
 normalized=raw/(raw.mean()+EPS)
 ess=float(raw.sum()**2/np.square(raw).sum()) if np.square(raw).sum()>0 else 0.
 return v,temperature,raw,normalized,ess

def all_pairs(d):
 from itertools import combinations
 rows=[]
 for (ds,ref,typ),g in d.groupby(['dataset','ref_id','distortion_type'],sort=True):
  for a,b in combinations(g.index,2):
   sa,sb=d.at[a,'severity'],d.at[b,'severity']
   if not np.isfinite(sa+sb) or sa==sb:continue
   if sa>sb:a,b=b,a
   rows.append((ds,ref,typ,a,b,d.at[a,'image_id'],d.at[b,'image_id'],d.at[a,'severity'],d.at[b,'severity']))
 return pd.DataFrame(rows,columns=['dataset','ref_id','distortion_type','mild_index','severe_index','image_id_mild','image_id_severe','sev_mild','sev_severe'])

def severity_metrics(d,q):
 rows=[]
 for (ds,t),g in d.groupby(['dataset','distortion_type']):
  rho=float(spearmanr(g[q],g.severity).statistic)
  rows.append({'dataset':ds,'distortion_type':t,'score':q,'N':len(g),'severity_SRCC':rho})
 tab=pd.DataFrame(rows)
 if not np.isfinite(tab.severity_SRCC).all():raise ValueError('Undefined distortion severity correlation')
 by=tab.groupby('dataset').severity_SRCC.mean().mul(-1)
 return float(by.mean()),tab,by.to_dict()

def pair_metrics(d,q,pairs=None):
 p=all_pairs(d) if pairs is None else pairs.copy()
 gap=d.loc[p.mild_index,q].to_numpy()-d.loc[p.severe_index,q].to_numpy()
 p['score']=q;p['gap']=gap;p['correct']=gap>0;p['tie']=gap==0;p['credit']=(gap>0)+.5*(gap==0)
 rows=[]
 for (ds,t),g in p.groupby(['dataset','distortion_type']):
  rows.append({'dataset':ds,'distortion_type':t,'score':q,'N_pairs':len(g),'correct':int(g.correct.sum()),'ties':int(g.tie.sum()),'accuracy':float(g.credit.mean()),'strict_accuracy':float(g.correct.mean())})
 tab=pd.DataFrame(rows);macro=float(tab.groupby('dataset').accuracy.mean().mean())
 for ds,g in p.groupby('dataset'):
  rows.append({'dataset':ds,'distortion_type':'ALL','score':q,'N_pairs':len(g),'correct':int(g.correct.sum()),'ties':int(g.tie.sum()),'accuracy':float(g.credit.mean()),'strict_accuracy':float(g.correct.mean()),'macro_accuracy':float(tab[tab.dataset.eq(ds)].accuracy.mean())})
 return macro,pd.DataFrame(rows),p

def select_specialist(table):
 t=table.copy();t['priority']=t.arm.map({a:i for i,a in enumerate(PRIORITY)})
 return t.sort_values(['S_blind','macro_pair_accuracy','priority','epoch'],ascending=[False,False,True,True]).iloc[0]

def clusters(d,n=5000,seed=20260910):
 rng=np.random.default_rng(seed);out={}
 for ds,g in d.groupby('dataset',sort=True):
  refs=sorted(g.ref_id.unique());sample=rng.integers(0,len(refs),(n,len(refs)))
  counts=np.array([(sample==j).sum(1) for j in range(len(refs))]).T;out[ds]=(refs,counts)
 return out

def weighted_ranks(x,w):
 x=np.asarray(x,float);w=np.atleast_2d(w).astype(float);o=np.argsort(x,kind='stable');v=x[o]
 starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];cnt=np.add.reduceat(w[:,o],starts,axis=1)
 ranks=np.cumsum(cnt,axis=1)-.5*cnt+.5;groups=np.cumsum(np.r_[False,v[1:]!=v[:-1]])
 return ranks[:,groups][:,np.argsort(o)]
def weighted_corr(x,y,w):
 w=np.atleast_2d(w).astype(float);n=w.sum(1,keepdims=True)
 x=np.broadcast_to(x,w.shape);y=np.broadcast_to(y,w.shape)
 x=x-(w*x).sum(1,keepdims=True)/n;y=y-(w*y).sum(1,keepdims=True)/n
 den=np.sqrt((w*x*x).sum(1)*(w*y*y).sum(1));num=(w*x*y).sum(1)
 return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
def weighted_srcc(x,y,w):return weighted_corr(weighted_ranks(x,w),weighted_ranks(y,w),w)
def ci(values):
 v=np.asarray(values,float);ok=np.isfinite(v);a=v[ok]
 return {'ci_low':float(np.quantile(a,.025)) if len(a) else None,'ci_high':float(np.quantile(a,.975)) if len(a) else None,'valid_draws':int(ok.sum()),'valid_fraction':float(ok.mean()),'n_draws':len(v)}
