"""Inherited Day-11 training and scoring with frozen references and resumable RNG."""
import os,random,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from common import *
sys.path.insert(0,str(PROJECT/'day11_day12'))
import day11_rankall_pipeline as base

def configure():
 torch.set_num_threads(4)
 if not torch.cuda.is_available():raise RuntimeError('Required CUDA device unavailable')
 torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=True  # Original PyTorch/Day-11 default, recorded explicitly.
 return torch.device('cuda')

def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def model_hash(model):
 h=hashlib.sha256()
 for name,p in model.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def atomic_torch(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pth.tmp');torch.save(obj,tmp);tmp.replace(path)

def training_configuration(name,teacher,manifest_hash,pairs_hash,init,weight_info):
 return {'experiment':'day13_blindspot_D0','run':name,'teacher':teacher,'seed':42,'optimizer':'Adam','lr':1e-4,'betas':[.5,.999],
 'batch_size_pairs':8,'rank_margin':.1,'lambda_rank':.1,'epochs':20 if teacher else 25,'save_every':5,
 'init_from':str(init),'init_sha256':sha(init),'input_manifest_sha256':manifest_hash,'pair_table_sha256':pairs_hash,
 'ldim':100,'img':256,'sz_mode':'mu_only','sigma_t_max':1.,'D0_epsilon':EPS,'amp':False,
 'loss':'L1 + 0.1*VGG19 + max(KL_raw,1.0) + 0.1*mean(fixed_weight*softplus(0.1-signed_gap))',
 'weight_info':weight_info,'reference_policy':'unchanged tensors from initialization checkpoint','MOS_DMOS_used':False,
 'ranking_direction':'EH_severe_minus_mild' if teacher else 'EA_mild_minus_severe'}

class WeightedPairs(base.PairDataset):
 def __getitem__(self,i):
  mild,severe=super().__getitem__(i)
  row=self.pairs.iloc[i]
  return mild,severe,i,float(row.get('weight',1.)),bool(row.get('EH_reversed',False)),bool(row.get('diagnostic_focus',True))


def train_run(name,pair_file,init,teacher=False,weight_info=None):
 lock=verify_inputs();device=configure();base.set_seed(42)
 d=read_csv(pair_file,dtype={'ref_id':str})
 assert set(d.dataset).issubset({'KADID-10k','TID2013'}) and (d.sev_mild<d.sev_severe).all()
 assert not any('mos' in c.lower() for c in d.columns)
 run=OUT/('crossfit_EH' if teacher else 'training')/name;run.mkdir(parents=True,exist_ok=True)
 config=training_configuration(name,teacher,sha(OUT/'input_manifest.json'),sha(pair_file),init,weight_info or {})
 if (run/'complete.json').exists():
  info=read_json(run/'complete.json');assert info['configuration']==config
  assert sha(info['checkpoint'])==info['checkpoint_sha256'];return Path(info['checkpoint'])
 model,initial,mr,vr,img,_=base.load_model(init,device);model.train()
 before_refs={k:tensor_hash(v) for k,v in [('mu_ref',mr),('Sigma_ref',vr)]}
 optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.5,.999))
 vgg=base.VGGPerceptualLoss(device)
 loader=DataLoader(WeightedPairs(d,img),batch_size=8,shuffle=True,num_workers=4,pin_memory=True,drop_last=False)
 ckdir=run/'checkpoints';ckdir.mkdir(exist_ok=True);last_path=ckdir/'last.pth';start=1;history=[]
 if last_path.exists():
  last=torch.load(last_path,map_location=device,weights_only=False)
  if last['config']!=config:raise RuntimeError('Resume configuration mismatch')
  model.load_state_dict(last['generator']);optimizer.load_state_dict(last['optimizer'])
  for k,v in [('mu_ref',mr),('Sigma_ref',vr)]:assert torch.equal(last[k].to(device),v)
  start=last['epoch']+1;history=last['epoch_history'];restore_rng(last['rng_state'])
  del last
 save_json(run/'configuration.json',config)
 for ep in range(start,config['epochs']+1):
  started=time.time();last_notice=started;sums=np.zeros(7);nb=0;nseen=0;zero_batches=0
  nc=nt=nr=rc=nf=fc=0;ids_seen=[]
  for mild,severe,ids,weights,reversed_mask,focus_mask in loader:
   mild=mild.to(device,non_blocking=True);severe=severe.to(device,non_blocking=True);b=len(mild)
   imgs=torch.cat([mild,severe],0);optimizer.zero_grad(set_to_none=True)
   recon,mu,lv=model(imgs)
   pixel=F.l1_loss(recon,imgs);percept=vgg(recon,imgs);klraw=base.kl_loss(mu,lv);kl=torch.clamp(klraw,min=1.)
   em=base.sz_from_stats(mu[:b],lv[:b],mr,vr,eps=EPS,sigma_t_max=1.,mu_only=True)
   es=base.sz_from_stats(mu[b:],lv[b:],mr,vr,eps=EPS,sigma_t_max=1.,mu_only=True)
   gap=es-em if teacher else em-es;perpair=F.softplus(.1-gap)
   # Teachers preserve the original float32 mean exactly; specialists apply detached weights only here.
   ranked=perpair.mean() if teacher else (weights.to(device).detach()*perpair).mean()
   unweighted=perpair.mean();loss=pixel+.1*percept+kl+.1*ranked
   if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss {name} epoch {ep}')
   loss.backward()
   if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
   optimizer.step()
   vals=[loss,pixel,percept,klraw,kl,ranked,unweighted];sums+=np.array([v.item() for v in vals]);nb+=1;nseen+=b;ids_seen.extend(ids.tolist())
   ok=(gap.detach()>0).cpu();tie=(gap.detach()==0).cpu();nc+=int(ok.sum());nt+=int(tie.sum())
   nr+=int(reversed_mask.sum());rc+=int(ok[reversed_mask].sum());nf+=int(focus_mask.sum());fc+=int(ok[focus_mask].sum())
   zero_batches+=int(not torch.any(weights>0))
   if time.time()-last_notice>=45:
    save_json(run/'progress.json',{'state':'TRAINING','run':name,'epoch':ep,'epochs':config['epochs'],'pairs_seen':nseen,'pairs_per_epoch':len(d),'epoch_elapsed_seconds':time.time()-started,'updated_utc':now()})
    print(f'{name} epoch {ep}/{config["epochs"]}: {nseen}/{len(d)} pairs',flush=True);last_notice=time.time()
  assert sorted(ids_seen)==list(range(len(d)))
  assert before_refs=={k:tensor_hash(v) for k,v in [('mu_ref',mr),('Sigma_ref',vr)]}
  row=dict(zip(['L_total','L_pixel','L_VGG','L_KL_raw','L_KL_clamped','L_rank_weighted','L_rank_unweighted'],sums/nb))
  row.update(epoch=ep,run=name,alpha_H=(weight_info or {}).get('alpha_H'),T_alpha=(weight_info or {}).get('T_alpha'),m_A=None if teacher else .1,ranking_margin=.1,lambda_rank=.1,
   rank_accuracy_all=nc/nseen,rank_ties=nt,EH_reversed_pair_N=nr,rank_accuracy_EH_reversed=rc/nr if nr else None,
   focus_pair_N=nf,rank_accuracy_focus=fc/nf if nf else None,zero_ranking_weight_batches=zero_batches,
   N_pairs=nseen,N_batches=nb,time_seconds=time.time()-started,input_manifest_sha256=config['input_manifest_sha256'],pair_table_sha256=config['pair_table_sha256'],init_sha256=config['init_sha256'])
  history.append(row)
  checkpoint={'generator':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':ep,'config':config,'seed':42,
   'mu_ref':mr.detach().cpu(),'Sigma_ref':vr.detach().cpu(),'sz_mode':'mu_only','sz_sigma_t_max':1.,
   'checkpoint_kind':'day13_crossfit_EH' if teacher else 'day13_blindspot_EA','rng_state':rng_state(),'epoch_history':history}
  atomic_torch(last_path,checkpoint)
  if ep%5==0:atomic_torch(ckdir/f'epoch_{ep:04d}.pth',checkpoint)
  pd.DataFrame(history).to_csv(run/'train_log.csv',index=False)
  print(f'{name} epoch {ep}/{config["epochs"]} complete: loss {row["L_total"]:.6f}, ranking {row["L_rank_weighted"]:.6f}, {row["time_seconds"]:.1f}s',flush=True)
  save_json(run/'progress.json',{'state':'EPOCH_COMPLETE','run':name,'epoch':ep,'epochs':config['epochs'],'updated_utc':now(),'time_seconds':row['time_seconds']})
 final=ckdir/f'epoch_{config["epochs"]:04d}.pth'
 assert sha(init)==config['init_sha256']
 save_json(run/'complete.json',{'status':'COMPLETE','checkpoint':str(final),'checkpoint_sha256':sha(final),'configuration':config,
 'reference_hashes':before_refs,'all_reference_tensors_unchanged':True,'completed_utc':now(),'no_validation_selection':teacher})
 del model,optimizer,vgg,initial;torch.cuda.empty_cache();verify_inputs();return final

class ScoreImages(base.PathDataset):
 def __init__(self,d,img):
  d=d.copy();d['distorted_path']=d.path;super().__init__(d,img)

@torch.no_grad()
def score_checkpoint(d,path,output):
 """Fresh full-forward D0 scoring; no source opinions enter this function."""
 output=Path(output);cksha=sha(path)
 if output.exists():
  info=read_json(output.with_suffix('.json'));assert info['checkpoint_sha256']==cksha and info['output_sha256']==sha(output)
  old=read_csv(output,dtype={'ref_id':str,'image_id':str})
  assert old[['dataset','image_id']].equals(d[['dataset','image_id']].reset_index(drop=True));return old
 device=configure();model,ck,mr,vr,img,_=base.load_model(path,device);model.eval();before=model_hash(model)
 loader=DataLoader(ScoreImages(d,img),batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
 result=[];seen=[]
 for x,idx in loader:
  _,mu,lv=model(x.to(device,non_blocking=True))
  result.extend(base.sz_from_stats(mu,lv,mr,vr,eps=EPS,sigma_t_max=1.,mu_only=True).cpu().numpy().astype(float));seen.extend(idx.tolist())
 assert seen==list(range(len(d))) and np.isfinite(result).all() and np.std(result)>EPS
 assert before==model_hash(model) and sha(path)==cksha
 out=d.copy().reset_index(drop=True);out['energy']=result;output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(output,index=False)
 save_json(output.with_suffix('.json'),{'checkpoint':str(path),'checkpoint_sha256':cksha,'epoch':ck['epoch'],'N':len(out),'score_std_population':float(np.std(result)),
 'model_state_unchanged':True,'output_sha256':sha(output),'completed_utc':now(),'scoring':'original full model forward, mean-only D0'})
 del model,ck;torch.cuda.empty_cache();return out
