"""Full original VAE training, frozen primary distances, epoch-boundary RNG resume."""
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import *

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def atomic_torch(path,obj):
 tmp=path.with_suffix('.pth.tmp');torch.save(obj,tmp);tmp.replace(path)
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
class IndexedPairs(base.PairDataset):
 def __getitem__(self,i):return (*super().__getitem__(i),i)

def run_configuration(pole,arm,lam):
 cfg=read(ROOT/'config.json');path=Path(cfg['checkpoint_'+pole]);stpath=ROOT/f'training_stats/{pole}_{arm}.json'
 return {'task':cfg['task'],'pole':pole,'distance':arm,'lambda_rank':lam,'seed':42,'epochs':25,'save_every':5,'init_from':str(path),'init_sha256':sha(path),'reference_hashes':refhashes(pole),'training_standardization':read(stpath),'training_standardization_sha256':sha(stpath),'pairs_sha256':sha(ROOT/'inputs/pairs_used.csv'),'config_sha256':sha(ROOT/'config.json'),'amendment_lock_sha256':sha(ROOT/'amendment_lock.json'),'gate_sha256':sha(ROOT/'preflight/gate.json'),'ldim':100,'img':256,'optimizer':'Adam','lr':1e-4,'betas':[.5,.999],'batch_pairs':8,'rank_margin':.1,'loss':'L1 + 0.1*VGG19 + clamp(KL_raw,min=1.0) + lambda_rank*mean(softplus(0.1-signed_standardized_gap))','primary_dtype':'float32' if arm=='D0_LEGACY' else 'float64','diagnostic_float64_used_for_training':False,'input_dedup_used_for_training':False,'MOS_DMOS_used':False}

def objective(model,vgg,images,b,pole,arm,ref,st,lam):
 recon,mu,lv=model(images)
 pixel=F.l1_loss(recon,images);percept=vgg(recon,images);klraw=base.kl_loss(mu,lv);kl=klraw.clamp_min(1.)
 mild=distance(mu[:b],lv[:b],ref,arm);severe=distance(mu[b:],lv[b:],ref,arm)
 z_mild=standardized(mild,st);z_severe=standardized(severe,st)
 gap=rank_gap(z_mild,z_severe,pole);ranked=F.softplus(.1-gap).mean();weighted=lam*ranked;vae=pixel+.1*percept+kl;loss=vae+weighted
 return [loss,vae,pixel,percept,klraw,kl,ranked,weighted],torch.cat([mild,severe]),torch.cat([z_mild,z_severe]),gap

def train_run(pole,arm,lam):
 verify_frozen();device=configure();base.set_seed(42);run=runpath(pole,arm,lam);run.mkdir(parents=True,exist_ok=True)
 cfg=run_configuration(pole,arm,lam);configpath=run/'configuration.json'
 if (run/'failure.json').exists():
  archive=run/'failure_history';archive.mkdir(exist_ok=True);(run/'failure.json').replace(archive/f'failure_{time.time_ns()}.json')
 if configpath.exists():require(read(configpath)==cfg,'Run configuration changed')
 else:save(configpath,cfg)
 if (run/'complete.json').exists():
  done=read(run/'complete.json');require(done['configuration']==cfg,'Completed run configuration changed')
  for f,h in done['saved_checkpoint_hashes'].items():require(sha(run/f)==h,'Completed checkpoint changed')
  return
 pairfile=ROOT/'inputs/pairs_used.csv';pairs=csvread(pairfile,dtype={'ref_id':str})
 require(len(pairs)==18080 and (pairs.sev_mild<pairs.sev_severe).all(),'Training pair inventory or orientation changed')
 require(not any('mos' in c.lower() for c in pairs.columns),'Opinion column in training table')
 model,initial,mr,vr,img,_=base.load_model(Path(cfg['init_from']),device);model.train()
 ref=reference(pole);st=cfg['training_standardization'];before_refs={'mu_ref':tensor_hash(mr),'Sigma_ref':tensor_hash(vr)}
 opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.5,.999));vgg=base.VGGPerceptualLoss(device)
 loader=DataLoader(IndexedPairs(pairs,img),batch_size=8,shuffle=True,num_workers=4,pin_memory=True,drop_last=False)
 ckdir=run/'checkpoints';ckdir.mkdir(exist_ok=True);last=ckdir/'last.pth';start=1;history=[]
 if last.exists():
  ck=torch.load(last,map_location=device,weights_only=False);require(ck['config']==cfg,'Resume configuration/hash mismatch')
  model.load_state_dict(ck['generator']);opt.load_state_dict(ck['optimizer']);history=ck['epoch_history'];start=int(ck['epoch'])+1
  require(torch.equal(ck['mu_ref'].to(device),mr) and torch.equal(ck['Sigma_ref'].to(device),vr),'Resume reference changed');restore_rng(ck['rng_state']);del ck
 save(run/'status.json',{'state':'TRAINING','pole':pole,'distance':arm,'lambda_rank':lam,'start_epoch':start,'updated_utc':now(),'gate_sha256':cfg['gate_sha256'],'holdout_opened':False})
 for ep in range(start,26):
  begun=time.time();notice=begun;values_sum=np.zeros(8);n=0;nb=0;correct=ties=0;rawlist=[];zlist=[];norms=[];seen=[]
  for mild,severe,ids in loader:
   mild=mild.to(device,non_blocking=True);severe=severe.to(device,non_blocking=True);b=len(mild);images=torch.cat([mild,severe])
   opt.zero_grad(set_to_none=True);parts,raw,z,gap=objective(model,vgg,images,b,pole,arm,ref,st,lam)
   require(all(torch.isfinite(v).all() for v in [*parts,raw,z,gap]),f'Nonfinite loss/distance: {pole} {arm} epoch {ep}')
   parts[0].backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
   gradnorm=torch.stack(torch._foreach_norm(grads)).norm();require(torch.isfinite(gradnorm),'Nonfinite gradient; stopped before optimizer step')
   opt.step();nb+=1;n+=b;seen.extend(ids.tolist());values_sum+=np.array([x.item() for x in parts])*b
   rawlist.append(raw.detach().double().cpu().numpy());zlist.append(z.detach().double().cpu().numpy());norms.append(float(gradnorm))
   correct+=int((gap>0).sum());ties+=int((gap==0).sum())
   if nb==1 or time.time()-notice>=40:
    progress={'state':'TRAINING','pole':pole,'distance':arm,'lambda_rank':lam,'epoch':ep,'epochs':25,'batches_completed':nb,'batches_per_epoch':len(loader),'pairs_seen':n,'pairs_per_epoch':len(pairs),'epoch_elapsed_seconds':time.time()-begun,'last_batch_loss':float(parts[0].detach()),'last_gradient_norm':float(gradnorm),'nan_count':0,'inf_count':0,'updated_utc':now()}
    save(run/'progress.json',progress);save(ROOT/'progress.json',progress);print(f'{pole} {arm} lambda={lam} epoch={ep}/25 pairs={n}/{len(pairs)} loss={float(parts[0].detach()):.5f}',flush=True);notice=time.time()
  require(sorted(seen)==list(range(len(pairs))),'Pair sampling changed, omitted, or repeated a row')
  require(before_refs=={'mu_ref':tensor_hash(mr),'Sigma_ref':tensor_hash(vr)},'Reference tensors changed')
  raw=np.concatenate(rawlist);z=np.concatenate(zlist)
  row=dict(zip(['L_total','L_VAE','L_pixel','L_VGG','L_KL_raw','L_KL_clamped','L_rank_unweighted','L_rank_lambda_weighted'],values_sum/n))
  row.update(pole=pole,distance=arm,lambda_rank=lam,seed=42,epoch=ep,N_pairs=n,N_batches=nb,N_scored_training_occurrences=len(raw),raw_mean=float(raw.mean()),raw_std_population=float(raw.std()),raw_minimum=float(raw.min()),raw_maximum=float(raw.max()),standardized_mean=float(z.mean()),standardized_std_population=float(z.std()),pair_order_accuracy=correct/n,pair_ties=ties,gradient_norm_mean=float(np.mean(norms)),gradient_norm_max=float(np.max(norms)),nan_count=0,inf_count=0,time_seconds=time.time()-begun,init_sha256=cfg['init_sha256'],reference_statistics_hashes=json.dumps(cfg['reference_hashes'],sort_keys=True),training_standardization_sha256=cfg['training_standardization_sha256'],gate_sha256=cfg['gate_sha256'],amendment_lock_sha256=cfg['amendment_lock_sha256'],all_pair_rows_seen_once=True)
  history.append(row)
  state={'generator':model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'config':cfg,'seed':42,'mu_ref':mr.detach().cpu(),'Sigma_ref':vr.detach().cpu(),'checkpoint_kind':'modified_task4_amended_distance','sz_mode':'mu_only' if arm in ARMS[:3] else arm,'rng_state':rng_state(),'epoch_history':history}
  atomic_torch(last,state)
  if ep%5==0:atomic_torch(ckdir/f'epoch_{ep:04d}.pth',state)
  pd.DataFrame(history).to_csv(run/'train_log.csv',index=False)
  save(run/'progress.json',{'state':'EPOCH_COMPLETE','pole':pole,'distance':arm,'lambda_rank':lam,'epoch':ep,'epochs':25,'time_seconds':row['time_seconds'],'updated_utc':now()})
  print(f'Epoch complete: {pole} {arm} lambda={lam} {ep}/25, {row["time_seconds"]:.1f}s, loss={row["L_total"]:.6f}',flush=True)
 hashes={str(p.relative_to(run)):sha(p) for p in sorted(ckdir.glob('epoch_*.pth'))};require(len(hashes)==5,'Expected five saved epochs')
 save(run/'complete.json',{'status':'COMPLETE','configuration':cfg,'saved_checkpoint_hashes':hashes,'epochs':25,'optimizer_steps':25*len(loader),'reference_tensors_unchanged':True,'completed_utc':now()})
 del model,opt,vgg,initial;torch.cuda.empty_cache();verify_frozen()

@torch.no_grad()
def score_checkpoint(d,checkpoint,pole,arm,output):
 require(not any('mos' in c.lower() for c in d.columns),'Scorer accepts only image/severity metadata')
 output=Path(output);cksha=sha(checkpoint)
 if output.exists() and output.with_suffix('.json').exists():
  meta=read(output.with_suffix('.json'));require(meta['checkpoint_sha256']==cksha and meta['scores_sha256']==sha(output),'Scoring cache hash mismatch')
  prior=csvread(output,dtype={'ref_id':str,'image_id':str});require(prior.path.tolist()==d.path.tolist(),'Scoring cache image order mismatch');return prior
 device=configure();base.set_seed(42);model,ck,_,_,_,_=base.load_model(checkpoint,device);ref=reference(pole)
 require(ck['config']['distance']==arm and ck['config']['pole']==pole,'Wrong matched checkpoint')
 loader=DataLoader(ReferenceDataset(d.to_dict('records'),256),batch_size=16,shuffle=False,num_workers=4,pin_memory=True)
 scores=[]
 for images in loader:
  mu,lv=posterior(model,images.to(device,non_blocking=True));scores.extend(distance(mu,lv,ref,arm).double().cpu().tolist())
 values=np.asarray(scores);require(len(values)==len(d) and np.isfinite(values).all() and values.std()>1e-12,'Invalid trained scores')
 out=d.copy().reset_index(drop=True);out['energy']=values;output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.csv.tmp');out.to_csv(temp,index=False);temp.replace(output)
 save(output.with_suffix('.json'),{'checkpoint_sha256':cksha,'scores_sha256':sha(output),'N':len(d),'pole':pole,'distance':arm,'epoch':ck['epoch'],'batch_size':16,'primary_precision':'float32' if arm=='D0_LEGACY' else 'float64','diagnostic_scores_used':False,'completed_utc':now()})
 del model,ck;torch.cuda.empty_cache();return out
