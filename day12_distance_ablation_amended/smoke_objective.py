"""Read-only integration probe: full VAE objective/backward, zero optimizer steps."""
from common import *
from engine import objective,tensor_hash
from torch.utils.data import DataLoader

def probe():
 device=configure();base.set_seed(42);cfg=read(ROOT/'config.json');pairs=csvread(ROOT/'inputs/pairs_used.csv').iloc[:8].copy()
 mild,severe=next(iter(DataLoader(base.PairDataset(pairs,256),batch_size=8,shuffle=False,num_workers=0)));images=torch.cat([mild,severe]).to(device)
 rows=[]
 for pole in ('EA','EH'):
  model,ck,mr,vr,_,_=base.load_model(Path(cfg['checkpoint_'+pole]),device);model.train();vgg=base.VGGPerceptualLoss(device);ref=reference(pole)
  for arm in ARMS:
   model.zero_grad(set_to_none=True);st=read(ROOT/f'training_stats/{pole}_{arm}.json');parts,raw,z,gap=objective(model,vgg,images,8,pole,arm,ref,st,.1)
   require(all(torch.isfinite(t).all() for t in [*parts,raw,z,gap]),'Nonfinite objective smoke test')
   parts[0].backward();g=[p.grad for p in model.parameters() if p.grad is not None];norm=torch.stack(torch._foreach_norm(g)).norm();require(torch.isfinite(norm) and norm>0,'Invalid full-model gradient')
   require(raw.dtype==(torch.float32 if arm=='D0_LEGACY' else torch.float64),'Primary distance precision changed')
   require(torch.allclose(parts[0],parts[1]+.1*parts[6]),'Objective weighting mismatch')
   rows.append({'pole':pole,'distance':arm,'primary_dtype':str(raw.dtype),'loss':float(parts[0]),'gradient_norm':float(norm),'finite':True,'optimizer_steps':0,'batch_pairs':8})
   print('Full objective/backward smoke PASS:',pole,arm,flush=True)
  del model,vgg,ck,parts,raw,z,gap,g,norm;torch.cuda.empty_cache()
 save(ROOT/'verification/full_objective_smoke.json',{'status':'PASS','checked_utc':now(),'probe_only':True,'optimizer_steps':0,'source_checkpoint_files_unchanged':True,'rows':rows})
if __name__=='__main__':probe()
