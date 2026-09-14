"""Validate exact inputs and freeze reference-disjoint folds before training."""
import sys
import torch
from common import *
from engine import base,tensor_hash

def main():
 if (OUT/'input_manifest.json').exists():verify_inputs();print('Existing preflight verified');return
 for k in ['folds','crossfit_EH','weights','training','validation','kadid_development']:(OUT/k).mkdir(parents=True,exist_ok=True)
 sources={};checks={};records=[]
 for label,p in CHECKPOINTS.items():
  assert p.is_file(),p
  h=sha(p);assert h==EXPECTED[label],f'Wrong checkpoint {label}'
  m,c,mr,vr,img,ld=base.load_model(p,torch.device('cpu'));cfg=c['config']
  assert img==256 and ld==100 and mr.shape==vr.shape==(100,) and c['sz_mode']=='mu_only'
  assert torch.isfinite(mr).all() and torch.isfinite(vr).all() and (vr>0).all()
  assert all(torch.isfinite(v).all() for v in m.state_dict().values())
  ep={'EA_init':5,'EH_init':20,'EA_rankall':5,'EH_ranked':20}[label];assert c['epoch']==ep
  expected_lambda={'EA_init':0.,'EH_init':.25,'EA_rankall':.1,'EH_ranked':.1}[label]
  assert cfg['lambda_rank']==expected_lambda
  if p.name.startswith('epoch_'):assert int(p.stem.split('_')[1])==ep
  sources[str(p)]=h;checks[label]={'path':str(p),'sha256':h,'stored_epoch':ep,'stored_lambda':cfg['lambda_rank'],'stored_seed':c.get('seed'),'training_seed_override':42,'image_size':img,'latent_dim':ld,'reference_hashes':{'mu_ref':tensor_hash(mr),'Sigma_ref':tensor_hash(vr)},'sz_mode':'mu_only'}
  del m,c
 assert checks['EH_init']['reference_hashes']==checks['EH_ranked']['reference_hashes']
 assert checks['EA_init']['reference_hashes']==checks['EA_rankall']['reference_hashes']
 d=manifest();assert not d.duplicated(['dataset','image_id']).any();assert (d.groupby('group_key').split.nunique()==1).all()
 train=d[d.split.eq('train')].reset_index(drop=True);val=d[d.split.eq('val')].reset_index(drop=True)
 assert len(train)==9040 and len(val)==1860
 pairs=read_csv(PAIRS,usecols=PAIR_COLS,dtype={'dataset':str,'ref_id':str,'image_id_mild':str,'image_id_severe':str})
 assert len(pairs)==18080 and not pairs.duplicated(['dataset','image_id_mild','image_id_severe']).any()
 assert pairs.groupby('dataset').size().to_dict()=={'KADID-10k':14000,'TID2013':4080}
 assert set(pairs.dataset)=={'KADID-10k','TID2013'} and (pairs.sev_mild<pairs.sev_severe).all()
 index=train.set_index(['dataset','image_id'])
 for side in ['mild','severe']:
  matched=index.loc[list(zip(pairs.dataset,pairs['image_id_'+side]))].reset_index()
  for a,b in [('ref_id','ref_id'),('distortion_type','distortion_type'),('severity','sev_'+side),('path','path_'+side)]:
   assert np.array_equal(matched[a].to_numpy(),pairs[b].to_numpy()),(side,a)
  assert pairs['path_'+side].map(lambda x:Path(x).is_file()).all()
 pairs['pair_id']=np.arange(len(pairs));pairs['group_key']=pairs.dataset+'::'+pairs.ref_id
 folds=construct_folds(train);assert len(folds)==73 and folds.groupby('dataset').size().to_dict()=={'KADID-10k':56,'TID2013':17}
 assert not folds.group_key.duplicated().any();mapping=folds.set_index('group_key').fold
 pairs['heldout_fold']=pairs.group_key.map(mapping);train['heldout_fold']=train.group_key.map(mapping)
 assert pairs.heldout_fold.notna().all() and train.heldout_fold.notna().all()
 folds.to_csv(OUT/'folds/reference_fold_assignments.csv',index=False)
 train.to_csv(OUT/'folds/train_unique_manifest.csv',index=False);val.to_csv(OUT/'folds/validation_manifest.csv',index=False)
 pairs.to_csv(OUT/'folds/locked_pairs.csv',index=False)
 covered=[];balance=[]
 for k in range(5):
  folder=OUT/'crossfit_EH'/f'fold_{k}';folder.mkdir(exist_ok=True)
  a=pairs[pairs.heldout_fold.ne(k)];b=pairs[pairs.heldout_fold.eq(k)]
  assert not set(a.group_key)&set(b.group_key);assert set(b.dataset)==set(a.dataset)=={'KADID-10k','TID2013'}
  covered.extend(b.pair_id);a.to_csv(folder/'teacher_training_pairs.csv',index=False);b.to_csv(folder/'heldout_pairs.csv',index=False)
  for side,p in [('training',a),('heldout',b)]:
   u=train[train.group_key.isin(p.group_key)].copy();u.to_csv(folder/(side+'_unique_manifest.csv'),index=False)
   for ds,g in p.groupby('dataset'):
    imgs=set(g.image_id_mild)|set(g.image_id_severe)
    balance.append({'fold':k,'side':side,'dataset':ds,'references':g.ref_id.nunique(),'distortion_types':g.distortion_type.nunique(),'images':len(imgs),'pairs':len(g)})
 assert sorted(covered)==list(range(18080))
 pd.DataFrame(balance).to_csv(OUT/'folds/fold_balance_report.csv',index=False)
 levels=train.groupby(['dataset','distortion_type']).severity.agg(['min','max']).reset_index();assert (levels['max']>levels['min']).all();levels.to_csv(OUT/'folds/training_severity_ranges.csv',index=False)
 for p in [PAIRS,SPLITS,PROJECT/'day11_day12/pair_stats.txt',PROJECT/'day11_day12/day11_rankall_pipeline.py',PROJECT/'day11_day12/task12/day11b_pristine_rank_pipeline.py',PROJECT/'score.py',PROJECT/'external/model.py',PROJECT/'external/dataloader.py',ROOT/'PROTOCOL.md',ROOT/'inputs/day13.pdf',ROOT/'common.py',ROOT/'engine.py',ROOT/'preflight.py',ROOT/'crossfit.py',ROOT/'weights.py',Path('/home/projectwork/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth')]:sources[str(p)]=sha(p)
 for folder in [OUT/'folds',OUT/'crossfit_EH']:
  for p in folder.rglob('*.csv'):sources[str(p)]=sha(p)
 config={'created_utc':now(),'checkpoints':checks,'source_hashes':sources,'D0_function':'score.sz_from_stats','D0_arguments':{'eps':EPS,'mu_only':True,'sigma_t_max':1.},
 'D0_formula':'sqrt(sum((mu_ref-mu_x)^2/(Sigma_ref+1e-8)))','training':{'seed':42,'optimizer':'Adam','lr':1e-4,'betas':[.5,.999],'batch_size_pairs':8,'margin_A':.1,'margin_H':.1,'lambda_A':.1,'lambda_H':.1,'teacher_epochs':20,'EA_epochs':25,'saved_epochs':[5,10,15,20,25],'pixel':'L1','VGG_weight':.1,'KL_min':1.,'amp':False,'scheduler':None,'cudnn_benchmark':False,'cudnn_allow_tf32':True,'matmul_allow_tf32':False},
 'preprocessing':'original RGB, upsize short side only if<256, center crop256, float32 [0,1]',
 'arms':ARMS,'selection_priority':PRIORITY,'fold_seed':20260910,'bootstrap_seed':20260910,'bootstrap_draws':5000,
 'fold_assignment_sha256':sha(OUT/'folds/reference_fold_assignments.csv'),'pair_count':18080,'train_images':9040,'validation_images':1860,
 'training_groups':73,'Task4_outputs_in_configuration':False,'reference_rebuild':False,'train_pristine_used':False,'CSIQ_access_allowed':False,'MOS_DMOS_used_before_freeze':False,
 'historical_abs_dEH_ignored':True,'locked_at_preflight':True}
 save_json(OUT/'input_manifest.json',config);(OUT/'input_manifest.sha256').write_text(sha(OUT/'input_manifest.json')+'  input_manifest.json\n')
 (OUT/'checkpoint_identity_report.txt').write_text(json.dumps(checks,indent=2)+'\nAll exact identities and inherited references verified.\n')
 save_json(ROOT/'verification/preflight.json',{'status':'PASS','N_pairs':18080,'N_train_images':9040,'N_val_images':1860,'groups':73,'folds':5,'each_pair_heldout_once':True,'folds_reference_disjoint':True,'no_opinion_columns_loaded':True,'all_paths_exist':True,'completed_utc':now()})
 print('PREFLIGHT PASS: 18,080 locked pairs, 9,040 training images, 73 groups, five hashed folds.',flush=True)
if __name__=='__main__':main()
