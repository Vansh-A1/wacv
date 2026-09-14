"""Five fixed EH teachers and train-calibrated out-of-fold scores."""
import argparse
from common import *
from engine import train_run,score_checkpoint,tensor_hash
import torch

def train_fold(k):
 verify_inputs();folder=OUT/'crossfit_EH'/f'fold_{k}'
 return train_run(f'fold_{k}',folder/'teacher_training_pairs.csv',CHECKPOINTS['EH_init'],teacher=True)

def score_fold(k):
 verify_inputs();folder=OUT/'crossfit_EH'/f'fold_{k}';info=read_json(folder/'complete.json');ckpath=Path(info['checkpoint'])
 assert info['no_validation_selection'] is True and info['configuration']['epochs']==20
 assert sha(ckpath)==info['checkpoint_sha256']
 ck=torch.load(ckpath,map_location='cpu',weights_only=False);init=read_json(OUT/'input_manifest.json')['checkpoints']['EH_init']
 assert ck['epoch']==20 and ck['config']['lambda_rank']==.1
 assert {t:tensor_hash(ck[t]) for t in ['mu_ref','Sigma_ref']}==init['reference_hashes']
 train=read_csv(folder/'training_unique_manifest.csv',dtype={'ref_id':str,'image_id':str})
 held=read_csv(folder/'heldout_unique_manifest.csv',dtype={'ref_id':str,'image_id':str})
 assert not set(train.group_key)&set(held.group_key)
 assert train.path.is_unique and held.path.is_unique
 ts=score_checkpoint(train,ckpath,folder/'training_unique_scores.csv')
 stats={'mean':float(ts.energy.mean()),'std_population':float(ts.energy.std(ddof=0)),'N':len(ts),'eps':EPS,'calibration':'unique teacher-training-side images only','fold':k,'teacher_sha256':sha(ckpath)}
 assert stats['std_population']>EPS and np.isfinite(list(stats[x] for x in ['mean','std_population'])).all()
 statpath=folder/'training_score_statistics.json'
 if statpath.exists():assert read_json(statpath)==stats
 else:save_json(statpath,stats)
 held_scores=score_checkpoint(held,ckpath,folder/'heldout_unique_scores.csv')
 held_scores['z_H_oof']=(held_scores.energy-stats['mean'])/(stats['std_population']+EPS)
 held_scores['fold']=k;held_scores['teacher_sha256']=sha(ckpath);held_scores['teacher_training_mean']=stats['mean'];held_scores['teacher_training_std']=stats['std_population']
 held_scores.to_csv(folder/'heldout_calibrated_scores.csv',index=False)
 print(f'fold {k}: calibrated {len(held_scores)} held-out images from {len(ts)} unique teacher-training images',flush=True)

def assemble():
 verify_inputs();frames=[];teacher_info={}
 for k in range(5):
  f=OUT/'crossfit_EH'/f'fold_{k}';info=read_json(f/'complete.json');teacher_info[str(k)]=info
  assert sha(info['checkpoint'])==info['checkpoint_sha256']
  a=read_csv(f/'teacher_training_pairs.csv',dtype={'ref_id':str});b=read_csv(f/'heldout_pairs.csv',dtype={'ref_id':str})
  assert not set(a.group_key)&set(b.group_key)
  s=read_csv(f/'heldout_calibrated_scores.csv',dtype={'ref_id':str,'image_id':str});assert set(s.group_key)==set(b.group_key)
  frames.append(s)
 unique=pd.concat(frames,ignore_index=True);assert len(unique)==9040 and not unique.duplicated(['dataset','image_id']).any()
 assert np.isfinite(unique[['energy','z_H_oof']].to_numpy()).all()
 pairs=read_csv(OUT/'folds/locked_pairs.csv',dtype={'ref_id':str,'image_id_mild':str,'image_id_severe':str})
 for side in ['mild','severe']:
  s=unique[['dataset','image_id','ref_id','distortion_type','severity','fold','energy','z_H_oof','teacher_sha256']].rename(columns={c:c+'_'+side for c in ['ref_id','distortion_type','severity','fold','energy','z_H_oof','teacher_sha256']})
  pairs=pairs.merge(s,left_on=['dataset','image_id_'+side],right_on=['dataset','image_id'],validate='many_to_one',how='left').drop(columns='image_id')
  for left,right in [('ref_id','ref_id_'+side),('distortion_type','distortion_type_'+side),('sev_'+side,'severity_'+side),('heldout_fold','fold_'+side)]:assert np.array_equal(pairs[left],pairs[right])
  pairs=pairs.drop(columns=['ref_id_'+side,'distortion_type_'+side,'severity_'+side,'fold_'+side])
 assert pairs.teacher_sha256_mild.equals(pairs.teacher_sha256_severe)
 pairs['g_H']=pairs.z_H_oof_severe-pairs.z_H_oof_mild
 levels=read_csv(OUT/'folds/training_severity_ranges.csv')
 pairs=pairs.merge(levels,on=['dataset','distortion_type'],validate='many_to_one')
 pairs['delta']=(pairs.sev_severe-pairs.sev_mild)/(pairs['max']-pairs['min'])
 assert pairs.pair_id.is_unique and len(pairs)==18080 and (pairs.delta>0).all() and (pairs.delta<=1).all()
 assert np.isfinite(pairs[['g_H','delta']].to_numpy()).all()
 unique.to_csv(OUT/'crossfit_EH/oof_unique_scores.csv',index=False)
 pairs.sort_values('pair_id').to_csv(OUT/'crossfit_EH/oof_pair_scores.csv',index=False)
 audit={'status':'PASS','unique_images':9040,'pairs':18080,'heldout_reference_leakage':False,'score_assignment':'exactly one teacher per image','teachers':teacher_info,'reference_tensors_unchanged':True,'fold_specific_selection':False,'all_teachers_epoch':20,'completed_utc':now()}
 save_json(OUT/'crossfit_EH/crossfit_integrity.json',audit)
 (OUT/'crossfit_EH/crossfit_integrity_report.txt').write_text(json.dumps(audit,indent=2)+'\n')
 print('Cross-fit integrity PASS: every training pair has exactly one valid held-out teacher assignment.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','score','assemble']);p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
 if a.stage=='train':train_fold(a.fold)
 elif a.stage=='score':score_fold(a.fold)
 else:assemble()
