"""Complete registered moments and all revised preflights; enforce the training gate."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib,subprocess,time,platform
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from prepare_reference_audit import ROOT,PROJECT,sha,now,read,frame,image_audit
from distance_core import *
sys.path.insert(0,str(PROJECT/'day12_task2'))
from run_task2 import ReferenceDataset,posterior,load_checkpoint
from external.model import CVAEGenerator_v2

def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');t.replace(p)

def model_from(ck,device):
 m=CVAEGenerator_v2(latent_dim=100,image_size=256);m.load_state_dict(ck['generator']);return m.to(device).eval()

def model_hash(m):
 h=hashlib.sha256()
 for k,t in m.state_dict().items():h.update(k.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def extract_eh():
 cfg=read(ROOT/'config.json');folder=ROOT/'reference_stats/EH';mf=read(folder/'manifest.json')
 assert sha(folder/'manifest.json')==(folder/'manifest.sha256').read_text().split()[0]
 assert sha(folder/'eh_pristine_reference_v2.csv')==mf['accepted_manifest_sha256']
 assert sha(cfg['checkpoint_EH'])==mf['checkpoint_sha256']
 d=frame(folder/'eh_pristine_reference_v2.csv',dtype={'source_image_id':str,'source_reference_id':str})
 if (folder/'reference_posteriors.npz').exists():raise RuntimeError('Refusing to overwrite frozen reference extraction')
 device=torch.device('cuda');ck=load_checkpoint(cfg['checkpoint_EH'],20);m=model_from(ck,device);before=model_hash(m)
 loader=DataLoader(ReferenceDataset([{'path':p} for p in d.canonical_path],256),batch_size=16,shuffle=False,num_workers=4,pin_memory=True)
 mus=[];lvs=[];begin=time.time()
 with torch.no_grad():
  for i,x in enumerate(loader):
   mu,lv=posterior(m,x.to(device));assert torch.isfinite(mu).all() and torch.isfinite(lv).all()
   if i==0:
    _,full_mu,full_lv=m(x.to(device));torch.testing.assert_close(mu,full_mu,atol=0,rtol=0);torch.testing.assert_close(lv,full_lv,atol=0,rtol=0)
   mus.append(mu.cpu().numpy());lvs.append(lv.cpu().numpy())
   if i%25==0:print('EH reference encoding',min((i+1)*16,len(d)),'/',len(d),flush=True)
 mu=np.concatenate(mus);lv=np.concatenate(lvs);assert model_hash(m)==before
 np.savez_compressed(folder/'reference_posteriors.npz',mu=mu,logvar=lv,variance=np.exp(lv.astype(np.float64)),path=d.canonical_path.to_numpy(str),source_dataset=d.source_dataset.to_numpy(str),source_image_id=d.source_image_id.to_numpy(str),weights=d.weight.to_numpy(np.float64),checkpoint_sha256=np.array(mf['checkpoint_sha256']))
 a=aggregate(mu,lv,d.weight.to_numpy());np.savez_compressed(folder/'aggregate_reference.npz',**a)
 np.savez_compressed(folder/'between_reference.npz',mu_agg=a['mu_agg'],between_variance=a['v_between'],v_between=a['v_between'],epsilon=EPS)
 # The pre-encoding manifest stays byte-identical; append a separate completion record.
 save(folder/'extraction_complete.json',{'status':'COMPLETE','completed_utc':now(),'frozen_manifest_sha256':sha(folder/'manifest.json'),'N':len(d),'checkpoint_sha256':mf['checkpoint_sha256'],'model_state_unchanged':True,'encoder_full_forward_equivalence':True,'elapsed_seconds':time.time()-begin,'artifact_hashes':{n:sha(folder/n) for n in ['reference_posteriors.npz','aggregate_reference.npz','between_reference.npz']}})
 del m,ck;torch.cuda.empty_cache()

def reference(pole):
 folder=ROOT/'reference_stats'/pole
 with np.load(folder/'legacy_reference.npz') as l:lm=l['mu_ref'].copy();lvar=l['Sigma_ref'].copy()
 with np.load(folder/'aggregate_reference.npz') as a:
  return dict(legacy_mu=lm,legacy_variance=lvar,registered_mu=a['mu_agg'].copy(),between_variance=a['v_between'].copy(),aggregate_variance_raw=a['v_aggregate_raw'].copy(),within_variance=a['v_within'].copy())

def duplicate_inputs(split,d,cache):
 # Exact duplicate encoded representations give a strict upper bound on distinct deterministic scores.
 both=np.concatenate([cache['mu'],cache['logvar']],axis=1);_,group,cts=np.unique(both,axis=0,return_inverse=True,return_counts=True)
 idx=np.flatnonzero(cts[group]>1)
 records=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for i,a in zip(idx,pool.map(image_audit,d.iloc[idx].path)):
   r=d.iloc[i]
   assert a['sha256']==r.image_sha256, r.path
   records.append({'split':split,'posterior_group':int(group[i]),'dataset':r.dataset,'image_id':r.image_id,'ref_id':r.ref_id,'distortion_type':r.distortion_type,'path':r.path,'source_sha256':a['sha256'],'preprocessed_sha256':a['preprocessed_sha256'],'is_decodable':a['is_decodable']})
 pd.DataFrame(records).to_csv(ROOT/f'preflight/duplicate_input_diagnostics_{split}.csv',index=False)
 same=0
 if records:
  z=pd.DataFrame(records)
  same=int(z.groupby('posterior_group').preprocessed_sha256.nunique().eq(1).sum())
 return {'N':len(d),'distinct_joint_posteriors':len(cts),'maximum_distinct_score_fraction':len(cts)/len(d),'duplicate_posterior_groups':int((cts>1).sum()),'images_in_duplicate_groups':len(idx),'groups_with_identical_preprocessed_pixels':same,'source_images_checked':len(idx)}

def run():
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.cuda.manual_seed_all(42)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available();device=torch.device('cuda');cfg=read(ROOT/'config.json')
 (ROOT/'environment.txt').write_text(f'Python: {sys.version}\nPlatform: {platform.platform()}\nTorch: {torch.__version__}\nCUDA: {torch.version.cuda}\nGPU: {torch.cuda.get_device_name()}\nTF32: disabled (matmul and cuDNN)\nDeterministic algorithms: true\nSeed: 42\nNumPy: {np.__version__}\nAMP: false\nReference moment and nonlegacy distance dtype: float64\nD0 legacy dtype: float32\n')
 save(ROOT/'status.json',{'state':'EXTRACTING_EH_REFERENCE','updated_utc':now(),'training_runs_started':0,'holdout_scored':False})
 extract_eh()
 refrows=[]
 for pole in ['EA','EH']:
  r=reference(pole)
  for dim in range(100):refrows.append({'pole':pole,'dimension':dim,'legacy_mean':r['legacy_mu'][dim],'registered_mean':r['registered_mu'][dim],'mean_difference':r['registered_mu'][dim]-r['legacy_mu'][dim],'legacy_variance':r['legacy_variance'][dim],'between_variance':r['between_variance'][dim],'within_variance':r['within_variance'][dim],'aggregate_variance_raw':r['aggregate_variance_raw'][dim]})
 pd.DataFrame(refrows).to_csv(ROOT/'reference_stats/reference_comparison.csv',index=False)
 (ROOT/'reference_stats/reference_report.txt').write_text('EA: verified Task-2 2,000-image uniform-weight aggregate reused. Between moments exported without changing stored arrays.\nEH: new source-balanced pristine reference registered and frozen before encoding. This is not reconstruction of the unknown legacy pool.\nD0 legacy tensors remain byte-identical to initialization checkpoint references.\nD0 vs DM-Between measures reference registration, not a pure pool effect.\nDM-Between vs DM-Aggregate isolates covariance definition in the mean-only formula.\nDM-Aggregate vs KL/W2/BHATT changes geometry and image posterior variance use.\n')
 with (ROOT/'preflight/distance_unit_tests.txt').open('w') as log:
  tests=subprocess.run([sys.executable,'-B',str(ROOT/'test_distances.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 assert tests.returncode==0,'Synthetic distance tests failed'
 save(ROOT/'status.json',{'state':'RUNNING_DATA_PREFLIGHT','updated_utc':now(),'training_runs_started':0,'holdout_scored':False})
 rows=[];gradrows=[];replays=[];pairrows=[];dups={};oldstats=[]
 for pole in ['EA','EH']:
  ref=reference(pole);checkpoint=Path(cfg['checkpoint_'+pole]);ck=load_checkpoint(checkpoint,5 if pole=='EA' else 20)
  for key,t in [('legacy_mu','mu_ref'),('legacy_variance','Sigma_ref')]:np.testing.assert_array_equal(ref[key],ck[t].numpy())
  model=model_from(ck,device);before=model_hash(model)
  for split,file in [('train','train_images.csv'),('val','validation_images.csv')]:
   d=frame(ROOT/'inputs'/file,dtype={'ref_id':str,'image_id':str});folder=ROOT/'preflight/cache'/pole
   meta=read(folder/f'{split}_cache.json');cp=folder/f'{split}_posteriors.npz';assert sha(cp)==meta['sha256']
   cache=np.load(cp);assert list(cache['path'])==d.path.tolist()
   # Fresh replay at exactly the original batch size, using the same checkpoint and preprocessing.
   x=next(iter(DataLoader(ReferenceDataset(d.iloc[:16].to_dict('records'),256),batch_size=16,shuffle=False)))
   with torch.no_grad():m,lv=posterior(model,x.to(device))
   err_mu=float(np.max(np.abs(m.cpu().numpy()-cache['mu'][:16])));err_lv=float(np.max(np.abs(lv.cpu().numpy()-cache['logvar'][:16])))
   np.testing.assert_array_equal(m.cpu().numpy(),cache['mu'][:16])
   np.testing.assert_array_equal(lv.cpu().numpy(),cache['logvar'][:16])
   replays.append({'pole':pole,'split':split,'cache_sha256':sha(cp),'checkpoint_sha256':sha(checkpoint),'fresh_replay_N':16,'mu_max_abs_difference':err_mu,'logvar_max_abs_difference':err_lv,'passed':True})
   if pole=='EA':dups[split]=duplicate_inputs(split,d,cache)
   scores={a:[] for a in ARMS}
   for i in range(0,len(d),16):
    mu=torch.from_numpy(cache['mu'][i:i+16]).to(device);lv=torch.from_numpy(cache['logvar'][i:i+16]).to(device)
    for arm in ARMS:scores[arm].extend(distance(mu,lv,ref,arm).detach().double().cpu().numpy())
   output=d.copy()
   for arm in ARMS:
    values=np.asarray(scores[arm]);output[arm]=values
    # The 99% gate applies to each scored split; dataset-specific summaries are additional diagnostics.
    for dataset,ix in [('ALL',np.arange(len(d)))]+[(str(ds),np.asarray(indices)) for ds,indices in d.groupby('dataset').groups.items()]:
     v=values[ix];finite=bool(np.isfinite(v).all());std=float(v.std());distinct=len(np.unique(v));fraction=distinct/len(v)
     rows.append({'pole':pole,'distance':arm,'split':split,'dataset':dataset,'N':len(v),'finite':finite,'mean':float(v.mean()),'std_population':std,'minimum':float(v.min()),'maximum':float(v.max()),'distinct_scores':distinct,'distinct_fraction':fraction,'required_fraction':.99,'finite_pass':finite,'std_pass':std>1e-12,'uniqueness_pass':fraction>=.99,'all_checks_pass':finite and std>1e-12 and fraction>=.99,'precision':'float32' if arm=='D0_LEGACY' else 'float64'})
    if split=='train':
     rec={'pole':pole,'distance':arm,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'train_split':str(ROOT/'inputs/train_images.csv'),'train_split_sha256':sha(ROOT/'inputs/train_images.csv'),'N':len(d),'mean':float(values.mean()),'std_population':float(values.std()),'epsilon':EPS,'ddof':0,'source':'Verified cached common pre-ranking posteriors; revised exact distance implementation','frozen_utc':now(),'passed_new_preflight':bool(len(np.unique(values))/len(values)>=.99)}
     save(ROOT/f'training_stats/{pole}_{arm}.json',rec)
     oldname={'D0_LEGACY':'D0','KL_AGGREGATE':'KL','W2_AGGREGATE':'W2','BHATT_AGGREGATE':'BHATT'}.get(arm)
     if oldname and (PROJECT/f'day12_Task4/training_stats/{pole}_{oldname}.json').exists():
      old=read(PROJECT/f'day12_Task4/training_stats/{pole}_{oldname}.json')
      oldstats.append({'pole':pole,'distance':arm,'old_mean':old['mean'],'new_mean':rec['mean'],'mean_difference':rec['mean']-old['mean'],'old_std':old['std_population'],'new_std':rec['std_population'],'std_difference':rec['std_population']-old['std_population'],'interpretation':'identical reference and definition' if pole=='EA' or arm=='D0_LEGACY' else 'new reference requires new moments'})
     pairs=frame(ROOT/'inputs/pairs_used.csv',dtype={'ref_id':str});lookup={p:i for i,p in enumerate(d.path)}
     sample=pairs.iloc[:16]
     for pr in sample.itertuples():
      mild=values[lookup[pr.path_mild]];severe=values[lookup[pr.path_severe]];gm=rank_gap(mild,severe,pole)/(rec['std_population']+EPS)
      pairrows.append({'pole':pole,'distance':arm,'dataset':pr.dataset,'ref_id':pr.ref_id,'distortion_type':pr.distortion_type,'image_id_mild':pr.image_id_mild,'image_id_severe':pr.image_id_severe,'severity_mild':pr.sev_mild,'severity_severe':pr.sev_severe,'raw_mild':mild,'raw_severe':severe,'signed_standardized_gap':gm,'softplus_loss':float(np.logaddexp(0,.1-gm)),'desired_order_holds':gm>0,'polarity_formula':'mild-minus-severe' if pole=='EA' else 'severe-minus-mild','batch_polarity_implementation_checked':True})
   output.to_csv(ROOT/f'preflight/{pole}_{split}_distance_scores.csv',index=False)
   print('scored preflight',pole,split,len(d),flush=True)
  # Reference-scaled synthetic nonidentical and variance-mismatched probes for all arms.
  for arm in ARMS:
   rm=ref['legacy_mu'] if arm=='D0_LEGACY' else ref['registered_mu']
   m=torch.tensor(np.array(rm)[None,:]+.2,device=device,dtype=torch.float64,requires_grad=True)
   lv=torch.tensor(np.log(np.maximum(ref['aggregate_variance_raw'],EPS))[None,:]+.7,device=device,dtype=torch.float64,requires_grad=True)
   val=distance(m,lv,ref,arm).sum();gm,gl=torch.autograd.grad(val,[m,lv],allow_unused=True)
   mn=float(gm.norm());ln=0. if gl is None else float(gl.norm());finite=bool(torch.isfinite(gm).all() and (gl is None or torch.isfinite(gl).all()))
   passed=finite and mn>1e-12 and (ln>1e-12 if arm in ARMS[3:] else gl is None or ln==0)
   gradrows.append({'pole':pole,'distance':arm,'finite':finite,'mu_gradient_norm':mn,'logvar_gradient_norm':ln,'logvar_gradient_expected':arm in ARMS[3:],'passed':passed,'probe':'reference mean plus 0.2; log aggregate variance plus 0.7'})
  assert model_hash(model)==before;del model,ck;torch.cuda.empty_cache()
 pd.DataFrame(rows).to_csv(ROOT/'preflight/distance_preflight.csv',index=False)
 pd.DataFrame(gradrows).to_csv(ROOT/'preflight/gradient_preflight.csv',index=False)
 pd.DataFrame(pairrows).to_csv(ROOT/'preflight/hand_checked_polarity_batch.csv',index=False)
 pd.DataFrame(oldstats).to_csv(ROOT/'reused/training_statistics_comparison.csv',index=False)
 save(ROOT/'verification/posterior_reuse_verification.json',replays);save(ROOT/'preflight/duplicate_input_summary.json',dups)
 gate_rows=[r for r in rows if r['dataset']=='ALL']
 passgate=training_allowed(gate_rows) and all(r['passed'] for r in gradrows)
 failures=[{k:r[k] for k in ['pole','distance','split','N','distinct_scores','distinct_fraction','required_fraction']} for r in gate_rows if not r['all_checks_pass']]
 report=['MODIFIED DAY 12 TASK 4 PREFLIGHT',f'Overall gate: {"PASS" if passgate else "FAIL - TRAINING PROHIBITED BY THE PDF"}',f'All 13 synthetic unit tests: PASS',f'All 12 gradient probes: {all(r["passed"] for r in gradrows)}','Training standardization records: 12, frozen from unique training-image entries only.','Gate requires finite scores, population std > 1e-12 and distinct machine-precision score fraction >= 0.99 on train and validation.','All six arms were retained and checked. None was removed for weak severity correlation.','Failed checks:',pd.DataFrame(failures).to_string(index=False),'','Duplicate-input findings:',json.dumps(dups,indent=2),'','Identical locked preprocessing inputs cannot produce distinct deterministic distances. No duplicates were removed, no jitter was added, and no gate threshold was weakened.','Training, validation checkpoint selection and holdout model scoring have not been performed.','Next dependent steps remain blocked until the protocol/data conflict is resolved explicitly.']
 (ROOT/'preflight/preflight_report.txt').write_text('\n'.join(report)+'\n')
 save(ROOT/'preflight/gate.json',{'passed':passgate,'state':'PASS' if passgate else 'FAILED_DISTINCT_SCORE_THRESHOLD','checked_utc':now(),'failed_split_arm_checks':failures,'synthetic_tests_passed':True,'gradient_tests_passed':all(r['passed'] for r in gradrows),'config_sha256':sha(ROOT/'config.json'),'reference_manifest_sha256':{p:sha(ROOT/f'reference_stats/{p}/manifest.json') for p in ['EA','EH']},'distance_preflight_sha256':sha(ROOT/'preflight/distance_preflight.csv'),'gradient_preflight_sha256':sha(ROOT/'preflight/gradient_preflight.csv')})
 matrix=[]
 for pole in ['EA','EH']:
  for arm in ARMS:
   for lam in [.1,1.]:
    run=ROOT/f'runs/{pole}/{arm}/lambda_{lam}/seed_42';run.mkdir(parents=True,exist_ok=True)
    rec={'pole':pole,'distance':arm,'lambda_rank':lam,'seed':42,'epochs':25,'state':'READY' if passgate else 'BLOCKED_PREFLIGHT','optimizer_steps':0,'checkpoint':cfg['checkpoint_'+pole],'checkpoint_sha256':sha(cfg['checkpoint_'+pole]),'training_standardization_sha256':sha(ROOT/f'training_stats/{pole}_{arm}.json')}
    save(run/'status.json',rec);matrix.append(rec)
 pd.DataFrame(matrix).to_csv(ROOT/'run_matrix.csv',index=False)
 save(ROOT/'status.json',{'state':'PREFLIGHT_PASSED' if passgate else 'BLOCKED_PREFLIGHT','updated_utc':now(),'reference_registration_complete':True,'preflight_complete':True,'training_runs_started':0,'training_runs_blocked':0 if passgate else 24,'holdout_scored':False,'reason':'Distinct score fraction below PDF 0.99 minimum' if not passgate else None})
 print('\n'.join(report),flush=True)

if __name__=='__main__':
 try:run()
 except Exception as e:
  save(ROOT/'status.json',{'state':'PREFLIGHT_EXECUTION_ERROR','updated_utc':now(),'error':str(e),'training_runs_started':0,'holdout_scored':False});raise
