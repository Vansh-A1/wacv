"""Register revised Task 4 inputs and audit candidates without reading opinion labels."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib,json,shutil,subprocess,re,sys,csv
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT))
from external.dataloader import _resize_short_side,center_crop

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def read(p):return json.loads(Path(p).read_text())
def canonical(p):return str(Path(str(p).replace('/data/projectwork/swati_mam/kadid10k/','/data/projectwork/swati_mam/new model data/kadid10k/')).resolve())
def frame(path,**kw):return pd.read_csv(path,float_precision='round_trip',**kw)
def image_audit(path):
 out={'canonical_path':canonical(path),'sha256':'','is_decodable':False,'rgb_sha256':'','preprocessed_sha256':'','width':None,'height':None,'decode_error':''}
 try:
  out['sha256']=sha(out['canonical_path'])
  with Image.open(out['canonical_path']) as im:
   rgb=im.convert('RGB');rgb.load();out['width'],out['height']=rgb.size
   out['rgb_sha256']=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
   crop=center_crop(_resize_short_side(rgb,256),256)
   x=(np.asarray(crop)/255.).astype('float32');assert x.shape==(256,256,3) and np.isfinite(x).all()
   out['preprocessed_sha256']=hashlib.sha256(x.tobytes()).hexdigest();out['is_decodable']=True
 except Exception as e:out['decode_error']=str(e)
 return out

def main():
 for d in ['inputs','reused/old_task4','reference_stats/EA','reference_stats/EH','preflight','training_stats','runs','selection','holdout','verification','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
 save(ROOT/'status.json',{'state':'PREPARING_REFERENCE_AUDIT','updated_utc':now(),'training_runs_started':0,'holdout_scored':False})
 pdf=Path('/home/projectwork/Downloads/modified_Day12_Task_4.pdf')
 if not (ROOT/'inputs/modified_Day12_Task_4.pdf').exists():shutil.copy2(pdf,ROOT/'inputs'/pdf.name)
 assert sha(pdf)==sha(ROOT/'inputs'/pdf.name)
 subprocess.run(['pdftotext','-layout',str(pdf),str(ROOT/'inputs/requirements.txt')],check=True)
 old=PROJECT/'day12_Task4';oldhash={str(p.relative_to(old)):sha(p) for p in old.rglob('*') if p.is_file()}
 save(ROOT/'verification/original_task4_hashes.json',oldhash)
 shutil.copytree(old,ROOT/'reused/old_task4',dirs_exist_ok=True)
 inherited=read(old/'config.json')['expected_hashes'];checked=[]
 for p,h in inherited.items():
  if sha(p)!=h:raise RuntimeError('Changed historical input '+p)
  checked.append(p)
 # Copy the verified inputs and both cached posterior sets without claiming their old gate passes.
 copying={}
 for f in ['train_images.csv','validation_images.csv','pairs_used.csv','source_hashes.json','run_identity.json','protocol.json','split_audit.json']:
  copying[f'inputs/{f}']=PROJECT/'day12_task3_preflight/inputs'/f
 for pole in ['EA','EH']:
  for name in ['train_posteriors.npz','val_posteriors.npz','train_cache.json','val_cache.json']:
   copying[f'preflight/cache/{pole}/{name}']=PROJECT/'day12_task3_preflight'/pole/name
  copying[f'reference_stats/{pole}/legacy_reference.npz']=PROJECT/'day12_task3_preflight'/pole/'legacy_reference.npz'
 for name in ['reference_posteriors.npz','aggregate_reference.npz','manifest.json']:
  copying[f'reference_stats/EA/{name}']=PROJECT/'day12_task2/reference_stats/EA'/name
 m=read(PROJECT/'day12_task2/reference_stats/EA/manifest.json')
 for n,key in [('reference_posteriors.npz','posterior_sha256'),('aggregate_reference.npz','aggregate_sha256')]:assert sha(PROJECT/'day12_task2/reference_stats/EA'/n)==m[key]
 records=[]
 for dest,source in copying.items():
  q=ROOT/dest;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,q)
  assert sha(q)==sha(source);records.append({'destination':dest,'source':str(source),'sha256':sha(source),'use':'Identical input/reference/posterior reused; revised distances and preflight recomputed'})
 agg=np.load(ROOT/'reference_stats/EA/aggregate_reference.npz')
 np.savez_compressed(ROOT/'reference_stats/EA/between_reference.npz',mu_agg=agg['mu_agg'],between_variance=agg['v_between'],v_between=agg['v_between'],epsilon=1e-8)
 save(ROOT/'reused/reuse_manifest.json',{'copied_old_task4_file_count':len(oldhash),'verified_historical_hash_count':len(checked),'copied_artifacts':records,'old_training_runs':0,'old_training_results_reusable':False,'reason':'Old task stopped before training; revised rule uses softplus, six arms, new EH reference and new preflight.'})
 config={'task':'Modified Day12 Task4 matched-distance ablation','source_pdf_sha256':sha(pdf),'seed':42,'epsilon':1e-8,'poles':['EA','EH'],'distances':['D0_LEGACY','DM_BETWEEN','DM_AGGREGATE','KL_AGGREGATE','W2_AGGREGATE','BHATT_AGGREGATE'],'lambdas':[.1,1.0],'epochs':25,'save_every':5,'primary_runs':24,'preflight_unique_score_fraction_min':.99,'preflight_std_min':1e-12,'reference_weights_EA':'1/2000','reference_weights_EH':'1/(accepted_source_count * accepted_image_count_for_source)','rank_loss':'mean(softplus(0.1 - signed_standardized_gap))','EA_gap':'mild - severe','EH_gap':'severe - mild','optimizer':'Adam','lr':1e-4,'betas':[.5,.999],'batch_pairs':8,'vae_loss':'L1 + 0.1*VGGPerceptualLoss + clamp(KL_raw,min=1.0)','preprocessing':'RGB; upscale only if short side <256; 256 center crop; float32 [0,1]; no augmentation or AMP','preflight_precision':'historical float32 D0; float64 registered distances/reference statistics','checkpoint_EA':str(PROJECT/'dataset_model2/checkpoints/wacv_ea_day7_seed42/epoch_0005.pth'),'checkpoint_EH':str(PROJECT/'checkpoints/hr_combined_ft1/best.pth'),'bootstrap_draws':5000,'bootstrap_seed':20260912,'holdout_opened_for_scoring':False,'MOS_DMOS_used':False}
 save(ROOT/'config.json',config)
 # Freeze source labels before any reference moments or encoder calls.
 base='/data/projectwork/swati_mam/HR_DATA/HR-train/'
 mapping=[{'prefix':base+'DIV2K/','source_dataset':'DIV2K','kind':'HR','filename_regex':r'\d{4}\.png','source_url':'https://data.vision.ee.ethz.ch/cvl/DIV2K/','evidence':'Local HR-train/DIV2K original-size PNG branch; source describes high-quality HR originals.'},{'prefix':base+'Flickr2K(1)/Flickr2K/Flickr2K_HR/','source_dataset':'Flickr2K','kind':'HR','filename_regex':r'\d{6}\.png','source_url':'https://github.com/LimBee/NTIRE2017','evidence':'Published Flickr2K_HR branch of the local source collection.'},{'prefix':base+'Flickr2K(1)/Flickr2K/Flickr2K_LR_bicubic/','source_dataset':'Flickr2K','kind':'LR_bicubic','evidence':'Explicit synthetically downsampled branch; excluded.'},{'prefix':base+'Flickr2K(1)/Flickr2K/Flickr2K_LR_unknown/','source_dataset':'Flickr2K','kind':'LR_unknown','evidence':'Explicit degraded/downsampled branch; excluded.'}]
 mapfile=ROOT/'reference_stats/EH/path_to_source_mapping.json'
 if mapfile.exists():assert read(mapfile)['rules']==mapping
 else:save(mapfile,{'created_utc':now(),'rules':mapping,'verification_scope':'Dataset/HR branch identity, original filename pattern, successful decode, exact and RGB hashes, protected reference disjointness. No MOS/DMOS and no per-image subjective quality threshold.','unknown_policy':'exclude','prohibited_datasets':['KADID-10k','TID2013','LIVE','CSIQ','KonIQ']})
 # Protected identities: metadata and content hashes only; no evaluation/model scores or opinion columns.
 protected=[]
 split=PROJECT/'dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv'
 df=frame(split,usecols=['dataset','image_id','ref_id','distortion_type','split','distorted_path','ref_path'],dtype=str,keep_default_na=False)
 for r in df.itertuples():
  if r.dataset in ['KADID-10k','TID2013']:
   protected.append({'source_dataset':r.dataset,'source_reference_id':r.ref_id,'path':canonical(r.ref_path),'reason':'all_KADID_TID_reference_content'})
  if r.split!='train':protected.append({'source_dataset':r.dataset,'source_reference_id':r.ref_id,'path':canonical(r.distorted_path),'reason':'protected_evaluation_image'})
 cs=frame(PROJECT/'day12_Task5/csiq/manifest.csv',usecols=['dataset','image_id','ref_id','path','ref_path'],dtype=str)
 for r in cs.itertuples():
  for k in ['path','ref_path']:protected.append({'source_dataset':'CSIQ','source_reference_id':r.ref_id,'path':canonical(getattr(r,k)),'reason':'all_CSIQ_content'})
 live=Path('/data/projectwork/swati_mam/FLIVE/databaserelease2')
 assert (live/'refimgs').is_dir()
 for p in (live/'refimgs').glob('*'):
  if p.suffix.lower() in ['.bmp','.png','.jpg']:protected.append({'source_dataset':'LIVE','source_reference_id':p.stem,'path':canonical(p),'reason':'all_LIVE_reference_content'})
 for branch in ['HR-test','HR-calibration']:
  for p in Path('/data/projectwork/swati_mam/HR_DATA',branch).rglob('*'):
   if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']:protected.append({'source_dataset':'DIV2K','source_reference_id':p.stem,'path':canonical(p),'reason':'protected_'+branch})
 prot=pd.DataFrame(protected).drop_duplicates(['path']).reset_index(drop=True)
 protpaths=prot.path.tolist()
 print('Auditing protected identities:',len(prot),flush=True)
 with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(image_audit,protpaths))
 pa=pd.concat([prot,pd.DataFrame(results).drop(columns=['canonical_path'])],axis=1)
 pa.to_csv(ROOT/'reference_stats/EH/protected_identity_audit.csv',index=False)
 if not pa.is_decodable.all():raise RuntimeError('Protected content unavailable; cannot certify disjointness')
 protected_paths=set(pa.path);protected_hashes=set(pa.sha256);protected_rgb=set(pa.rgb_sha256)
 protected_ids=set(zip(pa.source_dataset,pa.source_reference_id))
 candidate_file=PROJECT/'runs/hr_combined_ft1/train_files.txt';candidates=candidate_file.read_text().splitlines()
 assert len(candidates)==19300
 print('Auditing candidate images:',len(candidates),flush=True)
 results=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for i,out in enumerate(pool.map(image_audit,candidates)):
   out.update(candidate_index=i,original_path=candidates[i]);results.append(out)
   if (i+1)%1000==0:
    print('candidate audit',i+1,'/',len(candidates),flush=True)
    save(ROOT/'reference_stats/EH/audit_progress.json',{'candidates_checked':i+1,'total':len(candidates),'updated_utc':now()})
 rows=[];seen_paths=set();seen_hashes=set();seen_rgb=set()
 for out in results:
  p=out['canonical_path'];rule=next((r for r in mapping if p.startswith(r['prefix'])),None)
  ds=rule['source_dataset'] if rule else 'UNKNOWN';kind=rule['kind'] if rule else 'UNKNOWN'
  sid=Path(p).stem;rid=re.sub(r'x[234]$','',sid)
  verified=bool(rule and kind=='HR' and re.fullmatch(rule['filename_regex'],Path(p).name) and min(out['width'] or 0,out['height'] or 0)>=256)
  reasons=[]
  if not out['is_decodable']:reasons.append('not_decodable')
  if kind.startswith('LR'):reasons.append('synthetically_degraded_low_resolution')
  elif not verified:reasons.append('source_or_pristine_status_unverified')
  if p in seen_paths:reasons.append('duplicate_canonical_path')
  if out['sha256'] and out['sha256'] in seen_hashes:reasons.append('duplicate_exact_image_hash')
  if out['rgb_sha256'] and out['rgb_sha256'] in seen_rgb:reasons.append('duplicate_decoded_RGB_content')
  if p in protected_paths:reasons.append('protected_canonical_path')
  if out['sha256'] in protected_hashes:reasons.append('protected_exact_hash')
  if out['rgb_sha256'] in protected_rgb:reasons.append('protected_RGB_content')
  if (ds,rid) in protected_ids:reasons.append('protected_dataset_reference_id')
  accepted=not reasons
  if accepted:seen_paths.add(p);seen_hashes.add(out['sha256']);seen_rgb.add(out['rgb_sha256'])
  rows.append({**out,'source_dataset':ds,'source_image_id':sid,'source_reference_id':rid,'source_kind':kind,'is_verified_pristine':verified,'split_membership_decision':'INCLUDE' if accepted else 'EXCLUDE','exclusion_reason':';'.join(reasons),'accepted':accepted})
 audit=pd.DataFrame(rows);audit.to_csv(ROOT/'reference_stats/EH/candidate_audit.csv',index=False)
 audit[~audit.accepted].to_csv(ROOT/'reference_stats/EH/eh_pristine_reference_v2_exclusions.csv',index=False)
 accepted=audit[audit.accepted].copy()
 if len(accepted)==0:raise RuntimeError('NO_ELIGIBLE_EH_REFERENCE_SOURCE: exclusions must not be weakened')
 counts=accepted.source_dataset.value_counts().to_dict();S=len(counts)
 accepted['weight']=accepted.source_dataset.map(lambda s:1./(S*counts[s])).astype('float64')
 assert abs(accepted.weight.sum()-1)<=1e-12
 assert np.allclose(accepted.groupby('source_dataset').weight.sum(),1/S,rtol=0,atol=1e-12)
 cols=['source_dataset','source_image_id','source_reference_id','canonical_path','sha256','rgb_sha256','preprocessed_sha256','weight']
 accepted[cols].to_csv(ROOT/'reference_stats/EH/eh_pristine_reference_v2.csv',index=False)
 report=['EH-reference-v2: newly registered pristine reference pool; not a reconstruction of the unavailable legacy pool.',f'Candidate images: {len(audit)}',f'Accepted images: {len(accepted)}',f'Excluded images: {len(audit)-len(accepted)}',f'Included by source: {counts}',f'Weight sum: {accepted.weight.sum():.17g}',f'Source weight sums: {accepted.groupby("source_dataset").weight.sum().to_dict()}',f'Protected images/reference files audited: {len(pa)}','Accepted images are disjoint by canonical path, exact SHA-256, dataset-qualified reference ID and decoded RGB hash.','Pristine status uses the registered HR source branches, published dataset descriptions and valid full-size files.','No MOS/DMOS columns were read.','Exclusion counts by source and reason:',audit[~audit.accepted].groupby(['source_dataset','exclusion_reason']).size().to_string()]
 (ROOT/'reference_stats/EH/reference_audit_report.txt').write_text('\n'.join(report)+'\n')
 manifest={'status':'FROZEN_BEFORE_ENCODING','frozen_utc':now(),'declaration':report[0],'candidate_list':str(candidate_file),'candidate_list_sha256':sha(candidate_file),'source_mapping_sha256':sha(mapfile),'accepted_manifest_sha256':sha(ROOT/'reference_stats/EH/eh_pristine_reference_v2.csv'),'exclusions_sha256':sha(ROOT/'reference_stats/EH/eh_pristine_reference_v2_exclusions.csv'),'candidate_audit_sha256':sha(ROOT/'reference_stats/EH/candidate_audit.csv'),'protected_identity_audit_sha256':sha(ROOT/'reference_stats/EH/protected_identity_audit.csv'),'reference_count':len(accepted),'source_counts':counts,'source_weights':accepted.groupby('source_dataset').weight.sum().to_dict(),'weight_sum':float(accepted.weight.sum()),'checkpoint':config['checkpoint_EH'],'checkpoint_sha256':sha(config['checkpoint_EH']),'preprocessing':config['preprocessing'],'MOS_DMOS_used':False,'legacy_reconstruction':False}
 save(ROOT/'reference_stats/EH/manifest.json',manifest)
 (ROOT/'reference_stats/EH/manifest.sha256').write_text(sha(ROOT/'reference_stats/EH/manifest.json')+'  manifest.json\n')
 print('\n'.join(report),flush=True)
 save(ROOT/'status.json',{'state':'REFERENCE_MANIFEST_FROZEN','updated_utc':now(),'training_runs_started':0,'holdout_scored':False,'accepted_EH_images':len(accepted)})

if __name__=='__main__':
 try:main()
 except Exception as e:
  save(ROOT/'status.json',{'state':'REFERENCE_AUDIT_FAILED','updated_utc':now(),'error':str(e),'training_runs_started':0,'holdout_scored':False});raise
