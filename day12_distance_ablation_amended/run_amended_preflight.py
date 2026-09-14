"""Exhaustive input hash audit; no input, primary score, or statistic is overwritten."""
from concurrent.futures import ThreadPoolExecutor
import subprocess,time,platform
from PIL import Image
from torch.utils.data import DataLoader
from common import *
from amendment_gate import representative_indices,amended_training_allowed,check_row
from external.dataloader import _resize_short_side,center_crop
DIAG=ROOT/'preflight/amended'

def audit_image(record):
 file_sha=sha(record['path']);require(file_sha==record['image_sha256'],f'Image bytes changed: {record["path"]}')
 with Image.open(record['path']) as source:
  cropped=center_crop(_resize_short_side(source.convert('RGB'),256),256)
  hwc=(np.asarray(cropped)/255.).astype(np.float32)
 chw=np.ascontiguousarray(hwc.transpose(2,0,1))
 require(chw.shape==(3,256,256) and np.isfinite(chw).all(),'Invalid preprocessed tensor')
 return {'source_sha256':file_sha,'preprocessed_tensor_sha256':hashlib.sha256(chw.tobytes()).hexdigest(),'legacy_HWC_sha256':hashlib.sha256(hwc.tobytes()).hexdigest()}

def run():
 require(not (ROOT/'amendment_lock.json').exists(),'Amendment is frozen; use verify_results.py, not a new preflight')
 DIAG.mkdir(exist_ok=True);device=configure();base.set_seed(42)
 baseline=read(ROOT/'verification/baseline_snapshot.json')
 copied={}
 for rel,digest in baseline['files'].items():
  require(sha(ROOT/'prior_work'/rel)==digest,f'Prior snapshot changed: {rel}')
  require(sha(Path(baseline['source'])/rel)==digest,f'Original work changed: {rel}')
  if (ROOT/rel).is_file() and (rel.startswith(('inputs/','reference_stats/','training_stats/','preflight/')) or rel in ('distance_core.py','test_distances.py')):
   require(sha(ROOT/rel)==digest,f'Inherited primary file changed: {rel}');copied[rel]=digest
 save(ROOT/'verification/preserved_primary_artifacts.json',{'all_identical':True,'checked_utc':now(),'files':copied})
 with (DIAG/'unit_tests.txt').open('w') as log:
  result=subprocess.run([sys.executable,'-B','-m','unittest','test_distances','test_amendment','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 require(result.returncode==0,'Synthetic/amendment tests failed')
 groups={};dups={}
 for split,filename in [('train','train_images.csv'),('val','validation_images.csv')]:
  d=csvread(ROOT/'inputs'/filename,dtype={'ref_id':str,'image_id':str})
  records=[];last=time.time()
  with ThreadPoolExecutor(max_workers=6) as pool:
   for i,a in enumerate(pool.map(audit_image,d.to_dict('records'))):
    records.append(a)
    if time.time()-last>30:print(f'Input tensor audit {split}: {i+1}/{len(d)}',flush=True);last=time.time()
  out=pd.concat([d,pd.DataFrame(records)],axis=1);out.insert(0,'row_index',np.arange(len(d)))
  representatives,mapping=representative_indices(out.preprocessed_tensor_sha256.tolist())
  out['representative_row_index']=mapping
  counts=out.preprocessed_tensor_sha256.value_counts();out['group_size']=out.preprocessed_tensor_sha256.map(counts)
  out['is_representative']=out.row_index==out.representative_row_index
  out.to_csv(DIAG/f'{split}_input_tensor_groups.csv',index=False)
  out[out.group_size>1].to_csv(DIAG/f'{split}_duplicate_input_members.csv',index=False)
  duplicates=[]
  for h,g in out[out.group_size>1].groupby('preprocessed_tensor_sha256',sort=False):
   duplicates.append({'split':split,'preprocessed_tensor_sha256':h,'representative_row_index':int(g.row_index.iloc[0]),'size':len(g),'members':g[['row_index','dataset','image_id','ref_id','distortion_type','severity','path','source_sha256']].to_dict('records')})
  save(DIAG/f'{split}_duplicate_input_groups.json',duplicates)
  groups[split]=(out,representatives,mapping)
  dups[split]={'all_rows':len(d),'representatives':len(representatives),'duplicate_groups':len(duplicates),'rows_in_duplicate_groups':int((out.group_size>1).sum()),'redundant_input_rows':len(d)-len(representatives),'tensor_layout':'contiguous CHW little-endian float32, shape (3,256,256), raw bytes; no rounding'}
  print('Exhaustive tensor audit complete',split,dups[split],flush=True)
 require(not set(groups['train'][0].preprocessed_tensor_sha256)&set(groups['val'][0].preprocessed_tensor_sha256),'Cross-split exact tensor overlap')
 save(DIAG/'duplicate_input_summary.json',dups)
 rows=[];replay=[];gradrows=[];collision_summary=[];stats_verified=[]
 for pole in ('EA','EH'):
  ref=reference(pole);cfg=read(ROOT/'config.json');model,ck,mr,vr,_,_=base.load_model(Path(cfg['checkpoint_'+pole]),device)
  np.testing.assert_array_equal(ref['legacy_mu'],mr.cpu().numpy());np.testing.assert_array_equal(ref['legacy_variance'],vr.cpu().numpy())
  for split in ('train','val'):
   d,ri,mapping=groups[split];cachepath=ROOT/f'preflight/cache/{pole}/{split}_posteriors.npz';cache=np.load(cachepath)
   require(sha(cachepath)==read(cachepath.with_name(f'{split}_cache.json'))['sha256'],'Cache hash mismatch')
   require(list(cache['path'])==d.path.tolist(),'Cache order differs')
   for first in (0,(len(d)//16-1)*16):
    indices=np.arange(first,min(first+16,len(d)));batch=d.iloc[indices]
    x=next(iter(DataLoader(ReferenceDataset(batch.to_dict('records'),256),batch_size=16)))
    with torch.no_grad():m,lv=posterior(model,x.to(device))
    np.testing.assert_array_equal(m.cpu().numpy(),cache['mu'][indices]);np.testing.assert_array_equal(lv.cpu().numpy(),cache['logvar'][indices])
    replay.append({'pole':pole,'split':split,'first_row':int(first),'N':len(indices),'posterior_bit_exact':True})
   # All duplicate tensors must also have identical cached representations.
   np.testing.assert_array_equal(cache['mu'],cache['mu'][mapping]);np.testing.assert_array_equal(cache['logvar'],cache['logvar'][mapping])
   primary=csvread(ROOT/f'preflight/{pole}_{split}_distance_scores.csv')
   require(primary.path.tolist()==d.path.tolist(),'Primary score row order changed')
   recomputed={a:[] for a in ARMS}
   for i in range(0,len(d),16):
    mu=torch.from_numpy(cache['mu'][i:i+16]).to(device);lv=torch.from_numpy(cache['logvar'][i:i+16]).to(device)
    for a in ARMS:recomputed[a].extend(distance(mu,lv,ref,a).detach().double().cpu().tolist())
   diag0=[]
   for i in range(0,len(ri),16):
    ids=ri[i:i+16];mu=torch.from_numpy(cache['mu'][ids]).to(device).double();lv=torch.from_numpy(cache['logvar'][ids]).to(device).double()
    diag0.extend(distance(mu,lv,ref,'D0_LEGACY').cpu().tolist())
   diagnostic=d.iloc[ri][['row_index','preprocessed_tensor_sha256','path']].copy()
   for arm in ARMS:
    values=np.asarray(recomputed[arm],np.float64);old=primary[arm].to_numpy(np.float64)
    np.testing.assert_array_equal(values,old)
    rep=values[ri];dv=np.asarray(diag0) if arm=='D0_LEGACY' else rep
    diagnostic[arm+('_float64_diagnostic_only' if arm=='D0_LEGACY' else '_primary_precision')]=dv
    row={'pole':pole,'distance':arm,'split':split,'N_primary':len(values),'N_representatives':len(rep),'primary_precision':'float32' if arm=='D0_LEGACY' else 'float64','diagnostic_precision':'float64','finite':bool(np.isfinite(values).all()),'std_population':float(values.std()),'mean':float(values.mean()),'minimum':float(values.min()),'maximum':float(values.max()),'primary_full_distinct_scores':len(np.unique(values)),'primary_full_fraction':len(np.unique(values))/len(values),'primary_representative_distinct_scores':len(np.unique(rep)),'primary_representative_fraction':len(np.unique(rep))/len(rep),'diagnostic_representative_distinct_scores':len(np.unique(dv)),'diagnostic_representative_fraction':len(np.unique(dv))/len(dv),'diagnostic_finite':bool(np.isfinite(dv).all()),'primary_bit_exact_to_original':True,'required_fraction':.99,'acceptance_precision':'float64 diagnostic only' if arm=='D0_LEGACY' else 'primary float64'}
    row['passed']=check_row(row);rows.append(row)
    if split=='train':
     st=read(ROOT/f'training_stats/{pole}_{arm}.json');require(st['mean']==row['mean'] and st['std_population']==row['std_population'] and st['N']==len(values),'Frozen primary standardization changed')
     stats_verified.append({'pole':pole,'distance':arm,'N':st['N'],'mean_bit_exact':True,'std_bit_exact':True,'file_sha256':sha(ROOT/f'training_stats/{pole}_{arm}.json')})
    if arm=='D0_LEGACY':
     unique,inv,cts=np.unique(rep,return_inverse=True,return_counts=True);members=[];collision_groups=[]
     for gi in np.flatnonzero(cts>1):
      idx=np.flatnonzero(inv==gi);original_rows=ri[idx]
      collision_groups.append({'pole':pole,'split':split,'primary_float32_score':float(unique[gi]),'float32_bits_hex':hex(int(np.float32(unique[gi]).view(np.uint32))),'distinct_input_count':len(idx),'redundant_score_count':len(idx)-1,'pair_collision_count':len(idx)*(len(idx)-1)//2,'representative_row_indices':original_rows.tolist(),'diagnostic_float64_scores':dv[idx].tolist()})
      for j,k in zip(idx,original_rows):members.append({'pole':pole,'split':split,'float32_score':float(rep[j]),'float64_diagnostic_score':float(dv[j]),'representative_row_index':int(k),'preprocessed_tensor_sha256':d.iloc[k].preprocessed_tensor_sha256,'path':d.iloc[k].path,'input_group_size':int(d.iloc[k].group_size)})
     save(DIAG/f'{pole}_{split}_D0_float32_collision_groups.json',collision_groups)
     pd.DataFrame(members,columns=['pole','split','float32_score','float64_diagnostic_score','representative_row_index','preprocessed_tensor_sha256','path','input_group_size']).to_csv(DIAG/f'{pole}_{split}_D0_float32_collision_members.csv',index=False)
     collision_summary.append({'pole':pole,'split':split,'collision_groups':len(collision_groups),'distinct_inputs_in_collisions':sum(g['distinct_input_count'] for g in collision_groups),'excess_equal_scores_among_distinct_inputs':len(rep)-len(unique),'pair_collisions_among_distinct_inputs':sum(g['pair_collision_count'] for g in collision_groups),'full_row_excess_equal_scores':len(values)-len(unique),'duplicate_input_excess_rows':len(values)-len(rep)})
   diagnostic.to_csv(DIAG/f'{pole}_{split}_representative_diagnostic_scores.csv',index=False)
   print('All primary scores reproduced bit exactly:',pole,split,flush=True)
  for arm in ARMS:
   dtype=torch.float32 if arm=='D0_LEGACY' else torch.float64
   rm=ref['legacy_mu'] if arm=='D0_LEGACY' else ref['registered_mu']
   m=torch.tensor(np.asarray(rm)[None,:]+.2,device=device,dtype=dtype,requires_grad=True)
   lv=torch.tensor(np.log(np.maximum(ref['aggregate_variance_raw'],EPS))[None,:]+.7,device=device,dtype=dtype,requires_grad=True)
   gm,gl=torch.autograd.grad(distance(m,lv,ref,arm).sum(),[m,lv],allow_unused=True)
   good=bool(torch.isfinite(gm).all() and gm.norm()>1e-12 and (gl is None if arm in ARMS[:3] else torch.isfinite(gl).all() and gl.norm()>1e-12))
   gradrows.append({'pole':pole,'distance':arm,'dtype':str(dtype),'mu_gradient_norm':float(gm.norm()),'logvar_gradient_norm':0. if gl is None else float(gl.norm()),'passed':good})
  del model,ck;torch.cuda.empty_cache()
 oldgrad=csvread(ROOT/'preflight/gradient_preflight.csv');polarity=csvread(ROOT/'preflight/hand_checked_polarity_batch.csv')
 require(len(polarity)==192,'Incomplete inherited manual polarity checks')
 for r in polarity.itertuples():
  st=read(ROOT/f'training_stats/{r.pole}_{r.distance}.json');gap=(r.raw_mild-r.raw_severe if r.pole=='EA' else r.raw_severe-r.raw_mild)/(st['std_population']+EPS)
  require(np.isclose(gap,r.signed_standardized_gap,rtol=1e-14,atol=1e-14),'Manual polarity mismatch')
 pd.DataFrame(rows).to_csv(DIAG/'distance_preflight.csv',index=False);pd.DataFrame(gradrows).to_csv(DIAG/'gradient_preflight_primary_precision.csv',index=False)
 save(DIAG/'float32_collision_summary.json',collision_summary);save(DIAG/'posterior_replay_verification.json',replay);save(DIAG/'training_statistics_verification.json',stats_verified)
 others={'synthetic_and_amendment_tests':result.returncode==0,'original_gradient_tests':bool(oldgrad.passed.all()),'primary_precision_gradient_tests':all(r['passed'] for r in gradrows),'hand_checked_polarity':bool(polarity.batch_polarity_implementation_checked.all()),'split_audit':read(ROOT/'inputs/split_audit.json')['status']=='PASS','all_primary_scores_bit_exact':all(r['primary_bit_exact_to_original'] for r in rows),'all_input_bytes_and_tensor_hashes_checked':True,'training_statistics_unchanged':len(stats_verified)==12,'cache_replay_bit_exact':len(replay)==8,'no_train_val_tensor_overlap':True}
 passed=amended_training_allowed(rows,others)
 save(DIAG/'result.json',{'passed':passed,'checked_utc':now(),'checks':others,'rows':rows,'duplicate_inputs':dups,'float32_collision_summary':collision_summary,'training_runs_started':0,'holdout_opened':False})
 (ROOT/'environment.txt').write_text(f'Python {sys.version}\nPlatform {platform.platform()}\nTorch {torch.__version__}, CUDA {torch.version.cuda}\nGPU {torch.cuda.get_device_name()}\nNumPy {np.__version__}; pandas {pd.__version__}\nTF32 disabled; deterministic algorithms enabled; CUBLAS_WORKSPACE_CONFIG=:4096:8; seed42; AMP false.\n')
 print('AMENDED PREFLIGHT:', 'PASS' if passed else 'FAIL',flush=True);require(passed,'Amended preflight failed; training remains prohibited')
if __name__=='__main__':run()
