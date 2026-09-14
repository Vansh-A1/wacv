"""Independent NumPy checks of registered moments, scores, provenance and the gate."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
EPS=1e-8
ARMS=['D0_LEGACY','DM_BETWEEN','DM_AGGREGATE','KL_AGGREGATE','W2_AGGREGATE','BHATT_AGGREGATE']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def df(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def compare(a,b,rtol=1e-11,atol=1e-11):np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)
def main():
 checks={};old=read(R/'verification/original_task4_hashes.json')
 assert set(old)=={str(p.relative_to(R.parent/'day12_Task4')) for p in (R.parent/'day12_Task4').rglob('*') if p.is_file()}
 for f,h in old.items():assert sha(R.parent/'day12_Task4'/f)==h==sha(R/'reused/old_task4'/f)
 checks['old_task4_files_unchanged_and_copied']=len(old)
 reuse=read(R/'reused/reuse_manifest.json')
 for a in reuse['copied_artifacts']:assert sha(R/a['destination'])==a['sha256']==sha(a['source'])
 checks['reused_artifacts_identical']=len(reuse['copied_artifacts'])
 f=R/'reference_stats/EH';mf=read(f/'manifest.json');ex=read(f/'extraction_complete.json')
 assert sha(f/'manifest.json')==(f/'manifest.sha256').read_text().split()[0]==ex['frozen_manifest_sha256']
 for filename,key in [('eh_pristine_reference_v2.csv','accepted_manifest_sha256'),('eh_pristine_reference_v2_exclusions.csv','exclusions_sha256'),('path_to_source_mapping.json','source_mapping_sha256'),('candidate_audit.csv','candidate_audit_sha256'),('protected_identity_audit.csv','protected_identity_audit_sha256')]:assert sha(f/filename)==mf[key]
 for fn,h in ex['artifact_hashes'].items():assert sha(f/fn)==h
 assert mf['frozen_utc']<ex['completed_utc'];assert ex['model_state_unchanged'] and ex['encoder_full_forward_equivalence']
 accepted=df(f/'eh_pristine_reference_v2.csv',dtype={'source_reference_id':str});excluded=df(f/'eh_pristine_reference_v2_exclusions.csv');protected=df(f/'protected_identity_audit.csv',dtype={'source_reference_id':str})
 assert len(accepted)==3341 and len(excluded)==15959 and len(accepted)+len(excluded)==19300
 candidate_audit=df(f/'candidate_audit.csv');assert candidate_audit.loc[candidate_audit.accepted,'is_decodable'].all();assert set(accepted.sha256)==set(candidate_audit.loc[candidate_audit.accepted,'sha256'])
 for col,pc in [('canonical_path','path'),('sha256','sha256'),('rgb_sha256','rgb_sha256')]:
  assert accepted[col].is_unique;assert not set(accepted[col]) & set(protected[pc])
 assert not set(zip(accepted.source_dataset,accepted.source_reference_id)) & set(zip(protected.source_dataset,protected.source_reference_id))
 compare(accepted.weight.sum(),1,atol=1e-12,rtol=0)
 for src,g in accepted.groupby('source_dataset'):
  compare(g.weight.to_numpy(),1/(2*len(g)),atol=1e-15,rtol=0);compare(g.weight.sum(),.5,atol=1e-12,rtol=0)
 checks['EH_audit_disjointness_freeze_and_equal_source_weights']='PASS'
 moment_errors=[];score_errors=[];summaries=df(R/'preflight/distance_preflight.csv')
 for pole in ['EA','EH']:
  folder=R/'reference_stats'/pole;p=np.load(folder/'reference_posteriors.npz');a=np.load(folder/'aggregate_reference.npz');l=np.load(folder/'legacy_reference.npz')
  mu=p['mu'].astype('float64');lv=p['logvar'].astype('float64');w=p['weights'] if pole=='EH' else np.full(len(mu),1/len(mu))
  mean=np.einsum('n,nj->j',w,mu);between=np.einsum('n,nj->j',w,(mu-mean)**2);within=np.einsum('n,nj->j',w,np.exp(lv))
  for k,v in [('mu_agg',mean),('v_between',between),('v_within',within),('v_aggregate_raw',between+within)]:
   compare(a[k],v,rtol=1e-10,atol=1e-10);moment_errors.append({'pole':pole,'array':k,'max_abs_difference':float(np.max(np.abs(a[k]-v)))})
  b=np.load(folder/'between_reference.npz');np.testing.assert_array_equal(b['v_between'],a['v_between']);np.testing.assert_array_equal(b['mu_agg'],a['mu_agg'])
  for split in ['train','val']:
   cache=np.load(R/f'preflight/cache/{pole}/{split}_posteriors.npz');scores=df(R/f'preflight/{pole}_{split}_distance_scores.csv');assert scores.path.tolist()==list(cache['path'])
   mu=cache['mu'].astype('float64');lv=cache['logvar'].astype('float64');vx=np.maximum(np.exp(lv),EPS);vr=np.maximum(a['v_aggregate_raw'],EPS);delta=mu-a['mu_agg'];bar=.5*(vx+vr)
   expected={
    'D0_LEGACY':np.sqrt(np.sum((cache['mu']-l['mu_ref'])**2/(l['Sigma_ref']+np.float32(EPS)),axis=-1)),
    'DM_BETWEEN':np.sqrt(np.sum(delta**2/(a['v_between']+EPS),axis=-1)),
    'DM_AGGREGATE':np.sqrt(np.sum(delta**2/(a['v_aggregate_raw']+EPS),axis=-1)),
    'KL_AGGREGATE':.5*np.sum(np.log(vr/vx)+(vx+delta**2)/vr-1,axis=-1),
    'W2_AGGREGATE':np.sum(delta**2+(np.sqrt(vx)-np.sqrt(vr))**2,axis=-1),
    'BHATT_AGGREGATE':np.sum(delta**2/bar,axis=-1)/8+.5*np.sum(np.log(bar/np.sqrt(vx*vr)),axis=-1)}
   for arm,v in expected.items():
    got=scores[arm].to_numpy();compare(got,v,rtol=3e-7 if arm=='D0_LEGACY' else 1e-10,atol=1e-7 if arm=='D0_LEGACY' else 1e-11)
    score_errors.append({'pole':pole,'split':split,'distance':arm,'N':len(v),'max_abs_difference':float(np.max(np.abs(got-v)))})
    row=summaries[(summaries.pole==pole)&(summaries.split==split)&(summaries.distance==arm)&(summaries.dataset=='ALL')].iloc[0]
    assert row.N==len(got) and row.distinct_scores==len(np.unique(got));compare(row.std_population,got.std());assert bool(row.uniqueness_pass)==(len(np.unique(got))/len(got)>=.99)
    if split=='train':
     stats=read(R/f'training_stats/{pole}_{arm}.json');compare(stats['mean'],got.mean());compare(stats['std_population'],got.std());assert stats['N']==len(got) and stats['ddof']==0
 checks['independent_weighted_population_moments']='PASS';checks['independent_numpy_distances_all_images']='PASS';checks['independent_distinct_counts_and_12_frozen_stats']='PASS'
 pairs=df(R/'preflight/hand_checked_polarity_batch.csv')
 assert (pairs.severity_mild<pairs.severity_severe).all()
 for (pole,arm),g in pairs.groupby(['pole','distance']):
  stats=read(R/f'training_stats/{pole}_{arm}.json');sign=1 if pole=='EA' else -1;gap=sign*(g.raw_mild.to_numpy()-g.raw_severe.to_numpy())/(stats['std_population']+EPS)
  compare(g.signed_standardized_gap,gap);compare(g.softplus_loss,np.logaddexp(0,.1-gap))
 checks['real_mild_severe_polarity_and_softplus']='PASS'
 gate=read(R/'preflight/gate.json');grad=df(R/'preflight/gradient_preflight.csv');assert len(grad)==12 and grad.passed.all()
 allrows=summaries[summaries.dataset=='ALL'];assert len(allrows)==24 and allrows.finite.all() and allrows.std_pass.all();assert not allrows.uniqueness_pass.any() and not gate['passed']
 for a in read(R/'verification/posterior_reuse_verification.json'):assert a['passed'] and a['mu_max_abs_difference']==0 and a['logvar_max_abs_difference']==0
 checks['fresh_cached_posterior_replays_bit_exact']='PASS'
 matrix=df(R/'run_matrix.csv');assert len(matrix)==24 and matrix.state.eq('BLOCKED_PREFLIGHT').all() and matrix.optimizer_steps.eq(0).all()
 assert len(list((R/'runs').rglob('status.json')))==24 and not list((R/'runs').rglob('*.pth'))
 checks['mandatory_failed_gate_preserved_no_training']='PASS'
 for split in ['train','val']:
  diag=df(R/f'preflight/duplicate_input_diagnostics_{split}.csv');assert diag.groupby('posterior_group').preprocessed_sha256.nunique().eq(1).all()
 checks['all_examined_duplicate_posteriors_have_identical_input_crops']='PASS'
 out={'verification_result':'PASS_FOR_COMPLETED_PREPARATION_AND_REPORTED_BLOCKER','whole_task_complete':False,'training_preflight_passed':False,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'moment_errors':moment_errors,'score_errors':score_errors}
 (R/'verification/independent_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
