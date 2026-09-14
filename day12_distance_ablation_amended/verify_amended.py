"""Independent inspection of the authorized gate and preserved primary data."""
from common import *
from amendment_gate import representative_indices,amended_training_allowed

def verify():
 diag=ROOT/'preflight/amended';result=read(diag/'result.json');require(result['passed'],'Amended result failed')
 baseline=read(ROOT/'verification/baseline_snapshot.json')
 for rel,h in baseline['files'].items():
  require(sha(Path(baseline['source'])/rel)==h,'Old experiment changed: '+rel)
  require(sha(ROOT/'prior_work'/rel)==h,'Prior snapshot changed: '+rel)
 primary=read(ROOT/'verification/preserved_primary_artifacts.json')
 for rel,h in primary['files'].items():require(sha(ROOT/rel)==h,'Primary artifact changed: '+rel)
 rows=csvread(diag/'distance_preflight.csv');require(len(rows)==24,'Incomplete gate matrix')
 require(amended_training_allowed(rows.to_dict('records'),result['checks']),'Gate logic failed')
 summary={}
 for split in ['train','val']:
  d=csvread(diag/f'{split}_input_tensor_groups.csv',dtype={'image_id':str,'ref_id':str})
  inp=csvread(ROOT/'inputs'/('train_images.csv' if split=='train' else 'validation_images.csv'),dtype={'image_id':str,'ref_id':str})
  require(d.path.tolist()==inp.path.tolist(),'Input order changed');require(d.source_sha256.tolist()==inp.image_sha256.tolist(),'Input hashes changed')
  ri,mapping=representative_indices(d.preprocessed_tensor_sha256.tolist());require(np.array_equal(mapping,d.representative_row_index.to_numpy()),'Hash grouping differs')
  require(len(ri)==(8850 if split=='train' else 1820),'Unexpected unique tensor count')
  summary[split]={'original_images':len(d),'unique_tensors':len(ri),'duplicate_groups':int((d.groupby('preprocessed_tensor_sha256').size()>1).sum())}
  for pole in ('EA','EH'):
   c=np.load(ROOT/f'preflight/cache/{pole}/{split}_posteriors.npz');np.testing.assert_array_equal(c['mu'],c['mu'][mapping]);np.testing.assert_array_equal(c['logvar'],c['logvar'][mapping])
   p=csvread(ROOT/f'preflight/{pole}_{split}_distance_scores.csv');s=csvread(diag/f'{pole}_{split}_representative_diagnostic_scores.csv')
   require(s.row_index.tolist()==ri.tolist(),'Wrong representative scoring population')
   ref=reference(pole);m=c['mu'][ri].astype(float);d0=np.sqrt(np.sum((m-ref['legacy_mu'].astype(float))**2/(ref['legacy_variance'].astype(float)+EPS),axis=1))
   np.testing.assert_allclose(d0,s.D0_LEGACY_float64_diagnostic_only,rtol=1e-12,atol=1e-12)
   for arm in ARMS:
    r=rows[rows.pole.eq(pole)&rows.split.eq(split)&rows.distance.eq(arm)].iloc[0]
    v=p[arm].to_numpy();rep=v[ri];dv=s[arm+('_float64_diagnostic_only' if arm=='D0_LEGACY' else '_primary_precision')].to_numpy()
    if arm!='D0_LEGACY':np.testing.assert_array_equal(rep,dv)
    require(r.primary_full_distinct_scores==len(np.unique(v)) and r.primary_representative_distinct_scores==len(np.unique(rep)) and r.diagnostic_representative_distinct_scores==len(np.unique(dv)),'Distinct count mismatch')
    require(r.diagnostic_representative_fraction>=.99,'Diagnostic uniqueness below threshold')
    if split=='train':
     st=read(ROOT/f'training_stats/{pole}_{arm}.json');require(st['N']==len(v) and st['mean']==v.mean() and st['std_population']==v.std(),'Standardization not from original full image rows')
 for x in read(diag/'posterior_replay_verification.json'):require(x['posterior_bit_exact'],'Fresh replay differs')
 for (pole,split),g in rows[rows.distance.eq('D0_LEGACY')].groupby(['pole','split']):
  groups=read(diag/f'{pole}_{split}_D0_float32_collision_groups.json');r=g.iloc[0];require(sum(x['redundant_score_count'] for x in groups)==r.N_representatives-r.primary_representative_distinct_scores,'Collision counts not fully disclosed')
 out={'status':'PASS','verified_utc':now(),'source_experiment_unchanged':True,'primary_artifacts_byte_identical':len(primary['files']),'training_rows_removed':0,'training_statistics_unchanged':12,'independent_float64_D0_comparison':'PASS','diagnostic_rows_passed':24,'input_summary':summary,'all_other_preflight_checks':result['checks'],'holdout_opened':(ROOT/'holdout/opened.json').exists()}
 save(ROOT/'verification/independent_amended_verification.json',out);print(json.dumps(out,indent=2));return out
if __name__=='__main__':verify()
