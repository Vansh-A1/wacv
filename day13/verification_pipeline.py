"""Independent checks before holdout access and after all requested analysis."""
import argparse
import torch
from common import *
from engine import tensor_hash
from evaluation import verify_freeze

def verify_training_and_validation():
 frozen=verify_freeze();m=verify_inputs();report={'status':'PASS','checks':{},'checked_utc':now()}
 assign=read_csv(OUT/'folds/reference_fold_assignments.csv',dtype={'ref_id':str});train=read_csv(OUT/'folds/train_unique_manifest.csv',dtype={'ref_id':str,'image_id':str})
 pd.testing.assert_frame_equal(assign,construct_folds(train),check_dtype=False)
 oof=read_csv(OUT/'crossfit_EH/oof_unique_scores.csv',dtype={'ref_id':str,'image_id':str});assert len(oof)==9040 and not oof.duplicated(['dataset','image_id']).any()
 for k in range(5):
  folder=OUT/'crossfit_EH'/f'fold_{k}';a=read_csv(folder/'teacher_training_pairs.csv');b=read_csv(folder/'heldout_pairs.csv')
  assert not set(a.group_key)&set(b.group_key)
  stats=read_json(folder/'training_score_statistics.json');s=read_csv(folder/'training_unique_scores.csv');h=read_csv(folder/'heldout_calibrated_scores.csv')
  np.testing.assert_allclose([stats['mean'],stats['std_population']],[s.energy.mean(),s.energy.std(ddof=0)],atol=1e-12)
  np.testing.assert_allclose(h.z_H_oof,(h.energy-stats['mean'])/(stats['std_population']+EPS),atol=1e-12)
  info=read_json(folder/'complete.json');c=torch.load(info['checkpoint'],map_location='cpu',weights_only=False)
  assert c['epoch']==20 and c['config']['seed']==42 and c['config']['lambda_rank']==.1
  assert {t:tensor_hash(c[t]) for t in ['mu_ref','Sigma_ref']}==m['checkpoints']['EH_init']['reference_hashes']
  logs=read_csv(folder/'train_log.csv');assert logs.epoch.tolist()==list(range(1,21));assert (logs.N_pairs==len(a)).all()
  del c
 report['checks']['all_five_teachers_20_epochs_reference_exclusion_and_normalization']=True
 pairs=read_csv(OUT/'crossfit_EH/oof_pair_scores.csv');np.testing.assert_allclose(pairs.g_H,pairs.z_H_oof_severe-pairs.z_H_oof_mild,atol=1e-14)
 np.testing.assert_allclose(pairs.delta,(pairs.sev_severe-pairs.sev_mild)/(pairs['max']-pairs['min']),atol=1e-14)
 arms=read_json(OUT/'weights/arms.json')['arms']
 for arm,info in arms.items():
  p=read_csv(info['weight_file']);assert sha(info['weight_file'])==info['weight_sha256']
  v=info['alpha_H']*p.delta-p.g_H;T=max(float(np.median(abs(v-np.median(v)))),.05)
  raw=(v>=0).to_numpy(float) if info['kind']=='hard' else np.exp(-np.logaddexp(0,-v/T))
  np.testing.assert_allclose(p.raw_weight,raw,atol=1e-14);np.testing.assert_allclose(p.weight,raw/(np.mean(raw)+EPS),atol=1e-14)
  if not info['feasible']:continue
  run=OUT/'training'/arm;logs=read_csv(run/'train_log.csv');assert logs.epoch.tolist()==list(range(1,26));assert (logs.N_pairs==18080).all()
  for ep in [5,10,15,20,25]:
   c=torch.load(run/'checkpoints'/f'epoch_{ep:04d}.pth',map_location='cpu',weights_only=False)
   assert c['epoch']==ep and c['config']['init_sha256']==m['checkpoints']['EA_init']['sha256']
   assert {t:tensor_hash(c[t]) for t in ['mu_ref','Sigma_ref']}==m['checkpoints']['EA_init']['reference_hashes']
   assert c['config']['lambda_rank']==.1 and c['config']['rank_margin']==.1
   del c
 report['checks']['all_weights_and_feasible_arms_25_epochs_unchanged_Day7_initialization']=True
 val=read_csv(OUT/'validation/qz_scores.csv',dtype={'ref_id':str});table=read_csv(OUT/'validation/selection_table.csv');cal=read_json(OUT/'validation/calibration_statistics.json')
 for name,g in val.groupby('candidate'):
  assert len(g)==1860
  stats=cal[name];np.testing.assert_allclose([stats['mean'],stats['std_population']],[g.E_A.mean(),g.E_A.std(ddof=0)],atol=1e-12)
  za=(g.E_A-stats['mean'])/(stats['std_population']+EPS);zh=(g.E_H-cal['EH_ranked']['mean'])/(cal['EH_ranked']['std_population']+EPS)
  np.testing.assert_allclose(g.Q_z,za-zh,atol=1e-13)
  ds=[]
  for dataset,sub in g.groupby('dataset'):
   ds.append(-np.mean([spearmanr(t.Q_z,t.severity).statistic for _,t in sub.groupby('distortion_type')]))
  r=table[table.candidate.eq(name)].iloc[0];np.testing.assert_allclose(r.S_blind,np.mean(ds),atol=1e-14)
 selection=table[~table.control].copy();selection['priority']=selection.arm.map({a:i for i,a in enumerate(PRIORITY)})
 chosen=selection.sort_values(['S_blind','macro_pair_accuracy','priority','epoch'],ascending=[False,False,True,True]).iloc[0]
 assert chosen.candidate==frozen['selected']['candidate']
 report['checks']['all_validation_calibrations_Qz_severity_scores_and_deterministic_selection']=True
 d=read_csv(OUT/'validation/selected_and_control_scores.csv',dtype={'ref_id':str});boot=read_csv(OUT/'validation/bootstrap_differences.csv');info=read_json(OUT/'validation/bootstrap_provenance.json')
 with np.load(OUT/'validation/bootstrap_cluster_counts.npz') as counts:
  for b in range(12):
   score={q:[] for q in ['C0','C2','C3']}
   for ds,g in d.groupby('dataset'):
    sub=pd.concat([g[g.ref_id.eq(r)] for r,c in zip(info['reference_order'][ds],counts[ds][b]) for _ in range(c)],ignore_index=True)
    for q in score:score[q].append(-np.mean([spearmanr(t[q],t.severity).statistic for _,t in sub.groupby('distortion_type')]))
   for q in ['C0','C2']:np.testing.assert_allclose(boot['C3_minus_'+q][b],np.mean(score['C3'])-np.mean(score[q]),atol=2e-14)
 report['checks']['12_validation_bootstrap_draws_independent_replication']=True
 assert frozen['kadid_holdout_opened'] is False and frozen['csiq_opened'] is False
 save_json(ROOT/'verification/validation_verified.json',report);print(json.dumps(report,indent=2))


def verify_holdout():
 f=verify_freeze();folder=OUT/'kadid_development';complete=read_json(folder/'complete.json');assert complete['status']=='COMPLETE' and not complete['CSIQ_opened']
 opened=read_json(folder/'opened.json');assert f['frozen_utc']<opened['opened_utc']<complete['completed_utc']
 d=read_csv(folder/'control_and_selected_scores.csv',dtype={'ref_id':str});table=read_csv(folder/'opinion_correlations.csv');p=read_csv(folder/'conditional_pair_details.csv',dtype={'ref_id':str});saved=read_csv(folder/'bootstrap_differences.csv');binfo=read_json(folder/'bootstrap_provenance.json')
 for r in table.itertuples():
  np.testing.assert_allclose([r.SRCC,r.Pearson_raw],[spearmanr(d[r.score],d.MOS).statistic,pearsonr(d[r.score],d.MOS).statistic],atol=1e-13)
 for c in ['C1','C2','C3']:
  s=f['calibration'][c];np.testing.assert_allclose(d['z_A_'+c],(d['E_A_'+c]-s['mean'])/(s['std_population']+EPS),atol=1e-13)
  np.testing.assert_allclose(d[c],d['z_A_'+c]-d.z_H,atol=1e-13)
 np.testing.assert_allclose(p.g_Q,p.g_A+p.g_H,atol=1e-13)
 np.testing.assert_array_equal(p.failure_strict,p.g_H<=0);np.testing.assert_array_equal(p.failure_margin,p.g_H<=f['selected']['alpha_H']*p.delta)
 with np.load(folder/'bootstrap_cluster_counts.npz') as cts:
  for b in range(12):
   order=binfo['reference_order'];counts=cts['counts'][b]
   sub=pd.concat([d[d.ref_id.eq(r)] for r,c in zip(order,counts) for _ in range(c)],ignore_index=True)
   pp=pd.concat([p[p.ref_id.eq(r)] for r,c in zip(order,counts) for _ in range(c)],ignore_index=True)
   for c in ['C0','C2']:
    value=spearmanr(sub.C3,sub.MOS).statistic-spearmanr(sub[c],sub.MOS).statistic
    np.testing.assert_allclose(saved[f'C3_minus_{c}_SRCC'][b],value,atol=2e-14)
    def credit(q):
     gap=d.loc[pp.mild_index,q].to_numpy()-d.loc[pp.severe_index,q].to_numpy();return np.mean((gap>0)+.5*(gap==0))
    np.testing.assert_allclose(saved[f'C3_minus_{c}_pair_accuracy'][b],credit('C3')-credit(c),atol=1e-14)
   for scope in ['strict','margin']:
    h=pp['failure_'+scope]
    for label,gap in [('R_A','g_A'),('R_Q','g_Q')]:
     actual=float((pp.loc[h,gap]>0).mean()) if h.any() else np.nan
     np.testing.assert_allclose(saved[f'{label}_{scope}'][b],actual,atol=1e-14,equal_nan=True)
 intervals=read_csv(folder/'bootstrap_confidence_intervals.csv')
 for r in intervals.itertuples():
  v=saved[r.comparison].to_numpy();a=v[np.isfinite(v)];assert r.valid_draws==len(a)
  if len(a):np.testing.assert_allclose([r.ci_low,r.ci_high],np.quantile(a,[.025,.975]),atol=1e-14)
 report={'status':'PASS','checked_utc':now(),'holdout_opened_after_freeze':True,'frozen_calibration_unchanged':True,'all_point_correlations_verified':True,'conditional_gaps_and_masks_verified':True,'12_holdout_bootstrap_draws_explicit_replication':True,'all_bootstrap_intervals_verified':True,'CSIQ_opened':False}
 save_json(ROOT/'verification/holdout_verified.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['validation','holdout']);s=a.parse_args().stage
 verify_training_and_validation() if s=='validation' else verify_holdout()
