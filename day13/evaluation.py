"""Opinion-free validation selection, pre-holdout freeze, and KADID reporting."""
import argparse
from common import *
from engine import score_checkpoint


def norm_stats(v):
 a=np.asarray(v,float)
 if not np.isfinite(a).all() or a.std(ddof=0)<=EPS:raise ValueError('Nonfinite or degenerate calibration')
 return {'mean':float(a.mean()),'std_population':float(a.std(ddof=0)),'N':len(a),'epsilon':EPS}
def normalize(v,s):return (np.asarray(v,float)-s['mean'])/(s['std_population']+EPS)
def model_metrics(d,col,name):
 S,sev,by=severity_metrics(d,col);mac,pa,_=pair_metrics(d,col)
 return {'candidate':name,'S_blind':S,'S_KADID':by.get('KADID-10k'),'S_TID':by.get('TID2013'),'macro_pair_accuracy':mac},sev,pa

def validation_bootstrap(d):
 draws=clusters(d);values={q:np.zeros(5000) for q in ['C0','C2','C3']};datasets=len(draws)
 for ds,g in d.groupby('dataset',sort=True):
  refs,cts=draws[ds];lookup={r:i for i,r in enumerate(refs)};nt=g.distortion_type.nunique()
  for typ,sub in g.groupby('distortion_type',sort=True):
   w=cts[:,[lookup[r] for r in sub.ref_id]]
   for q in values:values[q]-=weighted_srcc(sub[q].to_numpy(),sub.severity.to_numpy(),w)/(nt*datasets)
 diffs={q:values['C3']-values[q] for q in ['C0','C2']}
 return diffs,draws

def validation():
 verify_inputs();folder=OUT/'validation'
 if (folder/'frozen_configuration.json').exists():verify_freeze();print('Validation already frozen and verified');return
 d=read_csv(OUT/'folds/validation_manifest.csv',dtype={'ref_id':str,'image_id':str})
 assert len(d)==1860 and set(d.split)=={'val'} and not any('mos' in c.lower() for c in d.columns)
 full_h=score_checkpoint(d,CHECKPOINTS['EH_ranked'],folder/'raw/EH_ranked.csv')
 hstats=norm_stats(full_h.energy);d['E_H']=full_h.energy.to_numpy();d['z_H']=normalize(d.E_H,hstats);d['C0']=-d.z_H
 choices=[];cal={'EH_ranked':hstats};raw=[];qz=[];sev=[];pair=[];models={};table=[]
 for tag,label in [('C1','EA_init'),('C2','EA_rankall')]:
  models[tag]={'checkpoint':str(CHECKPOINTS[label]),'arm':tag,'epoch':5,'control':True}
 arms=read_json(OUT/'weights/arms.json')['arms']
 for arm,info in arms.items():
  if not info['feasible']:continue
  done=read_json(OUT/'training'/arm/'complete.json');assert done['configuration']['epochs']==25
  for ep in [5,10,15,20,25]:models[f'{arm}_ep{ep:04d}']={'checkpoint':str(OUT/'training'/arm/'checkpoints'/f'epoch_{ep:04d}.pth'),'arm':arm,'epoch':ep,'control':False}
 if len(models)<=2:raise RuntimeError('No feasible specialist; C3 cannot be selected; do not evaluate holdout')
 for name,info in models.items():
  scored=score_checkpoint(d.drop(columns=['E_H','z_H','C0']),Path(info['checkpoint']),folder/'raw'/f'{name}.csv')
  stats=norm_stats(scored.energy);cal[name]=stats;tmp=d.copy();tmp['E_A']=scored.energy.to_numpy();tmp['z_A']=normalize(tmp.E_A,stats);tmp['Q_z']=tmp.z_A-tmp.z_H
  tmp['candidate']=name;tmp['arm']=info['arm'];tmp['epoch']=info['epoch'];tmp['checkpoint_sha256']=sha(info['checkpoint'])
  raw.append(tmp[['dataset','image_id','ref_id','distortion_type','severity','path','candidate','arm','epoch','checkpoint_sha256','E_A','E_H']]);qz.append(tmp)
  row,st,pt=model_metrics(tmp,'Q_z',name);row.update(info);row['checkpoint_sha256']=sha(info['checkpoint']);table.append(row)
  st['candidate']=name;pt['candidate']=name;sev.append(st);pair.append(pt)
  if info['control']:d[name]=tmp.Q_z.to_numpy();d['z_A_'+name]=tmp.z_A.to_numpy();d['E_A_'+name]=tmp.E_A.to_numpy()
  print(f'Validation {name}: S_blind={row["S_blind"]:.8f}, macro pairs={row["macro_pair_accuracy"]:.8f}',flush=True)
 fulltable=pd.DataFrame(table);candidates=fulltable[~fulltable.control].copy();chosen=select_specialist(candidates);name=chosen.candidate
 winner=next(f for f in qz if f.candidate.iloc[0]==name)
 d['C3']=winner.Q_z.to_numpy();d['z_A_C3']=winner.z_A.to_numpy();d['E_A_C3']=winner.E_A.to_numpy()
 c0,st,pt=model_metrics(d,'C0','C0');c0.update(arm='C0',epoch=20,control=True,checkpoint=str(CHECKPOINTS['EH_ranked']),checkpoint_sha256=sha(CHECKPOINTS['EH_ranked']))
 fulltable=pd.concat([pd.DataFrame([c0]),fulltable],ignore_index=True);fulltable['selected_specialist']=fulltable.candidate.eq(name)
 controls=[];control_severity=[];control_pairs=[]
 for c in ['C0','C1','C2','C3']:
  r,s,p=model_metrics(d,c,c);controls.append(r);control_severity.append(s);control_pairs.append(p)
 pd.concat(control_severity,ignore_index=True).to_csv(folder/'control_severity_by_distortion.csv',index=False)
 pd.concat(control_pairs,ignore_index=True).to_csv(folder/'control_pair_accuracy.csv',index=False)
 ctab=pd.DataFrame(controls);cvals=ctab.set_index('candidate').S_blind
 no_go=not (cvals['C3']>cvals['C0'] and cvals['C3']>cvals['C2'])
 diffs,draws=validation_bootstrap(d);unc=[]
 for c,v in diffs.items():unc.append({'comparison':'C3_minus_'+c,'point_difference':float(cvals['C3']-cvals[c]),**ci(v)})
 pd.DataFrame({'draw':np.arange(5000),**{'C3_minus_'+c:v for c,v in diffs.items()}}).to_csv(folder/'bootstrap_differences.csv',index=False)
 pd.DataFrame(unc).to_csv(folder/'bootstrap_confidence_intervals.csv',index=False)
 np.savez_compressed(folder/'bootstrap_cluster_counts.npz',**{ds:t[1] for ds,t in draws.items()})
 save_json(folder/'bootstrap_provenance.json',{'seed':20260910,'draws':5000,'reference_order':{ds:t[0] for ds,t in draws.items()},'fixed_candidate':name,'selection_repeated':False})
 pd.concat(raw,ignore_index=True).to_csv(folder/'raw_scores.csv',index=False);pd.concat(qz,ignore_index=True).to_csv(folder/'qz_scores.csv',index=False)
 pd.concat(sev,ignore_index=True).to_csv(folder/'severity_by_distortion.csv',index=False);pd.concat(pair,ignore_index=True).to_csv(folder/'pair_accuracy_by_distortion.csv',index=False)
 fulltable.to_csv(folder/'selection_table.csv',index=False);ctab.to_csv(folder/'control_comparison.csv',index=False);d.to_csv(folder/'selected_and_control_scores.csv',index=False)
 save_json(folder/'calibration_statistics.json',cal)
 sel={'candidate':name,'arm':chosen.arm,'epoch':int(chosen.epoch),'checkpoint':chosen.checkpoint,'checkpoint_sha256':chosen.checkpoint_sha256,
 'alpha_H':arms[chosen.arm]['alpha_H'],'T_alpha':arms[chosen.arm]['T_alpha'],'S_blind':float(chosen.S_blind),'macro_pair_accuracy':float(chosen.macro_pair_accuracy),
 'validation_no_go':no_go,'selection_rule':'maximum S_blind; higher macro pair accuracy; S10,H10,S00,H00,S25,H25; earlier epoch',
 'exact_S_tie_candidates':candidates[candidates.S_blind.eq(chosen.S_blind)].candidate.tolist(),'bootstrap_diagnostics':unc}
 save_json(folder/'selected_blindspot_model.json',sel)
 artifacts=[OUT/'input_manifest.json',OUT/'input_manifest.sha256',OUT/'weights/arms.json',OUT/'crossfit_EH/crossfit_integrity.json',ROOT/'evaluation.py',ROOT/'verification_pipeline.py']
 artifacts+=list((OUT/'folds').glob('*.csv'))+list((OUT/'weights').glob('*.csv'))+list((OUT/'crossfit_EH').glob('oof_*.csv'))
 artifacts+=list(folder.glob('*.csv'))+list(folder.glob('*.json'))+list(folder.glob('*.npz'))
 artifacts+=[Path(info['checkpoint']) for info in models.values()]+[CHECKPOINTS['EH_ranked']]
 for k in range(5):
  f=OUT/'crossfit_EH'/f'fold_{k}';artifacts +=[f/'complete.json',Path(read_json(f/'complete.json')['checkpoint']),f/'training_score_statistics.json']
 frozen={'frozen_utc':now(),'selected':sel,'Q_z':'z_A - z_H','controls':{'C0':'-z_H','C1':'Day7 z_A-z_H','C2':'rankall z_A-z_H','C3':name+' z_A-z_H'},
 'calibration':{'EH_ranked':hstats,'C1':cal['C1'],'C2':cal['C2'],'C3':cal[name]},'checkpoints':{'C0':str(CHECKPOINTS['EH_ranked']),'C1':str(CHECKPOINTS['EA_init']),'C2':str(CHECKPOINTS['EA_rankall']),'C3':chosen.checkpoint},
 'checkpoint_hashes':{k:sha(p) for k,p in {'C0':CHECKPOINTS['EH_ranked'],'C1':CHECKPOINTS['EA_init'],'C2':CHECKPOINTS['EA_rankall'],'C3':chosen.checkpoint}.items()},
 'all_candidate_identities':models,'training_manifest_sha256':sha(OUT/'input_manifest.json'),'frozen_artifact_hashes':{str(p):sha(p) for p in artifacts},
 'fold_assignment_sha256':sha(OUT/'folds/reference_fold_assignments.csv'),'bootstrap_seed':20260910,'bootstrap_draws':5000,'MOS_DMOS_unused_for_selection':True,
 'validation_no_go':no_go,'kadid_holdout_opened':False,'csiq_opened':False,'Task4_used':False,'references_rebuilt':False}
 save_json(folder/'frozen_configuration.json',frozen);(folder/'frozen_configuration.sha256').write_text(sha(folder/'frozen_configuration.json')+'  frozen_configuration.json\n')
 print('VALIDATION FROZEN:',name,'; validation no-go:',no_go,flush=True)


def verify_freeze():
 verify_inputs();folder=OUT/'validation';p=folder/'frozen_configuration.json'
 assert sha(p)==(folder/'frozen_configuration.sha256').read_text().split()[0]
 f=read_json(p)
 for path,h in f['frozen_artifact_hashes'].items():assert sha(path)==h,('Changed frozen artifact',path)
 return f

def conditional_table(pairs,d,alpha):
 p=pairs.copy();a=p.mild_index.to_numpy();b=p.severe_index.to_numpy()
 p['g_H']=d.loc[b,'z_H'].to_numpy()-d.loc[a,'z_H'].to_numpy()
 p['g_A']=d.loc[a,'z_A_C3'].to_numpy()-d.loc[b,'z_A_C3'].to_numpy()
 p['g_Q']=d.loc[a,'C3'].to_numpy()-d.loc[b,'C3'].to_numpy()
 ranges=read_csv(OUT/'folds/training_severity_ranges.csv');p=p.merge(ranges,on=['dataset','distortion_type'],validate='many_to_one')
 p['delta']=(p.sev_severe-p.sev_mild)/(p['max']-p['min']);p['failure_strict']=p.g_H<=0;p['failure_margin']=p.g_H<=alpha*p.delta;p['failure_A']=p.g_A<=0
 rows=[]
 for name in ['strict','margin']:
  h=p['failure_'+name];a=p.failure_A;count=int(h.sum());union=int((h|a).sum())
  rows.append({'failure_set':name,'alpha_H':alpha,'N_pairs':len(p),'N_H_failures':count,'N_A_failures':int(a.sum()),
   'R_A':float((p.loc[h,'g_A']>0).mean()) if count else np.nan,'R_Q':float((p.loc[h,'g_Q']>0).mean()) if count else np.nan,
   'failure_intersection':int((h&a).sum()),'failure_union':union,'failure_Jaccard':float((h&a).sum()/union) if union else np.nan})
 return p,pd.DataFrame(rows)

def holdout_bootstrap(d,p,alpha):
 refs,counts=clusters(d)['KADID-10k'];lookup={r:i for i,r in enumerate(refs)};w=counts[:,[lookup[r] for r in d.ref_id]]
 vals={};pairweights=counts[:,[lookup[r] for r in p.ref_id]];den=pairweights.sum(1)
 for c in ['C0','C2','C3']:
  x=d[c].to_numpy();y=d.MOS.to_numpy();sr=np.empty(5000)
  for j in range(0,5000,250):sr[j:j+250]=weighted_srcc(x,y,w[j:j+250])
  vals[c+'_SRCC']=sr
  gap=d.loc[p.mild_index,c].to_numpy()-d.loc[p.severe_index,c].to_numpy();credit=(gap>0)+.5*(gap==0)
  vals[c+'_pair_accuracy']=(pairweights@credit)/den
 diffs={}
 for c in ['C0','C2']:
  for metric in ['SRCC','pair_accuracy']:diffs[f'C3_minus_{c}_{metric}']=vals[f'C3_{metric}']-vals[f'{c}_{metric}']
 for name in ['strict','margin']:
  h=p['failure_'+name].to_numpy();eligible=pairweights@h.astype(float)
  for label,gap in [('R_A','g_A'),('R_Q','g_Q')]:
   num=pairweights@(h&(p[gap].to_numpy()>0)).astype(float)
   diffs[f'{label}_{name}']=np.divide(num,eligible,out=np.full(5000,np.nan),where=eligible>0)
 return diffs,refs,counts,vals


def holdout():
 frozen=verify_freeze();folder=OUT/'kadid_development'
 if (folder/'complete.json').exists():print('KADID holdout already completed');return
 save_json(folder/'opened.json',{'opened_utc':now(),'frozen_configuration_sha256':sha(OUT/'validation/frozen_configuration.json'),'selected':frozen['selected']['candidate'],'csiq_opened':False})
 # This is the first loader allowed to materialize KADID MOS for Day13.
 raw=pd.read_csv(SPLITS,usecols=META+['mos_or_dmos'],dtype=str,keep_default_na=False)
 raw=raw[raw.dataset.eq('KADID-10k')&raw.split.eq('holdout')].reset_index(drop=True)
 d=raw.drop(columns=['mos_or_dmos','distorted_path','severity_or_level']).copy();d['path']=raw.distorted_path.map(remap);d['ref_path']=d.ref_path.map(remap);d['severity']=pd.to_numeric(raw.severity_or_level);d['MOS']=pd.to_numeric(raw.mos_or_dmos);d['group_key']=d.dataset+'::'+d.ref_id
 assert len(d)==1625 and d.ref_id.nunique()==13 and np.isfinite(d.MOS).all()
 val=read_csv(OUT/'folds/validation_manifest.csv',dtype={'ref_id':str,'image_id':str});train=read_csv(OUT/'folds/train_unique_manifest.csv',dtype={'ref_id':str,'image_id':str})
 assert not set(d.group_key)&set(pd.concat([val,train]).group_key)
 h=score_checkpoint(d.drop(columns='MOS'),Path(frozen['checkpoints']['C0']),folder/'raw/C0_EH.csv');d['E_H']=h.energy.to_numpy();d['z_H']=normalize(d.E_H,frozen['calibration']['EH_ranked']);d['C0']=-d.z_H
 for c in ['C1','C2','C3']:
  a=score_checkpoint(d.drop(columns=['MOS','E_H','z_H','C0']+[c for c in d if c.startswith('E_A_') or c.startswith('z_A_') or c in ['C1','C2','C3']]),Path(frozen['checkpoints'][c]),folder/'raw'/(c+'_EA.csv'))
  d['E_A_'+c]=a.energy.to_numpy();d['z_A_'+c]=normalize(a.energy,frozen['calibration'][c]);d[c]=d['z_A_'+c]-d.z_H
 d.to_csv(folder/'control_and_selected_scores.csv',index=False)
 scores=['C0','C1','C2','C3','E_A_C1','E_A_C2','E_A_C3'];rows=[];sevs=[];pas=[];paird=[]
 p=all_pairs(d)
 for q in scores:
  x=d[q].to_numpy();_,cts=np.unique(x,return_counts=True);S,severity,by=severity_metrics(d,q);macro,pa,detail=pair_metrics(d,q,p)
  row={'score':q,'N_images':len(d),'N_references':d.ref_id.nunique(),'N_pairs':len(p),'SRCC':float(spearmanr(x,d.MOS).statistic),'Pearson_raw':float(pearsonr(x,d.MOS).statistic),
   'S_severity':S,'macro_severity_SRCC':-S,'pair_accuracy':float(detail.credit.mean()),'macro_pair_accuracy':macro,'min':float(x.min()),'max':float(x.max()),'range':float(np.ptp(x)),'variance_population':float(x.var()),'exact_equal_score_pairs':int((cts*(cts-1)//2).sum()),'eligible_pair_ties':int(detail.tie.sum())}
  rows.append(row);sevs.append(severity);pas.append(pa);paird.append(detail)
 results=pd.DataFrame(rows);results.to_csv(folder/'opinion_correlations.csv',index=False);pd.concat(sevs).to_csv(folder/'severity_correlations.csv',index=False);pd.concat(pas).to_csv(folder/'pair_accuracy.csv',index=False);pd.concat(paird).to_csv(folder/'pair_details.csv',index=False)
 pp,cond=conditional_table(p,d,frozen['selected']['alpha_H']);pp.to_csv(folder/'conditional_pair_details.csv',index=False);cond.to_csv(folder/'conditional_complementarity.csv',index=False)
 diffs,refs,counts,values=holdout_bootstrap(d,pp,frozen['selected']['alpha_H']);point=results.set_index('score');cpoint=cond.set_index('failure_set');intervals=[]
 for label,v in diffs.items():
  if label.startswith('C3_minus_'):
   rest=label[len('C3_minus_'):];control,metric=rest.split('_',1);estimate=float(point.loc['C3',metric]-point.loc[control,metric])
  else:
   metric,scope=label.rsplit('_',1);estimate=float(cpoint.loc[scope,metric])
  intervals.append({'comparison':label,'estimate':estimate,**ci(v)})
 pd.DataFrame({'draw':np.arange(5000),**diffs}).to_csv(folder/'bootstrap_differences.csv',index=False);pd.DataFrame(intervals).to_csv(folder/'bootstrap_confidence_intervals.csv',index=False)
 np.savez_compressed(folder/'bootstrap_cluster_counts.npz',counts=counts);np.savez_compressed(folder/'bootstrap_absolute_draws.npz',**values)
 save_json(folder/'bootstrap_provenance.json',{'seed':20260910,'draws':5000,'reference_order':refs,'fixed_selected':frozen['selected']['candidate'],'pair_accuracy':'overall half-credit exact ties'})
 verify_freeze();save_json(folder/'complete.json',{'status':'COMPLETE','completed_utc':now(),'N':1625,'N_references':13,'N_pairs':len(p),'selected_unchanged':frozen['selected']['candidate'],'validation_no_go_unchanged':frozen['validation_no_go'],'MOS_used_only_after_freeze':True,'CSIQ_opened':False,'normalization_refitted':False})
 print('KADID development holdout COMPLETE; validation decision unchanged.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['validation','holdout']);a=p.parse_args()
 validation() if a.stage=='validation' else holdout()
