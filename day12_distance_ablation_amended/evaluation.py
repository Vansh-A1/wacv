"""Severity-only selection, frozen matched calibration, then guarded holdout evaluation."""
from itertools import combinations
from scipy.stats import spearmanr,pearsonr
from common import *
from engine import score_checkpoint

TYPES={'blur':(('KADID-10k','Gaussian blur'),('TID2013','type_08')),'lens':(('KADID-10k','Lens blur'),),'sharpen':(('KADID-10k','High sharpen'),),'pixelate':(('KADID-10k','Pixelate'),)}

def finite_corr(x,y,kind='spearman'):
 x=np.asarray(x,float);y=np.asarray(y,float)
 require(np.isfinite(x).all() and np.isfinite(y).all(),'Nonfinite correlation input')
 if len(x)<3 or len(np.unique(x))<2 or len(np.unique(y))<2:return float('nan')
 return float((spearmanr if kind=='spearman' else pearsonr)(x,y).statistic)

def pair_table(d):
 rows=[]
 for (ds,ref,typ),g in d.groupby(['dataset','ref_id','distortion_type'],sort=True):
  for a,b in combinations(g.index,2):
   sa,sb=d.at[a,'severity'],d.at[b,'severity']
   if sa==sb:continue
   if sa>sb:a,b=b,a
   rows.append((ds,str(ref),typ,a,b))
 return pd.DataFrame(rows,columns=['dataset','ref_id','distortion_type','mild_index','severe_index'])

def pair_gap(energy,pairs,pole):
 v=np.asarray(energy);g=v[pairs.mild_index]-v[pairs.severe_index]
 return -g if pole=='EH' else g

def selection_value(pole,rhos):
 b,l,s,p=[rhos[k] for k in ('blur','lens','sharpen','pixelate')]
 require(np.isfinite([b,l,s,p]).all(),'Undefined registered severity selection score')
 return -b-l-max(0.,s)-max(0.,p) if pole=='EA' else b+l+s+p

def winner(table):
 require(len(table)>0,'No selection candidates')
 require(np.isfinite(table[['validation_score','pair_order_accuracy']].to_numpy()).all(),'Invalid selection candidates')
 return table.sort_values(['validation_score','pair_order_accuracy','epoch','lambda_rank'],ascending=[False,False,True,True],kind='stable').iloc[0]

def validate_and_freeze():
 verify_frozen();target=ROOT/'selection/selected_matched_models.json'
 if target.exists():verify_selected();return read(target)
 val=csvread(ROOT/'inputs/validation_images.csv',dtype={'image_id':str,'ref_id':str})
 require(not any('mos' in c.lower() for c in val.columns),'Opinions may not enter validation selection')
 require(set(val.split)=={'val'} and set(val.dataset)=={'KADID-10k','TID2013'},'Wrong validation population')
 pairs=pair_table(val);cfg=read(ROOT/'config.json');rows=[];all_scores=[];normalization={};selected={};ties={}
 for pole in ('EA','EH'):
  for arm in ARMS:
   for lam in (.1,1.):
    run=runpath(pole,arm,lam);done=read(run/'complete.json')
    require(done['status']=='COMPLETE' and done['epochs']==25,'All 24 runs must complete before selection')
    for ep in (5,10,15,20,25):
     checkpoint=run/f'checkpoints/epoch_{ep:04d}.pth';require(sha(checkpoint)==done['saved_checkpoint_hashes'][str(checkpoint.relative_to(run))],'Saved checkpoint altered')
     tag=f'{pole}_{arm}_lambda_{lam}_epoch_{ep:04d}';output=ROOT/'validation/raw'/f'{tag}.csv'
     scored=score_checkpoint(val,checkpoint,pole,arm,output);rhos={};counts={}
     for key,groups in TYPES.items():
      mask=np.zeros(len(scored),bool)
      for dataset,typ in groups:mask|=(scored.dataset.eq(dataset)&scored.distortion_type.eq(typ)).to_numpy()
      require(mask.any(),f'Missing selection distortion {key}')
      rhos[key]=finite_corr(scored.loc[mask,'energy'],scored.loc[mask,'severity']);counts[key]=int(mask.sum())
     gap=pair_gap(scored.energy,pairs,pole)
     row={'pole':pole,'distance':arm,'lambda_rank':lam,'epoch':ep,'seed':42,'validation_score':selection_value(pole,rhos),'pair_order_accuracy':float((gap>0).mean()),'pair_ties':int((gap==0).sum()),'N_pairs':len(gap),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'score_file':str(output),'score_file_sha256':sha(output),'MOS_DMOS_used':False}
     row.update({k+'_SRCC':v for k,v in rhos.items()});row.update({k+'_N':v for k,v in counts.items()});rows.append(row)
     all_scores.append(scored.assign(pole=pole,distance=arm,lambda_rank=lam,epoch=ep))
     print('Validation scored',tag,'score',row['validation_score'],flush=True)
     save(ROOT/'progress.json',{'state':'VALIDATION_SELECTION','completed_candidates':len(rows),'total_candidates':120,'last_candidate':tag,'updated_utc':now(),'holdout_opened':False})
   armrows=pd.DataFrame([r for r in rows if r['pole']==pole and r['distance']==arm])
   within=[]
   for lam,g in armrows.groupby('lambda_rank'):
    best=winner(g);within.append(best.to_dict());save(runpath(pole,arm,float(lam))/'checkpoint_selection.json',{**best.to_dict(),'selection_scope':'best saved epoch within this lambda','tie_break':['higher strict pair accuracy','earlier epoch','smaller lambda'],'MOS_DMOS_used':False})
    (runpath(pole,arm,float(lam))/'selected_checkpoint.sha256').write_text(best.checkpoint_sha256+'  '+str(best.checkpoint)+'\n')
   best=winner(pd.DataFrame(within));key=pole+'::'+arm;selected[key]=best.to_dict();ties[key]=armrows[armrows.validation_score.eq(best.validation_score)].to_dict('records')
   scored=csvread(best.score_file,dtype={'ref_id':str,'image_id':str});normalization[key]={}
   for dataset,g in scored.groupby('dataset',sort=True):
    std=float(g.energy.std(ddof=0));require(std>1e-12,'Degenerate validation normalization')
    normalization[key][dataset]={'mean':float(g.energy.mean()),'std_population':std,'N':len(g),'ddof':0,'epsilon':EPS,'source_sha256':best.score_file_sha256}
 table=pd.DataFrame(rows);require(len(table)==120 and len(selected)==12,'Incomplete validation matrix')
 for pole in ('EA','EH'):
  t=table[table.pole.eq(pole)].copy();t['selected']=[selected[pole+'::'+r.distance]['checkpoint_sha256']==r.checkpoint_sha256 for r in t.itertuples()]
  t.to_csv(ROOT/'selection'/f'{pole}_validation_selection.csv',index=False)
 pd.concat(all_scores,ignore_index=True).to_csv(ROOT/'validation/all_arm_scores.csv.gz',index=False,compression='gzip')
 save(ROOT/'validation/calibration_statistics.json',normalization);save(ROOT/'selection/selected_per_distance.json',selected)
 frozen={'state':'FROZEN_BEFORE_HOLDOUT','frozen_utc':now(),'selected':selected,'validation_normalization':normalization,'reference_hashes':{p:refhashes(p) for p in ('EA','EH')},'training_standardization':{p+'::'+a:read(ROOT/f'training_stats/{p}_{a}.json') for p in ('EA','EH') for a in ARMS},'gate_sha256':sha(ROOT/'preflight/gate.json'),'amendment_lock_sha256':sha(ROOT/'amendment_lock.json'),'selection_formulas':cfg['selection'],'distance_formula_source_sha256':sha(ROOT/'distance_core.py'),'Qz':'z_EA - z_EH, same distance arm','normalization_population':'separate validation dataset, pole and distance; only KADID validation used for KADID holdout','primary_D0_precision':'float32','D0_float64_diagnostic_used_for_selection_or_training':False,'tie_candidates':ties,'MOS_DMOS_used':False,'holdout_opened':False,'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'selection/EA_validation_selection.csv',ROOT/'selection/EH_validation_selection.csv',ROOT/'selection/selected_per_distance.json',ROOT/'validation/calibration_statistics.json',ROOT/'validation/all_arm_scores.csv.gz']}}
 save(target,frozen);target.with_suffix('.sha256').write_text(sha(target)+'  selected_matched_models.json\n')
 verify_selected();(ROOT/'selection/validation_report.txt').write_text('All 120 saved checkpoint candidates evaluated using only KADID+TID severity metadata. Twelve pole/distance checkpoints selected by the frozen formulas, strict pair-accuracy tie-break, earlier epoch then smaller lambda. Per-dataset calibration and selected checkpoint hashes frozen before holdout. No MOS/DMOS read.\n')
 return frozen

def verify_selected():
 verify_frozen();p=ROOT/'selection/selected_matched_models.json';require(p.is_file() and p.with_suffix('.sha256').is_file(),'Holdout blocked: selected checkpoints have not been frozen')
 require(sha(p)==p.with_suffix('.sha256').read_text().split()[0],'Selected configuration hash changed');f=read(p)
 require(f['state']=='FROZEN_BEFORE_HOLDOUT' and f['holdout_opened'] is False and f['MOS_DMOS_used'] is False,'Invalid pre-holdout freeze declaration')
 require(set(f['selected'])=={p+'::'+a for p in ('EA','EH') for a in ARMS},'Incomplete matched selected arms')
 for key,r in f['selected'].items():
  require(sha(r['checkpoint'])==r['checkpoint_sha256'],'Selected checkpoint changed')
  require(sha(r['score_file'])==r['score_file_sha256'],'Selected validation scores changed')
 for rel,h in f['artifacts'].items():require(sha(ROOT/rel)==h,'Frozen validation artifact changed: '+rel)
 return f

def weighted_ranks(x,w):
 x=np.asarray(x,float);w=np.atleast_2d(w).astype(float);o=np.argsort(x,kind='stable');v=x[o];starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
 count=np.add.reduceat(w[:,o],starts,axis=1);rank=np.cumsum(count,axis=1)-.5*count+.5;groups=np.cumsum(np.r_[False,v[1:]!=v[:-1]])
 return rank[:,groups][:,np.argsort(o)]
def weighted_corr(x,y,w):
 w=np.atleast_2d(w).astype(float);total=w.sum(1,keepdims=True);x=np.broadcast_to(x,w.shape);y=np.broadcast_to(y,w.shape)
 x=x-(w*x).sum(1,keepdims=True)/total;y=y-(w*y).sum(1,keepdims=True)/total;den=np.sqrt((w*x*x).sum(1)*(w*y*y).sum(1));num=(w*x*y).sum(1)
 return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
def weighted_srcc(x,y,w):return weighted_corr(weighted_ranks(x,w),weighted_ranks(y,w),w)
def bootstrap_correlations(x,y,counts,indices,kind):
 result=[]
 for i in range(0,len(counts),128):
  w=counts[i:i+128,indices];result.extend((weighted_srcc if kind=='SRCC' else weighted_corr)(x,y,w))
 return np.asarray(result)
def interval(v):
 ok=np.isfinite(v);vals=np.asarray(v)[ok]
 return {'ci_low':float(np.quantile(vals,.025)) if len(vals) else None,'ci_high':float(np.quantile(vals,.975)) if len(vals) else None,'valid_draws':int(ok.sum()),'draws':len(v),'valid_fraction':float(ok.mean())}

def holdout_evaluate():
 frozen=verify_selected() # Must precede even reading the holdout metadata or opinions.
 done=ROOT/'holdout/complete.json'
 if done.exists():
  r=read(done)
  for rel,h in r['artifact_hashes'].items():require(sha(ROOT/rel)==h,'Completed holdout artifact changed')
  return r
 split_path=PROJECT/'dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv'
 expected=read(ROOT/'amendment_lock.json')['external_files'][str(split_path)];require(sha(split_path)==expected,'Original split metadata changed')
 opening=ROOT/'holdout/opened.json'
 if not opening.exists():save(opening,{'opened_utc':now(),'selected_configuration_sha256':sha(ROOT/'selection/selected_matched_models.json'),'gate_sha256':sha(ROOT/'preflight/gate.json'),'all_selected_hashes_verified_before_open':True})
 save(ROOT/'status.json',{'state':'HOLDOUT_EVALUATION','training_runs_completed':24,'holdout_opened':True,'updated_utc':now(),'selected_configuration_sha256':sha(ROOT/'selection/selected_matched_models.json')})
 d=pd.read_csv(split_path,usecols=['dataset','image_id','ref_id','distortion_type','split','distorted_path','severity_or_level','mos_or_dmos'],dtype=str,keep_default_na=False)
 d=d[d.dataset.eq('KADID-10k')&d.split.eq('holdout')].copy().reset_index(drop=True)
 d['path']=d.distorted_path.map(base.remap_path);d['severity']=pd.to_numeric(d.severity_or_level);mos=pd.to_numeric(d.mos_or_dmos).to_numpy(float)
 require(len(d)==1625 and d.ref_id.nunique()==13 and np.isfinite(mos).all(),'Unexpected KADID holdout inventory')
 meta=d[['dataset','image_id','ref_id','distortion_type','split','path','severity']].copy();poleout=[];relative=meta.copy();energies={};qzs={}
 for arm in ARMS:
  zs={}
  for pole in ('EA','EH'):
   key=pole+'::'+arm;s=frozen['selected'][key];scored=score_checkpoint(meta,Path(s['checkpoint']),pole,arm,ROOT/'holdout/raw'/f'{pole}_{arm}.csv')
   norm=frozen['validation_normalization'][key]['KADID-10k'];v=scored.energy.to_numpy();z=(v-norm['mean'])/(norm['std_population']+EPS);zs[pole]=z;energies[(pole,arm)]=v
   poleout.append(scored.assign(pole=pole,distance=arm,z=z,MOS=mos))
  q=zs['EA']-zs['EH'];qzs[arm]=q;relative[arm]=q
 relative['MOS']=mos;relative.to_csv(ROOT/'holdout/matched_relative_scores.csv.gz',index=False,compression='gzip');pd.concat(poleout,ignore_index=True).to_csv(ROOT/'holdout/matched_pole_scores.csv.gz',index=False,compression='gzip')
 cfg=read(ROOT/'config.json');refs=sorted(meta.ref_id.unique());refindex={r:i for i,r in enumerate(refs)};idx=meta.ref_id.map(refindex).to_numpy(int)
 draws=np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(refs),(cfg['bootstrap_draws'],len(refs)));counts=np.stack([(draws==i).sum(1) for i in range(len(refs))],axis=1)
 np.savez_compressed(ROOT/'holdout/bootstrap_reference_counts.npz',counts=counts,references=np.asarray(refs));save(ROOT/'holdout/bootstrap_provenance.json',{'seed':cfg['bootstrap_seed'],'draws':cfg['bootstrap_draws'],'unit':'whole ref_id cluster, sampled with replacement','same_draws_for_all_arms':True,'reference_ids':refs,'counts_sha256':sha(ROOT/'holdout/bootstrap_reference_counts.npz')})
 pairs=pair_table(meta);pairidx=pairs.ref_id.map(refindex).to_numpy(int);corr=[];severity=[];accuracy=[];br={};points={}
 series={**energies,**{('Qz',a):v for a,v in qzs.items()}}
 for (pole,arm),v in series.items():
  sr=finite_corr(v,mos);pr=finite_corr(v,mos,'pearson')
  corr.append({'pole':pole,'distance':arm,'N':len(v),'N_references':len(refs),'SRCC':sr,'Pearson':pr,'minimum':float(v.min()),'maximum':float(v.max()),'range':float(np.ptp(v)),'std_population':float(v.std()),'distinct_scores':len(np.unique(v)),'excess_equal_scores':len(v)-len(np.unique(v))})
  for kind,point in [('SRCC',sr),('Pearson',pr)]:
   key=(pole,arm,kind,'ALL');points[key]=point;br[key]=bootstrap_correlations(v,mos,counts,idx,'SRCC' if kind=='SRCC' else 'Pearson')
  gaps=pair_gap(v,pairs,pole);correct=(gaps>0).astype(float);ties_mask=gaps==0
  subsets=[('ALL',np.arange(len(pairs)))]+[(str(t),np.asarray(ix)) for t,ix in pairs.groupby('distortion_type').groups.items()]
  for typ,ix in subsets:
   c=correct[ix];pi=pairidx[ix];point=float(c.mean());accuracy.append({'pole':pole,'distance':arm,'distortion_type':typ,'N_pairs':len(ix),'correct':int(c.sum()),'ties':int(ties_mask[ix].sum()),'strict_pair_accuracy':point,'tie_half_credit_accuracy':float((c+.5*ties_mask[ix]).mean())})
   key=(pole,arm,'pair_accuracy',typ);points[key]=point;values=[]
   for start in range(0,len(counts),128):
    w=counts[start:start+128,pi];values.extend((w*c).sum(1)/w.sum(1))
   br[key]=np.asarray(values)
  for typ,g in meta.groupby('distortion_type',sort=True):
   ix=g.index.to_numpy();point=finite_corr(v[ix],g.severity.to_numpy());severity.append({'pole':pole,'distance':arm,'distortion_type':typ,'N':len(ix),'severity_SRCC':point})
   key=(pole,arm,'severity_SRCC',str(typ));points[key]=point;br[key]=bootstrap_correlations(v[ix],g.severity.to_numpy(),counts,idx[ix],'SRCC')
  print('Holdout metrics/bootstrap',pole,arm,flush=True)
 boot=[]
 for (pole,arm,metric,typ),samples in br.items():
  basekey=(pole,'D0_LEGACY',metric,typ);key=(pole,arm,metric,typ);difference=samples-br[basekey]
  boot.append({'pole':pole,'distance':arm,'metric':metric,'distortion_type':typ,'baseline':'D0_LEGACY','estimate':points[key]-points[basekey],**interval(difference),'absolute_estimate':points[key],**{'absolute_'+k:v for k,v in interval(samples).items()}})
 ct=pd.DataFrame(corr);st=pd.DataFrame(severity);at=pd.DataFrame(accuracy);bt=pd.DataFrame(boot)
 ct.to_csv(ROOT/'holdout/holdout_correlations.csv',index=False);st.to_csv(ROOT/'holdout/severity_by_distortion.csv',index=False);at.to_csv(ROOT/'holdout/pair_order_accuracy.csv',index=False);bt.to_csv(ROOT/'holdout/ref_bootstrap_differences.csv',index=False)
 candidates=ct[ct.pole.eq('Qz')&ct.distance.ne('D0_LEGACY')].sort_values(['SRCC','distance'],ascending=[False,True]);best=candidates.iloc[0];b=bt[bt.pole.eq('Qz')&bt.distance.eq(best.distance)&bt.metric.eq('SRCC')&bt.distortion_type.eq('ALL')].iloc[0]
 interpretation='retain D0-Legacy' if b.estimate<=0 else ('positive estimate; inconclusive development-set evidence' if not np.isfinite(b.ci_low) or b.ci_low<=0 else 'reliable development-set improvement')
 dm=float(ct[ct.pole.eq('Qz')&ct.distance.eq('DM_AGGREGATE')].SRCC.iloc[0]);geometry=best.distance in ('KL_AGGREGATE','W2_AGGREGATE','BHATT_AGGREGATE') and float(best.SRCC)>dm
 decision={'best_nonlegacy_Qz_arm':best.distance,'SRCC':float(best.SRCC),'paired_SRCC_difference_from_D0':float(b.estimate),'ci_low':float(b.ci_low),'ci_high':float(b.ci_high),'interpretation':interpretation,'geometry_point_estimate_exceeds_DM_AGGREGATE':bool(geometry),'selection_changed_after_holdout':False}
 save(ROOT/'holdout/decision.json',decision)
 (ROOT/'holdout/holdout_report.txt').write_text('All six matched frozen arms evaluated on 1,625 KADID holdout images (13 references). KADID validation normalization only. 5,000 paired whole-reference bootstrap draws; no model selection based on holdout. Strict pair accuracy counts ties as incorrect, with separate half-credit diagnostics.\n'+json.dumps(decision,indent=2)+'\n')
 verify_selected();artifacts={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'holdout').rglob('*') if p.is_file() and p.name!='complete.json'}
 result={'status':'COMPLETE','completed_utc':now(),'selected_configuration_sha256':sha(ROOT/'selection/selected_matched_models.json'),'artifact_hashes':artifacts,'decision':decision};save(done,result);return result
