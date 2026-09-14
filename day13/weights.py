"""Signed OOF shortages, six fixed specialist weighting arms and diagnostics."""
from common import *

def summarize(d,arm,kind,alpha,T,scope,label):
 raw=d.raw_weight.to_numpy();w=d.weight.to_numpy();ess=float(raw.sum()**2/np.square(raw).sum()) if np.square(raw).sum()>0 else 0.
 row={'arm':arm,'kind':kind,'alpha_H':alpha,'T_alpha':T if kind=='soft' else None,'scope':scope,'label':label,'N':len(d),
 'EH_reversal_fraction':float((d.g_H<0).mean()),'EH_tie_fraction':float((d.g_H==0).mean()),'margin_violation_fraction':float((d.violation>=0).mean()),
 'hard_active_fraction':float((d.violation>=0).mean()),'raw_mean':float(raw.mean()),'normalized_mean':float(w.mean()),'soft_effective_sample_size':ess if kind=='soft' else None,'ESS_fraction':ess/len(d) if kind=='soft' else None,'numerical_zero_soft_weights':int((raw==0).sum()) if kind=='soft' else None}
 for name,v in [('raw',raw),('normalized',w)]:
  for p in [0,1,5,25,50,75,95,99,100]:row[f'{name}_p{p}']=float(np.percentile(v,p))
 return row

def main():
 verify_inputs();assert read_json(OUT/'crossfit_EH/crossfit_integrity.json')['status']=='PASS'
 src=OUT/'crossfit_EH/oof_pair_scores.csv';d=read_csv(src,dtype={'ref_id':str});diagnostics=[];arms={};examples=[]
 for arm,(kind,alpha) in ARMS.items():
  out=d.copy();v,T,raw,norm,ess=weight_values(d.g_H,d.delta,alpha,kind)
  out['violation']=v;out['alpha_H']=alpha;out['raw_weight']=raw;out['weight']=norm;out['EH_reversed']=out.g_H<0
  out['diagnostic_focus']=out.raw_weight>0 if kind=='hard' else out.raw_weight>=np.quantile(raw,.75)
  active=float((v>=0).mean());feasible=active>=.05 if kind=='hard' else ess/len(d)>=.05
  path=OUT/'weights'/(arm+'_pairs.csv');out.to_csv(path,index=False)
  arms[arm]={'kind':kind,'alpha_H':alpha,'T_alpha':T if kind=='soft' else None,'raw_mean':float(raw.mean()),'normalized_mean':float(norm.mean()),'active_fraction':active,'ESS':ess if kind=='soft' else None,'ESS_fraction':ess/len(d) if kind=='soft' else None,'feasible':feasible,'status':'FEASIBLE' if feasible else 'SKIPPED_INFEASIBLE','weight_file':str(path),'weight_sha256':sha(path),'soft_upper_quartile_threshold':float(np.quantile(raw,.75)) if kind=='soft' else None,'normalization_epsilon':EPS,'numerical_zero_soft_weights':int((raw==0).sum()) if kind=='soft' else None}
  diagnostics.append(summarize(out,arm,kind,alpha,T,'overall','ALL'))
  for ds,g in out.groupby('dataset'):diagnostics.append(summarize(g,arm,kind,alpha,T,'dataset',ds))
  for (ds,typ),g in out.groupby(['dataset','distortion_type']):diagnostics.append(summarize(g,arm,kind,alpha,T,'dataset_distortion',ds+'::'+typ))
  for gap,g in out.groupby('delta'):diagnostics.append(summarize(g,arm,kind,alpha,T,'normalized_severity_gap',str(gap)))
  for label,g in [('highest',out.nlargest(10,'weight',keep='first')),('lowest',out.nsmallest(10,'weight',keep='first'))]:
   g=g.copy();g['arm']=arm;g['example_extreme']=label;examples.append(g)
  print(arm,arms[arm]['status'],'active',active,'ESS fraction',ess/len(d),flush=True)
 pd.DataFrame(diagnostics).to_csv(OUT/'weights/weight_diagnostics.csv',index=False)
 pd.concat(examples,ignore_index=True).to_csv(OUT/'weights/highest_lowest_examples.csv',index=False)
 save_json(OUT/'weights/arms.json',{'created_utc':now(),'OOF_pair_scores_sha256':sha(src),'arms':arms,'no_margin_changes':True,'no_MOS_DMOS_used':True})
if __name__=='__main__':main()
