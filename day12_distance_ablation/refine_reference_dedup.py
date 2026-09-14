"""One-time correction: exclude re-encoded duplicate RGB images before training."""
from pathlib import Path
import shutil,json
from prepare_reference_audit import ROOT,sha,read,save,now,frame

def main():
 f=ROOT/'reference_stats/EH';mf=read(f/'manifest.json');assert sha(f/'candidate_audit.csv')==mf['candidate_audit_sha256']
 a=ROOT/'verification/attempts/before_RGB_dedup'
 if a.exists():raise RuntimeError('Correction already archived; refusing to repeat')
 a.mkdir(parents=True)
 for n in ['reference_stats/EH','training_stats','preflight','runs']:
  p=ROOT/n
  if n=='preflight':shutil.copytree(p,a/n,ignore=shutil.ignore_patterns('cache'))
  else:shutil.copytree(p,a/n)
 for n in ['status.json','run_matrix.csv','logs/reference_and_preflight.log','reference_stats/reference_comparison.csv','reference_stats/reference_report.txt','reused/training_statistics_comparison.csv']:
  p=ROOT/n
  if p.exists():q=a/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 audit=frame(f/'candidate_audit.csv',dtype={'source_image_id':str,'source_reference_id':str},keep_default_na=False)
 seen=set();removed=[]
 for i,r in audit.iterrows():
  if not r.accepted:continue
  if r.rgb_sha256 in seen:
   audit.loc[i,'accepted']=False;audit.loc[i,'split_membership_decision']='EXCLUDE';audit.loc[i,'exclusion_reason']='duplicate_decoded_RGB_content';removed.append(r.canonical_path)
  else:seen.add(r.rgb_sha256)
 assert len(removed)==59
 accepted=audit[audit.accepted].copy();counts=accepted.source_dataset.value_counts().to_dict();accepted['weight']=accepted.source_dataset.map(lambda s:1/(len(counts)*counts[s]))
 audit.to_csv(f/'candidate_audit.csv',index=False);audit[~audit.accepted].to_csv(f/'eh_pristine_reference_v2_exclusions.csv',index=False)
 accepted[['source_dataset','source_image_id','source_reference_id','canonical_path','sha256','rgb_sha256','preprocessed_sha256','weight']].to_csv(f/'eh_pristine_reference_v2.csv',index=False)
 report=['EH-reference-v2: newly registered pristine reference pool; not a reconstruction of the unavailable legacy pool.',f'Candidate images: {len(audit)}',f'Accepted images: {len(accepted)}',f'Excluded images: {len(audit)-len(accepted)}',f'Included by source: {counts}',f'Weight sum: {accepted.weight.sum():.17g}',f'Source weight sums: {accepted.groupby("source_dataset").weight.sum().to_dict()}',f'Protected images/reference files audited: {len(frame(f/"protected_identity_audit.csv"))}','Accepted images are unique by canonical path, exact file hash, and decoded RGB hash.','Accepted images are disjoint from protected content by canonical path, exact SHA-256, dataset-qualified reference ID and decoded RGB hash.','Pristine status uses registered HR source branches, published dataset descriptions and valid full-size files.','No MOS/DMOS columns were read.','Correction before training: excluded 59 RGB-identical re-encoded duplicates; first eligible candidate retained deterministically. Earlier artifacts preserved under verification/attempts/before_RGB_dedup.','Exclusions by source and reason:',audit[~audit.accepted].groupby(['source_dataset','exclusion_reason']).size().to_string()]
 (f/'reference_audit_report.txt').write_text('\n'.join(report)+'\n')
 previous=sha(f/'manifest.json');mf.update(frozen_utc=now(),reference_count=len(accepted),source_counts=counts,source_weights=accepted.groupby('source_dataset').weight.sum().to_dict(),weight_sum=float(accepted.weight.sum()),supersedes_manifest_sha256=previous,correction_reason='59 different PNG files contain duplicate decoded RGB images; removed deterministically before any training or evaluation selection.')
 for n,k in [('eh_pristine_reference_v2.csv','accepted_manifest_sha256'),('eh_pristine_reference_v2_exclusions.csv','exclusions_sha256'),('candidate_audit.csv','candidate_audit_sha256')]:mf[k]=sha(f/n)
 save(f/'manifest.json',mf);(f/'manifest.sha256').write_text(sha(f/'manifest.json')+'  manifest.json\n')
 for n in ['reference_posteriors.npz','aggregate_reference.npz','between_reference.npz','extraction_complete.json']:(f/n).unlink()
 save(ROOT/'status.json',{'state':'REFERENCE_MANIFEST_FROZEN','accepted_EH_images':len(accepted),'training_runs_started':0,'holdout_scored':False,'updated_utc':now()})
 print('\n'.join(report))
if __name__=='__main__':main()
