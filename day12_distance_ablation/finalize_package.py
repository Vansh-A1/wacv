"""Write the readable report and hash inventory, then package the new folder."""
from pathlib import Path
import json,hashlib,zipfile,datetime
import pandas as pd
R=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 verified=read(R/'verification/independent_verification.json');assert verified['verification_result']=='PASS_FOR_COMPLETED_PREPARATION_AND_REPORTED_BLOCKER'
 status=read(R/'status.json');assert status['state']=='BLOCKED_PREFLIGHT' and status['training_runs_started']==0
 rows=pd.read_csv(R/'preflight/distance_preflight.csv',float_precision='round_trip');rows=rows[rows.dataset=='ALL']
 table=['| Pole | Distance | Training distinct / N | Validation distinct / N |','|---|---|---:|---:|']
 for (pole,arm),g in rows.groupby(['pole','distance'],sort=False):
  a=g[g.split=='train'].iloc[0];b=g[g.split=='val'].iloc[0]
  table.append(f'| {pole} | {arm} | {a.distinct_scores:,} / {a.N:,} ({a.distinct_fraction:.4%}) | {b.distinct_scores:,} / {b.N:,} ({b.distinct_fraction:.4%}) |')
 report='''# Modified Day 12 Task 4: execution report

**Current outcome: reference registration and preflight are complete. The prescribed data preflight fails, so the 24 training runs, checkpoint selection and holdout evaluation have not been performed.**

## Scope and preservation

The source is the 16-page modified PDF copied into `inputs/`. Work is in the new `/home/projectwork/student_package/day12_distance_ablation` folder. All 22 files in the original `/home/projectwork/student_package/day12_Task4` are byte-identical to the pre-work snapshot, and are also copied under `reused/old_task4`.

The old task had no completed training runs. Its blocked prerequisite records are useful provenance; they cannot be relabeled as results of the revised experiment. We reused the unchanged image and pair manifests, pre-ranking posterior caches, legacy reference tensors and mandatory Task-2 EA reference. Revised distance scores and preflight checks were recomputed. The five previously available training-statistic results agree (exactly, except approximately 3.47e-18 EA KL mean roundoff); all 12 revised records are now saved.

## New reference and implementations

The unavailable original EH reference pool is replaced by a newly registered EH-v2 reference, as the modified PDF allows. All 19,300 candidates were audited without reading MOS/DMOS. Final accepted pool: 3,341 unique pristine HR images, comprising 750 DIV2K and 2,591 Flickr2K. Excluded: 15,900 downsampled/degraded files and 59 different PNG files with duplicate decoded RGB pixels. Every included DIV2K image has weight 1/1,500; every included Flickr2K image has weight 1/5,182. Both sources therefore contribute total weight 0.5. The weight sum is one within the specified 1e-12 tolerance.

The audit also checked 8,378 protected image/reference files, with no accepted overlap by canonical path, file SHA-256, decoded RGB hash or dataset-qualified reference identity. The final accepted manifest was frozen and hashed before final encoder extraction. Both legacy references remain unchanged; EA's existing 2,000-image aggregate is reused exactly.

The six implemented arms are D0-Legacy, DM-Between, DM-Aggregate, KL-Aggregate, squared W2-Aggregate and Bhattacharyya-Aggregate. Reference moments use weighted population formulas. KL uses image-posterior-to-reference direction. Mean-only denominators and variance clamps follow the PDF without double epsilon protection. EA uses mild-minus-severe ranking gaps; EH uses severe-minus-mild. The revised ranking formula uses softplus with margin 0.1 and frozen training-only standardization.

## Verified results before training

- 13 synthetic unit tests passed, covering formulas, finite/nonnegative distances, equal distributions, mean/variance sensitivity, direction, squaring, batching, gradients, weighted moments and the gate.
- All 12 pole/arm gradient probes passed with finite, nonzero required gradients.
- All 24 pooled pole/arm/split checks have finite scores and population standard deviation above 1e-12.
- Fresh deterministic GPU replay matches the copied posteriors bit for bit for 16 images per pole/split (64 image-pole replay cases).
- Independent NumPy recomputation verified all distance columns for every training and validation image entry, the population moments, 12 standardization records, source weights, artifact hashes and real-pair polarity.
- The 99% distinct-score test failed for every pole/arm/split combination:

'''+ '\n'.join(table)+'''

## Why the gate fails

Different dataset entries sometimes produce identical 256-by-256 RGB center crops. Training contains 56 such duplicate groups involving 246 entries; validation contains 12 groups involving 52 entries. Each investigated duplicate-posterior group was confirmed to have identical actual preprocessed pixels. At most 8,850 distinct deterministic posterior outputs are possible among the 9,040 training entries, and 1,820 among the 1,860 validation entries. These bounds are already below 99%, irrespective of reference choice. Primary legacy float32 arithmetic introduces some additional score collisions, most visibly for EH D0.

A separate float64 D0 diagnostic recovers 8,850 / 1,820 distinct scores for both poles. This supports a rounding explanation for the additional legacy collisions. It does not modify the primary D0 implementation, its training statistics or the failed gate.

Step 5 says: “Do not begin full training until every primary arm passes its applicable preflight checks.” Accordingly, all 24 planned run directories contain explicit `BLOCKED_PREFLIGHT` status with zero optimizer steps. No images were silently discarded, no primary score jitter was added, and no threshold was reduced.

## Corrections retained in the audit history

The first fresh replay detected that TF32 settings did not match the historical cache extraction. That attempt was archived; final extraction disables TF32 and uses deterministic seed-42 settings. A separate independent reference audit detected re-encoded duplicate RGB files after the first pool registration. The superseded 3,400-image artifacts were archived, the 59 duplicates excluded deterministically, the source weights recalculated, and the corrected manifest frozen before re-encoding. These are documented execution corrections, not alternate reported experiments or performance-based model choices.

## What remains and what can be discussed with the professor

Training quality and comparative image-quality performance cannot yet be assessed. No revised trained checkpoints, validation-selected winners, matched Qz holdout correlations, bootstrap intervals or distance-improvement claims exist.

The remaining work is the 24 matched 25-epoch training runs, validation-only checkpoint selection, per-dataset normalization, model freeze, all six matched KADID-holdout evaluations and the specified 5,000-draw reference-cluster bootstrap analysis. These depend on resolving the mandatory preflight conflict.

`PROPOSED_PREFLIGHT_AMENDMENT.md` contains a concrete **unapplied** proposal: make the uniqueness diagnostic account for identical preprocessed inputs and legacy arithmetic rounding while preserving the original training data, splits and primary calculations. This requires explicit approval of the changed acceptance rule. Keeping the PDF unchanged leaves training blocked on these inputs.

See `REQUIREMENTS_CHECKLIST.md` for exact completed/blocked requirements, `ASSUMPTIONS.md` for source and numerical interpretations, and `verification/independent_verification.json` for independent evidence. This package references the existing local dataset/checkpoint paths and requires the existing project code and environment for reproduction; it does not include the large source datasets or model checkpoints.
'''
 (R/'Modified_Task4_Report.md').write_text(report)
 cfg=read(R/'config.json');lock={'source_pdf':{'path':str(R/'inputs/modified_Day12_Task_4.pdf'),'sha256':sha(R/'inputs/modified_Day12_Task_4.pdf')},'config_sha256':sha(R/'config.json'),'checkpoints':{p:{'path':cfg['checkpoint_'+p],'sha256':sha(cfg['checkpoint_'+p])} for p in ['EA','EH']},'reference_manifests':{p:sha(R/f'reference_stats/{p}/manifest.json') for p in ['EA','EH']},'inherited_artifacts_manifest_sha256':sha(R/'reused/reuse_manifest.json'),'local_runtime_dependencies':{str(p):sha(p) for p in [R.parent/'external/model.py',R.parent/'external/dataloader.py',R.parent/'day12_task2/run_task2.py',R.parent/'day12_task2/reference_math.py']}}
 (R/'input_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
 inventory={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='final_integrity.json' and '__pycache__' not in p.parts}
 integrity={'status':'VERIFIED_PREPARATION_WITH_DOCUMENTED_PREFLIGHT_FAILURE','whole_task_complete':False,'original_task4_unchanged':True,'training_runs_completed':0,'training_runs_blocked':24,'holdout_scored':False,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'file_count':len(inventory),'sha256':inventory}
 (R/'verification/final_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
 archive=R.parent/'output/Modified_Day12_Task4_Work.zip';archive.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(R.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(R.parent)))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'archive_sha256':sha(archive),'files_in_inventory':len(inventory),'integrity':str(R/'verification/final_integrity.json')},indent=2))
if __name__=='__main__':main()
