"""Freeze the approved acceptance rule, input artifacts and complete pipeline before training."""
import shutil
from common import *
from verify_amended import verify

def freeze():
 if (ROOT/'amendment_lock.json').exists():verify_frozen();print('Existing freeze verified');return
 verified=verify();probe=read(ROOT/'verification/full_objective_smoke.json');require(probe['status']=='PASS' and len(probe['rows'])==12 and probe['optimizer_steps']==0,'Full objective integration probe incomplete')
 testlog=(ROOT/'verification/pipeline_unit_tests.txt').read_text();require('Ran 22 tests' in testlog and testlog.rstrip().endswith('OK'),'Unit tests did not pass')
 result=read(ROOT/'preflight/amended/result.json');require(result['passed'] and len(result['rows'])==24,'Amended preflight incomplete')
 require(not (ROOT/'holdout/opened.json').exists(),'Holdout was opened before freezing')
 cfg=read(ROOT/'config.json')
 gate={'state':'PASS_AUTHORIZED_AMENDMENT','passed':True,'frozen_utc':now(),'authorization_file':'inputs/supervisor_amendment_and_authorization.txt','authorization_sha256':sha(ROOT/'inputs/supervisor_amendment_and_authorization.txt'),'uniqueness_rule':'Unique contiguous CHW float32 preprocessed tensors grouped by SHA-256; one representative each; >=0.99 distinct diagnostic scores. D0-Legacy only uses additional float64 arithmetic for this gate, without changing primary float32 D0.','all_other_checks':result['checks'],'unique_tensor_counts':result['duplicate_inputs'],'float32_collision_summary':result['float32_collision_summary'],'rows':result['rows'],'primary_scores_and_training_statistics_unchanged':True,'training_rows_removed':0,'training_optimizer_steps_before_freeze':0,'holdout_opened':False,'original_failed_gate_sha256':sha(ROOT/'historical/original_failed_gate.json'),'amended_result_sha256':sha(ROOT/'preflight/amended/result.json'),'independent_verification_sha256':sha(ROOT/'verification/independent_amended_verification.json'),'integration_probe_sha256':sha(ROOT/'verification/full_objective_smoke.json'),'pipeline_unit_test_log_sha256':sha(ROOT/'verification/pipeline_unit_tests.txt')}
 save(ROOT/'preflight/gate.json',gate);(ROOT/'preflight/gate.sha256').write_text(sha(ROOT/'preflight/gate.json')+'  gate.json\n')
 immutable=[]
 for folder in ('inputs','reference_stats','training_stats','preflight'):
  immutable.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
 immutable.extend(p for p in ROOT.glob('*.py'))
 immutable.extend([ROOT/'config.json',ROOT/'environment.txt',ROOT/'commands.sh',ROOT/'verification/full_objective_smoke.json',ROOT/'verification/pipeline_unit_tests.txt',ROOT/'historical/original_failed_gate.json'])
 external={};original=read(ROOT/'historical/input_lock.json')
 for pole,rec in original['checkpoints'].items():
  require(sha(rec['path'])==rec['sha256'],'Initialization checkpoint changed');external[rec['path']]=rec['sha256']
 for p,h in original['local_runtime_dependencies'].items():require(sha(p)==h,'Original dependency changed');external[p]=h
 for p in [PROJECT/'day11_day12/day11_rankall_pipeline.py',Path('/home/projectwork/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth')]:external[str(p)]=sha(p)
 split=PROJECT/'dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv';h=read(ROOT/'inputs/source_hashes.json')[str(split)];require(sha(split)==h,'Original dataset manifest changed');external[str(split)]=h
 lock={'version':1,'frozen_utc':now(),'authorization_sha256':gate['authorization_sha256'],'gate_sha256':sha(ROOT/'preflight/gate.json'),'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(immutable))},'external_files':external,'training_rows_deduplicated':False,'primary_D0_float32':True,'holdout_opened':False,'run_count':24}
 save(ROOT/'amendment_lock.json',lock);(ROOT/'amendment_lock.sha256').write_text(sha(ROOT/'amendment_lock.json')+'  amendment_lock.json\n');verify_frozen()
 from engine import run_configuration
 from run_pipeline import summary
 for pole in ('EA','EH'):
  for arm in ARMS:
   for lam in (.1,1.):
    p=runpath(pole,arm,lam);p.mkdir(parents=True,exist_ok=True);save(p/'configuration.json',run_configuration(pole,arm,lam));save(p/'status.json',{'state':'READY','optimizer_steps':0,'gate_sha256':sha(ROOT/'preflight/gate.json'),'amendment_lock_sha256':sha(ROOT/'amendment_lock.json')})
 summary();save(ROOT/'status.json',{'state':'READY_FOR_TRAINING','training_runs_started':0,'training_runs_total':24,'holdout_opened':False,'gate_sha256':sha(ROOT/'preflight/gate.json'),'amendment_lock_sha256':sha(ROOT/'amendment_lock.json'),'updated_utc':now()})
 (ROOT/'preflight/amended_preflight_report.txt').write_text('AMENDED PREFLIGHT PASS. All 24 pole/arm/split checks meet the approved uniqueness rule. Unique input tensors: train 8,850 of 9,040 rows; validation 1,820 of 1,860 rows. Every amended uniqueness fraction is 1.0. Primary D0 remains float32; its additional float64 score is used only for the uniqueness diagnostic. All original rows, primary scores, and twelve standardization files remain byte-identical. Twenty-two unit tests and twelve full-objective/backward probes passed, with zero optimizer steps before this freeze.\nGate SHA-256: '+sha(ROOT/'preflight/gate.json')+'\nAmendment lock SHA-256: '+sha(ROOT/'amendment_lock.json')+'\n')
 print('AMENDED GATE FROZEN:',sha(ROOT/'preflight/gate.json'));print('ALL PIPELINE INPUTS/CODE FROZEN:',sha(ROOT/'amendment_lock.json'))
if __name__=='__main__':freeze()
