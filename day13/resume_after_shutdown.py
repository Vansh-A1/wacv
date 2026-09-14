"""Continue verified specialist training after shutdown, preserving completed preprocessing."""
import fcntl,os,traceback
from common import *
from run_all import run,verify_implementation

def main():
 with (ROOT/'.pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   verify_implementation();verify_inputs()
   assert read_json(ROOT/'verification/shutdown_resume_audit.json')['status']=='PASS'
   if not (OUT/'validation/frozen_configuration.json').exists():
    arms=read_json(OUT/'weights/arms.json')
    assert sha(OUT/'crossfit_EH/oof_pair_scores.csv')==arms['OOF_pair_scores_sha256']
    for arm,info in arms['arms'].items():
     assert sha(info['weight_file'])==info['weight_sha256']
     if info['feasible']:run('train_'+arm,['train_specialist.py',arm])
    run('validation',['evaluation.py','validation'])
   run('verify_validation',['verification_pipeline.py','validation'])
   run('holdout',['evaluation.py','holdout'])
   run('verify_holdout',['verification_pipeline.py','holdout'])
   run('report',['final_report.py'])
   save_json(ROOT/'run_status.json',{'state':'COMPLETE','completed_utc':now(),'runner_pid':os.getpid(),'report':str(ROOT/'Day13_Report.md'),'CSIQ_opened':False})
  except Exception as exc:
   save_json(ROOT/'run_status.json',{'state':'FAILED','failed_utc':now(),'runner_pid':os.getpid(),'error':str(exc),'traceback':traceback.format_exc()})
   raise
if __name__=='__main__':main()
