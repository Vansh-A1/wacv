"""Run every prescribed Day13 stage under one process lock; resume completed work."""
import fcntl,os,subprocess,sys,traceback
from common import *

def verify_implementation():
 p=ROOT/'implementation_lock.json';lock=read_json(p)
 for name,h in lock['source_hashes'].items():
  if sha(name)!=h:raise RuntimeError('Implementation changed after launch: '+name)

def run(label,args):
 verify_implementation();verify_inputs()
 save_json(ROOT/'run_status.json',{'state':'RUNNING','stage':label,'runner_pid':os.getpid(),'updated_utc':now(),'CSIQ_opened':False})
 print(now(),label,flush=True)
 log=ROOT/'logs'/(label+'.log')
 with log.open('a') as output:
  subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})

def main():
 ROOT.joinpath('logs').mkdir(exist_ok=True)
 save_json(ROOT/'run_status.json',{'state':'WAITING_FOR_ACTIVE_TEACHER','runner_pid':os.getpid(),'updated_utc':now(),'note':'The first teacher is already running; wait for the shared lock, then resume the complete pipeline.'})
 with (ROOT/'.pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   verify_implementation();run('preflight',['preflight.py'])
   if not (OUT/'validation/frozen_configuration.json').exists():
    for k in range(5):
     run(f'teacher_fold_{k}',['crossfit.py','train','--fold',str(k)])
     run(f'score_teacher_fold_{k}',['crossfit.py','score','--fold',str(k)])
    run('assemble_crossfit',['crossfit.py','assemble']);run('weights',['weights.py'])
    arms=read_json(OUT/'weights/arms.json')['arms']
    for arm,info in arms.items():
     if info['feasible']:run('train_'+arm,['train_specialist.py',arm])
    run('validation',['evaluation.py','validation'])
   run('verify_validation',['verification_pipeline.py','validation'])
   run('holdout',['evaluation.py','holdout'])
   run('verify_holdout',['verification_pipeline.py','holdout'])
   run('report',['final_report.py'])
   save_json(ROOT/'run_status.json',{'state':'COMPLETE','completed_utc':now(),'runner_pid':os.getpid(),'report':str(ROOT/'Day13_Report.md'),'CSIQ_opened':False})
  except Exception as exc:
   save_json(ROOT/'run_status.json',{'state':'FAILED','failed_utc':now(),'runner_pid':os.getpid(),'error':str(exc),'traceback':traceback.format_exc(),'resume':'python -B run_all.py; it resumes complete stages and epoch checkpoints after the issue is resolved.'})
   raise
if __name__=='__main__':main()
