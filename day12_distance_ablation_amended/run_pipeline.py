"""Run all 24 frozen experiments, validation selection, then guarded holdout."""
import argparse,fcntl,time,traceback,os
from common import *
from engine import train_run,run_configuration
from evaluation import validate_and_freeze,holdout_evaluate

def summary():
 records=[]
 for pole in ('EA','EH'):
  for arm in ARMS:
   for lam in (.1,1.):
    d=runpath(pole,arm,lam);record={'pole':pole,'distance':arm,'lambda_rank':lam,'seed':42,'epochs_required':25,'state':'PENDING','epochs_complete':0,'optimizer_steps_completed_epochs':0}
    if (d/'complete.json').exists():record.update(state='TRAINING_COMPLETE',epochs_complete=25,optimizer_steps_completed_epochs=read(d/'complete.json')['optimizer_steps'])
    elif (d/'progress.json').exists():
     p=read(d/'progress.json');record['state']=p['state'];record['epochs_complete']=max(0,int(p.get('epoch',0))-(0 if p['state']=='EPOCH_COMPLETE' else 1));record['optimizer_steps_completed_epochs']=record['epochs_complete']*2260
    if (d/'failure.json').exists():record.update(state='FAILED',failure=str(d/'failure.json'))
    if (d/'checkpoint_selection.json').exists():
     s=read(d/'checkpoint_selection.json');record.update(selected_epoch=s['epoch'],validation_score=s['validation_score'],selected_checkpoint_sha256=s['checkpoint_sha256'])
    records.append(record)
 pd.DataFrame(records).to_csv(ROOT/'run_summary.csv',index=False);return records

def finish():
 verify_frozen();hold=read(ROOT/'holdout/complete.json')
 out={'state':'COMPLETE','updated_utc':now(),'training_runs_completed':24,'validation_checkpoint_candidates':120,'selected_pole_distance_checkpoints':12,'matched_distances':6,'holdout_opened':True,'decision':hold['decision']}
 save(ROOT/'status.json',out);save(ROOT/'stage_summary.json',out)
 hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and 'prior_work' not in p.parts and 'checkpoints' not in p.parts and 'logs' not in p.parts and p.name not in ['final_integrity.json','.pipeline.lock','runner.pid']}
 save(ROOT/'verification/final_integrity.json',{'state':'COMPLETE','verified_utc':now(),'amendment_lock_sha256':sha(ROOT/'amendment_lock.json'),'selected_configuration_sha256':sha(ROOT/'selection/selected_matched_models.json'),'noncheckpoint_artifact_hashes':hashes,'checkpoint_hash_locations':'runs/*/*/*/seed_42/complete.json and selection/selected_matched_models.json','holdout_manifest_sha256':sha(ROOT/'holdout/complete.json')})
 (ROOT/'stage_report.txt').write_text('Approved modified Day12 Task4 completed. All 24 matched runs trained through 25 epochs; all 120 saved checkpoint candidates scored by validation-only severity rules. Twelve selected pole/distance checkpoints and calibration were frozen before KADID holdout was opened. All six matched Qz arms evaluated; 5,000 paired reference-cluster bootstrap draws. Original input rows, pair sampling, reference tensors, and primary float32 D0 calculations were preserved.\n'+json.dumps(hold['decision'],indent=2)+'\n')

def run(phase='all'):
 lockfile=(ROOT/'.pipeline.lock').open('a+')
 try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('The amended pipeline is already running')
 verify_frozen();save(ROOT/'launch.json',{'pid':os.getpid(),'launched_utc':now(),'gate_sha256':sha(ROOT/'preflight/gate.json'),'amendment_lock_sha256':sha(ROOT/'amendment_lock.json'),'phase':phase})
 current=None
 try:
  if phase in ('all','train'):
   for pole in ('EA','EH'):
    for arm in ARMS:
     for lam in (.1,1.):
      current=(pole,arm,lam);done=sum(r['state']=='TRAINING_COMPLETE' for r in summary());save(ROOT/'status.json',{'state':'TRAINING','current_run':{'pole':pole,'distance':arm,'lambda_rank':lam},'training_runs_completed':done,'training_runs_total':24,'holdout_opened':False,'updated_utc':now()})
      train_run(pole,arm,lam)
      records=summary();save(ROOT/'status.json',{'state':'TRAINING','training_runs_completed':sum(r['state']=='TRAINING_COMPLETE' for r in records),'training_runs_total':24,'holdout_opened':False,'updated_utc':now()})
   current=None
  if phase in ('all','validation'):
   save(ROOT/'status.json',{'state':'VALIDATION_SELECTION','training_runs_completed':24,'holdout_opened':False,'updated_utc':now()});validate_and_freeze();summary()
  if phase in ('all','holdout'):
   # This function independently verifies the freeze before any holdout read.
   holdout_evaluate();finish()
 except BaseException as e:
  error={'state':'FAILED','updated_utc':now(),'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'current_run':current,'holdout_opened':(ROOT/'holdout/opened.json').exists(),'resume_policy':'Last completed epoch and frozen configuration only; no automatic threshold or hyperparameter changes'}
  save(ROOT/'status.json',error)
  if current is not None:save(runpath(*current)/'failure.json',error)
  summary();print(error['traceback'],flush=True);raise
 finally:fcntl.flock(lockfile,fcntl.LOCK_UN);lockfile.close()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','train','validation','holdout'],default='all');args=parser.parse_args();run(args.phase)
