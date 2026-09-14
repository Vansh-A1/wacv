"""Read the running experiment without loading datasets or checkpoint tensors."""
from datetime import datetime,timezone
from common import *

def main():
 status=read_json(ROOT/'run_status.json') if (ROOT/'run_status.json').exists() else {'state':'TRAINING_FIRST_TEACHER'}
 out={'runner':status,'updated_utc':now(),'teachers':[],'specialists':[]}
 for group,labels in [('crossfit_EH',[f'fold_{k}' for k in range(5)]),('training',list(ARMS))]:
  for label in labels:
   folder=OUT/group/label;entry={'name':label,'state':'PENDING','completed_epochs':0}
   if (folder/'progress.json').exists():entry.update(read_json(folder/'progress.json'))
   if (folder/'train_log.csv').exists():
    log=read_csv(folder/'train_log.csv');entry['completed_epochs']=int(log.epoch.max());entry['mean_epoch_seconds']=float(log.time_seconds.mean())
   if (folder/'complete.json').exists():entry['state']='COMPLETE'
   out['teachers' if group=='crossfit_EH' else 'specialists'].append(entry)
 print(json.dumps(clean(out),indent=2))
 save_json(ROOT/'latest_status.json',out)
if __name__=='__main__':main()
