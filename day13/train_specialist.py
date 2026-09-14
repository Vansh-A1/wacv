import argparse
from common import *
from engine import train_run
p=argparse.ArgumentParser();p.add_argument('arm',choices=list(ARMS));arm=p.parse_args().arm
info=read_json(OUT/'weights/arms.json')['arms'][arm]
assert info['feasible'],'Infeasible arm is excluded before training'
assert sha(info['weight_file'])==info['weight_sha256']
train_run(arm,Path(info['weight_file']),CHECKPOINTS['EA_init'],teacher=False,weight_info=info)
