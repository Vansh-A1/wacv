#!/usr/bin/env python3
"""Staged frozen-score analysis. No optimizer or training operation exists here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT))
from external.model import CVAEGenerator_v2
from external.dataloader import _resize_short_side,center_crop
from score import sz_from_stats
from fusion_core import CANDIDATES, EPS, fit_calibration, apply_calibration, selection_table, pair_outputs, score_summary, severity_table, validation_bootstrap, percentile_ci, evaluation_bootstrap, correlations

SPLITS=ROOT/'dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv'
HISTORICAL=ROOT/'day12_task1_distance_ablation/stage0_controls/protocol.json'
META=['dataset','image_id','ref_id','distortion_type','severity_or_level','split','distorted_path','ref_path']


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    if isinstance(v,Path): return str(v)
    return v


def write_json(path,data,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as f: json.dump(clean(data),f,indent=2,allow_nan=False);f.write('\n')


def read_json(path): return json.loads(Path(path).read_text())


def remap(p): return p.replace('/data/projectwork/swati_mam/kadid10k','/data/projectwork/swati_mam/new model data/kadid10k')


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_sources(lock):
    for path,expected in lock['source_hashes'].items():
        if sha(path)!=expected: raise RuntimeError('Changed frozen input: '+path)


def opinion_free_manifest():
    d=pd.read_csv(SPLITS,usecols=META,dtype=str,keep_default_na=False)
    d=d[d.dataset.isin(['KADID-10k','TID2013'])].copy()
    d['path']=d.distorted_path.map(remap)
    d['ref_path']=d.ref_path.map(remap)
    d['severity']=pd.to_numeric(d.severity_or_level,errors='raise')
    d=d.drop(columns=['distorted_path','severity_or_level'])
    if d.duplicated(['dataset','image_id']).any(): raise RuntimeError('Duplicate dataset/image identity')
    if (d.groupby(['dataset','ref_id']).split.nunique()>1).any(): raise RuntimeError('Reference leakage')
    if (d.groupby('path').split.nunique()>1).any(): raise RuntimeError('Image path leakage')
    return d


def prepare():
    if (OUT/'input_lock.json').exists(): raise RuntimeError('Input lock exists; do not overwrite it')
    hist=read_json(HISTORICAL)
    used=[SPLITS,ROOT/'score.py',ROOT/'external/model.py',ROOT/'external/dataloader.py']
    expected={str(p):hist['source_hashes'][str(p)] for p in used}
    poles={}
    for pole in ('EA','EH'):
        old=hist['poles'][pole];path=Path(old['checkpoint'])
        expected[str(path)]=old['checkpoint_sha256']
        if sha(path)!=old['checkpoint_sha256']: raise RuntimeError('Checkpoint hash mismatch')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['epoch']!=old['epoch'] or float(ck['config']['lambda_rank'])!=.1: raise RuntimeError('Checkpoint epoch/lambda mismatch')
        if ck['checkpoint_kind']!=old['checkpoint_kind'] or ck.get('sz_mode')!='mu_only': raise RuntimeError('Wrong model kind')
        if f"epoch_{old['epoch']:04d}.pth"!=path.name: raise RuntimeError('Stored/filename epoch mismatch')
        for name in ('mu_ref','Sigma_ref'):
            if not torch.isfinite(ck[name]).all(): raise RuntimeError('Invalid reference')
        if not (ck['Sigma_ref']>0).all(): raise RuntimeError('Nonpositive reference variance')
        if not all(torch.isfinite(t).all() for t in ck['generator'].values()): raise RuntimeError('Invalid model state')
        ref_file=OUT/'inputs'/f'{pole}_legacy_reference.npz'
        np.savez(ref_file,mu_ref=ck['mu_ref'].numpy(),Sigma_ref=ck['Sigma_ref'].numpy())
        poles[pole]={k:old[k] for k in ('checkpoint','checkpoint_sha256','epoch','lambda','checkpoint_kind')}
        poles[pole].update(stored_epoch=ck['epoch'],stored_lambda=ck['config']['lambda_rank'],
            reference_file=str(ref_file),reference_sha256=sha(ref_file),sz_mode=ck['sz_mode'],
            seed=ck.get('seed'),source_config_seed=ck['config'].get('seed'))
    for p in [HISTORICAL,OUT/'inputs/day12_Task5.pdf',OUT/'ASSUMPTIONS.md',OUT/'fusion_core.py',OUT/'run_task5.py']:
        expected[str(p)]=sha(p)
    for path,value in expected.items():
        if sha(path)!=value: raise RuntimeError('Source hash mismatch: '+path)
    d=opinion_free_manifest();val=d[d.split.eq('val')].copy().reset_index(drop=True)
    if len(val)!=1860: raise RuntimeError('Wrong validation membership')
    val.to_csv(OUT/'inputs/validation_manifest.csv',index=False)
    # Membership metadata only: no test images, targets, or test scores opened.
    d[['dataset','image_id','ref_id','split','path']].to_csv(OUT/'inputs/split_membership_without_opinions.csv',index=False)
    refs=val[['dataset','ref_id','ref_path']].drop_duplicates()
    refs.to_csv(OUT/'inputs/validation_reference_manifest.csv',index=False)
    lock={'created_utc':now(),'poles':poles,'source_hashes':expected,
        'validation_manifest':str(OUT/'inputs/validation_manifest.csv'),
        'validation_manifest_sha256':sha(OUT/'inputs/validation_manifest.csv'),
        'metadata_only_split_audit':{'reference_disjoint':True,'path_disjoint':True,'duplicate_ids':0,
            'validation_counts':val.groupby('dataset').size().to_dict(),
            'validation_reference_counts':val.groupby('dataset').ref_id.nunique().to_dict()},
        'D0_formula_identifier':'historical_score.sz_from_stats_mu_only_float32_eps1e-8',
        'D0_formula':'sqrt(sum((mu_ref-mu_x)^2/(Sigma_ref+1e-8)))',
        'preprocessing':'RGB; inherited _resize_short_side only upscales short side<256; center_crop(256); float32 [0,1]',
        'batch_size':64,'device':'cuda','autocast':False,'tie_break_order':list(CANDIDATES),
        'bootstrap':{'draws':5000,'seed':20260908,'unit':'ref_id','validation_stratified_by_dataset':True},
        'no_MOS_DMOS_for_calibration_selection':True,'training':False,
        'kadid_holdout_opened':False,'CSIQ_opened':False}
    write_json(OUT/'input_lock.json',lock,True)
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
        'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'CUDA':torch.version.cuda}
    write_json(OUT/'environment.json',env)
    (OUT/'environment.txt').write_text('\n'.join(f'{k}: {v}' for k,v in env.items())+'\n')
    print('Input lock saved. Validation:',len(val),'images; all checkpoint identities verified.',flush=True)


def image_identity(path):
    with Image.open(path) as im:
        im=im.convert('RGB');h=hashlib.sha256()
        h.update(str(im.size).encode());h.update(im.tobytes())
    return sha(path),h.hexdigest()


class ImageDataset(Dataset):
    def __init__(self,frame): self.paths=frame.path.tolist()
    def __len__(self): return len(self.paths)
    def __getitem__(self,index):
        path=self.paths[index]
        with Image.open(path) as im:
            im=im.convert('RGB');h=hashlib.sha256();h.update(str(im.size).encode());h.update(im.tobytes())
            pixels=h.hexdigest();im=center_crop(_resize_short_side(im,256),256)
            arr=(np.asarray(im)/255.).astype('float32')
        return torch.from_numpy(arr.transpose(2,0,1).copy()),index,sha(path),pixels


def posterior(model,x):
    for i in range(1,7): x=F.relu(getattr(model,'enc'+str(i))(x))
    hidden=model.fc1(F.adaptive_avg_pool2d(x,1).reshape(len(x),-1))
    return model.fc_mu(hidden),model.fc_log_var(hidden)


def fresh_scores(frame,stage,lock):
    verify_sources(lock)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no silent fallback')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42)
    outdir=OUT/stage;outdir.mkdir(parents=True,exist_ok=True)
    df=frame.reset_index(drop=True).copy()
    if df.duplicated(['dataset','image_id']).any(): raise RuntimeError('Duplicate scoring rows')
    loader=DataLoader(ImageDataset(df),batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
    pieces=[];provenance={};identity0=None
    for pole in ('EA','EH'):
        info=lock['poles'][pole];path=Path(info['checkpoint'])
        if sha(path)!=info['checkpoint_sha256']: raise RuntimeError('Wrong checkpoint loaded')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['epoch']!=info['epoch'] or ck['config']['lambda_rank']!=info['lambda']: raise RuntimeError('Metadata changed')
        model=CVAEGenerator_v2(latent_dim=100,image_size=256).cuda().eval();model.load_state_dict(ck['generator'],strict=True)
        for p in model.parameters(): p.requires_grad_(False)
        before=state_hash(model);mus=[];lvs=[];vals=[];hashes=[];pixelhash=[];seen=[];lastlog=time.time()
        mr=ck['mu_ref'].cuda();vr=ck['Sigma_ref'].cuda()
        with torch.inference_mode():
            for batch,(x,ids,bh,ph) in enumerate(loader):
                x=x.cuda(non_blocking=True);mu,lv=posterior(model,x)
                if batch==0:
                    # Same batch shape for exact encoder/full-forward equivalence.
                    _,fm,fl=model(x)
                    if not torch.equal(mu,fm) or not torch.equal(lv,fl): raise RuntimeError('Encoder shortcut mismatch')
                if not torch.isfinite(mu).all() or not torch.isfinite(lv).all(): raise RuntimeError('Nonfinite latent')
                score=sz_from_stats(mu,lv,mr,vr,eps=EPS,sigma_t_max=1.0,mu_only=True)
                if not torch.isfinite(score).all(): raise RuntimeError('Nonfinite D0')
                mus.append(mu.cpu().numpy());lvs.append(lv.cpu().numpy());vals.append(score.cpu().numpy())
                hashes.extend(bh);pixelhash.extend(ph);seen.extend(ids.tolist())
                if time.time()-lastlog>20 or len(seen)==len(df):
                    print(f'{stage} {pole}: {len(seen)}/{len(df)} images',flush=True);lastlog=time.time()
        if seen!=list(range(len(df))): raise RuntimeError('Scoring row order mismatch')
        after=state_hash(model)
        if before!=after or sha(path)!=info['checkpoint_sha256']: raise RuntimeError('Frozen model changed')
        scores=np.concatenate(vals).astype(np.float64)
        if scores.std()<=EPS or np.ptp(scores)<=EPS: raise RuntimeError('Degenerate pole '+pole)
        part=df[['dataset','image_id']].copy();part[pole]=scores;pieces.append(part)
        current=list(zip(hashes,pixelhash))
        if identity0 is not None and identity0!=current: raise RuntimeError('Images changed between pole scoring')
        identity0=current
        np.savez_compressed(outdir/(pole+'_posteriors.npz'),mu=np.concatenate(mus),logvar=np.concatenate(lvs),
            image_id=df.image_id.to_numpy(str),dataset=df.dataset.to_numpy(str),mu_ref=ck['mu_ref'].numpy(),Sigma_ref=ck['Sigma_ref'].numpy())
        provenance[pole]={'checkpoint_sha256':sha(path),'epoch':ck['epoch'],'lambda':ck['config']['lambda_rank'],
            'model_state_sha256_before':before,'model_state_sha256_after':after,'encoder_forward_equivalence':True,
            'D0_finite':True,'score_std_population':float(scores.std()),'N':len(scores),'freshly_scored':True}
        del model,ck;torch.cuda.empty_cache()
    result=df.merge(pieces[0],on=['dataset','image_id'],validate='one_to_one').merge(pieces[1],on=['dataset','image_id'],validate='one_to_one')
    result['image_sha256']=[x[0] for x in identity0];result['RGB_pixel_sha256']=[x[1] for x in identity0]
    result.to_csv(outdir/'raw_pole_scores.csv',index=False)
    write_json(outdir/'scoring_provenance.json',{'completed_utc':now(),'poles':provenance,'training':False,'MOS_used_in_scoring':False})
    verify_sources(lock)
    return result


def verify_frozen():
    path=OUT/'frozen_configuration.json';expected=(OUT/'frozen_configuration.sha256').read_text().split()[0]
    if sha(path)!=expected: raise RuntimeError('Frozen configuration was changed')
    frozen=read_json(path)
    for file,expected in frozen['frozen_artifact_hashes'].items():
        if sha(file)!=expected: raise RuntimeError('Changed frozen artifact: '+file)
    verify_sources(read_json(OUT/'input_lock.json'))
    return frozen


def validate():
    if (OUT/'frozen_configuration.json').exists(): raise RuntimeError('Selection already frozen; never overwrite it')
    lock=read_json(OUT/'input_lock.json');verify_sources(lock)
    path=Path(lock['validation_manifest'])
    if sha(path)!=lock['validation_manifest_sha256']: raise RuntimeError('Validation manifest changed')
    val=pd.read_csv(path,dtype={'dataset':str,'image_id':str,'ref_id':str})
    val=fresh_scores(val,'validation',lock)
    refs=pd.read_csv(OUT/'inputs/validation_reference_manifest.csv',dtype=str)
    with ThreadPoolExecutor(max_workers=4) as pool: identities=list(pool.map(image_identity,refs.ref_path))
    refs['image_sha256']=[r[0] for r in identities];refs['RGB_pixel_sha256']=[r[1] for r in identities]
    refs.to_csv(OUT/'inputs/validation_reference_hashes.csv',index=False)
    stats,ecdfs=fit_calibration(val);scored=apply_calibration(val,stats,ecdfs)
    scored.to_csv(OUT/'validation/fusion_scores.csv',index=False)
    write_json(OUT/'calibration/z_statistics.json',stats)
    np.savez(OUT/'calibration/empirical_CDF_arrays.npz',**ecdfs)
    table,types,selected=selection_table(scored)
    table.to_csv(OUT/'selection/validation_selection.csv',index=False)
    types.to_csv(OUT/'validation/severity_by_distortion.csv',index=False)
    pair_summary,pair_details=pair_outputs(scored)
    pair_summary.to_csv(OUT/'validation/pair_accuracy.csv',index=False)
    pair_details.to_csv(OUT/'validation/pair_details.csv',index=False)
    score_summary(scored).to_csv(OUT/'validation/score_summary.csv',index=False)
    print('Validation-selected candidate:',selected,'; computing 5000 fixed-choice cluster draws.',flush=True)
    delta,draws=validation_bootstrap(scored,selected)
    pd.DataFrame({'draw':np.arange(5000),'delta_S_fusion_vs_L00':delta}).to_csv(OUT/'selection/validation_bootstrap_draws.csv',index=False)
    np.savez_compressed(OUT/'selection/validation_cluster_draws.npz',**{ds:cts for ds,(refs,cts) in draws.items()})
    ci=percentile_ci(delta);best=table[table.selected].iloc[0];baseline=table[table.candidate.eq('L00')].iloc[0]
    ci.update(selected=selected,delta_S_fusion=float(best.S_fusion-baseline.S_fusion),
        advantage_uncertain=ci['ci_low']<=0<=ci['ci_high'],selection_fixed_in_draws=True,
        bootstrap_seed=20260908,bootstrap_draws=5000,reference_order={ds:refs for ds,(refs,_) in draws.items()})
    write_json(OUT/'selection/validation_bootstrap.json',ci)
    frozen_files=[OUT/'input_lock.json',OUT/'inputs/validation_manifest.csv',OUT/'inputs/split_membership_without_opinions.csv',
        OUT/'inputs/validation_reference_hashes.csv',OUT/'validation/raw_pole_scores.csv',OUT/'validation/fusion_scores.csv',
        OUT/'calibration/z_statistics.json',OUT/'calibration/empirical_CDF_arrays.npz',
        OUT/'selection/validation_selection.csv',OUT/'selection/validation_bootstrap.json',OUT/'ASSUMPTIONS.md',OUT/'fusion_core.py',OUT/'run_task5.py']
    definitions={q:f'{w}*z_A - {1.-w}*z_H' for q,w in zip(CANDIDATES[:5],[0.,.25,.5,.75,1.])}
    definitions.update(PAVG='(u_A+u_H)/2',PHARM='2*u_A*u_H/(u_A+u_H+1e-8)')
    frozen={'frozen_utc':now(),'task':'Day 12 Task 5 frozen-score fusion','checkpoints':lock['poles'],
        'D0_formula_identifier':lock['D0_formula_identifier'],'D0_formula':lock['D0_formula'],
        'calibration_scope':'pooled locked KADID+TID validation, 1860 images',
        'z_calibration':stats,'ECDF_formula':'(n_below + 0.5*n_equal + 0.5)/(N_cal+1)',
        'percentile_polarity':{'u_A':'F_A(EA)','u_H':'1-F_H(EH)'},'fusion_definitions':definitions,
        'selected_Q_star':selected,'validation_scores':table.to_dict('records'),
        'tie_break_order':list(CANDIDATES),'tie_break_result':table[table.S_fusion.eq(best.S_fusion)].candidate.tolist(),
        'validation_bootstrap':ci,'bootstrap':lock['bootstrap'],
        'polarity':'Every candidate is higher=better; raw EA higher=better, raw EH higher=worse',
        'MOS_DMOS_unused_for_calibration_fusion_hyperparameter_selection':True,
        'kadid_holdout_opened':False,'CSIQ_opened':False,'training':False,
        'frozen_artifact_hashes':{str(p):sha(p) for p in frozen_files},
        'test_calibration':'Apply pooled validation z statistics and empirical CDF arrays unchanged to KADID and CSIQ',
        'CSIQ_primary_comparison':selected+' versus L00','CSIQ_target':'negative DMOS',
        'CSIQ_secondary_ablations':[q for q in CANDIDATES if q not in (selected,'L00')],
        'CSIQ_logistic_mapping':False}
    write_json(OUT/'frozen_configuration.json',frozen,True)
    (OUT/'frozen_configuration.sha256').write_text(sha(OUT/'frozen_configuration.json')+'  frozen_configuration.json\n')
    (OUT/'selection/validation_report.txt').write_text('Frozen-score validation selection\n'+table.to_string(index=False)+'\n\n'+json.dumps(clean(ci),indent=2)+'\n')
    print('FROZEN CONFIGURATION SAVED. Both test-open flags are false.',flush=True)


def evaluate(which,manifest=None):
    frozen=verify_frozen();lock=read_json(OUT/'input_lock.json');stage='kadid_holdout' if which=='kadid' else 'csiq'
    if (OUT/stage/'evaluation_complete.json').exists(): raise RuntimeError('Evaluation already complete; do not overwrite')
    write_json(OUT/stage/'opened.json',{'opened_utc':now(),'frozen_configuration_sha256':sha(OUT/'frozen_configuration.json'),
        'selected_Q_star':frozen['selected_Q_star'],'dataset':which},exclusive=not (OUT/stage/'opened.json').exists())
    if which=='kadid':
        # Only now materialize human scores and holdout images.
        raw=pd.read_csv(SPLITS,usecols=META+['mos_or_dmos'],dtype=str,keep_default_na=False)
        d=raw[raw.dataset.eq('KADID-10k')&raw.split.eq('holdout')].copy()
        d['path']=d.distorted_path.map(remap);d['ref_path']=d.ref_path.map(remap)
        d['severity']=pd.to_numeric(d.severity_or_level,errors='raise')
        d['MOS']=pd.to_numeric(d.mos_or_dmos,errors='raise');d['target_quality']=d.MOS
        d=d.drop(columns=['distorted_path','severity_or_level','mos_or_dmos'])
        if len(d)!=1625: raise RuntimeError('Wrong KADID holdout membership')
    else:
        if manifest is None: raise RuntimeError('CSIQ manifest required')
        d=pd.read_csv(manifest,dtype={'image_id':str,'ref_id':str,'dataset':str})
        required={'dataset','image_id','ref_id','distortion_type','severity','path','ref_path','DMOS'}
        if not required.issubset(d.columns) or set(d.dataset)!={'CSIQ'}: raise RuntimeError('Invalid CSIQ metadata')
        d['split']='confirmatory';d['target_quality']=-pd.to_numeric(d.DMOS,errors='raise')
        d['severity']=pd.to_numeric(d.severity,errors='coerce')
        write_json(OUT/stage/'external_manifest_provenance.json',{'path':str(manifest),'sha256':sha(manifest),'accepted_after_freeze':True})
    d=d.reset_index(drop=True)
    if not np.isfinite(d.target_quality).all(): raise RuntimeError('Nonfinite evaluation target')
    val=pd.read_csv(OUT/'validation/raw_pole_scores.csv',dtype={'image_id':str,'ref_id':str})
    if set(zip(d.dataset,d.image_id)) & set(zip(val.dataset,val.image_id)): raise RuntimeError('Validation identity overlap')
    if which=='kadid' and set(d.ref_id)&set(val[val.dataset.eq('KADID-10k')].ref_id): raise RuntimeError('Reference overlap')
    with ThreadPoolExecutor(max_workers=4) as pool: ids=list(pool.map(image_identity,d.path))
    if set(a for a,b in ids)&set(val.image_sha256) or set(b for a,b in ids)&set(val.RGB_pixel_sha256):
        raise RuntimeError('Validation/test duplicate image content')
    valrefs=pd.read_csv(OUT/'inputs/validation_reference_hashes.csv',dtype=str)
    testrefs=d[['ref_id','ref_path']].drop_duplicates()
    with ThreadPoolExecutor(max_workers=4) as pool: refids=list(pool.map(image_identity,testrefs.ref_path))
    if set(a for a,b in refids)&set(valrefs.image_sha256) or set(b for a,b in refids)&set(valrefs.RGB_pixel_sha256):
        raise RuntimeError('Validation/test duplicate reference content')
    d.to_csv(OUT/stage/'manifest.csv',index=False)
    write_json(OUT/stage/'split_exclusion_audit.json',{'validation_image_overlap':0,'validation_reference_overlap':0,
        'methods':['dataset-qualified identity','SHA-256 image file','SHA-256 decoded full RGB pixels'],
        'evaluated_images':len(d),'evaluated_references':len(testrefs),'checked_before_scoring':True})
    scored=fresh_scores(d,stage,lock)
    stats=read_json(OUT/'calibration/z_statistics.json')
    with np.load(OUT/'calibration/empirical_CDF_arrays.npz') as f: ecdf={k:f[k] for k in ('EA','EH')}
    scored=apply_calibration(scored,stats,ecdf)
    scored.to_csv(OUT/stage/'fusion_scores.csv',index=False)
    summary=score_summary(scored,'target_quality');summary['primary_comparison']=summary.candidate.isin([frozen['selected_Q_star'],'L00'])
    summary.to_csv(OUT/stage/'correlations_and_score_summary.csv',index=False)
    types=severity_table(scored);types.to_csv(OUT/stage/'severity_by_distortion.csv',index=False)
    prow=[]
    for (ds,typ),g in scored.groupby(['dataset','distortion_type']):
        for q in CANDIDATES:
            sr,pr,n=correlations(g[q],g.target_quality)
            prow.append({'dataset':ds,'distortion_type':typ,'candidate':q,'N':n,'SRCC':sr,'Pearson_raw':pr})
    pd.DataFrame(prow).to_csv(OUT/stage/'correlations_by_distortion.csv',index=False)
    pa,pdets=pair_outputs(scored);pa.to_csv(OUT/stage/'pair_accuracy.csv',index=False);pdets.to_csv(OUT/stage/'pair_details.csv',index=False)
    groups=scored.groupby(['dataset','ref_id','distortion_type'])
    valid_groups=sum(int(g.severity.notna().all() and g.severity.nunique()>1) for _,g in groups)
    covered_images=len(set(pdets.mild_index)|set(pdets.severe_index))
    write_json(OUT/stage/'severity_coverage.json',{'total_images':len(scored),'finite_severity_images':int(scored.severity.notna().sum()),
        'images_in_eligible_pairs':covered_images,'image_pair_coverage':covered_images/len(scored),
        'total_groups':len(groups),'complete_severity_groups':valid_groups,
        'eligible_pairs':int(len(pdets)/len(CANDIDATES)),'DMOS_used_to_infer_severity':False})
    print(stage+': computing 5000 paired reference-cluster bootstrap draws.',flush=True)
    boot,values,draws=evaluation_bootstrap(scored)
    boot.to_csv(OUT/stage/'bootstrap_confidence_intervals.csv',index=False)
    np.savez_compressed(OUT/stage/'bootstrap_metric_draws.npz',**values)
    np.savez_compressed(OUT/stage/'bootstrap_cluster_counts.npz',**{ds:v[1] for ds,v in draws.items()})
    write_json(OUT/stage/'bootstrap_provenance.json',{'seed':20260908,'n':5000,'ref_order':{ds:v[0] for ds,v in draws.items()},'paired':True})
    verify_frozen()
    (OUT/stage/'report.txt').write_text(f'{which} frozen-fusion evaluation\nPrimary selected candidate: {frozen["selected_Q_star"]}\n'+
        summary.to_string(index=False)+'\n\nPAIRED BOOTSTRAP DIFFERENCES FROM L00\n'+boot[boot.comparison.eq('difference_vs_L00')].to_string(index=False)+'\n')
    write_json(OUT/stage/'evaluation_complete.json',{'status':'COMPLETE','completed_utc':now(),'N':len(scored),
        'selected_Q_star_unchanged':frozen['selected_Q_star'],'frozen_configuration_sha256':sha(OUT/'frozen_configuration.json'),
        'normalization_refitted':False,'models_changed':False,'target':'MOS' if which=='kadid' else '-DMOS'})
    print(stage+' COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','validation','kadid','csiq']);p.add_argument('--manifest',type=Path)
    args=p.parse_args()
    try:
        if args.stage=='prepare': prepare()
        elif args.stage=='validation': validate()
        else: evaluate(args.stage,args.manifest)
    except Exception as exc:
        write_json(OUT/'logs'/f'failure_{args.stage}_{int(time.time())}.json',{'time':now(),'stage':args.stage,'error':str(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__': main()
