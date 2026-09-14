#!/usr/bin/env python3
"""Independent result verification; never fits or modifies the selected experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr,pearsonr

ROOT=Path(__file__).resolve().parent
_read_csv=pd.read_csv
def read_csv(*args,**kwargs):
    return _read_csv(*args,float_precision='round_trip',**kwargs)

Q=['L00','L25','L50','L75','L100','PAVG','PHARM']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(stage):
    report={'stage':stage,'checks':{},'method':'Independent NumPy/SciPy recomputation from saved arrays and tables'}
    frozen=json.loads((ROOT/'frozen_configuration.json').read_text())
    assert sha(ROOT/'frozen_configuration.json')==(ROOT/'frozen_configuration.sha256').read_text().split()[0]
    for p,h in frozen['frozen_artifact_hashes'].items():assert sha(p)==h,p
    assert frozen['kadid_holdout_opened'] is False and frozen['CSIQ_opened'] is False
    report['checks']['frozen_configuration_and_hashes']=True
    raw=read_csv(ROOT/stage/'raw_pole_scores.csv',dtype={'image_id':str,'ref_id':str})
    scored=read_csv(ROOT/stage/'fusion_scores.csv',dtype={'image_id':str,'ref_id':str})
    assert not raw.duplicated(['dataset','image_id']).any()
    assert raw[['dataset','image_id']].equals(scored[['dataset','image_id']])
    stats=json.loads((ROOT/'calibration/z_statistics.json').read_text())
    cal=read_csv(ROOT/'validation/raw_pole_scores.csv',dtype={'image_id':str})
    if stage=='validation':assert not any('mos' in c.lower() or 'dmos' in c.lower() for c in raw.columns)
    for p in ['EA','EH']:
        with np.load(ROOT/stage/(p+'_posteriors.npz'),allow_pickle=False) as a:
            assert a['image_id'].tolist()==raw.image_id.tolist()
            assert a['dataset'].tolist()==raw.dataset.tolist()
            reconstructed=np.sqrt(np.sum((a['mu_ref'][None,:]-a['mu'])**2/(a['Sigma_ref'][None,:]+np.float32(1e-8)),axis=1))
        err=float(np.max(np.abs(reconstructed-raw[p])))
        np.testing.assert_allclose(reconstructed,raw[p],rtol=2e-7,atol=1e-3)
        report['checks'][p+'_D0_max_abs_float32_reduction_difference']=err
        vals=cal[p].to_numpy(float)
        np.testing.assert_allclose([stats[p]['mean'],stats[p]['std_population']],[vals.mean(),vals.std(ddof=0)],rtol=2e-13)
        z=(raw[p]-vals.mean())/(vals.std(ddof=0)+1e-8)
        np.testing.assert_allclose(scored['z_'+p],z,rtol=1e-10,atol=1e-10)
        x=raw[p].to_numpy()
        # Count directly, independently of the production searchsorted implementation.
        f=np.array([(np.sum(vals<e)+.5*np.sum(vals==e)+.5)/(len(vals)+1) for e in x])
        np.testing.assert_allclose(scored['u_A' if p=='EA' else 'u_H'],f if p=='EA' else 1-f,atol=1e-14)
    for q,w in zip(Q[:5],[0,.25,.5,.75,1]):
        np.testing.assert_allclose(scored[q],w*scored.z_EA-(1-w)*scored.z_EH,atol=1e-13)
    np.testing.assert_allclose(scored.PAVG,(scored.u_A+scored.u_H)/2,atol=1e-14)
    np.testing.assert_allclose(scored.PHARM,2*scored.u_A*scored.u_H/(scored.u_A+scored.u_H+1e-8),atol=1e-14)
    report['checks']['all_calibration_and_fusion_formulas']=True
    pairs=read_csv(ROOT/stage/'pair_details.csv')
    assert (pairs.severity_mild<pairs.severity_severe).all()
    for q,g in pairs.groupby('candidate'):
        a=g.mild_index.to_numpy(int);b=g.severe_index.to_numpy(int)
        assert (scored.loc[a,'ref_id'].to_numpy()==scored.loc[b,'ref_id'].to_numpy()).all()
        assert (scored.loc[a,'dataset'].to_numpy()==scored.loc[b,'dataset'].to_numpy()).all()
        assert (scored.loc[a,'distortion_type'].to_numpy()==scored.loc[b,'distortion_type'].to_numpy()).all()
        gaps=scored.loc[a,q].to_numpy()-scored.loc[b,q].to_numpy()
        np.testing.assert_array_equal(g['correct'].to_numpy(),gaps>0)
        np.testing.assert_array_equal(g['tie'].to_numpy(),gaps==0)
    report['checks']['pair_order_and_exact_ties']=True
    if stage=='validation':
        table=read_csv(ROOT/'selection/validation_selection.csv');sel=frozen['selected_Q_star']
        for _,row in table.iterrows():
            dataset_scores=[]
            for ds,g in scored.groupby('dataset'):
                sr=[spearmanr(sub[row.candidate],sub.severity).statistic for _,sub in g.groupby('distortion_type')]
                dataset_scores.append(-np.mean(sr))
            np.testing.assert_allclose(row.S_fusion,np.mean(dataset_scores),atol=1e-14)
        assert table.iloc[0].candidate==sel
        vinfo=json.loads((ROOT/'selection/validation_bootstrap.json').read_text())
        saved=read_csv(ROOT/'selection/validation_bootstrap_draws.csv').delta_S_fusion_vs_L00.to_numpy()
        with np.load(ROOT/'selection/validation_cluster_draws.npz') as draws:
            for b in range(20):
                values=[]
                for ds,g in scored.groupby('dataset',sort=True):
                    ref_order=vinfo['reference_order'][ds];counts=draws[ds][b]
                    expanded=pd.concat([g[g.ref_id.eq(r)] for r,c in zip(ref_order,counts) for _ in range(c)],ignore_index=True)
                    diffs=[]
                    for _,sub in expanded.groupby('distortion_type'):
                        diffs.append(-spearmanr(sub[sel],sub.severity).statistic+spearmanr(sub.L00,sub.severity).statistic)
                    values.append(np.mean(diffs))
                np.testing.assert_allclose(saved[b],np.mean(values),atol=2e-14)
        np.testing.assert_allclose([vinfo['ci_low'],vinfo['ci_high']],np.quantile(saved[np.isfinite(saved)],[.025,.975]),atol=1e-14)
        report['checks']['20_bootstrap_draws_explicit_cluster_replication']=True
    else:
        summary=read_csv(ROOT/stage/'correlations_and_score_summary.csv')
        for _,r in summary.iterrows():
            g=scored[scored.dataset.eq(r.dataset)]
            np.testing.assert_allclose([r.SRCC,r.Pearson_raw],[spearmanr(g[r.candidate],g.target_quality).statistic,pearsonr(g[r.candidate],g.target_quality).statistic],atol=1e-13)
        info=json.loads((ROOT/stage/'bootstrap_provenance.json').read_text())
        with np.load(ROOT/stage/'bootstrap_cluster_counts.npz') as counts,np.load(ROOT/stage/'bootstrap_metric_draws.npz') as draws:
            for ds,g in scored.groupby('dataset'):
                for b in range(10):
                    ix=np.concatenate([g[g.ref_id.eq(r)].index.to_numpy() for r,c in zip(info['ref_order'][ds],counts[ds][b]) for _ in range(c)])
                    sub=scored.loc[ix]
                    for q in [frozen['selected_Q_star'],'L00']:
                        for metric,value in [('SRCC',spearmanr(sub[q],sub.target_quality).statistic),('Pearson_raw',pearsonr(sub[q],sub.target_quality).statistic)]:
                            np.testing.assert_allclose(draws[f'{ds}__{q}__{metric}__absolute'][b],value,atol=1e-13)
        report['checks']['10_test_bootstrap_draws_explicit_cluster_replication']=True
        report['checks']['raw_target_correlations']=True
    report['status']='PASS'
    out=ROOT/'verification'/(stage+'_independent_verification.json');out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['validation','kadid_holdout','csiq']);main(p.parse_args().stage)
