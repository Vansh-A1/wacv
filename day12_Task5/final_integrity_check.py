"""Read-only end-of-task integrity and full PDF requirement audit."""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_task5 import verify_frozen,sha

root=Path(__file__).resolve().parent
frozen=verify_frozen(); frozen_time=datetime.fromisoformat(frozen['frozen_utc'])
checks={'frozen_configuration_and_all_source_hashes_unchanged':True,'selected_Q_star':frozen['selected_Q_star'],'counts':{}}
assert sha(root/'inputs/day12_Task5.pdf')==sha('/home/projectwork/Downloads/day12_Task5.pdf')
for stage,n,nref in [('validation',1860,15),('kadid_holdout',1625,13),('csiq',866,30)]:
    d=pd.read_csv(root/stage/'fusion_scores.csv',float_precision='round_trip',dtype={'ref_id':str})
    assert len(d)==n and len(d[['dataset','ref_id']].drop_duplicates())==nref
    assert not d.duplicated(['dataset','image_id']).any()
    assert np.isfinite(d[['EA','EH','L00','L25','L50','L75','L100','PAVG','PHARM']].to_numpy()).all()
    assert json.loads((root/'verification'/(stage+'_independent_verification.json')).read_text())['status']=='PASS'
    provenance=json.loads((root/stage/'scoring_provenance.json').read_text())
    for pole in ['EA','EH']:
        p=provenance['poles'][pole]
        assert p['model_state_sha256_before']==p['model_state_sha256_after']
        assert p['checkpoint_sha256']==frozen['checkpoints'][pole]['checkpoint_sha256']
    checks['counts'][stage]={'images':n,'references':nref,'fresh_scoring_verified':True}
    if stage=='validation':continue
    opened=json.loads((root/stage/'opened.json').read_text())
    complete=json.loads((root/stage/'evaluation_complete.json').read_text())
    assert frozen_time<datetime.fromisoformat(opened['opened_utc'])<datetime.fromisoformat(complete['completed_utc'])
    assert complete['selected_Q_star_unchanged']==frozen['selected_Q_star']
    assert not complete['normalization_refitted'] and not complete['models_changed']
    boot=pd.read_csv(root/stage/'bootstrap_confidence_intervals.csv',float_precision='round_trip')
    assert len(boot)==42
    assert (boot.valid_draws==5000).all() and (boot.invalid_draws==0).all()
    info=json.loads((root/stage/'bootstrap_provenance.json').read_text())
    pairs=pd.read_csv(root/stage/'pair_details.csv')
    pa=pd.read_csv(root/stage/'pair_accuracy.csv',float_precision='round_trip')
    with np.load(root/stage/'bootstrap_metric_draws.npz') as draws,np.load(root/stage/'bootstrap_cluster_counts.npz') as counts:
        for r in boot.itertuples():
            key=f'{r.dataset}__{r.candidate}__{r.metric}__{r.comparison}'
            v=draws[key]
            assert len(v)==5000 and np.isfinite(v).all()
            np.testing.assert_allclose([r.ci_low,r.ci_high],np.quantile(v,[.025,.975]),atol=1e-14)
            if r.comparison=='difference_vs_L00':
                np.testing.assert_array_equal(v,draws[f'{r.dataset}__{r.candidate}__{r.metric}__absolute']-draws[f'{r.dataset}__L00__{r.metric}__absolute'])
        # Independently materialize repeated clusters of pair outcomes for first ten draws.
        for ds in d.dataset.unique():
            for q in ['L00',frozen['selected_Q_star']]:
                g=pairs[pairs.dataset.eq(ds)&pairs.candidate.eq(q)]
                row=pa[pa.dataset.eq(ds)&pa.candidate.eq(q)&pa.distortion_type.eq('all')].iloc[0]
                assert row.n_pairs==len(g) and row.correct==g.correct.sum() and row.ties==g.tie.sum()
                for b in range(10):
                    explicit=np.concatenate([g[g.ref_id.astype(str).eq(ref)].correct.to_numpy() for ref,c in zip(info['ref_order'][ds],counts[ds][b]) for _ in range(c)])
                    np.testing.assert_allclose(explicit.mean(),draws[f'{ds}__{q}__pair_accuracy__absolute'][b],atol=1e-14)
    checks[stage+'_all_42_CI_rows_and_paired_differences_verified']=True
    checks[stage+'_10_pair_bootstrap_draws_explicit_replication']=True
access=json.loads((root/'csiq/data_access_started.json').read_text())
assert frozen_time<datetime.fromisoformat(access['started_utc'])
audit=json.loads((root/'csiq/manifest_preparation_audit.json').read_text())
assert audit['original_reference_file_matches']==audit['original_reference_RGB_matches']==30
for entry in audit['original_source_downloads']:
    if entry['complete']:assert sha(entry['file'])==entry['sha256']
coverage=json.loads((root/'csiq/severity_coverage.json').read_text())
assert coverage['total_images']==coverage['finite_severity_images']==coverage['images_in_eligible_pairs']==866
checks.update(original_PDF_unchanged=True,official_CSIQ_metadata_and_reference_archives_unchanged=True,
    all_CSIQ_access_after_freeze=True,CSIQ_severity_coverage=1.0,status='PASS',checked_utc=datetime.now(timezone.utc).isoformat())
(root/'verification/final_integrity.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
