"""Assemble the finished research report from verified frozen experiment outputs."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
read=lambda p:pd.read_csv(root/p,float_precision='round_trip')
frozen=json.loads((root/'frozen_configuration.json').read_text());selected=frozen['selected_Q_star']
for stage in ('validation','kadid_holdout','csiq'):
 assert json.loads((root/'verification'/(stage+'_independent_verification.json')).read_text())['status']=='PASS'
for stage in ('kadid_holdout','csiq'):
 assert json.loads((root/stage/'evaluation_complete.json').read_text())['status']=='COMPLETE'
val=read('selection/validation_selection.csv');vboot=json.loads((root/'selection/validation_bootstrap.json').read_text())
Q=['L00','L25','L50','L75','L100','PAVG','PHARM'];summaries={};boots={};pairs={}
for stage in ('kadid_holdout','csiq'):
 summaries[stage]=read(stage+'/correlations_and_score_summary.csv').set_index('candidate')
 boots[stage]=read(stage+'/bootstrap_confidence_intervals.csv')
 pairs[stage]=read(stage+'/pair_accuracy.csv').query('distortion_type == "all"').set_index('candidate')

def md_table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def ci(stage,q,metric,kind='difference_vs_L00'):
 row=boots[stage].query('candidate == @q and metric == @metric and comparison == @kind').iloc[0]
 return float(row.ci_low),float(row.ci_high)

def fmt_ci(v):return f'[{v[0]:.6f}, {v[1]:.6f}]'

def interval_words(interval):
 if interval[0]>0:return 'The interval is entirely above zero.'
 if interval[1]<0:return 'The interval is entirely below zero.'
 return 'The interval includes zero, so an advantage is uncertain.'

primary=[]
for stage,label in [('kadid_holdout','KADID development holdout'),('csiq','CSIQ confirmatory')]:
 a=summaries[stage].loc[selected];b=summaries[stage].loc['L00'];p=pairs[stage]
 primary.append({'dataset':label,'N':int(a.N),'selected':selected,'selected_SRCC':float(a.SRCC),'L00_SRCC':float(b.SRCC),
 'delta_SRCC':float(a.SRCC-b.SRCC),'delta_SRCC_CI_low':ci(stage,selected,'SRCC')[0],'delta_SRCC_CI_high':ci(stage,selected,'SRCC')[1],
 'selected_Pearson_raw':float(a.Pearson_raw),'L00_Pearson_raw':float(b.Pearson_raw),
 'delta_Pearson_raw':float(a.Pearson_raw-b.Pearson_raw),
 'delta_Pearson_CI_low':ci(stage,selected,'Pearson_raw')[0],'delta_Pearson_CI_high':ci(stage,selected,'Pearson_raw')[1],
 'selected_pair_accuracy':float(p.loc[selected].pair_accuracy),'L00_pair_accuracy':float(p.loc['L00'].pair_accuracy),
 'delta_pair_accuracy':float(p.loc[selected].pair_accuracy-p.loc['L00'].pair_accuracy),
 'delta_pair_accuracy_CI_low':ci(stage,selected,'pair_accuracy')[0],'delta_pair_accuracy_CI_high':ci(stage,selected,'pair_accuracy')[1]})
pd.DataFrame(primary).to_csv(root/'primary_comparison.csv',index=False)
lines=['# Day 12 Task 5: Frozen-score fusion analysis','',
 '**Status: COMPLETE.** Validation, KADID development holdout, and CSIQ confirmatory evaluation are complete and independently verified.','',
 f'Validation selected **{selected}**, defined as **0.25 z_A - 0.75 z_H**. The selected score was fixed before either test was opened. No encoder was trained or fine-tuned.','',
 f'The validation advantage over EH alone (L00) was {vboot["delta_S_fusion"]:.6f}, with a reference-cluster bootstrap 95% interval {fmt_ci((vboot["ci_low"],vboot["ci_high"]))}. {interval_words((vboot["ci_low"],vboot["ci_high"]))}', '',
 '## Primary comparison','',
 md_table(['Evaluation','N','L00 SRCC',selected+' SRCC','Difference','95% CI for difference'],
 [[r['dataset'],r['N'],f'{r["L00_SRCC"]:.6f}',f'{r["selected_SRCC"]:.6f}',f'{r["delta_SRCC"]:+.6f}',fmt_ci((r['delta_SRCC_CI_low'],r['delta_SRCC_CI_high']))] for r in primary]),'']
for r in primary:lines.append(r['dataset']+': '+interval_words((r['delta_SRCC_CI_low'],r['delta_SRCC_CI_high'])))
lines += ['', 'KADID is a previously inspected development test. The primary external comparison is the preselected L25 versus L00 on CSIQ. Other candidates remain secondary ablations and do not replace L25.','',
 '## Models and scoring','',
 '- EA: rank-all, lambda 0.1, epoch 5.','- EH: corrected ranked pristine model, lambda 0.1, epoch 20. Both energies were freshly recomputed.',
 '- Legacy D0: `sqrt(sum((mu_ref-mu_x)^2/(Sigma_ref+1e-8)))`, using the original float32 implementation and embedded references.',
 '- RGB input, existing short-side upsize only if needed, deterministic 256 center crop, [0,1]. Encoder-only output was checked against the full model forward. Model-state and checkpoint hashes were unchanged.',
 '- Pool 1,500 KADID and 360 TID validation images for z-score and empirical-CDF calibration. Population standard deviation, epsilon=1e-8. No test-set calibration.',
 '- Retain the PDF percentile `(n_below+0.5*n_tied+0.5)/(N+1)`, with exact ties. All seven fusion scores are higher=better.',
 '- No MOS/DMOS was used for calibration, fusion selection, hyperparameter decisions, or model updates. KADID test target is its existing MOS column. CSIQ target is negative DMOS. Pearson is raw, with no logistic fit.','',
 '## Validation selection','',
 md_table(['Candidate','S_KADID','S_TID','S_fusion','Selected'],[[r.candidate,f'{r.S_KADID:.6f}',f'{r.S_TID:.6f}',f'{r.S_fusion:.6f}',str(bool(r.selected))] for r in val.itertuples()]),'',
 'The selection score averages severity SRCC over all 25 KADID distortion types and all 24 TID types separately, negates each average, then gives each dataset equal weight. TID has only three validation reference clusters. Exact maximum ties use the declared order L00, L25, L50, L75, L100, PAVG, PHARM; the winning score had no tie.','',
 'The bootstrap uses 5,000 draws, seed 20260908, sampling entire reference clusters independently within each validation dataset. The selected identity and calibration are fixed inside every draw. All validation bootstrap draws were finite. The interval does not change selection.']
for stage,title in [('kadid_holdout','KADID development holdout'),('csiq','CSIQ confirmatory test')]:
 s=summaries[stage];p=pairs[stage]
 lines += ['', '## '+title,'',md_table(['Candidate','SRCC','Raw Pearson','Pair accuracy','Pair ties','Score SD'],
 [[q,f'{s.loc[q].SRCC:.6f}',f'{s.loc[q].Pearson_raw:.6f}',f'{p.loc[q].pair_accuracy:.6f}',int(p.loc[q].ties),f'{s.loc[q].std_population:.6f}'] for q in Q]),'',
 md_table(['Metric',selected+' minus L00','95% CI'],
 [[metric,f'{(s.loc[selected,metric]-s.loc["L00",metric]) if metric!="pair_accuracy" else (p.loc[selected,metric]-p.loc["L00",metric]):+.6f}',fmt_ci(ci(stage,selected,metric))] for metric in ('SRCC','Pearson_raw','pair_accuracy')]),'',
 f'Complete per-image scores, per-distortion severity and opinion correlations, pair details, ranges/variances/ties, absolute confidence intervals, paired differences, and all bootstrap draws are in `{stage}/`.']
coverage=json.loads((root/'csiq/severity_coverage.json').read_text())
lines += ['', '## CSIQ source and coverage','',
 'The original [author-hosted DMOS workbook](https://s2.smu.edu/~eclarson/csiq/csiq.DMOS.xlsx) supplies all 866 final DMOS records and the published `dst_lev` metadata. Final DMOS values were cross-checked against the workbook\'s second summary sheet. Unrelated intermediate cells with existing #REF! errors were preserved and not used.','',
 'Images came from the [IQA-PyTorch CSIQ mirror](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/blob/main/csiq.tgz), verified against its published SHA-256. Metadata was not substituted from the mirror. The original source-image archive was also downloaded; reference-image content was checked against it where available. The slow original distorted-image download was retained as an incomplete acquisition attempt, not used as data.','',
 f'CSIQ includes {coverage["total_images"]} scored images. Severity is available for {coverage["finite_severity_images"]}, and {coverage["images_in_eligible_pairs"]} participate in valid within-reference/type pairs. Pair coverage is {coverage["image_pair_coverage"]:.1%}, with {coverage["eligible_pairs"]} eligible pairs. Severity was never inferred from DMOS.','',
 'Validation and test image/reference content were compared using file hashes and decoded full-RGB hashes before test scoring. No validation/test overlaps were found. These checks do not establish the undocumented contents of the original supplied pretraining pool. CSIQ is confirmatory for this frozen Task 5 decision, rather than proof of an entirely audited model ancestry.','',
 '## Verification and interpretation','',
 '- Ten unit tests passed, including tied percentiles, polarity, equal dataset weighting, complete-reference resampling, and weighted correlations versus explicit replication.',
 '- Independent verification recomputed D0 from saved posteriors; normalization and all fusion formulas; exact pair ties; selection scores; 20 validation bootstrap draws and 10 test bootstrap draws using explicitly repeated reference clusters.',
 '- Exact CSV round-trip parsing is required for reanalysis so serialization does not break floating-point ties. The initial verifier used the default CSV parser and showed a tiny PAVG discrepancy; correcting only the verifier to round-trip parsing resolved it. Experiment scores, selection, and frozen files were unchanged.',
 '- All evaluation bootstrap intervals use 5,000 paired whole-reference draws. The confidence-interval tables report valid and invalid draw counts.',
 '- No post-test score sign change, normalization fit, model update, or replacement of Q* occurred.',
 '- A small point improvement is not evidence of a reliable improvement when its paired confidence interval includes zero. The external comparison is the main confirmatory result.','',
 '## Files','',
 '- `primary_comparison.csv`: main L25-versus-L00 results and intervals.',
 '- `frozen_configuration.json` and `.sha256`: immutable pre-test selection/calibration record.',
 '- `calibration/`: pooled z-score statistics and empirical distributions.',
 '- `validation/`, `selection/`, `kadid_holdout/`, `csiq/`: complete tables and reports.',
 '- `verification/`: unit and independent checks, plus final integrity and confidence-interval audit.',
 '- `REQUIREMENTS_CHECKLIST.md`: each PDF requirement mapped to its saved evidence.',
 '- `inputs/`, `data/csiq/`, `logs/`: original task, manifests, source data, and acquisition logs.',
 '- `ASSUMPTIONS.md`: protocol choices written before scoring.','']
(root/'Task5_Report.md').write_text('\n'.join(lines))

# Standard scientific figures, fixed across every candidate; no data-dependent selection.
fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
colors=['#185f73' if q==selected else '#434950' if q=='L00' else '#a7b8c2' for q in Q]
v=val.set_index('candidate').loc[Q]
axes[0].barh(Q,v.S_fusion,color=colors);axes[0].invert_yaxis();axes[0].set_xlabel('Severity selection score (higher is better)');axes[0].set_title('Validation selection')
for ax,stage,title in zip(axes[1:],['kadid_holdout','csiq'],['KADID development holdout','CSIQ confirmatory test']):
 s=summaries[stage];point=np.array([s.loc[q].SRCC-s.loc['L00'].SRCC for q in Q])
 bounds=np.array([ci(stage,q,'SRCC') for q in Q])
 for i,q in enumerate(Q):
  ax.plot(bounds[i], [i,i],color=colors[i],lw=2);ax.scatter(point[i],i,color=colors[i],s=30,zorder=3)
 ax.axvline(0,color='#777',ls='--',lw=1);ax.set_yticks(range(7),Q);ax.invert_yaxis();ax.set_xlabel('SRCC difference from L00 (95% CI)');ax.set_title(title)
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Frozen-score fusion: L25 selected before test evaluation',fontsize=14)
fig.savefig(root/'fusion_results.png',dpi=180);fig.savefig(root/'fusion_results.svg');plt.close(fig)

status={'status':'COMPLETE','completed_utc':datetime.now(timezone.utc).isoformat(),'selected_Q_star':selected,
 'training_performed':False,'validation_N':1860,'KADID_holdout_N':1625,'CSIQ_N':866,
 'frozen_configuration_sha256':hashlib.file_digest((root/'frozen_configuration.json').open('rb'),'sha256').hexdigest(),
 'unit_tests_passed':10,'independent_verification':['validation','kadid_holdout','csiq'],
 'primary_comparison':primary}
(root/'task5_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
