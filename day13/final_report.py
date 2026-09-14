"""Write the completed Day13 research report only after independent verification."""
from common import *
from evaluation import verify_freeze

def main():
 f=verify_freeze();v=read_json(ROOT/'verification/validation_verified.json');h=read_json(ROOT/'verification/holdout_verified.json')
 assert v['status']==h['status']=='PASS'
 sel=f['selected'];control=read_csv(OUT/'validation/control_comparison.csv');hold=read_csv(OUT/'kadid_development/opinion_correlations.csv');cond=read_csv(OUT/'kadid_development/conditional_complementarity.csv');boot=read_csv(OUT/'kadid_development/bootstrap_confidence_intervals.csv');vb=read_csv(OUT/'validation/bootstrap_confidence_intervals.csv')
 def markdown(frame):
  headers=list(frame.columns);rows=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
  for row in frame.itertuples(index=False,name=None):rows.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in row)+' |')
  return '\n'.join(rows)
 lines=['# Day 13: Blind-spot EA training with D0','','**Status: COMPLETE; independently verified.**','',
 f"Selected specialist: **{sel['arm']}, epoch {sel['epoch']}**, alpha_H={sel['alpha_H']}; T_alpha={sel['T_alpha']}. Qz=zA-zH.",'',
 '**Validation decision: '+('NO-GO' if f['validation_no_go'] else 'GO')+'.** '+('The selected specialist did not strictly exceed both ranked EH alone and ranked-all dual fusion on validation S_blind.' if f['validation_no_go'] else 'The selected specialist strictly exceeded both ranked EH alone and ranked-all dual fusion on validation S_blind.'),'',
 'The validation decision is fixed. KADID is a development holdout and cannot reverse it. CSIQ was not opened or evaluated in this experiment.','',
 '## Validation controls','',markdown(control),'','## Validation uncertainty','',markdown(vb),'',
 'These intervals use 5,000 whole-reference resamples independently within KADID and TID, seed 20260910. The selected model and validation normalization stay fixed in all draws. An interval containing zero means the corresponding advantage remains uncertain.','',
 '## KADID development holdout','',markdown(hold[['score','N_images','SRCC','Pearson_raw','S_severity','pair_accuracy','macro_pair_accuracy']]),'',
 'C0 is ranked EH alone; C1 uses Day7 EA plus ranked EH; C2 uses ranked-all EA plus ranked EH; C3 uses selected blind-spot EA plus ranked EH. Individual EA rows are diagnostics. All Pearson correlations are raw, without a logistic mapping.','',
 '## Conditional complementarity','',markdown(cond),'',
 'Strict EH failures include ties and reversals (gH<=0). Margin failures additionally include gaps no larger than the selected alpha_H times the training-metadata severity distance. R_A and R_Q require strictly positive EA and fused gaps, respectively. Undefined conditional rates remain unavailable.','',
 '## Holdout confidence intervals','',markdown(boot),'',
 'Paired confidence intervals use 5,000 complete KADID reference-cluster draws, seed 20260910. Valid-draw fractions are reported when conditional failure sets are empty.','',
 '## Protocol and verification','',
 '- Exact Day7 EA and original pristine EH initialization checkpoints and ranked control checkpoints were hash-verified before training.',
 '- Five folds preserve complete dataset/reference groups. Every training image has one EH score from a teacher whose ranking training excluded that reference.',
 '- Each EH teacher trained for exactly 20 epochs. Each feasible specialist trained independently from Day7 EA for 25 epochs, with candidate checkpoints at 5,10,15,20,25. Infeasible arms were skipped by the predeclared rules.',
 '- Original architecture, preprocessing, reconstruction losses, Adam settings, ranking margin and coefficients were retained. Fixed weights affect only the EA ranking term. Both reference distributions remain unchanged.',
 '- MOS/DMOS were unused for folds, weights, training, selection and calibration. KADID MOS was loaded only after the final selection/calibration freeze.',
 '- Day13 uses its specified half-credit exact ties for validation/holdout pair accuracy; conditional rescue remains strict.',
 '- Independent verification checks all teachers and specialist checkpoints, reference tensors, signed shortages, weights, validation normalization/selection, and explicit replication of bootstrap draws.',
 '- These reference-fold exclusions audit ranking fine-tuning. They do not claim to erase or reconstruct the initial checkpoints\' earlier training history.','',
 '## Saved evidence','',
 'See `day12_blindspot_d0/input_manifest.json`, `folds/`, `crossfit_EH/`, `weights/`, `training/`, `validation/`, and `kadid_development/`. The pre-holdout selection is in `validation/frozen_configuration.json` and its SHA-256 sidecar. `verification/` contains the independent checks.','']
 text='\n'.join(lines);(ROOT/'Day13_Report.md').write_text(text);(OUT/'stage_report.txt').write_text(text)
 summary={'status':'COMPLETE','completed_utc':now(),'selected':sel,'validation_no_go':f['validation_no_go'],'validation_controls':control.to_dict('records'),'holdout_metrics':hold.to_dict('records'),'holdout_uncertainty':boot.to_dict('records'),'CSIQ_opened':False,'independent_verification':'PASS','report':str(ROOT/'Day13_Report.md')}
 save_json(OUT/'stage_summary.json',summary)
 print('DAY 13 COMPLETE. Report:',ROOT/'Day13_Report.md',flush=True)
if __name__=='__main__':main()
