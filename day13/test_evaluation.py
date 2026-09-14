"""Synthetic unit checks only; never writes real experimental outputs."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import evaluation as e
from common import all_pairs,clusters

class EvaluationTests(unittest.TestCase):
 def fixture(self):
  rows=[]
  for ref in ['r1','r2','r3']:
   for sev in [1,2,3]:
    j=int(ref[1]);h=sev*.2+j*.3;a=-sev*.3+j*.4
    if ref=='r2':h=-sev*.1+j*.3
    rows.append({'dataset':'KADID-10k','ref_id':ref,'distortion_type':'blur','image_id':ref+str(sev),'severity':sev,'z_H':h,'z_A_C3':a,'C0':-h,'C2':-sev*.25+j*.3,'C3':a-h,'MOS':4-sev+j*.05})
  return pd.DataFrame(rows)
 def test_conditional_polarities_and_bootstrap(self):
  d=self.fixture();previous=e.OUT
  with tempfile.TemporaryDirectory(prefix='day13_synthetic_') as td:
   e.OUT=Path(td);(e.OUT/'folds').mkdir();pd.DataFrame([{'dataset':'KADID-10k','distortion_type':'blur','min':1,'max':3}]).to_csv(e.OUT/'folds/training_severity_ranges.csv',index=False)
   p,c=e.conditional_table(all_pairs(d),d,.1)
   np.testing.assert_allclose(p.g_Q,p.g_A+p.g_H,atol=1e-14)
   self.assertEqual(int(p.failure_strict.sum()),3);self.assertTrue((c.R_A==1).all());self.assertTrue((c.R_Q==1).all())
   vals,refs,counts,absolute=e.holdout_bootstrap(d,p,.1)
   for b in range(12):
    sub=pd.concat([d[d.ref_id.eq(r)] for r,n in zip(refs,counts[b]) for _ in range(n)])
    expected=spearmanr(sub.C3,sub.MOS).statistic-spearmanr(sub.C0,sub.MOS).statistic
    np.testing.assert_allclose(vals['C3_minus_C0_SRCC'][b],expected,atol=1e-14)
   self.assertGreater(np.isnan(vals['R_A_strict']).sum(),0)
  e.OUT=previous
 def test_validation_dataset_weight_and_cluster_multiplicity(self):
  d=self.fixture();other=d.copy();other.dataset='TID2013';d=pd.concat([d,other],ignore_index=True)
  vals,draws=e.validation_bootstrap(d)
  for b in range(8):
   expected=[]
   for ds,g in d.groupby('dataset'):
    refs,counts=draws[ds];sub=pd.concat([g[g.ref_id.eq(r)] for r,n in zip(refs,counts[b]) for _ in range(n)])
    expected.append(-spearmanr(sub.C3,sub.severity).statistic+spearmanr(sub.C0,sub.severity).statistic)
   np.testing.assert_allclose(vals['C0'][b],np.mean(expected),atol=1e-14)
if __name__=='__main__':unittest.main(verbosity=2)
