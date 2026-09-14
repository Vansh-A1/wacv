import unittest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from evaluation import winner,selection_value,weighted_srcc,weighted_corr,pair_table,pair_gap
from common import ROOT
class PipelineTests(unittest.TestCase):
 def test_EH_positive_rule_and_EA_penalties(self):
  r=dict(blur=.2,lens=.3,sharpen=-.4,pixelate=.1)
  self.assertAlmostEqual(selection_value('EA',r),-.6);self.assertAlmostEqual(selection_value('EH',r),.2)
 def test_selection_ties_accuracy_epoch_lambda(self):
  t=pd.DataFrame([dict(validation_score=1.,pair_order_accuracy=.8,epoch=5,lambda_rank=.1),dict(validation_score=1.,pair_order_accuracy=.9,epoch=10,lambda_rank=.1),dict(validation_score=1.,pair_order_accuracy=.9,epoch=5,lambda_rank=1.),dict(validation_score=1.,pair_order_accuracy=.9,epoch=5,lambda_rank=.1)])
  self.assertEqual(int(winner(t).name),3)
 def test_cluster_bootstrap_matches_explicit_replication(self):
  x=np.array([1.,1.,3.,2.,5.,6.]);y=np.array([2.,3.,1.,3.,6.,6.]);w=np.array([[2,2,0,0,1,1],[0,0,3,3,1,1],[1,1,1,1,1,1]])
  a=weighted_srcc(x,y,w);b=weighted_corr(x,y,w)
  for i,n in enumerate(w):
   xx=np.repeat(x,n);yy=np.repeat(y,n);self.assertAlmostEqual(a[i],spearmanr(xx,yy).statistic,places=12);self.assertAlmostEqual(b[i],pearsonr(xx,yy).statistic,places=12)
 def test_pair_groups_and_polarity(self):
  d=pd.DataFrame({'dataset':['A']*4,'ref_id':['R','R','R','X'],'distortion_type':['t']*4,'severity':[1,2,2,3]});p=pair_table(d)
  self.assertEqual(len(p),2);np.testing.assert_array_equal(pair_gap([3.,2.,1.,9.],p,'EA'),[1,2]);np.testing.assert_array_equal(pair_gap([3.,2.,1.,9.],p,'EH'),[-1,-2])
 def test_holdout_read_guard_precedes_opinion_read(self):
  from unittest.mock import patch
  import evaluation
  with patch.object(evaluation,'verify_selected',side_effect=RuntimeError('not frozen')):
   with patch.object(evaluation.pd,'read_csv') as reader:
    with self.assertRaisesRegex(RuntimeError,'not frozen'):evaluation.holdout_evaluate()
    reader.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
