import unittest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import construct_folds,weight_values,select_specialist,pair_metrics,weighted_srcc,clusters,ARMS

class RuleTests(unittest.TestCase):
 def test_folds_keep_all_reference_images_together(self):
  rows=[{'dataset':ds,'ref_id':str(r),'group_key':ds+'::'+str(r)} for ds in ['KADID-10k','TID2013'] for r in range(12) for _ in range(3)]
  a=construct_folds(pd.DataFrame(rows));b=construct_folds(pd.DataFrame(rows)[::-1]);pd.testing.assert_frame_equal(a,b)
  self.assertEqual(len(a),24);self.assertFalse(a.group_key.duplicated().any());self.assertEqual(set(a.fold),set(range(5)))
 def test_signed_reversal_is_not_absolute_gap(self):
  v,T,r,w,e=weight_values(np.array([-.4,0,.4]),np.ones(3),0,'hard')
  np.testing.assert_array_equal(r,[1,1,0]);self.assertGreater(v[0],0)
 def test_margin_uses_normalized_severity(self):
  v,*_=weight_values([.04,.04],[.25,1],.1,'hard');np.testing.assert_allclose(v,[-.015,.06])
 def test_soft_midpoint_temperature_and_global_normalization(self):
  v,T,r,w,e=weight_values([-.1,0,.1],[1,1,1],0,'soft')
  self.assertEqual(T,.1);self.assertEqual(r[1],.5);self.assertTrue(np.all((r>0)&(r<1)))
  np.testing.assert_allclose(w,r/(r.mean()+1e-8));self.assertLessEqual(e,3)
 def test_temperature_floor_and_zero_hard_arm(self):
  v,T,r,w,e=weight_values([1,1,1],[1,1,1],0,'hard')
  self.assertEqual(T,.05);self.assertEqual(e,0);np.testing.assert_array_equal(w,[0,0,0])
 def test_pair_exact_ties_half_credit_and_group_boundaries(self):
  d=pd.DataFrame({'dataset':['KADID-10k']*4,'ref_id':['a','a','b','b'],'distortion_type':['t']*4,'image_id':list('wxyz'),'severity':[1,2,1,2],'q':[5,5,4,2]})
  macro,tab,p=pair_metrics(d,'q');self.assertEqual(len(p),2);self.assertEqual(macro,.75);self.assertEqual(p.tie.sum(),1)
 def test_selection_uses_macro_accuracy_then_arm_then_earlier_epoch(self):
  t=pd.DataFrame([{'arm':a,'epoch':ep,'S_blind':1.,'macro_pair_accuracy':.5} for a in ARMS for ep in [5,10]])
  self.assertEqual(select_specialist(t).arm,'S10');self.assertEqual(select_specialist(t).epoch,5)
  t.loc[t.arm.eq('H25'),'macro_pair_accuracy']=.6;self.assertEqual(select_specialist(t).arm,'H25')
 def test_weighted_spearman_matches_duplicate_cluster_replication(self):
  x=np.array([1,1,2,5,6]);y=np.array([6,5,5,2,1]);w=np.array([[2,2,0,1,1],[0,0,1,3,3]])
  actual=weighted_srcc(x,y,w)
  expected=[spearmanr(np.repeat(x,a),np.repeat(y,a)).statistic for a in w]
  np.testing.assert_allclose(actual,expected,atol=1e-14)
 def test_bootstrap_sample_is_complete_reference(self):
  d=pd.DataFrame({'dataset':['d']*4,'ref_id':['a','a','b','b']})
  refs,cts=clusters(d,n=100)['d'];self.assertEqual(refs,['a','b']);self.assertTrue(np.all(cts.sum(1)==2))

if __name__=='__main__':unittest.main(verbosity=2)
