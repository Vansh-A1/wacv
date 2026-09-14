import unittest
import numpy as np
import torch
from distance_core import *
class DistanceTests(unittest.TestCase):
 def setUp(self):
  self.m=torch.tensor([[.3,-.7],[1.,2.]],dtype=torch.float64)
  self.lv=torch.tensor([[.6,1.4],[2.,.8]],dtype=torch.float64).log()
  self.ref={'legacy_mu':np.zeros(2),'legacy_variance':np.array([1.,2.]),'registered_mu':np.zeros(2),'between_variance':np.array([1.,2.]),'aggregate_variance_raw':np.array([1.5,2.5])}
 def test_finite_nonnegative(self):
  for a in ARMS:
   d=distance(self.m,self.lv,self.ref,a);self.assertTrue(torch.isfinite(d).all());self.assertTrue((d>=-1e-10).all())
 def test_identical_distribution_zero(self):
  m=torch.zeros(1,2,dtype=torch.float64);lv=torch.tensor([[1.5,2.5]],dtype=torch.float64).log()
  for a in ARMS[3:]:self.assertLessEqual(abs(distance(m,lv,self.ref,a).item()),1e-10)
 def test_mean_change_all_arms(self):
  for a in ARMS:self.assertFalse(torch.allclose(distance(self.m,self.lv,self.ref,a),distance(self.m+1,self.lv,self.ref,a)))
 def test_variance_change_only_distribution_arms(self):
  for a in ARMS:
   x=distance(self.m,self.lv,self.ref,a);y=distance(self.m,self.lv+1,self.ref,a)
   self.assertEqual(torch.equal(x,y),a in ARMS[:3])
 def test_mean_gradients(self):
  for a in ARMS:
   m=self.m.clone().requires_grad_(True);g=torch.autograd.grad(distance(m,self.lv,self.ref,a).sum(),m)[0]
   self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.norm().item(),1e-12)
 def test_log_variance_gradients(self):
  for a in ARMS[3:]:
   lv=self.lv.clone().requires_grad_(True);g=torch.autograd.grad(distance(self.m,lv,self.ref,a).sum(),lv)[0]
   self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.norm().item(),1e-12)
 def test_KL_direction(self):
  ref={'registered_mu':np.zeros(1),'aggregate_variance_raw':np.array([4.])}
  got=distance(torch.tensor([[1.]],dtype=torch.float64),torch.zeros(1,1,dtype=torch.float64),ref,'KL_AGGREGATE')
  expected=.5*(np.log(4)+1/4+1/4-1);self.assertAlmostEqual(got.item(),expected,places=12)
 def test_squared_W2(self):
  r={'registered_mu':np.zeros(1),'aggregate_variance_raw':np.ones(1)}
  self.assertEqual(distance(torch.tensor([[3.]],dtype=torch.float64),torch.zeros(1,1,dtype=torch.float64),r,'W2_AGGREGATE').item(),9.)
 def test_batch_matches_single(self):
  for a in ARMS:
   got=distance(self.m,self.lv,self.ref,a);one=torch.cat([distance(self.m[i:i+1],self.lv[i:i+1],self.ref,a) for i in range(2)])
   torch.testing.assert_close(got,one,rtol=0,atol=1e-12)
 def test_denominator_no_double_protection(self):
  r={**self.ref,'between_variance':np.zeros(2)};m=torch.ones(1,2,dtype=torch.float64)
  self.assertAlmostEqual(distance(m,torch.zeros_like(m),r,'DM_BETWEEN').item(),np.sqrt(2/EPS),places=9)
 def test_weighted_population_moments(self):
  m=np.array([[0.,2.],[2.,4.],[10.,6.]]);lv=np.zeros_like(m);w=np.array([.25,.25,.5]);a=aggregate(m,lv,w)
  np.testing.assert_allclose(a['mu_agg'],[5.5,4.5]);np.testing.assert_allclose(a['v_within'],[1,1]);np.testing.assert_allclose(a['v_aggregate_raw'],a['v_between']+1)
 def test_polarity_and_softplus(self):
  mild=torch.tensor([2.]);severe=torch.tensor([1.])
  self.assertGreater(rank_gap(mild,severe,'EA').item(),0);self.assertLess(rank_gap(mild,severe,'EH').item(),0)
  for pole,lo,hi in [('EA',mild,severe),('EH',severe,mild)]:
   g=rank_gap(lo,hi,pole);self.assertLess(torch.nn.functional.softplus(.1-g).item(),torch.nn.functional.softplus(.1+g).item())
 def test_uniqueness_gate_cannot_be_bypassed(self):
  good={'finite':True,'std_population':1.,'distinct_fraction':.99}
  self.assertTrue(training_allowed([good]));self.assertFalse(training_allowed([{**good,'distinct_fraction':.989999}]))
  self.assertFalse(training_allowed([{**good,'finite':False}]));self.assertFalse(training_allowed([]))
if __name__=='__main__':unittest.main(verbosity=2)
