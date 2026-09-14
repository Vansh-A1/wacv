import unittest
import numpy as np
from amendment_gate import *
from distance_core import ARMS
class AmendmentTests(unittest.TestCase):
 def good(self,arm='D0_LEGACY'):
  return dict(distance=arm,finite=True,std_population=1.,diagnostic_finite=True,primary_representative_fraction=.95,diagnostic_representative_fraction=1.)
 def test_only_exact_hashes_grouped_and_order_preserved(self):
  r,m=representative_indices(['A','B','A','C','B']);np.testing.assert_array_equal(r,[0,1,3]);np.testing.assert_array_equal(m,[0,1,0,3,1])
 def test_float64_exception_only_legacy(self):
  self.assertTrue(check_row(self.good()))
  for arm in ARMS[1:]:self.assertFalse(check_row(self.good(arm)))
 def test_every_other_condition_still_blocks(self):
  for patch in ({'finite':False},{'std_population':1e-12},{'diagnostic_finite':False},{'diagnostic_representative_fraction':.989999}):self.assertFalse(check_row({**self.good(),**patch}))
 def test_complete_matrix_and_other_checks_required(self):
  rows=[{**self.good(a),'primary_representative_fraction':1.,'pole':p,'split':s} for p in ('EA','EH') for a in ARMS for s in ('train','val')]
  self.assertTrue(amended_training_allowed(rows,{'synthetic':True}))
  self.assertFalse(amended_training_allowed(rows[:-1],{'synthetic':True}))
  self.assertFalse(amended_training_allowed(rows,{'synthetic':False}))
  self.assertFalse(amended_training_allowed(rows,{}))
if __name__=='__main__':unittest.main(verbosity=2)
