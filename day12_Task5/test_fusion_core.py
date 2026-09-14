import unittest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from fusion_core import *

class FusionTests(unittest.TestCase):
    def test_midrank_ties_and_endpoints(self):
        np.testing.assert_allclose(midrank_percentile([-1,0,1,2,3],[0,1,1,2]),[.1,.2,.5,.8,.9])
    def test_ecdf_uses_fixed_calibration(self):
        self.assertEqual(midrank_percentile([1000],[0,1,2])[0],.875)
    def test_forbid_test_calibration(self):
        f=pd.DataFrame({'split':['holdout'],'dataset':['KADID-10k'],'image_id':['a'],'EA':[1],'EH':[2]})
        with self.assertRaises(ValueError):fit_calibration(f)
    def test_fusion_polarities(self):
        f=pd.DataFrame({'EA':[0.,2.],'EH':[0.,2.]})
        stats={p:{'mean':1.,'std_population':1.} for p in ['EA','EH']};ecdf={p:np.array([0.,2.]) for p in ['EA','EH']}
        s=apply_calibration(f,stats,ecdf)
        np.testing.assert_allclose(s.L00,-s.z_EH);np.testing.assert_allclose(s.L100,s.z_EA)
        np.testing.assert_allclose(s.L50,0.)
        np.testing.assert_allclose(s.PHARM,2*s.u_A*s.u_H/(s.u_A+s.u_H+1e-8))
    def test_weighted_ranks_match_explicit_replication_with_ties(self):
        x=np.array([1.,2.,2.,7.,3.]);y=np.array([5.,1.,3.,3.,8.])
        ws=np.array([[2,0,3,1,2],[0,2,1,1,0],[1,1,1,1,1]])
        for w,r in zip(ws,weighted_spearman(x,y,ws)):
            ids=np.repeat(np.arange(len(x)),w)
            self.assertAlmostEqual(r,spearmanr(x[ids],y[ids]).statistic,places=13)
    def test_weighted_pearson_matches_replication(self):
        x=np.array([1.,4.,5.]);y=np.array([2.,1.,7.]);w=np.array([[2,0,3]])
        ids=np.repeat(np.arange(3),w[0])
        self.assertAlmostEqual(weighted_pearson(x,y,w)[0],pearsonr(x[ids],y[ids]).statistic,places=13)
    def test_pairs_do_not_cross_refs_or_equal_severity(self):
        f=pd.DataFrame({'dataset':['A']*4,'ref_id':['1','1','1','2'],'distortion_type':['b']*4,
                        'image_id':['a','b','c','d'],'severity':[1,2,2,3]})
        p=make_pairs(f);self.assertEqual(len(p),2);self.assertTrue((p.severity_mild<p.severity_severe).all())
    def test_dataset_equal_weight_selection(self):
        rows=[]
        for ds,n in [('KADID-10k',10),('TID2013',3)]:
            for i in range(n):
                rows.append({'dataset':ds,'ref_id':'r','distortion_type':'t','image_id':str(i),'split':'val','severity':i,**{q:-float(i) for q in CANDIDATES}})
        t,_,selected=selection_table(pd.DataFrame(rows))
        self.assertEqual(selected,'L00');np.testing.assert_allclose(t.S_fusion,1.)
    def test_bootstrap_clusters_keep_complete_references(self):
        f=pd.DataFrame({'dataset':['A']*4,'ref_id':['r1','r1','r2','r2']})
        refs,counts=cluster_draws(f,n=50,seed=42)['A'];self.assertEqual(refs,['r1','r2'])
        self.assertTrue((counts.sum(axis=1)==2).all())
    def test_identical_selected_baseline_bootstrap_difference_zero(self):
        rows=[]
        for ds in ('KADID-10k','TID2013'):
            for ref in ('a','b'):
                for s in (1,2,3): rows.append({'dataset':ds,'ref_id':ref,'distortion_type':'x','severity':s,'L00':-s})
        delta,_=validation_bootstrap(pd.DataFrame(rows),'L00',50,42)
        np.testing.assert_array_equal(delta,np.zeros(50))

if __name__=='__main__': unittest.main()
