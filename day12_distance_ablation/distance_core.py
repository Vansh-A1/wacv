"""Exact registered six-arm distances; constants are never learned or refitted."""
import numpy as np
import torch
EPS=1e-8
ARMS=('D0_LEGACY','DM_BETWEEN','DM_AGGREGATE','KL_AGGREGATE','W2_AGGREGATE','BHATT_AGGREGATE')

def distance(mu,logvar,reference,arm):
 if arm not in ARMS:raise ValueError(arm)
 if mu.shape!=logvar.shape:raise ValueError('Posterior shape mismatch')
 dtype=mu.dtype if arm=='D0_LEGACY' else torch.float64
 m=mu.to(dtype);lv=logvar.to(dtype)
 def tensor(k):return torch.as_tensor(reference[k],device=m.device,dtype=dtype).detach()
 mr=tensor('legacy_mu' if arm=='D0_LEGACY' else 'registered_mu');delta2=(m-mr).square()
 if arm in ARMS[:3]:
  var=tensor({'D0_LEGACY':'legacy_variance','DM_BETWEEN':'between_variance','DM_AGGREGATE':'aggregate_variance_raw'}[arm])
  return (delta2/(var+EPS)).sum(-1).sqrt()
 vx=lv.exp().clamp_min(EPS);vr=tensor('aggregate_variance_raw').clamp_min(EPS)
 if arm=='KL_AGGREGATE':return .5*(vr.log()-vx.log()+vx/vr+delta2/vr-1).sum(-1)
 if arm=='W2_AGGREGATE':return (delta2+(vx.sqrt()-vr.sqrt()).square()).sum(-1)
 bar=(vx+vr)/2
 return (delta2/bar).sum(-1)/8+.5*(bar.log()-.5*(vx.log()+vr.log())).sum(-1)

def aggregate(mu,logvar,weights):
 mu=np.asarray(mu,np.float64);lv=np.asarray(logvar,np.float64);w=np.asarray(weights,np.float64)
 if mu.shape!=lv.shape or w.shape!=(len(mu),):raise ValueError('Moment geometry')
 if not all(np.isfinite(x).all() for x in [mu,lv,w]) or (w<=0).any() or abs(w.sum()-1)>1e-12:raise ValueError('Invalid weights/posteriors')
 mean=np.sum(w[:,None]*mu,axis=0);between=np.sum(w[:,None]*(mu-mean)**2,axis=0)
 within=np.sum(w[:,None]*np.exp(lv),axis=0);raw=between+within
 if not all(np.isfinite(x).all() for x in [mean,between,within,raw]):raise ValueError('Invalid moments')
 return dict(mu_agg=mean,v_between=between,v_within=within,v_aggregate_raw=raw,v_agg=np.maximum(raw,EPS),variance_epsilon=np.array(EPS),ddof=np.array(0),n_images=np.array(len(mu)))

def rank_gap(mild,severe,pole):
 if pole=='EA':return mild-severe
 if pole=='EH':return severe-mild
 raise ValueError(pole)

def standardized(raw,stats):return (raw-stats['mean'])/(stats['std_population']+EPS)

def training_allowed(rows):
 return bool(rows and all(r['finite'] and r['std_population']>1e-12 and r['distinct_fraction']>=.99 for r in rows))
