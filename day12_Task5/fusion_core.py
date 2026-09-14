"""Opinion-unaware frozen calibration, fusion and exact cluster-bootstrap statistics."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

EPS = 1e-8
CANDIDATES = ('L00', 'L25', 'L50', 'L75', 'L100', 'PAVG', 'PHARM')
WEIGHTS = (0., .25, .5, .75, 1.)


def midrank_percentile(values, sorted_calibration):
    cal = np.asarray(sorted_calibration, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(cal) < 2 or not np.isfinite(cal).all() or np.any(cal[1:] < cal[:-1]):
        raise ValueError('Invalid frozen empirical distribution')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite score')
    lo = np.searchsorted(cal, values, side='left')
    hi = np.searchsorted(cal, values, side='right')
    return (lo + .5 * (hi - lo) + .5) / (len(cal) + 1.)


def fit_calibration(validation):
    if set(validation['split']) != {'val'} or set(validation['dataset']) != {'KADID-10k','TID2013'}:
        raise ValueError('Calibration is allowed only on pooled KADID/TID validation')
    if validation.duplicated(['dataset','image_id']).any():
        raise ValueError('Duplicate calibration identities')
    stats, ecdfs = {}, {}
    for pole in ('EA','EH'):
        values = validation[pole].to_numpy(np.float64)
        if not np.isfinite(values).all() or values.std() <= EPS or np.ptp(values) <= EPS:
            raise ValueError('Nonfinite or degenerate raw pole: ' + pole)
        stats[pole] = {'mean':float(values.mean()), 'std_population':float(values.std(ddof=0)),
                       'N':len(values), 'ddof':0, 'epsilon':EPS}
        ecdfs[pole] = np.sort(values)
    return stats, ecdfs


def apply_calibration(frame, stats, ecdfs):
    out = frame.copy()
    for p in ('EA','EH'):
        out['z_'+p] = (out[p] - stats[p]['mean']) / (stats[p]['std_population'] + EPS)
    out['u_A'] = midrank_percentile(out['EA'], ecdfs['EA'])
    out['u_H'] = 1. - midrank_percentile(out['EH'], ecdfs['EH'])
    for name,w in zip(CANDIDATES[:5],WEIGHTS):
        out[name] = w*out['z_EA'] - (1.-w)*out['z_EH']
    out['PAVG'] = (out['u_A'] + out['u_H']) / 2.
    out['PHARM'] = 2.*out['u_A']*out['u_H'] / (out['u_A']+out['u_H']+EPS)
    if not np.isfinite(out[list(CANDIDATES)].to_numpy()).all():
        raise ValueError('Nonfinite fusion score')
    return out


def correlations(x,y):
    x,y = np.asarray(x,float),np.asarray(y,float)
    ok = np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.ptp(x[ok])==0 or np.ptp(y[ok])==0:
        return np.nan,np.nan,int(ok.sum())
    return float(spearmanr(x[ok],y[ok]).statistic),float(pearsonr(x[ok],y[ok]).statistic),int(ok.sum())


def severity_table(frame):
    rows=[]
    for (dataset,typ),g in frame.groupby(['dataset','distortion_type'],sort=True):
        for q in CANDIDATES:
            sr,pr,n=correlations(g[q],g['severity'])
            rows.append({'dataset':dataset,'distortion_type':typ,'candidate':q,'N':n,
                         'severity_SRCC':sr,'severity_Pearson':pr})
    return pd.DataFrame(rows)


def selection_table(frame):
    if set(frame['split'])!={'val'}:
        raise ValueError('Selection accepts validation only')
    table=severity_table(frame)
    rows=[]
    for q in CANDIDATES:
        row={'candidate':q,'tie_priority':CANDIDATES.index(q)}
        for ds,key in [('KADID-10k','KADID'),('TID2013','TID')]:
            vals=table.loc[table.candidate.eq(q)&table.dataset.eq(ds),'severity_SRCC'].to_numpy()
            row['S_'+key]=-float(vals.mean()) if np.isfinite(vals).all() and len(vals)>0 else np.nan
            row['n_types_'+key]=len(vals)
        row['S_fusion']=(row['S_KADID']+row['S_TID'])/2.
        rows.append(row)
    result=pd.DataFrame(rows).sort_values(['S_fusion','tie_priority'],ascending=[False,True],na_position='last')
    if not np.isfinite(result.iloc[0]['S_fusion']):
        raise ValueError('No valid fusion candidate')
    selected=result.iloc[0]['candidate']
    result['selected']=result.candidate.eq(selected)
    return result,table,selected


def make_pairs(frame):
    """All different-severity pairs; never infer severity from human opinion scores."""
    rows=[]
    for (ds,ref,typ),g in frame.groupby(['dataset','ref_id','distortion_type'],sort=True):
        g=g[np.isfinite(g['severity'])]
        for a,b in combinations(g.index,2):
            sa,sb=frame.at[a,'severity'],frame.at[b,'severity']
            if sa==sb: continue
            if sa>sb: a,b=b,a
            rows.append({'dataset':ds,'ref_id':ref,'distortion_type':typ,
                'mild_index':a,'severe_index':b,'image_id_mild':frame.at[a,'image_id'],
                'image_id_severe':frame.at[b,'image_id'],'severity_mild':frame.at[a,'severity'],
                'severity_severe':frame.at[b,'severity']})
    return pd.DataFrame(rows,columns=['dataset','ref_id','distortion_type','mild_index','severe_index',
        'image_id_mild','image_id_severe','severity_mild','severity_severe'])


def pair_outputs(frame):
    pairs=make_pairs(frame)
    details=[];rows=[]
    for q in CANDIDATES:
        d=pairs.copy(); a=d.mild_index.to_numpy(int); b=d.severe_index.to_numpy(int)
        d['candidate']=q
        d['quality_gap']=frame.loc[a,q].to_numpy()-frame.loc[b,q].to_numpy()
        d['correct']=d.quality_gap>0;d['tie']=d.quality_gap==0
        details.append(d)
        scopes=[(ds,'all',g) for ds,g in d.groupby('dataset',sort=True)]
        scopes += [(ds,t,g) for (ds,t),g in d.groupby(['dataset','distortion_type'],sort=True)]
        for ds,t,g in scopes:
            rows.append({'dataset':ds,'distortion_type':t,'candidate':q,'n_pairs':len(g),
                'correct':int(g.correct.sum()),'ties':int(g.tie.sum()),
                'pair_accuracy':float(g.correct.mean()),'tie_rate':float(g.tie.mean())})
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def score_summary(frame,target=None):
    rows=[]
    for ds,g in frame.groupby('dataset',sort=True):
        for q in CANDIDATES:
            v=g[q].to_numpy(float); _,cts=np.unique(v,return_counts=True)
            row={'dataset':ds,'candidate':q,'N':len(g),'min':float(v.min()),'max':float(v.max()),
                'range':float(np.ptp(v)),'variance_population':float(v.var()),'std_population':float(v.std()),
                'equal_score_unordered_pairs':int((cts*(cts-1)//2).sum()),
                'duplicate_score_images':int(cts[cts>1].sum())}
            if target:
                sr,pr,n=correlations(v,g[target]);row.update(SRCC=sr,Pearson_raw=pr,target_N=n)
            rows.append(row)
    return pd.DataFrame(rows)


def weighted_midrank(values, weights):
    """Ranks in the explicitly cluster-replicated sample, including exact ties."""
    values=np.asarray(values,float); weights=np.atleast_2d(weights).astype(float)
    order=np.argsort(values,kind='stable'); sorted_values=values[order]
    starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
    group_counts=np.add.reduceat(weights[:,order],starts,axis=1)
    ranks=np.cumsum(group_counts,axis=1)-.5*group_counts+.5
    groups=np.cumsum(np.r_[False,sorted_values[1:]!=sorted_values[:-1]])
    inverse=np.argsort(order)
    return ranks[:,groups][:,inverse]


def weighted_pearson(x,y,weights):
    w=np.atleast_2d(weights).astype(float); x=np.broadcast_to(x,w.shape);y=np.broadcast_to(y,w.shape)
    n=w.sum(axis=1,keepdims=True)
    xc=x-(w*x).sum(axis=1,keepdims=True)/n;yc=y-(w*y).sum(axis=1,keepdims=True)/n
    cov=(w*xc*yc).sum(axis=1)
    den=np.sqrt((w*xc*xc).sum(axis=1)*(w*yc*yc).sum(axis=1))
    return np.divide(cov,den,out=np.full_like(cov,np.nan),where=den>0)


def weighted_spearman(x,y,weights):
    return weighted_pearson(weighted_midrank(x,weights),weighted_midrank(y,weights),weights)


def cluster_draws(frame,n=5000,seed=20260908):
    rng=np.random.default_rng(seed); result={}
    for ds,g in frame.groupby('dataset',sort=True):
        refs=sorted(g.ref_id.unique()); sampled=rng.integers(0,len(refs),size=(n,len(refs)))
        counts=np.array([(sampled==j).sum(axis=1) for j in range(len(refs))]).T
        result[ds]=(refs,counts)
    return result


def validation_bootstrap(frame,selected,n=5000,seed=20260908):
    """Fixed selected identity and calibration, stratified whole-reference draws."""
    draws=cluster_draws(frame,n,seed); by_dataset={}
    for ds,g in frame.groupby('dataset',sort=True):
        refs,counts=draws[ds]; lookup={r:i for i,r in enumerate(refs)}
        data=np.zeros((n,2)); ntypes=0
        for typ,sub in g.groupby('distortion_type',sort=True):
            w=counts[:,[lookup[r] for r in sub.ref_id]]
            for k,q in enumerate((selected,'L00')):
                data[:,k] -= weighted_spearman(sub[q].to_numpy(),sub.severity.to_numpy(),w)
            ntypes+=1
        by_dataset[ds]=data/ntypes
    values=sum(by_dataset.values())/len(by_dataset)
    return values[:,0]-values[:,1],draws


def percentile_ci(values):
    values=np.asarray(values,float);valid=values[np.isfinite(values)]
    return {'estimate_draw_mean':float(valid.mean()) if len(valid) else None,
        'ci_low':float(np.quantile(valid,.025)) if len(valid) else None,
        'ci_high':float(np.quantile(valid,.975)) if len(valid) else None,
        'valid_draws':len(valid),'invalid_draws':len(values)-len(valid),'draws':len(values)}


def evaluation_bootstrap(frame,target='target_quality',n=5000,seed=20260908):
    """Paired ref-cluster CIs for SRCC, raw Pearson, and within-ref pair accuracy."""
    draws=cluster_draws(frame,n,seed); pairs=make_pairs(frame);summaries=[];saved={}
    for ds,g in frame.groupby('dataset',sort=True):
        refs,counts=draws[ds]; lookup={r:i for i,r in enumerate(refs)}
        w=counts[:,[lookup[r] for r in g.ref_id]]
        y=g[target].to_numpy(float)
        pg=pairs[pairs.dataset.eq(ds)]
        pair_counts=np.array([int(pg.ref_id.eq(ref).sum()) for ref in refs])
        pair_denom=counts@pair_counts
        qdata={}
        for q in CANDIDATES:
            x=g[q].to_numpy(float); sr=np.empty(n);pr=np.empty(n)
            for start in range(0,n,250):
                sl=slice(start,min(start+250,n));sr[sl]=weighted_spearman(x,y,w[sl]);pr[sl]=weighted_pearson(x,y,w[sl])
            a=pg.mild_index.to_numpy(int);b=pg.severe_index.to_numpy(int)
            correct=frame.loc[a,q].to_numpy()>frame.loc[b,q].to_numpy()
            byref=np.array([correct[pg.ref_id.eq(ref).to_numpy()].sum() for ref in refs])
            pa=np.divide(counts@byref,pair_denom,out=np.full(n,np.nan),where=pair_denom>0)
            qdata[q]={'SRCC':sr,'Pearson_raw':pr,'pair_accuracy':pa}
        for q in CANDIDATES:
            for metric,vals in qdata[q].items():
                for kind,values in [('absolute',vals),('difference_vs_L00',vals-qdata['L00'][metric])]:
                    summaries.append({'dataset':ds,'candidate':q,'metric':metric,'comparison':kind,**percentile_ci(values)})
                    saved[f'{ds}__{q}__{metric}__{kind}']=values
    return pd.DataFrame(summaries),saved,draws
