"""The authorized replacement changes only the uniqueness acceptance population/precision."""
import numpy as np

def representative_indices(hashes):
 seen={};indices=[];mapping=[]
 for i,h in enumerate(hashes):
  if h not in seen:seen[h]=i;indices.append(i)
  mapping.append(seen[h])
 return np.asarray(indices,dtype=np.int64),np.asarray(mapping,dtype=np.int64)

def check_row(row):
 # The float64 exception is narrowly restricted to D0-Legacy.
 field='diagnostic_representative_fraction' if row['distance']=='D0_LEGACY' else 'primary_representative_fraction'
 return bool(row['finite'] and row['std_population']>1e-12 and row['diagnostic_finite'] and row[field]>=.99)

def amended_training_allowed(rows,other_checks):
 expected={(p,a,s) for p in ('EA','EH') for a in ('D0_LEGACY','DM_BETWEEN','DM_AGGREGATE','KL_AGGREGATE','W2_AGGREGATE','BHATT_AGGREGATE') for s in ('train','val')}
 keys=[(r['pole'],r['distance'],r['split']) for r in rows]
 return bool(len(keys)==24 and set(keys)==expected and other_checks and all(other_checks.values()) and all(check_row(r) for r in rows))
