"""Join official CSIQ metadata to verified mirror images after the freeze."""
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_task5 import verify_frozen, sha, image_identity

root=Path(__file__).resolve().parent
frozen=verify_frozen()
meta=pd.read_csv(root/'csiq/published_metadata.csv',dtype={'ref_id':str},float_precision='round_trip')
assert len(meta)==866 and meta.ref_id.nunique()==30
assert len(meta.distortion_type_original.unique())==6
assert np.isfinite(meta[['DMOS','severity']].to_numpy(float)).all()
assert not meta.duplicated(['ref_id','distortion_type_original','severity']).any()
base=root/'data/csiq/mirror/CSIQ'
paths=list((base/'dst_imgs').glob('*.png'))
lookup={p.name.lower():p for p in paths}
assert len(lookup)==len(paths)==866
refs=list((base/'src_imgs').glob('*.png'))
ref_lookup={p.name.lower():p for p in refs}
assert len(ref_lookup)==30
mapping={'noise':'awgn','jpeg':'jpeg','jpeg 2000':'jpeg2000','fnoise':'fnoise','blur':'blur','contrast':'contrast'}
rows=[]
for r in meta.to_dict('records'):
    assert int(r['severity'])==r['severity']
    name=f"{r['ref_id']}.{mapping[r['distortion_type_original']]}.{int(r['severity'])}.png"
    path=lookup[name.lower()]
    ref=ref_lookup[(r['ref_id']+'.png').lower()]
    rows.append({'dataset':'CSIQ','image_id':path.name,'ref_id':r['ref_id'],
        'distortion_type':r['distortion_type_original'],'severity':int(r['severity']),
        'path':str(path),'ref_path':str(ref),**{k:v for k,v in r.items() if k not in ['ref_id','severity']}})
df=pd.DataFrame(rows)
assert df.path.nunique()==866 and df.ref_path.nunique()==30
assert set(df.path)==set(map(str,paths))
checks=[]
for p in refs:
    original=root/'data/csiq/src_imgs'/p.name
    assert original.exists(),original
    orig_hash,orig_pixels=image_identity(original)
    mirror_hash,mirror_pixels=image_identity(p)
    assert orig_pixels==mirror_pixels, p.name
    checks.append({'reference':p.name,'original_sha256':orig_hash,'mirror_sha256':mirror_hash,
        'identical_file_bytes':orig_hash==mirror_hash,'identical_decoded_RGB':True})
pd.DataFrame(checks).to_csv(root/'csiq/original_reference_crosscheck.csv',index=False)
out=root/'csiq/evaluation_manifest.csv'
if out.exists():
    assert out.read_text()==df.to_csv(index=False),'Existing manifest differs; refusing overwrite'
else:
    df.to_csv(out,index=False)
files=[]
for name in ('csiq.DMOS.xlsx','src_imgs.zip'):
    p=root/'data/csiq'/name
    files.append({'file':str(p),'source_url':'https://s2.smu.edu/~eclarson/csiq/'+name,'sha256':sha(p),'bytes':p.stat().st_size,'complete':True})
partial=root/'data/csiq/dst_imgs.zip.partial'
if partial.exists():
    files.append({'file':str(partial),'source_url':'https://s2.smu.edu/~eclarson/csiq/dst_imgs.zip','bytes':partial.stat().st_size,'complete':False,'used':False,'reason':'Slow transfer stopped after the verified documented mirror completed.'})
audit={'completed_utc':datetime.now(timezone.utc).isoformat(),'frozen_configuration_verified':True,
    'selected_Q_star':frozen['selected_Q_star'],'manifest_sha256':sha(out),'metadata_sha256':sha(root/'csiq/published_metadata.csv'),
    'N':len(df),'references':df.ref_id.nunique(),'distortion_counts':df.distortion_type.value_counts().to_dict(),
    'all_archive_distorted_images_matched_once':True,'original_reference_RGB_matches':len(checks),
    'original_reference_file_matches':sum(x['identical_file_bytes'] for x in checks),
    'filename_distortion_mapping':mapping,'DMOS_source':'Official author workbook; mirror labels unused',
    'original_source_downloads':files}
(root/'csiq/manifest_preparation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
verify_frozen()
print(json.dumps(audit,indent=2))
