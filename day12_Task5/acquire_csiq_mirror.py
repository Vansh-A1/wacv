"""Acquire the documented IQA-PyTorch dataset mirror and verify its published hash."""
import hashlib,json,time,tarfile
from pathlib import Path
from datetime import datetime,timezone
import requests
from run_task5 import verify_frozen
root=Path(__file__).resolve().parent;out=root/'data/csiq';verify_frozen()
url='https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main/csiq.tgz?download=true'
expected='e34c372c98e751c5490dadeb536e37666296b6c4f651df30cc04c0547ffee5b0'
path=out/'csiq_mirror.tgz';h=hashlib.sha256();n=0;start=time.time();last=start
with requests.get(url,stream=True,timeout=(20,60)) as r:
 r.raise_for_status()
 with path.with_suffix('.tgz.partial').open('wb') as f:
  for block in r.iter_content(1024*1024):
   if block:f.write(block);h.update(block);n+=len(block)
   if time.time()-last>20:print('Mirror',n,'bytes',round(time.time()-start,1),'seconds',flush=True);last=time.time()
actual=h.hexdigest()
if actual!=expected:raise RuntimeError('Published CSIQ mirror SHA-256 mismatch: '+actual)
path.with_suffix('.tgz.partial').rename(path)
folder=out/'mirror';folder.mkdir(exist_ok=True)
with tarfile.open(path,'r:gz') as tar:
 for member in tar.getmembers():
  if not (member.isdir() or member.isfile()):raise RuntimeError('Unexpected non-file in mirror')
 tar.extractall(folder,filter='data')
record={'source_url':url,'source_documentation':'https://github.com/chaofengc/IQA-PyTorch#-download-benchmark-datasets',
 'published_hash_source':'https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/blob/main/csiq.tgz',
 'sha256':actual,'published_sha256':expected,'published_hash_verified':True,'bytes':n,
 'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.time()-start,
 'metadata_policy':'Use original author-hosted csiq.DMOS.xlsx, not substitute labels',
 'reason':'Original distorted-image transfer was too slow; use documented research mirror with published hash.'}
(root/'csiq/mirror_download_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print('CSIQ MIRROR DOWNLOADED AND HASH VERIFIED',n,flush=True)
verify_frozen()
