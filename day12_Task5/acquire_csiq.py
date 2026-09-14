"""Download the unchanged original CSIQ archives after selection has been frozen."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,time,zipfile
from pathlib import Path
import requests
from run_task5 import verify_frozen

root=Path(__file__).resolve().parent;dest=root/'data/csiq'
verify_frozen()
start=datetime.now(timezone.utc).isoformat()
(root/'csiq/data_access_started.json').write_text(json.dumps({'started_utc':start,'frozen_configuration_verified_before_access':True})+'\n')
def download(name):
 url='https://s2.smu.edu/~eclarson/csiq/'+name;path=dest/name
 if path.exists():raise RuntimeError('Refusing to overwrite downloaded data')
 t=time.time();h=hashlib.sha256();total=0;last=t
 with requests.get(url,stream=True,timeout=(20,60)) as r:
  r.raise_for_status();length=int(r.headers.get('Content-Length','0'))
  with path.with_suffix(path.suffix+'.partial').open('wb') as f:
   for chunk in r.iter_content(1024*1024):
    if chunk:f.write(chunk);h.update(chunk);total+=len(chunk)
    if time.time()-last>20:print(name,total,'bytes',flush=True);last=time.time()
 if length and total!=length:raise RuntimeError('Incomplete download '+name)
 path.with_suffix(path.suffix+'.partial').rename(path)
 if name.endswith('.zip'):
  folder=dest/path.stem;folder.mkdir(exist_ok=True)
  with zipfile.ZipFile(path) as z:
   if z.testzip() is not None:raise RuntimeError('Archive CRC failure')
   for item in z.infolist():
    target=(folder/item.filename).resolve()
    if not target.is_relative_to(folder.resolve()):raise RuntimeError('Unsafe archive path')
    if ((item.external_attr>>16)&0o170000)==0o120000:raise RuntimeError('Unexpected symlink in archive')
   z.extractall(folder)
 print('Downloaded',name,total,'bytes',flush=True)
 return {'name':name,'source_url':url,'sha256':h.hexdigest(),'bytes':total,'download_seconds':time.time()-t,'file':str(path)}
with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(download,['csiq.DMOS.xlsx','src_imgs.zip','dst_imgs.zip']))
(root/'csiq/download_manifest.json').write_text(json.dumps({'started_utc':start,'completed_utc':datetime.now(timezone.utc).isoformat(),'files':results},indent=2)+'\n')
verify_frozen()
