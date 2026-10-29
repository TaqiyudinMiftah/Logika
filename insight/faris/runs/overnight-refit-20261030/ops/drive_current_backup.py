"""Private, hash-verified experiment archives streamed from lab2001 to Drive.

Credentials stay on personal host119. No model weights, virtualenvs, OAuth state,
SSH keys or downloaded foundation checkpoints enter the archives. No sharing or
deletion calls are made. Uploads are resumable and the returned MD5 is checked.
"""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,time,urllib.request,urllib.parse,urllib.error,traceback
R=Path('/home/faris/logika-overnight-refit-20261030');P=R/'private/drive';P.mkdir(parents=True,exist_ok=True)
PROJECT='gen-lang-client-0779760775';LAB='faris@100.68.29.126';KEY='/home/faris/.ssh/id_ed25519_logika_lab'
def dump(v,p):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def token():
 c=json.load(open(Path.home()/'.config/colab-cli/accounts/2/token.json'))
 body=urllib.parse.urlencode(dict(client_id=c['client_id'],client_secret=c['client_secret'],refresh_token=c['refresh_token'],grant_type='refresh_token')).encode()
 return json.load(urllib.request.urlopen('https://oauth2.googleapis.com/token',data=body,timeout=45))['access_token']
def request(url,body=None,method=None,extra=None):
 headers={'Authorization':'Bearer '+token(),'x-goog-user-project':PROJECT,**(extra or{})}
 return urllib.request.urlopen(urllib.request.Request(url,data=body,method=method,headers=headers),timeout=150)
def ssh(code,timeout=900):
 return subprocess.run(['ssh','-T','-i',KEY,'-o','BatchMode=yes',LAB,'python3 -'],input=code,text=True,capture_output=True,check=True,timeout=timeout)
def build(root):
 code=r'''
from pathlib import Path
import hashlib,json,tarfile,re,io,time,os
os.nice(15)
r=Path(ROOT);base=Path('/mnt/home/faris/logika-overnight-refit-20261030/private/drive-archives');base.mkdir(parents=True,exist_ok=True)
archive=base/(r.name+'.tar.gz');manifest=base/(r.name+'.manifest.json')
if archive.exists()and manifest.exists():
 v=json.load(open(manifest));h=hashlib.sha256()
 with archive.open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert v['archive_sha256']==h.hexdigest();print(json.dumps(v));raise SystemExit
exclude={'private','env','env311','env_gpu','.venv','venv','storage','huggingface','hf','pretrained','pretrained-cache','checkpoints','__pycache__','.git','.ssh','raw','raw_data','data','datasets','dataset','supplied_data','INCREMENTAL_EVIDENCE','FINAL_EVIDENCE','publication_ready','node_modules'}
extensions={'.py','.json','.jsonl','.md','.npz','.npy','.csv','.parquet','.txt','.log','.yaml','.yml','.toml','.ipynb','.sh','.pdf'}
secret=re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----|KGAT_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|[\"\x27](?:refresh_token|access_token|client_secret)[\"\x27]\s*:\s*[\"\x27][A-Za-z0-9._/-]{12,}')
included={};excluded=[]
with tarfile.open(archive.with_suffix('.tmp'),'w:gz',compresslevel=1)as t:
 for parent,dirs,files in os.walk(r,followlinks=False):
  dirs[:]=[d for d in dirs if d not in exclude and not d.startswith('env')and not Path(parent,d).is_symlink()]
  for name in sorted(files):
   p=Path(parent,name);rel=str(p.relative_to(r))
   if p.suffix not in extensions or any(s in name.lower()for s in ['token','credential','secret','sessions','known_hosts','client_oauth','client-oauth']):continue
   if p.is_symlink()or not p.resolve().is_relative_to(Path('/mnt/home/faris')):continue
   if p.suffix in {'.json','.jsonl','.txt','.log','.md','.py','.ipynb','.toml','.yaml','.yml'}:
    b=p.read_bytes()
    if secret.search(b):excluded.append(dict(path=rel,reason='credential_pattern'));continue
    i=tarfile.TarInfo(rel);i.size=len(b);i.mode=0o600;i.mtime=int(p.stat().st_mtime);t.addfile(i,io.BytesIO(b));digest=hashlib.sha256(b).hexdigest()
   else:
    # Source is sealed or a completed scientific output, not a live fitted cache.
    b=p.read_bytes();i=tarfile.TarInfo(rel);i.size=len(b);i.mode=0o600;t.addfile(i,io.BytesIO(b));digest=hashlib.sha256(b).hexdigest()
   included[rel]=dict(bytes=len(b),sha256=digest)
 archive_info=json.dumps(dict(source=str(r),included=included,excluded=excluded,scope='Accessible experiment source/parameters/datafeatures/compact predictions/reports; private machine state and weights excluded'),indent=2).encode();i=tarfile.TarInfo('ARCHIVE_CONTENT_MANIFEST.json');i.size=len(archive_info);i.mode=0o600;t.addfile(i,io.BytesIO(archive_info))
archive.with_suffix('.tmp').rename(archive)
hs=hashlib.sha256();hm=hashlib.md5()
with archive.open('rb')as f:
 for b in iter(lambda:f.read(8*1024**2),b''):hs.update(b);hm.update(b)
v=dict(source=str(r),archive=str(archive),bytes=archive.stat().st_size,archive_sha256=hs.hexdigest(),archive_md5=hm.hexdigest(),included_files=len(included),credential_pattern_exclusions=excluded,epoch=time.time(),private_machine_state_excluded=True,weights_excluded=True)
manifest.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
'''.replace('ROOT',repr(root))
 return json.loads(ssh(code,timeout=1800).stdout)
def upload(v,folder):
 receipt=P/(Path(v['source']).name+'.json')
 if receipt.exists():
  old=json.load(open(receipt))
  if old.get('archive_sha256')==v['archive_sha256']and old.get('status')=='UPLOADED_HASH_VERIFIED':return old
 meta=json.dumps(dict(name=Path(v['archive']).name,parents=[folder],description='Private Logika experiment evidence; SHA256 '+v['archive_sha256'])).encode()
 response=request('https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size,md5Checksum,webViewLink',meta,'POST',{'Content-Type':'application/json; charset=UTF-8','X-Upload-Content-Type':'application/gzip','X-Upload-Content-Length':str(v['bytes'])})
 location=response.headers['Location']
 # Resumable URL is a capability: retain only in this private directory.
 session=P/(Path(v['source']).name+'.upload-private.json');dump(dict(url=location,source=v),session);os.chmod(session,0o600)
 cmd=['ssh','-T','-i',KEY,'-o','BatchMode=yes',LAB,'cat '+shlex.quote(v['archive'])]
 process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE);offset=0;hasher=hashlib.md5();result=None;bearer=token();last_refresh=time.time()
 try:
  while offset<v['bytes']:
   assert time.time()<json.load(open(R/'CLOCK.json'))['deadline_epoch']-60,'Original absolute archive deadline reached; retain resumable evidence'
   block=process.stdout.read(min(16*1024**2,v['bytes']-offset));assert block,'Truncated source archive stream';hasher.update(block);end=offset+len(block)-1
   if time.time()-last_refresh>1800:bearer=token();last_refresh=time.time()
   headers={'Authorization':'Bearer '+bearer,'x-goog-user-project':PROJECT,'Content-Type':'application/gzip','Content-Length':str(len(block)),'Content-Range':f'bytes {offset}-{end}/{v["bytes"]}'}
   for attempt in range(3):
    try:
     response=urllib.request.urlopen(urllib.request.Request(location,data=block,method='PUT',headers=headers),timeout=240)
     payload=response.read();result=json.loads(payload)if payload else result;break
    except urllib.error.HTTPError as e:
     if e.code==308:break
     if attempt==2:raise
     time.sleep(5)
    except (TimeoutError,OSError):
     if attempt==2:raise
     time.sleep(5)
   offset=end+1;dump(dict(epoch=time.time(),source=v['source'],uploaded_bytes=offset,total_bytes=v['bytes'],resumable_url_private=True),P/'LIVE_STATE.json')
  assert process.wait(timeout=60)==0
  assert hasher.hexdigest()==v['archive_md5']and result['md5Checksum']==v['archive_md5']and int(result['size'])==v['bytes']
  final=dict(v,status='UPLOADED_HASH_VERIFIED',Drive=result,verified_epoch=time.time(),private_folder=True);dump(final,receipt);return final
 finally:
  if process.poll()is None:process.terminate();process.wait(timeout=30)
def main():
 C=json.load(open(R/'CLOCK.json'));folder=json.load(open(Path('/home/faris/logika-twelvehour-next-20261029/private/DRIVE_FOLDER.json')))['id']
 about=json.load(request('https://www.googleapis.com/drive/v3/about?fields=user(emailAddress),storageQuota'));assert about['user']['emailAddress']=='helma.eviiii@gmail.com';assert int(about['storageQuota']['limit'])>=5*10**12;dump(about,P/'VERIFIED_ACCOUNT.json')
 while time.time()<C['model_deadline_epoch']+300:
  # An early private snapshot is safe once every registered scientific job is terminal.
  code="from pathlib import Path;import json;r=Path('/mnt/home/faris/logika-overnight-refit-20261030');p=json.load(open(r/'UNION_PLAN.json'));names=p['component_studies']+p.get('meta_composition_studies',[]);print(all((r/n/'COMPLETION.json').exists()for n in names))"
  if ssh(code,timeout=60).stdout.strip()=='True':break
  time.sleep(30)
 current=upload(build('/mnt/home/faris/logika-overnight-refit-20261030'),folder)
 dump(dict(status='CURRENT_ARCHIVE_HASH_VERIFIED',bytes=current['bytes'],source_sha256=current['archive_sha256'],scope='All registered terminal scientific source/parameters/compact predictions snapshot; later root audit/submission receipts uploaded as a separate delivery archive; private machine state excluded'),P/'CURRENT_SUMMARY.json')
 while time.time()<C['deadline_epoch']-120:
  complete=ssh("from pathlib import Path;print(Path('/mnt/home/faris/logika-overnight-refit-20261030/SUBMITTED.json').exists())").stdout.strip()=='True'
  if complete:
   code="from pathlib import Path;import shutil;r=Path('/mnt/home/faris/logika-overnight-refit-20261030');d=r/'final_delivery';d.mkdir(exist_ok=True);[shutil.copy2(p,d/p.name)for p in r.glob('*.json')];[shutil.copy2(p,d/p.name)for p in r.glob('submission_CURRENT_*.csv')]"
   ssh(code);upload(build('/mnt/home/faris/logika-overnight-refit-20261030/final_delivery'),folder);break
  time.sleep(20)

if __name__=='__main__':
 try:main()
 except BaseException:(P/'BACKUP_FAILURE_PRIVATE.log').write_text(traceback.format_exc());raise
