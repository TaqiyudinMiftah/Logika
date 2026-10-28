"""Late node380 admission. Private connection material stays outside campaign snapshots."""
import base64,hashlib,json,os,sys,tarfile,time,urllib.request,subprocess,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'));from utils import read,write,sha
sys.path.insert(0,str(ROOT/'.credentials'))

def client(purpose='control'):
 import jupyter_remote as r
 private=read(ROOT/'.credentials/node380.json');c=r.JupyterClient(private['base_url'],private['token'])
 def request(method,endpoint,payload=None):
  from urllib.parse import urljoin
  data=json.dumps(payload).encode() if payload is not None else None;req=urllib.request.Request(urljoin(c.base_url,endpoint),data=data,method=method,headers={'Authorization':'token '+c.token,'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=10) as response:body=response.read()
  return json.loads(body) if body else None
 c.request=request;c.RETRIES=1
 # Reuse only this campaign's own kernel, never interrupt the user's terminal.
 statepath=ROOT/'.credentials'/('node-kernel.json' if purpose=='control' else 'node-kernel-'+purpose+'.json')
 if statepath.exists():
  kernel=read(statepath)['id']
  try:c.request('GET','api/kernels/'+kernel)
  except urllib.error.HTTPError as error:
   if error.code!=404:raise
   kernel=c.create_kernel()['id'];write(statepath,{'id':kernel})
 else:
  kernel=c.create_kernel()['id'];write(statepath,{'id':kernel})
 return c,kernel

def node_command(command,timeout=30):
 c,k=client();code,out=c.run_python(k,'import subprocess;p=subprocess.run('+repr(command)+',shell=True,capture_output=True,text=True,timeout='+str(timeout)+');print(p.stdout);print(p.stderr if p.returncode else "")',timeout=timeout+3)
 if code:raise RuntimeError('node380 command failed')
 return out.strip()

def upload_file(c,k,local,relative):
 """Bounded Contents-API chunks; never use another workflow's transfer kernel."""
 from urllib.parse import quote
 local=Path(local);wanted=sha(local);directory='faris/exploration-12h-20261017/upload-parts'
 for rel in ['faris','faris/exploration-12h-20261017',directory]:
  try:c.request('GET','api/contents/'+quote(rel,safe='/'))
  except urllib.error.HTTPError as e:
   if e.code!=404:raise
   c.request('PUT','api/contents/'+quote(rel,safe='/'),{'type':'directory'})
 total=0;part=ROOT/'spool/node-upload.part';part.parent.mkdir(exist_ok=True)
 with local.open('rb') as f:
  while data:=f.read(2*1024*1024):
   c.request('PUT','api/contents/'+directory+'/chunk_'+str(total),{'type':'file','format':'base64','content':base64.b64encode(data).decode()});total+=1
 snippet="""from pathlib import Path
import hashlib,os
parts=Path(PARTS);dest=Path(DEST);tmp=dest.with_suffix('.partial');digest=hashlib.sha256()
with tmp.open('wb') as f:
 for i in range(COUNT):
  block=(parts/('chunk_'+str(i))).read_bytes();f.write(block);digest.update(block)
assert digest.hexdigest()==WANTED
os.replace(tmp,dest)
for i in range(COUNT):(parts/('chunk_'+str(i))).unlink()
print('verified-upload')
""".replace('PARTS',repr('/home/taqiyudinmiftah/'+directory)).replace('DEST',repr('/home/taqiyudinmiftah/'+relative)).replace('COUNT',repr(total)).replace('WANTED',repr(wanted))
 status,out=c.run_python(k,snippet,timeout=60)
 if status:raise RuntimeError('Upload assembly failed: '+out[-500:])
 return {'sha256':wanted,'chunks':total,'bytes':local.stat().st_size}

def download_file(c,k,remote,local):
 from urllib.parse import quote
 local=Path(local);temporary=local.with_suffix('.partial');temporary.unlink(missing_ok=True)
 code="from pathlib import Path;import hashlib,json;p=Path("+repr(remote)+");h=hashlib.sha256();f=p.open('rb');[h.update(b) for b in iter(lambda:f.read(8*1024*1024),b'')];print(json.dumps({'size':p.stat().st_size,'sha256':h.hexdigest()}))"
 status,out=c.run_python(k,code,timeout=120)
 if status:raise RuntimeError('Download fingerprint failed')
 proof=json.loads(out);digest=hashlib.sha256();offset=0;chunkpath='faris/exploration-12h-20261017/download-chunk.bin'
 with temporary.open('wb') as f:
  while offset<proof['size']:
   if (ROOT/'clock.json').exists() and time.time()>=read(ROOT/'clock.json')['deadline']:raise TimeoutError('Collection deadline; immutable source retained on node380')
   code="from pathlib import Path;f=open("+repr(remote)+",'rb');f.seek("+str(offset)+");Path("+repr('/home/taqiyudinmiftah/'+chunkpath)+").write_bytes(f.read(2*1024*1024))"
   status,out=c.run_python(k,code,timeout=20)
   if status:raise RuntimeError('Download chunk failed')
   payload=c.request('GET','api/contents/'+quote(chunkpath,safe='/')+'?format=base64&type=file');block=base64.b64decode(payload['content']);assert block;f.write(block);digest.update(block);offset+=len(block)
 assert digest.hexdigest()==proof['sha256'];os.replace(temporary,local)

def download_directory(source,dest):
 c,k=client('transfer');remote='/home/taqiyudinmiftah/faris/exploration-12h-20261017/transfer-'+hashlib.sha256(source.encode()).hexdigest()[:20]+'.tar.gz'
 snippet='import tarfile;from pathlib import Path;p=Path('+repr(source)+');t=tarfile.open('+repr(remote)+',"w:gz",compresslevel=1);[t.add(x,arcname=str(x.relative_to(p)),recursive=False) for x in p.rglob("*") if x.is_file()];t.close()'
 status,out=c.run_python(k,snippet,timeout=180)
 if status:raise RuntimeError('Node immutable artifact packing failed')
 temp=Path(dest)/'.transfer.tar.gz';download_file(c,k,remote,temp)
 with tarfile.open(temp) as t:
  assert all((m.isfile() or m.islnk()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and (not m.islnk() or (not Path(m.linkname).is_absolute() and '..' not in Path(m.linkname).parts)) for m in t.getmembers());t.extractall(dest,filter='data')
 temp.unlink()
 c.run_python(k,'from pathlib import Path;Path('+repr(remote)+').unlink()',timeout=15)

def maybe_join(root,clock):
 root=Path(root)
 if (root/'node-admitted.json').exists() or time.time()>=clock['start']+8*3600:return
 try:
  c,k=client();code,out=c.run_python(k,'import subprocess,json,os;print(json.dumps({"gpu":subprocess.run(["nvidia-smi","--query-gpu=name,memory.free,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout,"cpu":os.cpu_count()}))',timeout=15)
  if code:raise RuntimeError('node probe failed')
  # Install/download exclusively in this node's new output directory before queue admission.
  archive=root/'node-payload.tar.gz'
  if not archive.exists():
   with tarfile.open(archive,'w:gz') as t:
    for name in ['source','evidence','toolkit','extras','host_prepare.py','bootstrap.py','deployment.json','clock.json','node_prepare.py','supervisor.py','node_join.py','rpc.py','archive_ops.py','launch.py','freeze.py','operate.py','service.py','source_manifest.json','management_manifest.json','reserved']:
     p=root/name
     for f in ([p] if p.is_file() else p.rglob('*')):
      if f.is_file() and '__pycache__' not in f.parts:t.add(f,arcname=str(f.relative_to(root)),recursive=False)
  receipt=upload_file(c,k,archive,'faris/exploration-12h-payload.tar.gz');write(root/'evidence/node-transfer.json',receipt)
  nr='/home/taqiyudinmiftah/faris/exploration-12h-20261017'
  snippet="""from pathlib import Path
import json,subprocess,os,tarfile
r=Path(NODE_ROOT);r.mkdir(exist_ok=True);a=Path('/home/taqiyudinmiftah/faris/exploration-12h-payload.tar.gz')
with tarfile.open(a) as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(r)
assert json.loads((r/'clock.json').read_text())==CLOCK
(r/'logs').mkdir(exist_ok=True);(r/'weights').mkdir(exist_ok=True)
old='/home/taqiyudinmiftah/faris/depth-20261017/env/bin/python'
env=os.environ.copy();env.update(HF_HOME='/home/taqiyudinmiftah/faris/kumo-large-node380-20261016/hf_cache',MPLBACKEND='Agg')
p=subprocess.Popen([old,str(r/'node_prepare.py')],cwd=r,env=env,stdout=(r/'logs/bootstrap.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'root':str(r),'deadline':CLOCK['deadline']}))
""".replace('NODE_ROOT',repr(nr)).replace('CLOCK',repr(clock))
  code,out=c.run_python(k,snippet,timeout=25)
  if code:raise RuntimeError('node bootstrap failed')
  write(root/'node-admitted.json',{'time':time.time(),'state':'preparing','detail':out,'same_deadline':clock['deadline']})
 except Exception as e:
  write(root/'node-status.json',{'time':time.time(),'state':'unavailable','error':type(e).__name__+': '+str(e)[-200:],'deadline':clock['deadline']})

if __name__=='__main__':maybe_join(ROOT,read(ROOT/'clock.json'))
