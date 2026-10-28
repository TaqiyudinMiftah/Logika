"""Local setup driver. PC3 commands and transfers always originate on PC2."""
from __future__ import annotations
import json
from pathlib import Path
import shlex
import subprocess
import time
from .common import read,write,sha,code_hash

ID='meta-20261016-v1'
REL='faris/logika-meta-experiments/'+ID
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12']

def machine_config():
    return {'id':ID,'metric':'KS','direction':'higher','hours':12,
      'machines':{
       'pc1':{'ssh':'agung@100.89.211.119','port':22,'root':'/home/agung/'+REL,'workers':16,'cpu_quota':1600,'memory_gib':58,'admission_gib':54,'run_disk_gib':3,'opencl':True},
       'pc2':{'ssh':'faris@labkc-comm1','port':22,'root':'/home/faris/'+REL,'workers':7,'cpu_quota':700,'memory_gib':20,'admission_gib':17,'run_disk_gib':30,'gpu_uuid':'GPU-24a9fffc-77a7-e8dd-d49b-d6f06cb1966a'},
       'pc3':{'ssh':'faris@gpu-labkc.ub.ac.id','port':2003,'root':'/mnt/home/faris/'+REL,'workers':7,'cpu_quota':700,'memory_gib':20,'admission_gib':17,'run_disk_gib':20,'gpu_uuid':'GPU-598ded97-fa88-1d7a-6808-88b81fd4de4e'}}}

def remote(host,command,**kwargs):
    if host=='pc3':
        command=shlex.join(SSH+['-p','2003','faris@gpu-labkc.ub.ac.id',command]);target='faris@labkc-comm1'
    else:target='agung@100.89.211.119' if host=='pc1' else 'faris@labkc-comm1'
    p=subprocess.run(SSH+[target,command],**kwargs)
    if p.returncode:raise RuntimeError(host+' remote command failed: '+str(p.stderr or '')[-3000:])
    return p

def python_remote(host,code,**kwargs):return remote(host,'python3 -c '+shlex.quote(code),**kwargs)

def install_key(root):
    code=f'''from pathlib import Path
import subprocess,json,os
r=Path.home()/{REL!r}; d=r/'.credentials';d.mkdir(mode=0o700,exist_ok=True);d.chmod(0o700);key=d/'pc1_ed25519'
if not key.exists():subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-C','metaexp-{ID}','-f',str(key)],check=True)
key.chmod(0o600);print(key.with_suffix('.pub').read_text().strip())
'''
    pub=python_remote('pc2',code,capture_output=True,text=True).stdout.strip()
    command='cd /home/agung/'+REL+' && .venv/bin/python -m metaexp rpc'
    line='restrict,command="'+command+'" '+pub
    script=f'''from pathlib import Path
import os,json
root=Path.home()/{REL!r}; d=Path.home()/'.ssh';d.mkdir(mode=0o700,exist_ok=True);p=d/'authorized_keys'
old=p.read_text() if p.exists() else '';backup=root/'.authorized_keys.before'
if not backup.exists():backup.write_text(old);backup.chmod(0o600)
line={line!r}
if line not in old.splitlines():
 with p.open('a') as f:f.write((chr(10) if old and not old.endswith(chr(10)) else '')+line+chr(10))
p.chmod(0o600);print(json.dumps({{'installed':True,'public_key_comment':{('metaexp-'+ID)!r}}}))
'''
    python_remote('pc1',script,capture_output=True,text=True)
    write(root/'deployment_key.json',{'public_key':pub,'forced_command':command,'comment':'metaexp-'+ID})

def sync_source(root):
    files=['metaexp','tests','vendor','history_sources','requirements.txt','README.md','machines.json','catalogue.json','history_inventory.json','RULES.md','TRAINING_BRIEF.md']
    manifest={str(p.relative_to(root)):sha(p) for d in ['metaexp','tests','vendor','history_sources'] for p in sorted((root/d).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write(root/'source_manifest.json',manifest);files+=['source_manifest.json']
    for host in ['pc1','pc2']:
        target='agung@100.89.211.119' if host=='pc1' else 'faris@labkc-comm1'
        subprocess.run(['rsync','-a','--exclude=__pycache__',*[str(root/f) for f in files],target+':'+REL+'/'],check=True)
    remote('pc2','cd '+REL+' && rsync -a --exclude=__pycache__ -e '+shlex.quote('ssh -p 2003')+' '+shlex.join(files)+' faris@gpu-labkc.ub.ac.id:'+REL+'/',capture_output=True,text=True)

def transfer_bank():
    # Capture the exact builder which produced the bank, before any runtime edits.
    code=f'''from pathlib import Path
import json,hashlib,shutil,sys
r=Path.home()/{REL!r};b=r/'bank';m=json.loads((b/'manifest.json').read_text())
if 'builder_source_sha256' not in m:
 shutil.copyfile(r/'metaexp/features.py',b/'builder_source.py')
 h=hashlib.sha256((b/'builder_source.py').read_bytes()).hexdigest();m['builder_source_sha256']=h;m['files']['builder_source.py']=h
if 'lineage_policy_source_sha256' not in m:
 sys.path.insert(0,str(r));from metaexp.features import lineage
 for name,info in m['info'].items():
  if info.get('legacy'):info['sources']=lineage(name)
 shutil.copyfile(r/'metaexp/features.py',b/'lineage_policy_source.py');h=hashlib.sha256((b/'lineage_policy_source.py').read_bytes()).hexdigest()
 m['lineage_policy_source_sha256']=h;m['files']['lineage_policy_source.py']=h
 m['lineage_policy']='New features carry explicit sources. Legacy inferred lineage is conservative: ambiguous dependencies require every possibly involved source.'
m.pop('bank_id',None);m['bank_id']=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();(b/'manifest.json').write_text(json.dumps(m,indent=2)+chr(10))
print(m['bank_id'])
'''
    remote('pc1',REL+'/.venv/bin/python -c '+shlex.quote(code),capture_output=True,text=True)
    code=f'''from pathlib import Path
import json,subprocess,tarfile
from metaexp.coordinator import Transport
from metaexp.common import read,sha
r=Path.cwd();tx=Transport(r,read(r/'machines.json'));archive=r/'bank.transfer.tar.gz'
if not (r/'bank/manifest.json').exists():
 with archive.open('wb') as f:subprocess.run(tx.command('pc1'),input=json.dumps({{'action':'bank_export'}}).encode(),stdout=f,check=True,timeout=1200)
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar.getmembers():
   p=Path(member.name)
   if not member.isfile() or p.is_absolute() or '..' in p.parts or p.parts[0]!='bank':raise ValueError('Unsafe bank archive')
  tar.extractall(r,filter='data')
 archive.unlink()
m=read(r/'bank/manifest.json')
for file,h in m['files'].items():
 if sha(r/'bank'/file)!=h:raise ValueError('Changed bank '+file)
print(json.dumps({{'bank_id':m['bank_id'],'verified_files':len(m['files'])}}))
'''
    remote('pc2','cd '+REL+' && .venv/bin/python -c '+shlex.quote(code),capture_output=True,text=True,timeout=1500)
    remote('pc2','cd '+REL+' && rsync -a -e '+shlex.quote('ssh -p 2003')+' bank faris@gpu-labkc.ub.ac.id:'+REL+'/',capture_output=True,text=True,timeout=1200)

def execute_preflight(host):
    cfg=machine_config();m=cfg['machines'][host];unit='metaexp-preflight-'+ID+'-'+host+'-'+str(int(time.time()))
    args=['systemd-run','--user','--wait','--pipe','--unit='+unit,'--property=CPUQuota='+str(m['cpu_quota'])+'%',
          '--property=MemoryMax='+str(m['memory_gib'])+'G','--property=MemorySwapMax=0','--property=WorkingDirectory='+m['root'],
          m['root']+'/.venv/bin/python','-m','metaexp','preflight','--host',host,'--exercise']
    result=remote(host,shlex.join(args),capture_output=True,text=True,timeout=900)
    return result.stdout

def collect_preflights(root):
    for host in ['pc1','pc2']:
        target='agung@100.89.211.119' if host=='pc1' else 'faris@labkc-comm1'
        dest=root/'preflight_hosts'/host;dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a',target+':'+REL+'/preflight/',str(dest)+'/'],check=True)
    remote('pc2','cd '+REL+' && mkdir -p .preflight_collection/pc3 && rsync -a -e '+shlex.quote('ssh -p 2003')+' faris@gpu-labkc.ub.ac.id:'+REL+'/preflight/ .preflight_collection/pc3/',capture_output=True,text=True)
    dest=root/'preflight_hosts/pc3';dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','faris@labkc-comm1:'+REL+'/.preflight_collection/pc3/',str(dest)+'/'],check=True)
    publish_preflights(root)
    subprocess.run(['rsync','-a',*[str(root/'preflight'/f'{host}.json') for host in ['pc1','pc2','pc3']],'faris@labkc-comm1:'+REL+'/preflight/'],check=True)

def publish_preflights(root):
    # A host's own fresh proof is authoritative. Previously collected copies of
    # another host's proof must never overwrite it during consolidation.
    for host in ['pc1','pc2','pc3']:write(root/'preflight'/f'{host}.json',read(root/'preflight_hosts'/host/f'{host}.json'))

def deploy(root,workspace):
    root=Path(root).resolve();workspace=Path(workspace).resolve()
    from .design import design
    from .reporting import audit
    if not (root/'catalogue.json').exists():design(root)
    if not (root/'history_inventory.json').exists():audit(root,workspace)
    cfg=machine_config()
    # AMD success must be observed, never inferred from merely finding an ICD.
    result=remote('pc1','grep AMD_SMOKE_PASSED '+REL+'/amd_build_retry.log',capture_output=True,text=True)
    if 'AMD_SMOKE_PASSED' not in result.stdout:raise ValueError('AMD backend has not passed its real training smoke')
    write(root/'machines.json',cfg);sync_source(root);install_key(root);transfer_bank()
    for host in ['pc1','pc2','pc3']:
        print('Preflight '+host,flush=True);execute_preflight(host)
    collect_preflights(root)
    expected=code_hash(root)
    bank_ids=set();environments=[]
    for host in ['pc1','pc2','pc3']:
        p=read(root/'preflight'/f'{host}.json')
        if not p['passed'] or p['code_hash']!=expected:raise ValueError('Host preflight/source identity mismatch')
        bank_ids.add(p['bank_id'])
        environments.append({k:v for k,v in p['environment'].items() if k!='hostname'})
    if len(bank_ids)!=1:raise ValueError('Feature banks differ between machines')
    if any(e!=environments[0] for e in environments[1:]):raise ValueError('Locked runtime versions differ between machines')
    verified={'time':time.time(),'code_hash':expected,'bank_id':next(iter(bank_ids)),'environment':environments[0],'hosts':['pc1','pc2','pc3'],'preflight':'passed'}
    write(root/'deployment_verified.json',verified)
    subprocess.run(['rsync','-a',str(root/'deployment_verified.json'),'faris@labkc-comm1:'+REL+'/'],check=True)
    print('Deployment verified on all three machines. Launch runs on PC2.',flush=True)
