"""Stage, verify and launch a separate research wave on the parent's clock."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from metaexp.common import read,write,sha,code_hash
from metaexp.deployment import remote,machine_config,REL,SSH

LOCAL=Path(__file__).resolve().parent;SOURCE=LOCAL/'extensions/wave2';ID='meta-20261016-v1-wave2';NEWREL='faris/logika-meta-experiments/'+ID

def run(host,code,timeout=120):
    base=machine_config()['machines'][host]['root']
    return remote(host,shlex.quote(base+'/.venv/bin/python')+' -c '+shlex.quote(code),capture_output=True,text=True,timeout=timeout).stdout

def stage():
    for host in ['pc1','pc2','pc3']:
        code=f'''from pathlib import Path
r=Path({machine_config()['machines'][host]['root']!r});n=r.with_name({ID!r});n.mkdir(exist_ok=True)
for name in ['bank','.venv','vendor']:
 p=n/name
 if not p.exists():p.symlink_to(r/name,target_is_directory=True)
if {host!r}=='pc2':
 p=n/'.credentials'
 if not p.exists():p.symlink_to(r/'.credentials',target_is_directory=True)
print(n)
'''
        print(run(host,code).strip(),flush=True)
    for host,target in [('pc1','agung@100.89.211.119'),('pc2','faris@labkc-comm1')]:
        subprocess.run(['rsync','-a','--exclude=__pycache__',str(SOURCE)+'/',target+':'+NEWREL+'/'],check=True)
    remote('pc2','cd '+NEWREL+' && rsync -a --exclude=__pycache__ -e '+shlex.quote('ssh -p 2003')+' metaexp tests *.py faris@gpu-labkc.ub.ac.id:'+NEWREL+'/',capture_output=True,text=True,timeout=180)

def prepare():
    old=machine_config();machines=old['machines']
    for host in machines:machines[host]['root']=str(Path(machines[host]['root']).with_name(ID))
    machines['pc2']['run_disk_gib']=48
    code=f'''from pathlib import Path
import json,sqlite3,time,subprocess,os
import sys
p=Path.home()/{REL!r};r=p.with_name({ID!r});sys.path.insert(0,str(r))
from metaexp.common import read,write,sha,code_hash
if (r/'launch.json').exists():raise ValueError('Already launched; preparation cannot replace an existing research campaign')
cfg=read(p/'campaign.json');old=read(p/'finalists.json')
if time.time()>=cfg['started']+9.5*3600:raise ValueError('Search reserve already expired')
if not (p/'reports/confirmation.json').exists():raise ValueError('Parent confirmation statistics are not checkpointed')
db=sqlite3.connect(p/'ledger.sqlite')
if db.execute("select count(*) from tasks where state='assigned'").fetchone()[0]:raise ValueError('Parent work still active')
unit='metaexp-'+cfg['id']+'-coordinator';pid=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True))
if pid:
 cmd=Path('/proc',str(pid),'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 if str(p) not in cmd:raise ValueError('Parent process ownership guard failed')
 subprocess.run(['systemctl','--user','stop',unit],check=True,timeout=40)
write(r/'parent_evidence.json',{{'parent_id':cfg['id'],'started':cfg['started'],'deadline':cfg['deadline'],'configuration_sha256':sha(p/'campaign.json'),'catalogue_sha256':sha(p/'catalogue.json'),'finalists_sha256':sha(p/'finalists.json'),'confirmation_sha256':sha(p/'reports/confirmation.json'),'parent_source_hash':code_hash(p),'parent_nomination':old['primary']}})
new=dict(cfg);new.update(id={ID!r},parent_campaign=cfg['id'],parent_root=str(p),research_only=True,machines={machines!r},code_hash=code_hash(r))
write(r/'machines.json',{{'id':{ID!r},'machines':new['machines']}})
from design_extension import design
c=design(r,p,4096);new['catalogue_sha256']=sha(r/'catalogue.json');write(r/'campaign.json',new)
sources={{str(f.relative_to(r)):sha(f) for f in list((r/'metaexp').glob('*.py'))+list(r.glob('*.py'))+list((r/'tests').glob('*.py'))}}
write(r/'source_manifest.json',sources)
revision={{'kind':'development_only_extension','execution':'autonomous_per_node','started':cfg['started'],'deadline':cfg['deadline'],'training_code_hash':code_hash(r),
 'script_sha256':sha(r/'scheduler_parallel.py'),'rpc_sha256':sha(r/'rpc_parallel.py'),'node_executor_sha256':sha(r/'node_executor.py'),'shards_sha256':sha(r/'autonomous_shards.py'),
 'extension_sources':{{n:sha(r/n) for n in ['research_coordinator.py','research_worker.py','design_extension.py']}}}}
write(r/'scheduler_revision.json',revision)
samples={{}}
for tid,task in db.execute("select id,task from tasks where verified=1 and mode='cv'"):
 score=read(p/'runs'/tid/'scores.json');samples.setdefault(score['model']+':'+score['device'],[]).append(score['elapsed_seconds'])
write(r/'runtime_history.json',{{'parent_ledger':'Parent verified CV timing only; no confirmation metrics','samples':samples}})
write(p/'research_extension.json',{{'id':new['id'],'root':str(r),'deadline':cfg['deadline'],'parent_report':'Rebuilt under the new capped PC2 service if unfinished','new_pipelines':c['new_pipeline_recipes']}})
print(json.dumps({{'catalogue':c['new_pipeline_recipes'],'source_hash':new['code_hash'],'deadline':new['deadline']}}))
'''
    print(run('pc2',code,timeout=600).strip(),flush=True)
    files=['campaign.json','machines.json','catalogue.json','scheduler_revision.json','source_manifest.json','parent_evidence.json','design_summary.json','runtime_history.json']
    subprocess.run(['rsync','-a',*['faris@labkc-comm1:'+NEWREL+'/'+f for f in files],str(SOURCE)+'/'],check=True)
    subprocess.run(['rsync','-a',*[str(SOURCE/f) for f in files],'agung@100.89.211.119:'+NEWREL+'/'],check=True)
    remote('pc2','cd '+NEWREL+' && rsync -a -e '+shlex.quote('ssh -p 2003')+' '+shlex.join(files)+' faris@gpu-labkc.ub.ac.id:'+NEWREL+'/',capture_output=True,text=True,timeout=180)

def preflight():
    def one(host):
        m=read(SOURCE/'machines.json')['machines'][host];root=m['root'];unit='metaexp-wave2-preflight-'+host
        command=[root+'/.venv/bin/python','-u',root+'/extension_preflight.py','--host',host]
        args=['systemd-run','--user','--wait','--pipe','--collect','--unit='+unit,'--property=CPUQuota='+str(m['cpu_quota'])+'%',
              '--property=MemoryMax='+str(m['memory_gib'])+'G','--property=MemorySwapMax=0','--property=WorkingDirectory='+root,
              '--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',*command]
        result=remote(host,shlex.join(args),capture_output=True,text=True,timeout=600)
        print(host+' preflight passed',flush=True);return host,result.stdout
    with ThreadPoolExecutor(max_workers=3) as pool:
        for host,output in pool.map(one,['pc1','pc2','pc3']):(SOURCE/(host+'_preflight.log')).write_text(output)
    remote('pc2','cd '+NEWREL+' && mkdir -p preflight && rsync -a -e '+shlex.quote('ssh -p 2003')+' faris@gpu-labkc.ub.ac.id:'+NEWREL+'/preflight/pc3.json preflight/',capture_output=True,text=True,timeout=60)
    subprocess.run(['rsync','-a','agung@100.89.211.119:'+NEWREL+'/preflight/pc1.json',str(SOURCE)+'/'],check=True)
    subprocess.run(['rsync','-a',str(SOURCE/'pc1.json'),'faris@labkc-comm1:'+NEWREL+'/preflight/'],check=True)

def seal():
    files=[SOURCE/'metaexp',SOURCE/'tests',*SOURCE.glob('*.py')]
    for target in ['agung@100.89.211.119','faris@labkc-comm1']:
        subprocess.run(['rsync','-a','--exclude=__pycache__',*[str(f) for f in files],target+':'+NEWREL+'/'],check=True)
    remote('pc2','cd '+NEWREL+' && rsync -a --exclude=__pycache__ -e '+shlex.quote('ssh -p 2003')+' metaexp tests *.py faris@gpu-labkc.ub.ac.id:'+NEWREL+'/',capture_output=True,text=True,timeout=180)
    code=f'''from pathlib import Path
import sys
r=Path.home()/{NEWREL!r};sys.path.insert(0,str(r))
from metaexp.common import read,write,sha,code_hash
if (r/'launch.json').exists():raise ValueError('Cannot reseal a launched wave')
cfg=read(r/'campaign.json');cfg['code_hash']=code_hash(r);write(r/'campaign.json',cfg)
revision=read(r/'scheduler_revision.json');revision['training_code_hash']=cfg['code_hash']
for key,name in [('script_sha256','scheduler_parallel.py'),('rpc_sha256','rpc_parallel.py'),('node_executor_sha256','node_executor.py'),('shards_sha256','autonomous_shards.py')]:revision[key]=sha(r/name)
revision['extension_sources']={{n:sha(r/n) for n in ['research_coordinator.py','research_worker.py','design_extension.py']}}
write(r/'scheduler_revision.json',revision)
write(r/'source_manifest.json',{{str(f.relative_to(r)):sha(f) for f in list((r/'metaexp').glob('*.py'))+list(r.glob('*.py'))+list((r/'tests').glob('*.py'))}})
print(cfg['code_hash'])
'''
    print(run('pc2',code).strip(),flush=True)
    names=['campaign.json','scheduler_revision.json','source_manifest.json']
    subprocess.run(['rsync','-a',*['faris@labkc-comm1:'+NEWREL+'/'+f for f in names],str(SOURCE)+'/'],check=True)
    subprocess.run(['rsync','-a',*[str(SOURCE/f) for f in names],'agung@100.89.211.119:'+NEWREL+'/'],check=True)
    remote('pc2','cd '+NEWREL+' && rsync -a -e '+shlex.quote('ssh -p 2003')+' '+shlex.join(names)+' faris@gpu-labkc.ub.ac.id:'+NEWREL+'/',capture_output=True,text=True,timeout=120)

def launch():
    public=run('pc2',f"from pathlib import Path;print((Path.home()/{REL!r}/'.credentials/pc1_ed25519.pub').read_text().strip())").strip().split()
    key=' '.join(public[:2]);command='cd /home/agung/'+NEWREL+' && .venv/bin/python rpc_parallel.py'
    line='restrict,command="'+command+'" '+key+' metaexp-'+ID
    print(run('pc1',f'''from pathlib import Path
p=Path.home()/'.ssh/authorized_keys';old=p.read_text();key={public[1]!r};line={line!r}
p.write_text(chr(10).join(v for v in old.splitlines() if key not in v.split())+chr(10)+line+chr(10));p.chmod(0o600)
print('Existing dedicated campaign key restricted to new research root')
''').strip(),flush=True)
    code=f'''from pathlib import Path
import sys,json,time,subprocess
r=Path.home()/{NEWREL!r};sys.path.insert(0,str(r))
from metaexp.common import read,write,sha,code_hash
cfg=read(r/'campaign.json');proof=read(r/'parent_evidence.json');p=Path(cfg['parent_root'])
if time.time()>=cfg['started']+9.5*3600:raise ValueError('Search phase expired')
for name,key in [('campaign.json','configuration_sha256'),('catalogue.json','catalogue_sha256'),('finalists.json','finalists_sha256'),('reports/confirmation.json','confirmation_sha256')]:
 if sha(p/name)!=proof[key]:raise ValueError('Parent evidence changed')
env=[]
for host in ['pc1','pc2','pc3']:
 check=read(r/'preflight'/f'{{host}}.json')
 if not check['passed'] or check['code_hash']!=cfg['code_hash'] or check['bank_id']!=cfg['bank_id']:raise ValueError('Missing source/data preflight '+host)
 if not check.get('original_acceptance_tests_passed') or not check.get('extension_guard_tests_passed'):raise ValueError('Missing acceptance or holdout-guard tests '+host)
 if host=='pc2' and not check.get('guarded_complete_three_fold_worker'):raise ValueError('Missing guarded full-CV verification')
 env.append({{k:v for k,v in check['environment'].items() if k!='hostname'}})
if any(v!=env[0] for v in env):raise ValueError('Locked environments differ')
for name,h in read(r/'source_manifest.json').items():
 if sha(r/name)!=h:raise ValueError('Source changed after preflight')
unit='metaexp-'+cfg['id']+'-coordinator';condition=str(r/'.venv/bin/python')+' -c '+__import__('shlex').quote('import time,sys;sys.exit(0 if time.time()<'+repr(cfg['deadline'])+' else 1)')
if subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip()=='active':raise ValueError('Already active; refusing duplicate launch')
args=['systemd-run','--user','--unit='+unit,'--property=CPUQuota=700%','--property=MemoryMax=20G','--property=MemorySwapMax=0','--property=OOMScoreAdjust=1000',
 '--property=Restart=on-failure','--property=RestartSec=5','--property=KillMode=control-group','--property=WorkingDirectory='+str(r),'--property=ExecCondition='+condition,
 '--property=RuntimeMaxSec='+str(int(cfg['deadline']-time.time()+5)),str(r/'.venv/bin/python'),'-u',str(r/'research_coordinator.py'),'--root',str(r)]
subprocess.run(['systemctl','--user','reset-failed',unit],capture_output=True);result=subprocess.run(args,capture_output=True,text=True,timeout=20)
if result.returncode:raise RuntimeError(result.stderr)
write(r/'launch.json',{{'unit':unit,'command':args,'started':cfg['started'],'deadline':cfg['deadline'],'wave_launched':time.time()}})
print(json.dumps({{'launched':True,'unit':unit,'same_deadline':cfg['deadline']}}))
'''
    print(run('pc2',code,timeout=60).strip(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stage','prepare','seal','preflight','launch']);a=p.parse_args();globals()[a.command]()
