"""Explicit setup driver; PC3 SSH always originates on PC2."""
import argparse,concurrent.futures,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'logika-meta-experiments'))
from metaexp.deployment import remote,SSH
ID=ROOT.name
PATHS={'pc1':'/home/agung/faris/'+ID,'pc2':'/mnt/home/logika-faris-exploration/'+ID,'pc3':'/mnt/home/faris/faris/'+ID}
OLD={'pc1':'/home/agung/faris/','pc2':'/home/faris/faris/','pc3':'/mnt/home/faris/faris/'}
def run(host,code,timeout=50):return remote(host,'python3 -c '+shlex.quote(code),capture_output=True,text=True,timeout=timeout).stdout

def setup(host):
 root=PATHS[host];old=OLD[host];code=f'''from pathlib import Path
import os,subprocess,json
r=Path({root!r});r.mkdir(parents=True,exist_ok=True)
for n in ['source','evidence','raw','extras','archive','spool','logs']: (r/n).mkdir(exist_ok=True)
old=Path({old!r})
python=old/'depth-20261017/.venv/bin/python';uv=old/'logika-meta-experiments/meta-20261016-v1/.tools/uv'
assert python.exists() and uv.exists()
subprocess.run([str(uv),'venv','--python',str(python),str(r/'.venv')],check=True)
# All downloads/caches stay inside the dedicated campaign storage.
env=os.environ.copy();env['UV_CACHE_DIR']=str(r/'.uv-cache')
script='set -e\\n'+str(uv)+' pip install --python '+str(r/'.venv/bin/python')+' numpy==1.26.4 pandas==1.5.3 scipy==1.14.1 scikit-learn==1.6.1 pyarrow==22.0.0 lightgbm==4.6.0 catboost==1.2.8 xgboost==3.0.5 joblib==1.4.2 optuna==4.5.0 psutil==7.2.2 cloudpickle==3.1.1 interpret-core==0.7.5\\n'
if {host!r}!='pc1':
 script+=str(uv)+' pip install --python '+str(r/'.venv/bin/python')+' torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126\\n'
 script+=str(uv)+' pip install --python '+str(r/'.venv/bin/python')+' tabm rtdl_num_embeddings\\n'
script+=str(r/'.venv/bin/python')+' -c "from pathlib import Path;Path(\\''+str(r/'environment-ready')+'\\').write_text(\\'ready\\')"\\n'
p=subprocess.Popen(['/bin/bash','-c',script],env=env,stdout=(r/'logs/install.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'root':str(r),'pid':p.pid}}))
'''
 return host,run(host,code)

def sync():
 for host in ['pc1','pc2']:
  target='agung@100.89.211.119' if host=='pc1' else 'faris@labkc-comm1'
  subprocess.run(['rsync','-a','--exclude=__pycache__',str(ROOT/'source'),str(ROOT/'evidence'),str(ROOT/'toolkit'),str(ROOT/'host_prepare.py'),str(ROOT/'supervisor.py'),str(ROOT/'node_join.py'),str(ROOT/'bootstrap.py'),str(ROOT/'rpc.py'),str(ROOT/'archive_ops.py'),str(ROOT/'node_prepare.py'),str(ROOT/'launch.py'),str(ROOT/'freeze.py'),str(ROOT/'operate.py'),str(ROOT/'service.py'),str(ROOT/'source_manifest.json'),str(ROOT/'management_manifest.json'),target+':'+PATHS[host]+'/'],check=True)
 remote('pc2','rsync -a -e '+shlex.quote('ssh -p 2003')+' '+shlex.join([PATHS['pc2']+'/'+n for n in ['source','evidence','toolkit','host_prepare.py','supervisor.py','node_join.py','bootstrap.py','rpc.py','archive_ops.py','node_prepare.py','launch.py','freeze.py','operate.py','service.py','source_manifest.json','management_manifest.json']])+' faris@gpu-labkc.ub.ac.id:'+PATHS['pc3']+'/',capture_output=True,text=True,timeout=120)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['environments','sync']);a=p.parse_args()
 if a.action=='environments':
  with concurrent.futures.ThreadPoolExecutor(3) as pool:
   for h,out in pool.map(setup,['pc1','pc2','pc3']):print(h,out)
 else:sync()
