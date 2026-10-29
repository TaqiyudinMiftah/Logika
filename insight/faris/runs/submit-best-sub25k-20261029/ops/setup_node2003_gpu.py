from pathlib import Path
import subprocess,json,os,time,traceback
R=Path('/mnt/home/faris/logika-submit-best-sub25k-20261029');u=str(R/'private/uv-bootstrap/bin/uv');py=str(R/'env_gpu/bin/python')
try:
 subprocess.run([u,'venv','--python','/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python',str(R/'env_gpu')],check=True,timeout=120)
 subprocess.run([u,'pip','install','--python',py,'--index-url','https://download.pytorch.org/whl/cu126','torch==2.7.1'],check=True,timeout=900)
 subprocess.run([u,'pip','install','--python',py,'structured-data-models==0.2.1','numpy==2.2.6','pandas==2.3.3','pyarrow==24.0.0','scipy==1.16.3','scikit-learn==1.7.2','huggingface-hub==0.35.3','safetensors==0.6.2'],check=True,timeout=600)
 code='import torch,sdm,json,importlib.metadata as m;assert torch.cuda.is_available();print(json.dumps({"Torch":torch.__version__,"SDM":m.version("structured-data-models"),"GPU":torch.cuda.get_device_name(0)}))'
 result=subprocess.run([py,'-c',code],check=True,capture_output=True,text=True,timeout=60)
 (R/'NODE2003_ENVIRONMENT.json').write_text(json.dumps(dict(status='PASS',epoch=time.time(),measured=json.loads(result.stdout)),indent=2)+'\n')
 c=json.load(open(R/'DEPLOYMENT_CLOCK.json'));assert time.time()<c['deadline_epoch']-600
 subprocess.run(['systemd-run','--user','--quiet','--unit=logika-submit25-controller-2003-20261029','--setenv=LOGIKA_REFIT_GPU_PYTHON='+py,'--setenv=LOGIKA_REFIT_RAW_ONLY=1','--property=MemoryMax=2G','--property=CPUQuota=100%','--property=RuntimeMaxSec='+str(int(c['deadline_epoch']-time.time())),'--property=StandardOutput=append:'+str(R/'logs/controller2003.log'),'--property=StandardError=append:'+str(R/'logs/controller2003.log'),'/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python','-u',str(R/'ops/refit_controller.py')],check=True)
 print('Pinned CUDA environment verified; fixed deployment started under original deadline',flush=True)
except Exception:
 (R/'NODE2003_SETUP_FAILURE_PRIVATE.log').write_text(traceback.format_exc());raise
