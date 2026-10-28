"""Two explicitly assigned GPU1 production fits, each with an1800sec bound."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/home/faris/logika-antigravity-refit-20261028');assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
for packet in ['fe_cov/fold0','fe_mat/fold0']:
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    line=[x for x in gpu.splitlines()if x.split(',')[0].strip()=='1'][0];_,mem,util=[int(x.strip())for x in line.split(',')];assert mem<256 and util<5,('GPU1 occupied; leaveunrelatedworkalone',line)
    print(json.dumps({'stage':'assigned_production_start','packet':packet,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'perfit_timeout_seconds':1800}),flush=True)
    subprocess.run([sys.executable,str(R/'source/engine_packets.py'),'--packets',str(R/'packets'),'--out',str(R/'predictions'),'--checkpoint','/mnt/home/faris/logika-model-arena-20261025/storage/huggingface/hub/models--nvidia--Kumo-Tabular/blobs/cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613','--only',packet],check=True,timeout=1800)
    time.sleep(2)
print(json.dumps({'stage':'ASSIGNED_PRODUCTION_COMPLETE'}),flush=True)
