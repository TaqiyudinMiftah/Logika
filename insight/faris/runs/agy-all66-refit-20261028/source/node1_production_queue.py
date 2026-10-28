"""Continue only preassigned legacy production fits on idle physicalGPU1."""
import os,json,subprocess,sys,time,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
for model in ['fe_cov','fe_mat']:
 for fold in range(1,5):
  packet=f'{model}/fold{fold}';out=R/'predictions'/packet
  if(out/'COMPLETE.json').exists():continue
  if out.exists():raise RuntimeError(f'Preservefailedattempt: {out}')
  line=[x for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()if x.split(',')[0].strip()=='1'][0]
  _,mem,util=[int(x.strip())for x in line.split(',')]
  wait_start=time.monotonic()
  while mem>=256 or util>=5:
   if time.monotonic()-wait_start>14400:raise RuntimeError('GPUcapacitywaitexceeded; nootherworkinterfered')
   if (R/'STOP').exists():raise RuntimeError('OwnedstudySTOP')
   time.sleep(30)
   line=[x for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()if x.split(',')[0].strip()=='1'][0]
   _,mem,util=[int(x.strip())for x in line.split(',')]
  print(json.dumps({'stage':'production_start','packet':packet,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
  subprocess.run([sys.executable,str(R/'source/engine_packets.py'),'--packets',str(R/'packets'),'--out',str(R/'predictions'),'--checkpoint','/mnt/home/faris/logika-model-arena-20261025/storage/huggingface/hub/models--nvidia--Kumo-Tabular/blobs/cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613','--only',packet],check=True,timeout=1800);time.sleep(2)
print(json.dumps({'stage':'NODE1_ASSIGNED_PRODUCTION_COMPLETE'}),flush=True)
