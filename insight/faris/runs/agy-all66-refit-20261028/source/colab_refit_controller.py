"""Fixed requested refit dispatch and compact result collection, with safe failure stops.

Uses only our account2 runtime. No recipe search and no uncontrolled retry.
"""
from pathlib import Path
import subprocess,json,time,hashlib,tarfile,shutil,datetime,sys
R=Path(__file__).resolve().parents[1];P=R/'private';cfg=str((P/'colab-sessions.json').resolve());name='logika-all66-retry2-20261028';prefix=['colab','--account','2','--config',cfg];gate=json.loads((R/'A100_REPLAY_GATE.json').read_text());assert gate['status']=='PASSED'
def record(stage,**kwargs):
 v={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,**kwargs};(R/'COLAB_CONTROLLER_STATE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
def call(args,label,timeout=300):
 out=P/(label+'.log')
 with out.open('w')as log:r=subprocess.run([*prefix,*args],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 assert r.returncode==0,('Colabactionfailed',label,'Inspectprivatelog');return out

def launch(stage,file,sha):
 adm={'stage':stage,'bundle':'/content/'+file.name,'bundle_sha256':sha,'A100_replay_gate':gate};path=P/('PRODUCTION_ADMISSION_'+stage+'.json');path.write_text(json.dumps(adm,indent=2)+'\n')
 call(['upload','--session',name,str(path),'/content/PRODUCTION_ADMISSION.json'],'admit-'+stage)
 call(['exec','--session',name,'--file',str(R/'source/colab_production_launch.py'),'--timeout','120'],'launch-'+stage,timeout=180)
 record('production_stage_launched',production_stage=stage)

def transfer(stage,file,sha,size):
 cmd=['python3',str(R/'source/transfer_colab_chunks.py'),'--file',str(file),'--sha256',sha,'--remote','/content/'+file.name,'--config',cfg,'--session',name,'--log',str(P/('chunks-'+stage+'-raw.log')),'--streaming','--bytes',str(size)]
 with(P/('chunks-'+stage+'-progress.log')).open('w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
 record('production_bundle_uploaded',production_stage=stage)

def wait_count(count):
 deadline=time.monotonic()+10800
 while time.monotonic()<deadline:
  log=call(['exec','--session',name,'--file',str(R/'source/colab_refit_status.py'),'--timeout','90'],'status',timeout=150)
  lines=[x.removeprefix('STATUS_JSON ')for x in log.read_text(errors='replace').splitlines()if x.startswith('STATUS_JSON ')]
  assert lines,'Statusreceiptmissing';v=json.loads(lines[-1]);assert not v['failures'],('Productionfailed',v['failures']);record('production_progress',**v)
  if v['complete_count']>=count:return v
  time.sleep(45)
 raise RuntimeError('Declaredexecutiontimeoutreached')

record('waiting_first_fold_chunk_transfer')
while True:
 text=(P/'chunk-transfer-f0-progress.log').read_text(errors='replace')
 if 'TRANSFER_COMPLETE'in text:break
 if 'Traceback'in text:raise RuntimeError('Firstfoldpackettransferfailed')
 time.sleep(3)
f=P/'production_fold0_A100.tar.gz';sha=hashlib.sha256(f.read_bytes()).hexdigest();launch('first_fold',f,sha)
record('transfer_remaining_main_packets_while_first_fold_fits')
f=P/'production_A100_all.tar.gz';sha='74c2768b422ff6f0617e71df6333a5b7b48dbac20266a6c8cb8e168f0e8a8c16';transfer('main',f,sha,844230572);wait_count(5);launch('main',f,sha)
# The two completedlabfold0fits are retained separately. The remaining8legacy
# fits were reassigned only after confirming the lab'sGPUs belongedtootherwork.
meta=json.loads((P/'A100_LEGACY_REASSIGNMENT.json').read_text());f=P/'production_A100_legacy_remaining.tar.gz';transfer('legacy',f,meta['sha256'],meta['bytes']);wait_count(35);launch('legacy',f,meta['sha256']);wait_count(43)
record('download_compact_predictions')
out=P/'A100_production_results_43.tar.gz';call(['download','--session',name,'/content/logika-all66/results_43.tar.gz',str(out)],'download-results',timeout=300)
folder=P/'A100_production_results';folder.mkdir(exist_ok=True)
with tarfile.open(out)as tar:tar.extractall(folder,filter='data')
complete=list((folder/'predictions').rglob('COMPLETE.json'));assert len(complete)==43
for p in complete:assert hashlib.sha256(p.with_name('prediction.npz').read_bytes()).hexdigest()==json.loads(p.read_text())['prediction_sha256']
remote='faris@100.68.29.126:/mnt/home/faris/logika-antigravity-refit-20261028/A100_production_results_43.tar.gz';subprocess.run(['scp',str(out),remote],check=True,timeout=300)
receipt=P/'A100_RESULTS_RECEIPT.json';receipt.write_text(json.dumps({'predictions':43,'bundle_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'all_prediction_hashes_verified':True},indent=2)+'\n')
subprocess.run(['scp',str(receipt),'faris@100.68.29.126:/mnt/home/faris/logika-antigravity-refit-20261028/A100_RESULTS_RECEIPT.json'],check=True,timeout=60)
# Stop only the runtime created by this task after locally verified collection.
call(['stop','--session',name],'stop-owned-runtime',timeout=180);record('COMPLETE_COMPACT_RESULTS_VERIFIED_AND_COPIED_TO_NODE1',predictions=43,result_bundle_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),owned_runtime_stopped=True)
