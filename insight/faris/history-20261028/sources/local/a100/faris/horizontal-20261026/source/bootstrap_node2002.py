"""Replicate the exact numerical runtime and immutable inputs into node2002's home."""
from pathlib import Path
import datetime,hashlib,json,subprocess,traceback
R=Path(__file__).resolve().parents[1]
DEST='/home/faris/logika-horizontal-20261026'
SOCK=R/'runtime/node2002/ssh.sock'
SSH=['ssh','-S',str(SOCK),'-p','2002','faris@gpu-labkc.ub.ac.id']
E=f'ssh -S {SOCK} -p 2002'
def state(stage,**kw):
    record={'stage':stage,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
    (R/'NODE2002_BOOTSTRAP.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
def remote(cmd):return subprocess.run(SSH+[cmd],check=True,text=True,capture_output=True,timeout=300)
def copy(src,dest):
    state('copying',source=str(src),destination=dest)
    remote('mkdir -p '+DEST+'/'+dest)
    with (R/'node2002_transfer.log').open('a') as log:
        subprocess.run(['rsync','-aL','--partial','-e',E,str(src).rstrip('/')+'/',
                        'faris@gpu-labkc.ub.ac.id:'+DEST+'/'+dest+'/'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
def main():
    plan=json.loads((R/'PLAN.json').read_text())
    assert datetime.datetime.now().timestamp()<plan['search_deadline_epoch']
    copy('/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/.python/cpython-3.11.13-linux-x86_64-gnu','runtime/python')
    copy('/mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/.venv/lib/python3.11/site-packages','runtime/site-base')
    copy('/mnt/home/faris/logika-faris-exploration/ranking-pipeline-20261020/.venv/lib/python3.11/site-packages','runtime/site-models')
    bank=Path('/mnt/home/faris/faris/logika-meta-experiments/meta-20261016-v1/bank')
    remote('mkdir -p '+DEST+'/bank')
    subprocess.run(['rsync','-a','-e',E,*[str(bank/n) for n in ['numeric.npy','categorical.parquet','metadata.parquet']],
                    'faris@gpu-labkc.ub.ac.id:'+DEST+'/bank/'],check=True,timeout=300)
    copy(R/'packets','packets');copy(R/'source','source')
    metadata=['PLAN.json','CATALOG.json','BANK_MANIFEST.json','ANCHOR_SOURCE.json','DATA_GATE.json','PILOT_GATE.json','FORMULA_GATE.json','REPAYMENT_TIMING_AUDIT.json','EXTRA_SOURCES.json','extra.parquet','SOURCE_LEDGER.json','SYNTHESIS_GATE.json']
    subprocess.run(['rsync','-a','-e',E,*[str(R/n) for n in metadata],'faris@gpu-labkc.ub.ac.id:'+DEST+'/'],check=True,timeout=180)
    revision='b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f'
    cp=f'hub/models--nvidia--Kumo-Tabular/snapshots/{revision}'
    expected={}
    for size in ['small','medium','large']:
        source=Path('/mnt/home/faris/logika-model-arena-20261025/storage/huggingface')/cp/size/'classifier.pt'
        expected[size]=hashlib.sha256(source.read_bytes()).hexdigest()
        remote('mkdir -p '+DEST+'/cache/'+cp+'/'+size)
        subprocess.run(['rsync','-aL','-e',E,str(source),'faris@gpu-labkc.ub.ac.id:'+DEST+'/cache/'+cp+'/'+size+'/classifier.pt'],check=True,timeout=300)
    remote(DEST+'/runtime/python/bin/python3.11 -m venv '+DEST+'/runtime/env')
    verify='''
from pathlib import Path
import hashlib,json,sys
root=Path(DESTINATION)
site=root/'runtime/env/lib/python3.11/site-packages'
(site/'logika_runtime.pth').write_text(str(root/'runtime/site-models')+'\\n'+str(root/'runtime/site-base')+'\\n')
manifest=json.loads((root/'BANK_MANIFEST.json').read_text())
for name in ['numeric.npy','categorical.parquet','metadata.parquet']:
 h=hashlib.sha256()
 with (root/'bank'/name).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==manifest['files'][name],name
for size,want in CHECKPOINTS.items():
 checkpoint=root/'cache'/CHECKPOINT_RELATIVE/size/'classifier.pt'
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==want,size
print('All bank and checkpoint hashes verified')
'''.replace('DESTINATION',repr(DEST)).replace('CHECKPOINTS',repr(expected)).replace('CHECKPOINT_RELATIVE',repr(cp))
    result=subprocess.run(SSH+[DEST+'/runtime/env/bin/python -'],input=verify,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,(result.stdout,result.stderr)
    package=remote(DEST+'/runtime/env/bin/python -c "import importlib.metadata as m,torch,sdm; print({k:m.version(k) for k in [\'torch\',\'structured-data-models\',\'numpy\',\'pandas\']}); print(torch.cuda.get_device_name(0))"')
    state('inputs_and_runtime_verified',verification=result.stdout,packages=package.stdout,checkpoint_sha256=expected,
          destination=DEST,model_weights_saved=False)
if __name__=='__main__':
    try:main()
    except BaseException:state('failed',error=traceback.format_exc());raise
