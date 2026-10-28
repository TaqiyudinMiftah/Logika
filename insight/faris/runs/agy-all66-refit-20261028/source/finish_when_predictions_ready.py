"""Verify returned compact A100 artifacts, assemble fixed folds and verify requested CSV."""
from pathlib import Path
import time,json,tarfile,hashlib,subprocess,sys,datetime
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def state(stage,**kw):
 v={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':stage,**kw};(R/'FINALIZE_STATE.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
state('waiting_A100_verified_result_receipt');deadline=time.monotonic()+18000
while not(R/'A100_RESULTS_RECEIPT.json').exists():
 if (R/'STOP').exists():raise RuntimeError('OwnedstudySTOP')
 if time.monotonic()>deadline:raise RuntimeError('A100resultwaitexceeded')
 time.sleep(15)
receipt=json.loads((R/'A100_RESULTS_RECEIPT.json').read_text());bundle=R/'A100_production_results_43.tar.gz';assert sha(bundle)==receipt['bundle_sha256']and receipt['all_prediction_hashes_verified']
private=R/'A100_returned';private.mkdir(exist_ok=True)
with tarfile.open(bundle)as tar:
 for member in tar.getmembers():assert not Path(member.name).is_absolute()and'..'not in Path(member.name).parts and(member.isfile()or member.isdir())
 tar.extractall(private,filter='data')
import shutil
for p in(private/'predictions').rglob('COMPLETE.json'):
 v=json.loads(p.read_text());assert sha(p.with_name('prediction.npz'))==v['prediction_sha256'];rel=p.parent.relative_to(private/'predictions');dest=R/'predictions'/rel
 assert not dest.exists(),('Nooverwritingexistingattempts',dest);shutil.copytree(p.parent,dest)
assert len(list((R/'predictions').rglob('COMPLETE.json')))==45
state('assemble_fixed_test_folds',completed_component_fits=45)
for fold in range(5):
 done=R/f'assembly/production/fold{fold}/complete.json'
 if done.exists():assert json.loads(done.read_text())['status']=='passed';continue
 cmd=[sys.executable,str(R/'source/assembly_runner.py'),'assemble','--fold',str(fold),'--features',str(R/f'features/fold{fold}/expanded.parquet'),'--features70',str(R/'features70/expanded.parquet'),'--prediction-root',str(R/'predictions')]
 subprocess.run(cmd,check=True,timeout=1200);state('assembled_fold',fold=fold)
subprocess.run([sys.executable,str(R/'source/assembly_runner.py'),'finish'],check=True,timeout=120)
verification=json.loads((R/'assembly/CSV_VERIFICATION.json').read_text());assert verification['status']=='passed'and verification['rows']==30000
state('CSV_COMPLETE_VERIFIED',verification=verification,independent_gain_claimed=False)
