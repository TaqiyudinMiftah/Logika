"""Collect terminal audited finalist A and submit once during this authorized run."""
from pathlib import Path
import csv,datetime,hashlib,json,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/mnt/home/logika-faris-exploration/transfer-20261025'
HOST='faris@100.68.29.126'
plan=json.loads((ROOT/'FULL_REFIT_PLAN.json').read_text())
deadline=plan['submission_deadline_epoch']
local_status=ROOT/'COLLECT_SUBMIT_A_STATUS.json'

def status(phase,**extra):
    data=dict(phase=phase,updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    local_status.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data),flush=True)

status('waiting_for_independent_audit')
while time.time()<deadline:
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,
                           f'test -f {REMOTE}/FINAL_INDEPENDENT_AUDIT.json'],capture_output=True,timeout=30)
    if result.returncode==0:break
    time.sleep(20)
else:
    status('deadline_reached_without_audited_candidate')
    raise SystemExit(1)

status('collecting_audited_candidate')
paths=['FINAL_INDEPENDENT_AUDIT.json','FULL_REFIT_DELIVERY.json',plan['submission_filename'],'final']
subprocess.run(['scp','-r',*[f'{HOST}:{REMOTE}/{name}'for name in paths],str(ROOT)+'/'],check=True,timeout=180)
audit=json.loads((ROOT/'FINAL_INDEPENDENT_AUDIT.json').read_text())
delivery=json.loads((ROOT/'FULL_REFIT_DELIVERY.json').read_text())
file=ROOT/plan['submission_filename']
digest=hashlib.sha256(file.read_bytes()).hexdigest()
assert audit['status']=='passed'and delivery['status']=='passed'
assert digest==audit['submission_sha256']==delivery['submission_sha256']
with file.open(newline='')as f:
    reader=csv.DictReader(f);assert reader.fieldnames==['application_id','default_probability'];rows=list(reader)
with (ROOT.parents[2]/'dataset/sample_submission(3).csv').open(newline='')as f:
    sample=list(csv.DictReader(f))
assert [r['application_id']for r in rows]==[r['application_id']for r in sample]
assert len(rows)==len({r['application_id']for r in rows})==30000
assert all(0<float(r['default_probability'])<1 for r in rows)
status('audited_local_hash_and_schema_passed',sha256=digest,rows=len(rows))
subprocess.run([sys.executable,str(ROOT/'source/submit_verified.py'),'A'],check=True)
status('submission_command_complete',sha256=digest,receipt='kaggle_receipts/submit_A_response.json')
