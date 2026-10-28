"""Submit an independently audited CSV once; preserve an uncertain attempt."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
COMPETITION='data-science-competition-logika-ui-2026'
parser=argparse.ArgumentParser()
parser.add_argument('candidate',choices=['A','B'])
options=parser.parse_args()
directory=ROOT/('candidate_B'if options.candidate=='B'else'.')
plan=json.loads((directory/'FULL_REFIT_PLAN.json').read_text())
audit=json.loads((directory/'FINAL_INDEPENDENT_AUDIT.json').read_text())
delivery=json.loads((directory/'FULL_REFIT_DELIVERY.json').read_text())
file=directory/plan['submission_filename']
digest=hashlib.sha256(file.read_bytes()).hexdigest()
assert audit['status']=='passed'and delivery['status']=='passed'
assert digest==audit['submission_sha256']==delivery['submission_sha256']
assert datetime.datetime.now(datetime.timezone.utc).timestamp()<plan['submission_deadline_epoch']
with file.open(newline='')as f:
    reader=csv.DictReader(f)
    assert reader.fieldnames==['application_id','default_probability']
    rows=list(reader)
assert len(rows)==len({r['application_id']for r in rows})==30000
assert all(0<float(r['default_probability'])<1 for r in rows)
ledger_path=ROOT/'SUBMISSIONS.json'
ledger=json.loads(ledger_path.read_text())
intents=ROOT/'kaggle_receipts'/f'submit_{options.candidate}_intent.json'
assert not intents.exists(),'An earlier intent exists; inspect its remote status instead of repeating upload'
used=[r for r in ledger['records']if r['phase']in['submitting','uploaded','complete']]
assert len(used)<3 and not any(r['sha256']==digest for r in used)
message=(f'Logika {options.candidate}: {plan["variant"]}; {plan["averaging"]}; full70k; '
         f'OOF {plan["research_pooled_OOF_KS"]:.9f}; audited; sha256 {digest[:16]}')
intent=dict(candidate=options.candidate,competition=COMPETITION,file=str(file),sha256=digest,
            created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),description=message,
            rows=30000,full_refit_outer_validation_score=None,phase='submitting')
intents.parent.mkdir(parents=True,exist_ok=True)
intents.write_text(json.dumps(intent,indent=2)+'\n')
record=next((r for r in ledger['records']if r['candidate']==options.candidate),None)
if record is None:
    record=dict(candidate=options.candidate,filename=file.name,sha256=digest,recipe_OOF_KS=plan['research_pooled_OOF_KS'],public_score=None)
    ledger['records'].append(record)
record.update(phase='submitting',intent_path=str(intents),description=message)
ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
arguments=[sys.executable,str(ROOT/'source/kaggle_private_cli.py'),'--record',f'submit_{options.candidate}_response.json','--',
           'competitions','submit',COMPETITION,'-f',str(file),'-m',message,'--wait','300','--poll-interval','20']
result=subprocess.run(arguments)
raise SystemExit(result.returncode)
