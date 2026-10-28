"""Launch the second declared Engine replay only after the first replay qualifies."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/home/faris/logika-antigravity-refit-20261028')
legacy=ROOT/'pilots/legacy_fe_cov_fold0_attempt2/RESULT.json';start=time.monotonic()
while True:
    if legacy.exists():
        r=json.loads(legacy.read_text());assert r['passed'],'Legacy replay failed; concat not admitted';break
    status=subprocess.run(['systemctl','--user','is-active','logika-agy-replay-fecov-20261028-attempt2.service'],capture_output=True,text=True)
    assert status.stdout.strip()in{'active','activating'},'Legacy unit ended without qualifying result'
    assert time.monotonic()-start<1800,'Legacy replay wait exceeded bound'
    time.sleep(5)
while subprocess.run(['systemctl','--user','is-active','logika-agy-replay-fecov-20261028-attempt2.service'],capture_output=True,text=True).stdout.strip()in{'active','activating'}:time.sleep(2)
print(json.dumps({'stage':'legacy_gate_passed_launching_concat','legacy_max_abs':r['max_abs_prediction_error']}),flush=True)
subprocess.run([str(Path('/mnt/home/faris/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python')),str(ROOT/'source/pilot_replay.py'),'--model','champion90_s7731','--fold','0','--out',str(ROOT/'pilots/concat_champion90_s7731_fold0_attempt1')],check=True,timeout=1800)
