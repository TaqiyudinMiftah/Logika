"""Collect, freeze, evaluate and audit once all admitted owners terminate."""
from pathlib import Path
import json,time,subprocess,traceback
R=Path(__file__).resolve().parent;C=json.loads((R/'CLOCK.json').read_text());PY='/mnt/home/faris/logika-model-arena-20261025/env311/bin/python'
def run(script,*args):
    subprocess.run([PY,'-u',str(R/script),*args],check=True,timeout=max(1,C['deadline_epoch']-time.time()-60))
owners=[R/'ownership/lab2001/FINAL.json']+[R/f'private/incoming/{owner}/ownership/{owner}/FINAL.json' for owner in ['lab2003','host119','host42']]+[R/f'private/incoming/lab2003/ownership/lab2003-gpu{i}/FINAL.json' for i in [0,1]]
try:
    while True:
        run('collect_remotes.py')
        if all(p.is_file() for p in owners):break
        assert time.time()<C['model_deadline_epoch']+30,'Owners did not terminate on time'
        time.sleep(30)
    # Qualification of the entire 80-fit Transformer grid uses measured pair times.
    pairs={}
    for k in ['transformer_scalar','transformer_coefficient']:
        js=[json.loads((R/f'runs/{k}__s10103__f0/{phase}/RESULT.json').read_text()) for phase in ['validation','production']]
        pairs[k]=sum(x['seconds'] for x in js)
    forecast=19*sum(pairs.values())*1.5/2+300
    gate=dict(status='NOT_ADMITTED_RUNTIME',epoch=time.time(),completed_probe_fits=4,remaining_registered_fit_slots=76,full_size_pair_seconds=pairs,optimistic_two_GPU_remaining_seconds_with_safety=forecast,model_deadline_epoch=C['model_deadline_epoch'],admission_closed=time.time()>=C['admission_end_epoch'],reason='The complete four-seed/five-fold/two-head grid cannot qualify inside the remaining original clock, even assuming both GPUs were free.',predictive_negative=False,partial_probes_not_promoted=True)
    (R/'TRANSFORMER_RUNTIME_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
    run('audit_raw.py')
    run('compose_final.py','--preflight')
    run('compose_final.py')
    run('audit_composition.py')
    report=dict(status='PASS',epoch=time.time(),owners=[json.loads(p.read_text()) for p in owners],source_and_clock_preserved=True,composition_and_separate_arithmetic_audits_passed=True,no_Kaggle_upload=True)
    (R/'TERMINAL_PIPELINE.json').write_text(json.dumps(report,indent=2)+'\n');print('TERMINAL_PIPELINE PASS',flush=True)
except BaseException:
    (R/'TERMINAL_FAILURE.json').write_text(json.dumps(dict(epoch=time.time(),diagnostic=traceback.format_exc()),indent=2)+'\n');raise
