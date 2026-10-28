"""Run the predeclared secondary control without depending on the workstation."""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import time
import traceback

R=Path(__file__).resolve().parents[1]


def main():
    lock=(R/'diagnostic_watcher.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    clock=json.loads((R/'CLOCK.json').read_text())
    output=R/'diagnostics/constant_gate';output.mkdir(parents=True,exist_ok=True)
    while time.time()<clock['deadline_epoch']-60:
        state=json.loads((R/'LIVE_STATE.json').read_text())
        if state['stage']=='failed':
            (output/'RESULT.json').write_text(json.dumps({'status':'SKIPPED_CORE_FAILURE','primary_changed':False})+'\n');return
        if state['stage']in{'complete','refit_required'}:
            if(output/'RESULT.json').exists():return
            if time.time()>clock['deadline_epoch']-600:break
            try:
                with(R/'ADDITIONAL_DIAGNOSTIC_DISPATCH.json').open('x')as stream:
                    json.dump({'started_epoch':time.time(),'attempt':1,'dispatcher':'remote_watchdog','primary_model_change':False},stream)
            except FileExistsError:return
            with(output/'execution.log').open('ab')as log:
                result=subprocess.run([sys.executable,str(R/'ops/constant_gate_diagnostic.py')],
                                      stdout=log,stderr=subprocess.STDOUT,timeout=540)
            if result.returncode and not(output/'RESULT.json').exists():
                (output/'RESULT.json').write_text(json.dumps({'status':'EXECUTION_FAILED','returncode':result.returncode,'scientific_conclusion':'none'})+'\n')
            return
        time.sleep(30)
    (output/'RESULT.json').write_text(json.dumps({'status':'SKIPPED_DEADLINE','primary_changed':False})+'\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        output=R/'diagnostics/constant_gate';output.mkdir(parents=True,exist_ok=True)
        (output/'private_failure.log').write_text(traceback.format_exc())
        if not(output/'RESULT.json').exists():
            (output/'RESULT.json').write_text(json.dumps({'status':'EXECUTION_FAILED','scientific_conclusion':'none','private_diagnostic':'private_failure.log'})+'\n')
        raise
