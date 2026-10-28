from pathlib import Path
import subprocess,sys,time,json
R=Path(__file__).resolve().parents[1];p=json.loads((R/'plan.json').read_text());D=p['original_deadline']
while not(R/'INDEPENDENT_AUDIT.json').exists():
 assert time.time()<D-10800 and not(R/'STOP').exists()
 if(R/'FAILURE.json').exists():raise RuntimeError('Feature construction failed; retain evidence, do not admit models')
 time.sleep(10)
assert json.loads((R/'INDEPENDENT_AUDIT.json').read_text())['status']=='passed'
subprocess.run([sys.executable,str(R/'source/stage.py')],check=True)
