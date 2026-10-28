"""Supply the omitted composition invocation before sealed development scoring."""
from pathlib import Path
import json,sys,time,subprocess
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import check_seal
C=check_seal();last=-1
while time.time()<C['deadline_epoch']-60:
 count=len(list((R/'runs').glob('*/attempt*/RESULT.json')))
 if count!=last:
  with(R/'logs/composition.log').open('ab')as log:
   code=subprocess.run([str(R/'env311/bin/python'),str(R/'ops/compose_adapter.py')],stdout=log,stderr=subprocess.STDOUT,timeout=max(20,min(900,int(C['deadline_epoch']-time.time()-30)))).returncode
  if code==0:last=count
 time.sleep(30)
