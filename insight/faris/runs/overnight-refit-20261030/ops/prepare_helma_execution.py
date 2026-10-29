from pathlib import Path
import json,hashlib,py_compile
P=Path('a100/faris/overnight-refit-20261030');R=P/'helma-book-context10k'
# Physical checkpoint identity is verified without creating paths outside /root.
b=(P/'ops/bootstrap_colab_owned.py').read_text().replace('/root/logika-overnight-refit-20261030','/root/logika-helma-books-20261030').replace("for size in ['large']:","for size in ['large','medium']:")
a=b.index(' logical=Path(');z=b.index(" out['checkpoint_",a)
b=b[:a]+" assert p.resolve().is_relative_to(Path.home())\n out['checkpoint_'+size]=str(p.resolve())\n"+b[z:]
(R/'ops/bootstrap_colab_owned.py').write_text(b)
s=Path('a100/faris/twelvehour-next-20261029/ops/colab_backend_controller.py').read_text()
s=s.replace("R=Path('/home/faris/logika-twelvehour-next-20261029');VROOT='/root/logika-twelvehour-next-20261029';LAB='faris@100.68.29.126';CANON='/mnt/home/faris/logika-twelvehour-next-20261029'", "R=Path('/home/faris/logika-overnight-refit-20261030/helma-book-context10k');VROOT='/root/logika-helma-books-20261030';LAB='faris@100.68.29.126';CANON='/mnt/home/faris/logika-overnight-refit-20261030/helma-book-context10k'")
s=s.replace("P=R/f'private/colab/account{N}'", "P=R.parent/f'private/helma-books/account{N}'")
s=s.replace("NAME=f'logika-next-a{N}-20261029'", "NAME='logika-0630-helma-books-20261030'")
s=s.replace("f'next-a{N}'", "f'helma-books-a{N}'")
s=s.replace("receipt=json.load(open(R/'RESOURCE_PREPUBLICATION.json'));assert len(receipt['remote_commit'])==40 and receipt['extra_designs_sha256']==sha(R/'EXTRA_DESIGNS.jsonl')", "receipt=json.load(open(R/'PREPUBLICATION.json'));assert len(receipt['remote_commit'])==40 and receipt['config_sha256']==sha(R/'CONFIG.json') and receipt['source_seal_sha256']==sha(R/'SOURCE_SEAL.json')")
s=s.replace("R/'private/colab/SOURCE.tar.gz'", "R/'private/SOURCE.tar.gz'").replace("R/'private/colab/PACKETS.tar'", "R/'private/PACKETS.tar'")
# Preserve namespace-scoped dataset in personal host's existing cache, hardlink archive.
(R/'ops/colab_backend_controller.py').write_text(s)
coordinator='''"""Collect/score only completed predeclared Helma designs; never trains on evaluator host."""
from pathlib import Path
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='2'
import json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];P=R.parent;sys.path.insert(0,str(R/'source'))
from common import load,dump,sha,utc,tasks,check_seal
PY='/mnt/home/faris/logika-sixhour-broad-20261028/env311/bin/python'
def module(name,args=()):
 with(R/'logs'/f'{name}.log').open('ab')as f:subprocess.run([PY,str(R/'source'/f'{name}.py'),*args],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,int(load('CLOCK.json')['deadline_epoch']-time.time()-10)))
try:
 check_seal();last=0;clock=load('CLOCK.json')
 while time.time()<clock['model_deadline_epoch']-90:
  count=len(list((R/'runs').glob('*/attempt1/RESULT.json')))
  if time.time()-last>60:module('compose');module('evaluate');last=time.time()
  dump(dict(stage='running',UTC=utc(),completed_fits=count,failed_attempts=len(list((R/'runs').glob('*/attempt1/FAILURE.json'))),planned_fits=len(tasks()),owner='helma_account2',original_deadline_epoch=clock['deadline_epoch']),R/'LIVE_STATE.json')
  if(R/'BACKEND_COLLECTION_account2_COMPLETE.json').exists():break
  time.sleep(15)
 # Controller has its own original-clock bounded final collection and owned release.
 while not(R/'BACKEND_COLLECTION_account2_COMPLETE.json').exists() and time.time()<clock['model_deadline_epoch']+90:time.sleep(10)
 assert(R/'BACKEND_COLLECTION_account2_COMPLETE.json').exists(),'Final collection/release must be reconciled before terminal selection'
 module('compose');module('evaluate',['--final']);module('audit')
 v=dict(stage='complete',UTC=utc(),completed_fits=len(list((R/'runs').glob('*/attempt1/RESULT.json'))),failed_attempts=len(list((R/'runs').glob('*/attempt1/FAILURE.json'))),planned_fits=len(tasks()),audit_sha256=sha(R/'ARTIFACT_CONSISTENCY_AUDIT.json'),original_deadline_epoch=clock['deadline_epoch']);dump(v,R/'COMPLETION.json');dump(v,R/'LIVE_STATE.json')
except BaseException:
 p=R/'private/COORDINATOR_FAILURE.log';p.parent.mkdir(exist_ok=True);p.write_text(traceback.format_exc());dump(dict(status='execution_review_required',UTC=utc(),trace_sha256=sha(p)),R/'COORDINATOR_FAILURE.json');raise
'''
(R/'ops/helma_coordinator.py').write_text(coordinator)
for p in (R/'ops').glob('*.py'):py_compile.compile(str(p),doraise=True)
print('Prepared physical-home pinned bootstrap, persistent collection/release and coordinator')
