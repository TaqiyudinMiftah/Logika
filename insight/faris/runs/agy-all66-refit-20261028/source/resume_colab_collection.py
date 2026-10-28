"""Resume only monitoring/collection after a transient connection loss.

Reconciles live outputs first; never relaunches or changes an existing fit.
"""
from pathlib import Path
import datetime,shutil
source=Path(__file__).with_name('colab_refit_controller.py');text=source.read_text();ns={'__file__':str(source),'__name__':'recovery_collection'}
exec(compile(text.split("record('waiting_first_fold_chunk_transfer')")[0],str(source),'exec'),ns)
p=ns['P'];failure=p/'execution_failures/status_connection_20261028';failure.mkdir(parents=True,exist_ok=True)
for name in ['status.log','colab-controller.log']:
 f=p/name
 if f.exists():shutil.copy2(f,failure/name)
original_call=ns['call']
def resilient_call(args,label,timeout=300):
 import subprocess,time
 for attempt in range(3):
  try:return original_call(args,'recovery-'+label+f'-attempt{attempt+1}',timeout)
  except (subprocess.TimeoutExpired,AssertionError)as error:
   log=p/('recovery-'+label+f'-attempt{attempt+1}.log');data=log.read_text(errors='replace')if log.exists()else''
   transport=isinstance(error,subprocess.TimeoutExpired)or any(s in data.lower()for s in ['connection was lost','websocket','timed out','connection reset'])
   if not transport or attempt==2:raise
   ns['record']('transient_collection_transport_retry',attempt=attempt+1);time.sleep(10)
ns['call']=resilient_call
ns['record']('RESUMED_COLLECTION_ONLY_NO_FITS_RELAUNCHED',last_live_verified_A100_count=41)
ns['wait_count'](43)
exec(compile("record('download_compact_predictions')"+text.split("record('download_compact_predictions')",1)[1],str(source),'exec'),ns)
