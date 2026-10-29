"""Atomic prediction publication for this campaign's own future worker units.

Execution I/O only: identical arrays/compression, no model/RNG/feature change.
Prevents an incremental collector from observing an unfinished ZIP file.
"""
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[_k]='2'
from pathlib import Path
import datetime,hashlib,json
import numpy as np

_original=np.savez_compressed
_root=Path('/mnt/home/faris/logika-twelvehour-20261029')
def _atomic(file,*args,**kwargs):
    if isinstance(file,(str,bytes,os.PathLike)) and Path(file).name=='prediction.npz':
        target=Path(file);temporary=target.with_name('prediction.'+str(os.getpid())+'.atomic-tmp')
        try:
            with temporary.open('wb')as stream:_original(stream,*args,**kwargs)
            temporary.replace(target)
        finally:
            if temporary.exists():temporary.unlink()
        return None
    return _original(file,*args,**kwargs)
np.savez_compressed=_atomic
_log=_root/'private/ATOMIC_IO_HOOK_RECORDS.jsonl';_log.parent.mkdir(exist_ok=True)
with _log.open('a')as _f:_f.write(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),only_prediction_file_atomicity_changed=True))+'\n')
