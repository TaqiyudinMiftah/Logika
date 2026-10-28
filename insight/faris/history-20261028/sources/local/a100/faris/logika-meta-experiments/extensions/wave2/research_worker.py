"""Hard development-only guard around the unchanged fitting implementation."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys
import numpy as np


def install_guard():
    original=np.load
    def load(path,*args,**kwargs):
        if isinstance(path,(str,os.PathLike)) and Path(path).name.startswith('confirmation_'):
            raise PermissionError('Wave two cannot access the previously opened confirmation partition')
        return original(path,*args,**kwargs)
    np.load=load


def main():
    from metaexp.common import read
    root=Path(sys.argv[1]);task=read(sys.argv[2])
    if task.get('mode','cv') not in ['cv','repeat','temporal','nested']:
        raise PermissionError('Wave two is development-only research')
    install_guard()
    from metaexp.worker import execute
    execute(root,task)

if __name__=='__main__':main()
