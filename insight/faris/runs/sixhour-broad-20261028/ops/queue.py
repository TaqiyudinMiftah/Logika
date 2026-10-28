"""Compatibility entrypoint without shadowing Python's standard queue module."""
from pathlib import Path as _Path
if __name__ == '__main__':
    import runpy as _runpy
    _runpy.run_path(str(_Path(__file__).with_name('worker_queue.py')), run_name='__main__')
else:
    import sysconfig as _sysconfig
    _stdlib_queue = _Path(_sysconfig.get_path('stdlib')) / 'queue.py'
    exec(compile(_stdlib_queue.read_text(), str(_stdlib_queue), 'exec'), globals())
