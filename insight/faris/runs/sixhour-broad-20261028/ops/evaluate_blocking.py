"""Run the sealed evaluator with blocking execution lock, identical mathematics."""
from pathlib import Path
import sys,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import evaluate
from common import check_seal,dump,sha,utc
check_seal();original=fcntl.flock
def blocking(fd,mode):
    # Only replace nonblocking exclusive admission with a wait. No predicate,
    # prediction, calibration, ranking or final selection code is modified.
    if mode==fcntl.LOCK_EX|fcntl.LOCK_NB:mode=fcntl.LOCK_EX
    return original(fd,mode)
fcntl.flock=blocking
try:evaluate.main(final=False)
finally:fcntl.flock=original
dump(dict(UTC=utc(),operational_wrapper_sha256=sha(__file__),sealed_evaluator_sha256=sha(R/'source/evaluate.py'),change='Wait for existing evaluate.lock instead of treating healthy composition contention as failure.',numerical_scientific_source_changed=False),R/'EVALUATION_LOCK_ADAPTER.json')
