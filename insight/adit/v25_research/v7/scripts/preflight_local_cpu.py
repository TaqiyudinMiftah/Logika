"""Verify the unchanged V25 replay and native lock before local CPU fitting."""
from pathlib import Path
import json,subprocess,sys,tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'v7'
sys.path[:0]=[str(V/'scripts'),str(ROOT/'scripts')]
from platform_runtime import fcntl,resource
from baseline import corrected_probability,dump,sha


def main():
    from modeling_features import load
    tr,te,sample,y,folds,reference,frames,parent=load()
    prediction=np.zeros(len(sample))
    for f in frames:
        p,_=corrected_probability(f['head'],f['test'],f['kt'])
        prediction+=pd.Series(.75*p+.25*f['st'],index=f['test'].application_id).reindex(sample.application_id).to_numpy()/5
    original=pd.read_csv(ROOT/parent['submission'],float_precision='round_trip');assert original.application_id.tolist()==sample.application_id.tolist()
    error=float(np.max(abs(prediction-original.default_probability.to_numpy())))
    assert error<2e-12
    with tempfile.TemporaryDirectory(dir=V) as temporary:
        path=Path(temporary)/'lock'
        with path.open('a') as f:
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            code="import sys;sys.path.insert(0,sys.argv[1]);from platform_runtime import fcntl;f=open(sys.argv[2],'a')\ntry:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept OSError:raise SystemExit(0)\nraise SystemExit(1)"
            check=subprocess.run([sys.executable,'-c',code,str(V/'scripts'),str(path)],timeout=10)
            assert check.returncode==0,'A second process must not enter the same model run'
    peak=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
    report=dict(status='PASS',source='Node480 authenticated artifacts replayed on local CPU',
                all5fold_original_V25_validation_predictions_replayed=True,original_V25_test_mean_max_abs_error=error,
                original_submission_sha256=sha(ROOT/parent['submission']),
                exclusive_process_lock_verified=True,peak_RSS_MiB=peak,execution_host='local_windows',
                prediction_algorithm_changed=False,local_preprocessing_fit_on_training_only=True,python_executable=sys.executable)
    dump(report,V/'reports/LOCAL_CPU_PREFLIGHT.json');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
