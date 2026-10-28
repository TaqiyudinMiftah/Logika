"""Finite final reporting/build stages; no search, upload or deadline reset."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def state(value):
    path=ROOT/'finish_status.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def stage(script,args=()):
    state({'phase':script,'time':time.time()})
    with (ROOT/(script+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixed-secondary',action='store_true');a=parser.parse_args()
    try:
        result=json.loads((ROOT/'benchmark_result.json').read_text())
        stage('assess_benchmark.py')
        if not result['promotion_gate_passed'] and not a.fixed_secondary:
            state({'phase':'benchmark_gate_failed','candidate_generated':False,'reason':'Frozen nested benchmark did not support the new addition. No gate or recipe changed.','time':time.time()})
            return
        stage('build_submission.py',['--fixed-secondary'] if a.fixed_secondary else [])
        for script in ['verify_run.py','export_lineage.py','report_results.py']:stage(script)
        state({'phase':'completed','candidate_generated':True,'verified':True,'no_upload':True,'time':time.time()})
    except BaseException:
        state({'phase':'failed','error':traceback.format_exc(),'time':time.time()});raise


if __name__=='__main__':main()
