"""Isolated dependency setup followed by a bounded real gradient preflight."""
from pathlib import Path
import subprocess,os,time,json
ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path.home()/'faris/exploration-12h-20261017/.venv/bin/python'
def status(state,**values):
    (ROOT/'PREPARE_STATUS.json').write_text(json.dumps(dict(state=state,time=time.time(),**values),indent=2)+'\n')
def main():
    plan=json.loads((ROOT/'plan.json').read_text());assert time.time()<plan['original_deadline']-3600
    status('installing_isolated_dependencies')
    packages=['structured-data-models==0.2.1','numpy==2.2.6','pandas==2.3.3','scipy==1.16.3',
        'scikit-learn==1.7.2','joblib==1.6.0','pyarrow==22.0.0','huggingface-hub==0.35.3',
        'safetensors==0.6.2','typing-extensions==4.16.0']
    with (ROOT/'dependencies.log').open('ab')as file:
        subprocess.run([str(PYTHON),'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
            '--no-deps','--target',str(ROOT/'deps'),*packages],stdout=file,stderr=subprocess.STDOUT,check=True,timeout=900)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'deps')+':'+str(ROOT/'source')
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',KUMO_MEMORY_FRACTION='.30')
    status('gradient_preflight_running',main_model_training_admitted=False)
    with (ROOT/'preflight.log').open('ab')as file:
        result=subprocess.run([str(PYTHON),str(ROOT/'source/preflight.py')],env=env,
            stdout=file,stderr=subprocess.STDOUT,cwd=ROOT,timeout=900)
    status('preflight_passed'if result.returncode==0 else 'preflight_failed',
        returncode=result.returncode,main_model_training_admitted=False)
if __name__=='__main__':
    try:main()
    except Exception as e:
        status('preparation_failed',error=repr(e),main_model_training_admitted=False);raise
