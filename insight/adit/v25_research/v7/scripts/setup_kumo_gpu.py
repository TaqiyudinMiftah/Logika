"""Prepare isolated GPU runtime, reusing installed Torch without changing peers."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.metadata
import json
import os
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'v7/kumo_gpu'
ENV=ROOT.parents[2]/'.venv-kumo-gpu-20261027'


def state(status,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'SETUP_STATUS.json').write_text(json.dumps(dict(status=status,pid=os.getpid(),UTC=datetime.now(timezone.utc).isoformat(),**kw),indent=2))


def run(args,**kw):
    print('RUN',str(args[0]),' '.join(str(x) for x in args[1:4]),flush=True)
    return subprocess.run(list(map(str,args)),check=True,**kw)


def main():
    state('CREATING_ISOLATED_ENV')
    source=ROOT.parents[2]/'.venv-logika-offset-20261027'
    gpu=ROOT.parents[2]/'.venv-datascape26'
    if not (ENV/'bin/python').exists():run([source/'bin/python','-m','venv',ENV])
    site=ENV/'lib/python3.11/site-packages'
    (site/'existing_runtime_paths.pth').write_text(str(source/'lib/python3.11/site-packages')+'\n'+str(gpu/'lib/python3.11/site-packages')+'\n')
    constraints=OUT/'constraints.txt'
    constraints.write_text('numpy==2.2.6\npandas==2.3.3\nscikit-learn==1.7.2\nscipy==1.16.3\njoblib==1.6.0\npyarrow==22.0.0\n')
    state('INSTALLING_SDM_AND_HF_CLI',environment=str(ENV),reused_torch_environment=str(gpu))
    run([ENV/'bin/python','-m','pip','install','--no-cache-dir','--constraint',constraints,'structured-data-models==0.2.1','huggingface_hub[cli]==0.36.0'])
    code="import torch,sdm,json;print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'sdm':getattr(sdm,'__version__','installed')}));assert torch.cuda.is_available()"
    run([ENV/'bin/python','-c',code])
    state('INSPECTING_PINNED_WEIGHTS')
    request=urllib.request.Request('https://huggingface.co/api/models/nvidia/Kumo-Tabular/revision/v1.0.1',headers={'User-Agent':'Logika-research'})
    with urllib.request.urlopen(request,timeout=30) as response:model=json.load(response)
    files=[x['rfilename'] for x in model.get('siblings',[]) if 'large' in x['rfilename'].lower() or x['rfilename'].lower().endswith(('.yaml','.yml','.json'))]
    assert files,'No matching pinned Kumo weights found'
    cache=OUT/'hf_cache'
    state('DOWNLOADING_PINNED_WEIGHTS',revision=model.get('sha'),files=files)
    run([ENV/'bin/hf','download','nvidia/Kumo-Tabular',*files,'--revision','v1.0.1','--cache-dir',cache,'--max-workers','1'])
    (OUT/'WEIGHTS_MANIFEST.json').write_text(json.dumps(dict(repository='nvidia/Kumo-Tabular',revision=model.get('sha'),tag='v1.0.1',files=files,cache=str(cache)),indent=2))
    state('READY',python=str(ENV/'bin/python'),hf_cache=str(cache),revision=model.get('sha'))


if __name__=='__main__':
    try:main()
    except Exception as e:
        state('FAILED',error=repr(e));raise
