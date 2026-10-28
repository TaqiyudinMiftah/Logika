from pathlib import Path
import os,json,time,subprocess,sys,hashlib
from huggingface_hub import hf_hub_download
R=Path(__file__).resolve().parents[1];p=json.loads((R/'plan.json').read_text());assert time.time()<p['original_deadline']-10800
for n,h in p['source_sha256'].items():assert hashlib.sha256((R/'source'/n).read_bytes()).hexdigest()==h
path=Path(hf_hub_download('nvidia/Kumo-Tabular',p['filename'],revision=p['source_checkpoint_revision'],cache_dir=R/'hf-cache',token=False));h=hashlib.sha256(path.read_bytes()).hexdigest();p.update(checkpoint=str(path),checkpoint_sha256=h);(R/'plan.json').write_text(json.dumps(p,indent=2));(R/'WEIGHT.json').write_text(json.dumps(dict(path=str(path),sha256=h,revision=p['source_checkpoint_revision'],filename=p['filename']),indent=2))
subprocess.run([sys.executable,str(R/'source/experiment.py'),'preflight'],check=True)
assert json.loads((R/'PREFLIGHT.json').read_text())['status']=='passed';p['training_admitted']=True;(R/'plan.json').write_text(json.dumps(p,indent=2));subprocess.run([sys.executable,str(R/'source/runner.py')],check=True)
