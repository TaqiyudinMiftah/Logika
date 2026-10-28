"""Admit fixed production packets only after the recorded A100 replay gate passed."""
from pathlib import Path
import os,sys,json,hashlib,tarfile,subprocess,datetime
r=Path('/content/logika-all66');spec=json.loads(Path('/content/PRODUCTION_ADMISSION.json').read_text());gate=spec['A100_replay_gate'];assert gate['status']=='PASSED'
assert all(c['passed']for c in gate['checks'])
bundle=Path(spec['bundle']);h=hashlib.sha256()
with bundle.open('rb')as f:
 for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
assert h.hexdigest()==spec['bundle_sha256'],'Production packet bundle changed'
# A second bundle can add remaining fixed packets; never overwrite existing files
# with different bytes or alter a live fit's inputs.
with tarfile.open(bundle)as tar:
 for member in tar:
  assert not Path(member.name).is_absolute()and'..'not in Path(member.name).parts and(member.isfile()or member.isdir()),member.name
  if member.isdir():continue
  relative=Path(member.name)
  if relative.parts[0]!='packets':relative=Path('packets')/relative
  data=tar.extractfile(member).read();dest=r/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==data,('Existing packet changed',member.name)
  else:dest.write_bytes(data)
assert list((r/'packets').rglob('meta.json')),'No production packets; abort before spawning'
env=json.loads((r/'ENVIRONMENT.json').read_text());mode=spec['stage'];log=(r/f'production-{mode}.log').open('w')
cmd=[sys.executable,str(r/'engine_packets.py'),'--packets',str(r/'packets'),'--out',str(r/'predictions'),'--checkpoint',env['checkpoint'],'--mem-fraction','0.90']
# Every packet is a fixed previously selected component. No validation labels or
# adaptive metric choices are staged on this runtime.
proc=subprocess.Popen(['timeout','10800',*cmd],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1'},cwd=r,start_new_session=True)
receipt={'started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':proc.pid,'stage':mode,'bundle_sha256':h.hexdigest(),'fixed_refit_no_selection':True,'fitted_weights_serialized':False,'KV_caches_serialized':False}
(r/f'PRODUCTION_START_{mode}.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
