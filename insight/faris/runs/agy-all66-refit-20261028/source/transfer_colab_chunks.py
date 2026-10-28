"""Transfer large private packets below Jupyter's single-request size limit."""
from pathlib import Path
import argparse,json,hashlib,subprocess,tempfile,time,os,sys
p=argparse.ArgumentParser();p.add_argument('--file',required=True);p.add_argument('--sha256',required=True);p.add_argument('--remote',required=True);p.add_argument('--config',required=True);p.add_argument('--session',required=True);p.add_argument('--log',required=True);p.add_argument('--streaming',action='store_true');p.add_argument('--bytes',type=int);a=p.parse_args()
source=Path(a.file);logpath=Path(a.log);prefix=['colab','--account','2','--config',a.config];chunk_size=16*1024**2
expected=a.bytes if a.streaming else source.stat().st_size;assert expected is not None
parts=[];temporary=logpath.parent/'transfer_chunks';temporary.mkdir(exist_ok=True)
def invoke(args):
 with logpath.open('a')as log:
  result=subprocess.run([*prefix,*args],stdout=log,stderr=subprocess.STDOUT,timeout=240)
 assert result.returncode==0,('Colabtransferactionfailed; inspectprivatelog',logpath)
for i,start in enumerate(range(0,expected,chunk_size)):
 length=min(chunk_size,expected-start)
 deadline=time.monotonic()+1800
 while not source.exists()or source.stat().st_size<start+length:
  assert time.monotonic()<deadline,'Source download didnotprogress';time.sleep(2)
 with source.open('rb')as f:f.seek(start);data=f.read(length)
 assert len(data)==length
 local=temporary/(source.name+f'.part{i:04d}');local.write_bytes(data);remote=a.remote+f'.part{i:04d}'
 invoke(['upload','--session',a.session,str(local),remote]);local.unlink();parts.append(remote)
 print(json.dumps({'stage':'chunk_uploaded','file':source.name,'chunk':i,'bytes_uploaded':min(start+length,expected),'total_bytes':expected}),flush=True)
code='''from pathlib import Path
import hashlib,json
paths=PARTS
out=Path(DEST)
h=hashlib.sha256()
with out.open('wb')as f:
 for name in paths:
  p=Path(name)
  with p.open('rb')as source:
   for block in iter(lambda:source.read(8*1024**2),b''):f.write(block);h.update(block)
assert h.hexdigest()==SHA,'Joined packet hash mismatch'
for name in paths:Path(name).unlink()
print(json.dumps({'joined':str(out),'bytes':out.stat().st_size,'sha256':h.hexdigest()}),flush=True)
'''.replace('PARTS',repr(parts)).replace('DEST',repr(a.remote)).replace('SHA',repr(a.sha256))
join=temporary/(source.name+'.join.py');join.write_text(code);invoke(['exec','--session',a.session,'--file',str(join),'--timeout','120']);join.unlink()
print(json.dumps({'stage':'TRANSFER_COMPLETE','file':source.name,'bytes':expected,'sha256':a.sha256}),flush=True)
