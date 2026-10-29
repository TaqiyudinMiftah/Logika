from pathlib import Path
import datetime, hashlib, json, os
R=Path(__file__).resolve().parents[1]
P=Path('/mnt/home/faris/logika-twelvehour-20261029')
A=Path('/mnt/home/faris/logika-faris-exploration')
PROD=Path('/mnt/home/faris/logika-antigravity-refit-20261028')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(value,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def sources():return {p.name:sha(p)for p in sorted((R/'source').glob('*.py'))}
