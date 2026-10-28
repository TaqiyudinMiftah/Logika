"""Upload a compact replay pilot through the existing Faris terminal kernel."""
from pathlib import Path
import hashlib,json,sys,tempfile

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R.parents[1]))
from connection_profiles import activated,load_profile

REL='Faris/projects/logika-horizontal-20261026-a100'
files=[R/'a100-pilot/a100_packet.tar.gz',R/'a100-pilot/sdm021.tar.gz',
       R/'source/a100_pilot.py',R/'source/a100_guard.py']

with activated(load_profile('faris')) as remote:
    c=remote.JupyterClient(remote.DEFAULT_BASE_URL,remote.load_token(Path(remote.DEFAULT_SYSINFO)))
    kernel=c.session_kernel('term')
    manifest={}
    for file in files:
        digest=hashlib.sha256(file.read_bytes()).hexdigest();parts=[]
        with file.open('rb') as source,tempfile.TemporaryDirectory() as temp:
            index=0
            while chunk:=source.read(2*1024*1024):
                part=Path(temp)/f'{file.name}.part{index:03d}';part.write_bytes(chunk)
                dest=REL+'/uploads/'+part.name;c.upload_small(part,dest)
                parts.append(dest);index+=1
                print(f'{file.name}: {index} chunks',flush=True)
        manifest[file.name]={'sha256':digest,'parts':parts,
                             'target':('source/' if file.suffix=='.py' else '')+file.name}
    code='''
from pathlib import Path
import hashlib,json,os,subprocess,tarfile,time
root=Path('/home/fariselhakim')/RELATIVE
manifest=MANIFEST
assert not (root/'BOOTSTRAP.json').exists(), 'Already launched; inspect existing pilot'
for name,item in manifest.items():
    target=root/item['target'];target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as out:
        for rel in item['parts']:
            with (Path('/home/fariselhakim')/rel).open('rb') as source:
                while block:=source.read(1024*1024):out.write(block)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'],name
for name,dest in [('a100_packet.tar.gz',root),('sdm021.tar.gz',root/'dependencies')]:
    dest.mkdir(exist_ok=True)
    with tarfile.open(root/name) as archive:
        assert all(m.isfile() or m.isdir() for m in archive.getmembers())
        assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in archive.getmembers())
        archive.extractall(dest)
env=os.environ.copy();env['PYTHONPATH']=str(root/'dependencies')
py='/home/fariselhakim/gipta/projects/logika-ui-26/.venv/bin/python'
checkpoint='/home/fariselhakim/gipta/projects/logika-ui-26/artifacts/hf_cache/hub/models--nvidia--Kumo-Tabular/snapshots/3c3e10bbdb590ace29e7026847f92db3c603096d/small/classifier.pt'
args=[py,str(root/'source/a100_guard.py'),py,str(root/'source/a100_pilot.py'),
      '--packet',str(root/'a100_packet'),'--checkpoint',checkpoint,'--out',str(root/'pilot_result')]
with (root/'guard.log').open('xb') as log:
    p=subprocess.Popen(args,env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':p.pid,'proc_start_ticks':Path(f'/proc/{p.pid}/stat').read_text().split()[21],
        'launched_epoch':time.time(),'manifest':manifest,'existing_term_kernel_reused':True}
(root/'BOOTSTRAP.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps({'pid':p.pid,'root':str(root),'status':'launched'}))
'''.replace('RELATIVE',repr(REL)).replace('MANIFEST',repr(manifest))
    rc,out=c.run_python(kernel,code,timeout=45)
    print(out)
    assert rc==0,'Remote pilot bootstrap failed'
    (R/'a100-pilot/DEPLOYMENT.json').write_text(json.dumps({'output':out,'manifest':manifest},indent=2)+'\n')
