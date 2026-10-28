"""Relocate copied environment launchers; preserve a private rollback copy."""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,time,re

HOME=Path('/mnt/home/faris');OUT=HOME/'.home-migration-20261025'
MAP=[('/home/faris','/mnt/home/faris'),('/mnt/home/logika-faris-exploration','/mnt/home/faris/logika-faris-exploration')]
SKIP={'.git','.cache','.python','node_modules','data','models','archive','runs','input','checkpoints','storage','site-packages','__pycache__'}
def replace(s):
    for old,new in MAP:s=re.sub(r'(?<![A-Za-z0-9_.-])'+re.escape(old)+r'(?=/|$|[^A-Za-z0-9_.-])',lambda _:new,s)
    return s
def envs():
    result=[]
    for base,ds,fs in os.walk(HOME,followlinks=False):
        if 'pyvenv.cfg'in fs:
            result.append(Path(base));ds[:]=[];continue
        ds[:]=[d for d in ds if d not in SKIP and not d.startswith('.home-migration-')and not Path(base,d).is_symlink()]
    return sorted(result)
def rewrite(p,changes):
    if p.is_symlink():
        old=os.readlink(p);new=replace(old)
        if old==new:return
        if not Path(new).exists():return
        backup=OUT/'env-backup'/p.relative_to(HOME);backup.parent.mkdir(parents=True,exist_ok=True)
        if not os.path.lexists(backup):backup.symlink_to(old)
        tmp=p.with_name(p.name+'.home-relocate');tmp.symlink_to(new);os.replace(tmp,p)
        changes.append({'path':str(p),'kind':'symlink','before':old,'after':new});return
    if not p.is_file()or p.stat().st_size>2*1024*1024:return
    b=p.read_bytes()
    if b'\x00'in b:return
    try:s=b.decode('utf8')
    except UnicodeError:return
    new=replace(s)
    if s==new:return
    backup=OUT/'env-backup'/p.relative_to(HOME);backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():shutil.copy2(p,backup)
    mode=p.stat().st_mode;tmp=p.with_name(p.name+'.home-relocate');tmp.write_text(new);os.chmod(tmp,mode&0o7777);os.replace(tmp,p)
    changes.append({'path':str(p),'kind':'text','before_sha256':hashlib.sha256(b).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()})
def repair():
    changes=[];found=envs()
    for e in found:
        rewrite(e/'pyvenv.cfg',changes)
        if(e/'bin').is_dir():
            for p in(e/'bin').iterdir():rewrite(p,changes)
        for site in(e/'lib').glob('python*/site-packages'):
            for pattern in ['*.pth','__editable__*.py','*.dist-info/direct_url.json']:
                for p in site.glob(pattern):rewrite(p,changes)
    (OUT/'environment-path-changes.json').write_text(json.dumps(changes,indent=2))
    (OUT/'environments.json').write_text(json.dumps([str(p)for p in found],indent=2))
    print(json.dumps({'environments':len(found),'launcher_or_path_repairs':len(changes)}),flush=True)
PROBE=r'''
import sys,json,importlib,importlib.util,ssl,secrets,sqlite3,encodings.latin_1,os
mods=['numpy','pandas','scipy','sklearn','joblib','lightgbm','catboost','xgboost','torch']
result={'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'python':sys.version.split()[0],'modules':{},'errors':{}}
for m in mods:
 if importlib.util.find_spec(m) is None:continue
 try:
  obj=importlib.import_module(m);result['modules'][m]=str(getattr(obj,'__version__','available'))
 except Exception as e:result['errors'][m]=type(e).__name__+': '+str(e)
print(json.dumps(result))
sys.exit(bool(result['errors']))
'''
def verify():
    found=json.loads((OUT/'environments.json').read_text());results=[]
    # Check both invocation paths, without fitting models or allocating GPUs.
    for name in found:
        paths=[name]
        if name.startswith('/mnt/home/faris/logika-faris-exploration/'):
            paths.append(name.replace('/mnt/home/faris/logika-faris-exploration/','/mnt/home/logika-faris-exploration/',1))
        elif name.startswith('/mnt/home/faris/'):
            paths.append(name.replace('/mnt/home/faris/','/home/faris/',1))
        for env in paths:
            p=Path(env)/'bin/python'
            if not os.path.lexists(p):continue
            try:
                run=subprocess.run([str(p),'-c',PROBE],capture_output=True,text=True,timeout=120,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                lines=run.stdout.strip().splitlines();data=json.loads(lines[-1])if lines else {'errors':{'startup':run.stderr[-2000:]}}
                results.append({'environment':env,'returncode':run.returncode,**data})
                print('ENV_CHECK',env,'OK'if run.returncode==0 else data.get('errors'),flush=True)
            except Exception as e:results.append({'environment':env,'returncode':-1,'errors':{'probe':str(e)}})
    (OUT/'environment-checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'invocations_checked':len(results),'failed':sum(x['returncode']!=0 for x in results)}),flush=True)
if __name__=='__main__':
    OUT.mkdir(mode=0o700,exist_ok=True)
    if sys.argv[1]=='repair':repair()
    elif sys.argv[1]=='verify':verify()
