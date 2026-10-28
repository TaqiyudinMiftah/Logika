"""Durable PC2 archive collector; training queues remain independent."""
from pathlib import Path
import subprocess,json,time,shlex,hashlib,sys,fcntl
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/mnt/home/faris/faris/kumo-coverage-placement-20261021'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','2003','faris@gpu-labkc.ub.ac.id']
VARIANTS=['control90','coverage96','null96']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def rpc(code):return subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)],text=True,timeout=45)
def receive(f,v):
    relative=f'conditions/fold{f}/{v}';dest=ROOT/relative
    if not (dest/'ARCHIVE_RECEIPT.json').exists():
        # A previous collector may have completed the atomic move but died
        # before writing its receipt. Verify that retained directory in place.
        incoming=dest if dest.exists() else ROOT/'incoming'/f'{f}_{v}'
        if incoming!=dest:
            incoming.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-a','--partial','-e','ssh -o BatchMode=yes -o ConnectTimeout=8 -p 2003',
                'faris@gpu-labkc.ub.ac.id:'+REMOTE+'/'+relative+'/',str(incoming)+'/'],check=True,timeout=1200)
        complete=read(incoming/'complete.json')
        for name,digest in complete['small_files'].items():
            assert Path(name).name==name and sha(incoming/name)==digest
        cache=incoming/'fitted_context.pt';assert sha(cache)==complete['large_cache']['sha256']
        assert cache.stat().st_size==complete['large_cache']['bytes']
        cache_bytes=cache.stat().st_size
        if incoming!=dest:
            assert not dest.exists(),'Do not overwrite existing completed/partial training artifacts'
            dest.parent.mkdir(parents=True,exist_ok=True);incoming.rename(dest)
        write(dest/'ARCHIVE_RECEIPT.json',dict(status='verified',source_host='pc3',source=REMOTE+'/'+relative,
            cache_sha256=complete['large_cache']['sha256'],bytes=cache_bytes,time=time.time()))
    receipt=read(dest/'ARCHIVE_RECEIPT.json')
    # Only this campaign-owned cache is removed, after full independent archive
    # hashing. Every distinct trained cache remains retained at the PC2 path.
    code='from pathlib import Path;import json;r=Path('+repr(REMOTE)+');d=r/'+repr(relative)+';c=json.loads((d/"complete.json").read_text());assert c["large_cache"]["sha256"]=='+repr(receipt['cache_sha256'])+';p=d/"fitted_context.pt";p.unlink(missing_ok=True);(d/"ARCHIVED.json").write_text('+repr(json.dumps(dict(archive_path=str(dest),sha256=receipt['cache_sha256'],status='verified_and_retained_on_pc2')))+')'
    rpc(code)
def main():
    lock=(ROOT/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ROOT/'plan.json');collected=set();errors=[]
    while time.time()<plan['original_deadline'] and not (ROOT/'STOP').exists():
        remote=None
        try:
            code='from pathlib import Path;p=Path('+repr(REMOTE+'/controller-state.json')+');print(p.read_text()if p.exists()else "{}")'
            remote=json.loads(rpc(code))
            for job in remote.get('jobs',{}).values():
                f,v=job['fold'],job['variant'];assert f in [3,4] and v in VARIANTS
                if job['status']=='complete'and (f,v)not in collected:receive(f,v);collected.add((f,v))
        except Exception as e:errors.append(dict(time=time.time(),error=repr(e)));errors=errors[-20:]
        complete=sum((ROOT/f'conditions/fold{f}/{v}/complete.json').exists()for f in range(5)for v in VARIANTS)
        write(ROOT/'COLLECTION_STATUS.json',dict(time=time.time(),complete_conditions=complete,total=15,
            pc3_verified_archives=len(collected),recent_errors=errors))
        if complete==15:
            for action in ['finish','audit_caches']:
                with (ROOT/(action+'.log')).open('ab')as file:
                    subprocess.run([plan['python'],str(ROOT/'source/experiment.py'),action],stdout=file,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            write(ROOT/'FINISHED.json',dict(status='complete',time=time.time(),conditions=15,
                result_sha256=sha(ROOT/'result.json'),cache_audit_sha256=sha(ROOT/'CACHE_AUDIT.json')))
            return
        local=read(ROOT/'controller-state.json')if (ROOT/'controller-state.json').exists()else {}
        if local.get('status')=='terminal'and remote and remote.get('status')=='terminal':
            write(ROOT/'FINISHED.json',dict(status='partial',time=time.time(),complete_conditions=complete,
                reason='At least one predeclared full-fold condition failed or did not start. No completed pipeline score is fabricated.'))
            return
        time.sleep(15)
    write(ROOT/'FINISHED.json',dict(status='deadline_or_stop',time=time.time(),target_unproven=True))
if __name__=='__main__':main()
