"""Authorized switch of two user services and 11 existing CVAT services."""
from pathlib import Path
import os,json,subprocess,hashlib,time,shutil,sys
H=Path('/mnt/home/faris');OUT=H/'.home-migration-20261025';APP=H/'license-plate'
SERVICES=['plate-split-coordinator.service','plate-split-api.service']
def run(args,**kw):return subprocess.run(args,check=True,text=True,**kw)
def output(args,**kw):return subprocess.check_output(args,text=True,**kw)
def prepare():
    mounts=json.loads((OUT/'docker-home-mounts.json').read_text());names=sorted({x['container'].lstrip('/')for x in mounts})
    raw=json.loads(output(['sudo','-n','docker','inspect',*names]));before=[];images={};targets=[]
    for c in raw:
        label=c['Config']['Labels'];s=label['com.docker.compose.service'];assert label['com.docker.compose.project']=='cvat'
        images[s]={'image':c['Image']};targets.append(s)
        before.append({k:c[k]for k in ['Id','Name','Image','Mounts']})
    (OUT/'cvat-before.json').write_text(json.dumps(before,indent=2))
    (OUT/'preserve-images.override.json').write_text(json.dumps({'services':images},indent=2))
    (OUT/'cvat-service-names.json').write_text(json.dumps(targets,indent=2))
    for name in ['docker-compose.yml','docker-compose.override.yml']:
        p=APP/'cvat'/name;b=p.read_bytes();s=b.decode();new=s.replace('/home/faris/','/mnt/home/faris/')
        # Protect already canonical paths against a duplicated /mnt prefix.
        if '/mnt/mnt/home/faris/'in new:new=new.replace('/mnt/mnt/home/faris/','/mnt/home/faris/')
        backup=OUT/'compose-backup'/name;backup.parent.mkdir(exist_ok=True)
        if not backup.exists():shutil.copy2(p,backup)
        if new!=s:p.write_text(new)
    cmd=compose()+['config','--format','json']
    resolved=json.loads(output(cmd,cwd=APP/'cvat'));summaries=[]
    for s in targets:
        cfg=resolved['services'][s];assert cfg['image']==images[s]['image']
        binds=[v for v in cfg.get('volumes',[])if v.get('type')=='bind']
        assert not any(v.get('source','').startswith('/home/faris/')for v in binds)
        summaries.append({'service':s,'image':cfg['image'],'binds':binds})
    (OUT/'cvat-prepared-config.json').write_text(json.dumps(summaries,indent=2))
    print('Prepared exact existing images and new-home binds for',len(targets),'CVAT services',flush=True)
def compose():
    return ['sudo','-n','docker','compose','-p','cvat','-f',str(APP/'cvat/docker-compose.yml'),'-f',str(APP/'cvat/docker-compose.override.yml'),'-f',str(OUT/'preserve-images.override.json')]
def stop():
    run(['systemctl','--user','stop',*SERVICES]);print('Stopped only the two authorized user services',flush=True)
def final_sync():
    with(OUT/'rsync-final-license-plate.log').open('w')as f:
        run(['rsync','-ax','--stats','--exclude=/.venv','--exclude=/.venv-sam31','--exclude=/.venv-ocr','--exclude=/cvat/docker-compose.yml','--exclude=/cvat/docker-compose.override.yml','/home/faris/license-plate/',str(APP)+'/'],stdout=f)
    print('Final license-plate data sync complete',flush=True)
def units():
    api=f'''[Service]
WorkingDirectory={APP}
ExecStart=
ExecStart={APP}/.venv/bin/python {APP}/scripts/sam_split_server.py --root {APP}
Environment=PYTHONPATH={APP}/scripts
'''
    coord=f'''[Service]
WorkingDirectory={APP}
ExecStart=
ExecStart={APP}/.venv/bin/python -u {APP}/scripts/run_split_coordinator.py --root {APP}
Environment=PYTHONPATH={APP}/scripts:{APP}/vendor/sam3
'''
    for home in [Path('/home/faris'),H]:
        for unit,text in [('plate-split-api.service',api),('plate-split-coordinator.service',coord)]:
            p=home/'.config/systemd/user'/(unit+'.d')/'10-home-migration.conf';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    print('Two service path overrides prepared',flush=True)
def cvat():
    names=json.loads((OUT/'cvat-service-names.json').read_text())
    run(compose()+['up','-d','--no-deps','--no-build','--pull','never','--force-recreate',*names],cwd=APP/'cvat')
    print('Recreated authorized CVAT services with existing images',flush=True)
def start():
    run(['systemctl','--user','set-environment','HOME='+str(H)])
    run(['systemctl','--user','daemon-reload'])
    run(['systemctl','--user','start','plate-split-api.service','plate-split-coordinator.service'])
    print('Started two services from the new home',flush=True)
if __name__=='__main__':
    {'prepare':prepare,'stop':stop,'sync':final_sync,'units':units,'cvat':cvat,'start':start}[sys.argv[1]]()
