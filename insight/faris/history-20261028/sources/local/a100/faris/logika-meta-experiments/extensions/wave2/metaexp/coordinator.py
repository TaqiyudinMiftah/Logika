"""Resident scheduler on PC2. SSH is transport; every execution is ordinary Python."""
from __future__ import annotations
import concurrent.futures
import io
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import tarfile
import time
import traceback
from .common import read,write,event,sha,identity,code_hash

class Transport:
    def __init__(self,root,cfg):self.root=Path(root);self.cfg=cfg
    def command(self,host):
        m=self.cfg['machines'][host]
        if host=='pc1':return ['ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-i',str(self.root/'.credentials/pc1_ed25519'),m['ssh'],'rpc']
        return ['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-p',str(m['port']),m['ssh'],
                'cd '+shlex.quote(m['root'])+' && .venv/bin/python -m metaexp rpc']
    def call(self,host,request,timeout=30):
        if host=='pc2':
            from .runtime import status,enqueue,start_daemon,stop_daemon
            action=request['action']
            if action=='status':return status(self.root)
            if action=='push':return {'accepted':enqueue(self.root,request['tasks'])}
            if action=='start':return start_daemon(self.root,host)
            if action=='stop':return stop_daemon(self.root)
            if action=='freeze':write(self.root/'finalists.json',request['finalists']);return {'frozen':True}
        p=subprocess.run(self.command(host),input=json.dumps(request),capture_output=True,text=True,timeout=timeout)
        if p.returncode:raise RuntimeError(p.stderr[-1500:])
        return json.loads(p.stdout)
    def collect(self,host,tid):
        if host=='pc2':return
        p=subprocess.run(self.command(host),input=json.dumps({'action':'pack','id':tid}).encode(),capture_output=True,timeout=90)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1000:])
        with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:gz') as tar:
            total=0
            for member in tar.getmembers():
                rel=Path(member.name)
                if not member.isfile() or rel.is_absolute() or '..' in rel.parts or rel.parts[:2]!=('runs',tid):raise ValueError('Unsafe result archive')
                total+=member.size
                if total>200*1024**2:raise ValueError('Unexpected result archive size')
            tar.extractall(self.root,filter='data')

class Ledger:
    def __init__(self,path):
        self.db=sqlite3.connect(path);self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, recipe_id TEXT, host TEXT, mode TEXT, state TEXT, task TEXT, assigned REAL, finished REAL, verified INTEGER DEFAULT 0)');self.db.commit()
    def insert(self,task,host):
        self.db.execute('INSERT INTO tasks(id,recipe_id,host,mode,state,task,assigned) VALUES(?,?,?,?,?,?,?)',
            (task['id'],task['recipe']['id'] if 'id' in task['recipe'] else task['id'],host,task.get('mode','cv'),'assigned',json.dumps(task,sort_keys=True),time.time()));self.db.commit()
    def all(self):
        self.db.row_factory=sqlite3.Row;return [dict(r) for r in self.db.execute('SELECT * FROM tasks')]
    def finish(self,tid,state,verified):
        self.db.execute('UPDATE tasks SET state=?,finished=?,verified=? WHERE id=?',(state,time.time(),int(verified),tid));self.db.commit()
    def exists(self,tid):return self.db.execute('SELECT 1 FROM tasks WHERE id=?',(tid,)).fetchone() is not None

class Coordinator:
    def __init__(self,root):
        self.root=Path(root).resolve();self.cfg=read(self.root/'campaign.json');self.catalogue=read(self.root/'catalogue.json')
        self.tx=Transport(self.root,self.cfg);self.ledger=Ledger(self.root/'ledger.sqlite');self.host_status={};self.errors={};self.bank=None;self.costs={};self.hold=None;self.hold_log=None
    def phase_end(self,hours):return self.cfg['started']+hours*3600
    def sync_one(self,host):
        status=self.tx.call(host,{'action':'status'});self.host_status[host]=status
        active=status.get('heartbeat',{}).get('active',[]) if status.get('heartbeat') else []
        records={r['id']:r for r in self.ledger.all() if r['host']==host}
        for tid,s in status['runs'].items():
            if tid not in records or records[tid]['state'] not in ['assigned','running'] or tid in active:continue
            self.tx.collect(host,tid);verified=False
            if s['state']=='completed':
                from .reporting import verify_run
                try:
                    score=verify_run(self.root,tid,self.bank);verified=True
                    if score.get('model'):self.costs.setdefault(score['model'],[]).append(score.get('elapsed_seconds',300))
                except Exception as e:
                    event(self.root/'coordinator.jsonl',event='verification_failed',task=tid,error=str(e));s={'state':'failed'}
            self.ledger.finish(tid,s['state'],verified)
        return status
    def sync(self):
        # SQLite remains on the coordinator thread; network calls are bounded.
        if self.hold_log is not None and (self.hold is None or self.hold.poll() is not None):self.hold_pc1()
        for host in ['pc2','pc1','pc3']:
            try:
                status=self.sync_one(host);self.errors.pop(host,None)
                hb=status.get('heartbeat') or {}
                if time.time()<self.phase_end(11.5) and time.time()-hb.get('time',0)>30:self.tx.call(host,{'action':'start','host':host})
            except Exception as e:self.errors[host]=str(e);event(self.root/'coordinator.jsonl',event='transport_error',host=host,error=str(e))
        write(self.root/'campaign_status.json',{'state':'running','time':time.time(),'deadline':self.cfg['deadline'],
            'phase':read(self.root/'phase.json') if (self.root/'phase.json').exists() else {},'hosts':{h:s.get('heartbeat') for h,s in self.host_status.items()},'errors':self.errors,
            'counts':{k:sum(r['state']==k for r in self.ledger.all()) for k in ['assigned','completed','failed','partial','timed_out','resource_limited']}})
    def submit(self,host,recipe,until,mode='cv',tid=None,**extra):
        device='cuda' if recipe.get('model') in ['cat','xgb'] and host in ['pc2','pc3'] else 'cpu'
        if recipe.get('model')=='lgb' and host=='pc1' and self.cfg['machines'][host].get('opencl'):
            hb=self.host_status.get(host,{}).get('heartbeat') or {};active=hb.get('active',[])
            assigned=[r for r in self.ledger.all() if r['host']==host and r['state']=='assigned' and json.loads(r['task']).get('device')=='opencl']
            if not assigned:device='opencl'
        if mode in ['nested','ensemble_confirmation'] and host in ['pc2','pc3']:device='cuda'
        task={'id':tid or recipe['id'],'mode':mode,'recipe':recipe,'device':device,'until':until,**extra}
        if self.ledger.exists(task['id']):return False
        # Persist intent first; a failed upload is retried without assigning another host.
        self.ledger.insert(task,host)
        try:self.tx.call(host,{'action':'push','tasks':[task]})
        except Exception as e:event(self.root/'coordinator.jsonl',event='push_pending',task=task['id'],host=host,error=str(e))
        return True
    def retry_delivery(self):
        for host in ['pc1','pc2','pc3']:
            if host in self.errors:continue
            pending=[json.loads(r['task']) for r in self.ledger.all() if r['host']==host and r['state']=='assigned']
            if pending:
                try:self.tx.call(host,{'action':'push','tasks':pending})
                except Exception:pass
    def wait_phase(self,until,mode=None):
        while time.time()<until:
            self.sync();self.retry_delivery()
            live=[r for r in self.ledger.all() if r['state']=='assigned' and (mode is None or r['mode'] in mode)]
            if not live:return
            time.sleep(10)
    def run_search(self):
        end=self.phase_end(9.5);recipes=self.catalogue['recipes'];start=time.time()
        write(self.root/'phase.json',{'name':'full_cv_search','until':end})
        controls=[r for r in recipes if r['representation']['id']==self.catalogue['control']]
        for r in controls:
            host='pc3' if r['model']=='cat' else 'pc2' if r['model']=='xgb' else 'pc1'
            self.submit(host,r,end,tid=r['id'])
        # A CPU LGB control is needed for fair CPU/OpenCL comparisons.
        lgb=next(r for r in controls if r['model']=='lgb')
        self.submit('pc2',lgb,end,tid=lgb['id']+'_cpu')
        self.submit('pc1',{'id':'reference_v3'},self.phase_end(2),mode='reference',tid='reference_v3')
        cursor=0
        while time.time()<end:
            self.sync();self.retry_delivery();records=self.ledger.all();assigned_ids={r['recipe_id'] for r in records}
            for host in ['pc1','pc2','pc3']:
                if host in self.errors:continue
                jobs=[r for r in records if r['host']==host and r['state']=='assigned'];slots=self.cfg['machines'][host]['workers']
                gpu=sum(json.loads(r['task']).get('device') in ['cuda','opencl'] for r in jobs)
                capacity=max(0,slots+2-len(jobs))
                for r in recipes:
                    if capacity<=0:break
                    if r['id'] in assigned_ids:continue
                    is_gpu=r['model'] in ['cat','xgb']
                    if is_gpu and (host=='pc1' or gpu>=2):continue
                    recent=self.costs.get(r['model'],[])[-64:]
                    forecast=max(60,float(sorted(recent)[min(len(recent)-1,int(.9*len(recent)))])*1.4) if recent else 300
                    if time.time()+forecast>end:continue
                    # Fixed per-recipe timeout; no fold-count reduction or score-based pruning.
                    self.submit(host,r,min(end,time.time()+max(900,forecast*2)))
                    assigned_ids.add(r['id']);capacity-=1
                    if is_gpu:gpu+=1
            if len(assigned_ids)>=len(recipes)+1 and not any(r['state']=='assigned' for r in self.ledger.all()):break
            if time.time()-start>60 and not (self.root/'reports/eda_summary.json').exists():
                from .reporting import eda
                eda(self.root,self.bank)
            if int(time.time())%3600<12:
                from .reporting import report
                report(self.root)
            time.sleep(10)
        self.wait_phase(self.phase_end(9.55),mode=['cv','reference'])
    def accepted_pipelines(self):
        out=[]
        for r in self.ledger.all():
            if r['verified'] and r['mode']=='cv':out.append((r['id'],read(self.root/'runs'/r['id']/'scores.json')))
        return sorted(out,key=lambda v:v[1]['pooled_oof']['ks'],reverse=True)
    def run_ensembles(self):
        from .ensembles import fixed
        import numpy as np
        end=self.phase_end(10.5);write(self.root/'phase.json',{'name':'ensembles','until':end})
        pool=[];seen=set()
        for tid,s in self.accepted_pipelines():
            if s['recipe_id'] in seen:continue
            p=np.load(self.root/'runs'/tid/'prediction.npy')
            if np.std(p)<1e-12:continue
            if pool and all(np.corrcoef(p,np.load(self.root/'runs'/v/'prediction.npy'))[0,1]>.998 for v in pool):continue
            pool.append(tid);seen.add(s['recipe_id'])
            if len(pool)>=32:break
        # Preserve learner and feature-family diversity before selecting blends.
        for tid,s in self.accepted_pipelines():
            if len(pool)>=32:break
            if tid not in pool and np.std(np.load(self.root/'runs'/tid/'prediction.npy'))>=1e-12:pool.append(tid)
        write(self.root/'ensemble_pool.json',{'ids':pool,'frozen':time.time()})
        bases=[];models=set()
        for tid,s in self.accepted_pipelines():
            if s['model'] not in models:bases.append(read(self.root/'runs'/tid/'task.json')['recipe']);models.add(s['model'])
            if len(bases)==4:break
        if len(bases)>=2:
            for i,method in enumerate(['weights','logistic','routing']):
                spec={'bases':bases,'method':method};tid='n_'+identity(spec)[:20]
                self.submit('pc2' if i%2==0 else 'pc3',spec,end,mode='nested',tid=tid)
        if len(pool)>=2:fixed(self.root,self.bank,pool,8192,min(end,time.time()+3000))
        self.wait_phase(end,mode=['nested'])
    def confirmation(self):
        end=self.phase_end(11.5);write(self.root/'phase.json',{'name':'confirmation','until':end})
        if not (self.root/'finalists.json').exists():
            singles=self.accepted_pipelines();chosen=[];ensembles=[]
            if singles:chosen=[singles[0][0]]
            candidates=[]
            for p in (self.root/'runs').glob('*/scores.json'):
                s=read(p)
                if s['kind'] in ['fixed_ensemble','nested_ensemble']:candidates.append((p.parent.name,s))
            for tid,s in sorted(candidates,key=lambda v:v[1]['pooled_oof']['ks'],reverse=True):
                if s['kind']=='fixed_ensemble':parents=s['recipe']['parents']
                else:
                    base_ids={r['id'] for r in s['recipe']['bases']};parents=[tid for tid,ps in singles if ps['recipe_id'] in base_ids]
                merged=list(dict.fromkeys(chosen+parents))
                if len(merged)<=8:chosen=merged;ensembles.append(tid)
                if len(ensembles)==4:break
            for tid,_ in singles:
                if len(chosen)>=8:break
                if tid not in chosen:chosen.append(tid)
            all_ids=chosen+ensembles+['reference_v3'];primary=max([(tid,read(self.root/'runs'/tid/'scores.json')['pooled_oof']['ks']) for tid in chosen+ensembles],key=lambda v:v[1])[0] if chosen else None
            final={'ids':all_ids,'singles':chosen,'ensembles':ensembles,'primary':primary,'frozen':time.time(),'reason':'Development CV only; no confirmation predictions or labels used'}
            for host in ['pc1','pc2','pc3']:self.tx.call(host,{'action':'freeze','finalists':final})
        final=read(self.root/'finalists.json')
        for host in ['pc1','pc3']:
            try:self.tx.call(host,{'action':'freeze','finalists':final})
            except Exception as e:event(self.root/'coordinator.jsonl',event='freeze_delivery_pending',host=host,error=str(e))
        for i,tid in enumerate(final['singles']):
            r=read(self.root/'runs'/tid/'task.json')['recipe'];host='pc2' if r['model'] in ['cat','xgb'] else 'pc1'
            self.submit(host,r,end,mode='confirmation',tid='c_'+tid,parent_id=tid)
            if i<4:
                self.submit('pc3' if r['model'] in ['cat','xgb'] else 'pc1',r,end,mode='repeat',tid='r_'+tid,parent_id=tid)
                self.submit('pc3' if r['model'] in ['cat','xgb'] else 'pc1',r,end,mode='temporal',tid='t_'+tid,parent_id=tid)
        self.submit('pc1',{'id':'reference_v3'},end,mode='reference_confirmation',tid='c_reference_v3')
        self.wait_phase(end,mode=['confirmation','reference_confirmation'])
        for i,tid in enumerate(final['ensembles']):
            spec=read(self.root/'runs'/tid/'scores.json')['recipe']
            # Fixed confirmation components have already been collected on PC2.
            self.submit('pc2',spec,end,mode='ensemble_confirmation',tid='c_'+tid,parent_id=tid)
        self.wait_phase(end)
    def run(self):
        from .features import Bank
        from .reporting import report,final_statistics,eda
        self.bank=Bank(self.root)
        # Keep PC1's login session alive without changing the account's linger setting.
        self.hold_log=(self.root/'pc1_hold.log').open('ab');self.hold_pc1()
        try:
            for host in ['pc1','pc2','pc3']:self.tx.call(host,{'action':'start','host':host})
            if time.time()<self.phase_end(9.5):self.run_search()
            if time.time()<self.phase_end(10.5):self.run_ensembles()
            if time.time()<self.phase_end(11.5):self.confirmation()
            write(self.root/'phase.json',{'name':'verification','until':self.cfg['deadline']})
            self.sync()
            for host in ['pc1','pc2','pc3']:
                try:self.tx.call(host,{'action':'stop'})
                except Exception as e:event(self.root/'coordinator.jsonl',event='stop_failed',host=host,error=str(e))
            time.sleep(3);self.sync()
            if not (self.root/'reports/eda_summary.json').exists():eda(self.root,self.bank)
            if (self.root/'finalists.json').exists():final_statistics(self.root,self.cfg['deadline']-120)
            summary=report(self.root,final=True)
            write(self.root/'campaign_status.json',{'state':'completed','time':time.time(),'deadline':self.cfg['deadline'],'summary':summary,'transport_errors':self.errors})
            try:self.tx.call('pc1',{'action':'cleanup_key'})
            except Exception as e:event(self.root/'coordinator.jsonl',event='key_cleanup_pending',error=str(e))
        finally:
            if self.hold is not None and self.hold.poll() is None:self.hold.terminate()
            self.hold_log.close()

    def hold_pc1(self):
        self.hold=subprocess.Popen(self.tx.command('pc1'),stdin=subprocess.PIPE,stdout=self.hold_log,stderr=self.hold_log)
        self.hold.stdin.write(json.dumps({'action':'hold'}).encode());self.hold.stdin.close()

def launch(root,hours=12):
    root=Path(root).resolve()
    if (root/'campaign.json').exists():
        cfg=read(root/'campaign.json')
        if time.time()>=cfg['deadline']:raise ValueError('Expired deadline; refusing to reset it')
        if cfg.get('code_hash')!=code_hash(root):raise ValueError('Source changed; refusing to resume against a different implementation')
        if cfg.get('catalogue_sha256')!=sha(root/'catalogue.json') or cfg.get('bank_id')!=read(root/'bank/manifest.json')['bank_id']:raise ValueError('Frozen catalogue or data identity changed')
    else:
        cfg=read(root/'machines.json');now=time.time();cfg.update(started=now,deadline=now+hours*3600,code_hash=code_hash(root))
        verified=read(root/'deployment_verified.json')
        if verified['code_hash']!=cfg['code_hash'] or verified.get('preflight')!='passed':raise ValueError('Deployment has not passed its launch gates')
        cfg.update(catalogue_sha256=sha(root/'catalogue.json'),bank_id=verified['bank_id'])
        if hours!=12:raise ValueError('This campaign requires its authorized 12-hour duration')
        for host in ['pc1','pc2','pc3']:
            pf=read(root/'preflight'/f'{host}.json')
            if not pf['passed'] or pf['code_hash']!=cfg['code_hash']:raise ValueError('Preflight is missing or stale: '+host)
        write(root/'campaign.json',cfg)
        from datetime import datetime,timezone,timedelta
        local=timezone(timedelta(hours=7))
        (root/'DEADLINE.md').write_text('# Immutable campaign deadline\n\nStart (Asia/Jakarta): '+datetime.fromtimestamp(cfg['started'],local).isoformat()+'\n\nDeadline (Asia/Jakarta): '+datetime.fromtimestamp(cfg['deadline'],local).isoformat()+'\n\nThe 12-hour window includes final verification. Resume preserves this deadline.\n')
    tx=Transport(root,cfg)
    for host in ['pc1','pc3']:tx.call(host,{'action':'configure','campaign':cfg})
    unit='metaexp-'+cfg['id']+'-coordinator'
    active=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True)
    if active.stdout.strip()=='active':return {'unit':unit,'deadline':cfg['deadline'],'already_running':True}
    subprocess.run(['systemctl','--user','reset-failed',unit],capture_output=True)
    condition=str(root/'.venv/bin/python')+' -c '+shlex.quote('import time,sys;sys.exit(0 if time.time()<'+repr(cfg['deadline'])+' else 1)')
    args=['systemd-run','--user','--unit='+unit,'--property=CPUQuota=700%','--property=MemoryMax=20G','--property=MemorySwapMax=0',
          '--property=OOMScoreAdjust=1000',
          '--property=Restart=on-failure','--property=RestartSec=5','--property=KillMode=control-group','--property=WorkingDirectory='+str(root),
          '--property=ExecCondition='+condition,
          '--property=RuntimeMaxSec='+str(int(cfg['deadline']-time.time()+5)),str(root/'.venv/bin/python'),'-u','-m','metaexp','coordinate','--root',str(root)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=20)
    if p.returncode:raise RuntimeError(p.stderr)
    write(root/'launch.json',{'unit':unit,'started':cfg['started'],'deadline':cfg['deadline'],'command':args})
    return {'unit':unit,'deadline':cfg['deadline']}
