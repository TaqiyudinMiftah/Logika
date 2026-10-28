"""Scheduler-only upgrade. Frozen metaexp training code is deliberately unchanged."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(name,'1')
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import io
import math
import shlex
import subprocess
import sys
import tarfile
import time
from metaexp.common import read,write,event,sha,identity,code_hash
from metaexp.coordinator import Coordinator,Transport

HOSTS=['pc1','pc2','pc3']

class BatchTransport(Transport):
    def __init__(self,root,cfg):
        super().__init__(root,cfg);self.buffer={h:{} for h in HOSTS};self.delivered=set();self.buffering=True
        runtime=Path(os.environ.get('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid())))
        self.sockets=runtime/('mx_'+identity(str(self.root))[:10]);self.sockets.mkdir(mode=0o700,parents=True,exist_ok=True);self.sockets.chmod(0o700)
    def command(self,host):
        base=super().command(host)
        if host=='pc3':base[-1]='cd '+shlex.quote(self.cfg['machines'][host]['root'])+' && .venv/bin/python rpc_parallel.py'
        return [base[0],'-o','ControlMaster=auto','-o','ControlPersist=60','-o','ControlPath='+str(self.sockets/'s-%C'),*base[1:]]
    def call(self,host,request,timeout=30):
        if request['action']!='push' or not self.buffering:return super().call(host,request,timeout)
        for task in request['tasks']:
            key=(host,task['id'])
            if key in self.delivered:continue
            old=self.buffer[host].get(task['id'])
            if old is not None and old!=task:raise ValueError('Changed buffered task')
            self.buffer[host][task['id']]=task
        return {'buffered':len(request['tasks'])}
    def flush(self):
        batches={h:list(v.values()) for h,v in self.buffer.items() if v};results={}
        def send(host,tasks):
            result=Transport.call(self,host,{'action':'push','tasks':tasks},timeout=45)
            if host=='pc2':
                from rpc_parallel import prioritize_gpu
                prioritize_gpu(self.root,tasks)
            return result
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(send,h,tasks):h for h,tasks in batches.items()}
            for future in as_completed(futures):
                host=futures[future]
                try:
                    future.result();results[host]={'delivered':len(batches[host])}
                    for task in batches[host]:self.delivered.add((host,task['id']));self.buffer[host].pop(task['id'],None)
                except Exception as exc:results[host]={'error':str(exc)}
        return results
    def collect_many(self,host,ids):
        if host=='pc2':return
        result=subprocess.run(self.command(host),input=json.dumps({'action':'pack_many','ids':ids}).encode(),capture_output=True,timeout=180)
        if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-1500:])
        with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
            total=0
            for member in archive.getmembers():
                rel=Path(member.name)
                if not member.isfile() or rel.is_absolute() or '..' in rel.parts or len(rel.parts)<3 or rel.parts[0]!='runs' or rel.parts[1] not in ids:raise ValueError('Unsafe batch archive')
                total+=member.size
                if total>512*1024**2:raise ValueError('Unexpected batch archive size')
            archive.extractall(self.root,filter='data')

class ParallelCoordinator(Coordinator):
    def __init__(self,root):
        super().__init__(root);revision=read(self.root/'scheduler_revision.json')
        if revision['script_sha256']!=sha(Path(__file__)):raise ValueError('Scheduler revision differs')
        if revision['rpc_sha256']!=sha(self.root/'rpc_parallel.py'):raise ValueError('Batch transport revision differs')
        if revision['training_code_hash']!=code_hash(self.root) or self.cfg['code_hash']!=code_hash(self.root):raise ValueError('Frozen training source changed')
        if revision['started']!=self.cfg['started'] or revision['deadline']!=self.cfg['deadline']:raise ValueError('Original clock changed')
        self.revision=revision['script_sha256'];self.tx=BatchTransport(self.root,self.cfg);self.queue_counts=Counter()
        self.last_report=0;self.report_process=None;self.archive_backlog=[]
        for record in self.ledger.all():
            path=self.root/'runs'/record['id']/'scores.json'
            if record['verified'] and record['mode']=='cv' and path.exists():
                score=read(path);self.costs.setdefault(score['model'],[]).append(score.get('elapsed_seconds',300))
            complete=self.root/'runs'/record['id']/'complete.json'
            if record['verified'] and record['host']=='pc1' and complete.exists():self.archive_backlog.append({'id':record['id'],'complete_sha256':sha(complete)})
    def flush(self):
        outcomes=self.tx.flush()
        if outcomes:event(self.root/'scheduler_parallel.jsonl',event='batch_delivery',outcomes=outcomes)
    def sync(self):
        start=time.monotonic();self.flush()
        if self.hold_log is not None and (self.hold is None or self.hold.poll() is not None):self.hold_pc1()
        terminal=[];records={r['id']:r for r in self.ledger.all()}
        with ThreadPoolExecutor(max_workers=3) as pool:
            requests={pool.submit(self.tx.call,h,{'action':'status'}):h for h in HOSTS}
            for future in as_completed(requests):
                host=requests[future]
                try:
                    status=future.result();self.host_status[host]=status;self.errors.pop(host,None)
                    heartbeat=status.get('heartbeat') or {};active=set(heartbeat.get('active',[]))
                    if time.time()<self.phase_end(11.5) and time.time()-heartbeat.get('time',0)>30:self.tx.call(host,{'action':'start','host':host})
                    for tid,state in status['runs'].items():
                        record=records.get(tid)
                        if record and record['host']==host and record['state'] in ['assigned','running'] and tid not in active:terminal.append((host,tid,state))
                except Exception as exc:self.errors[host]=str(exc)
        # Network transfers run concurrently. SQLite and verification stay on
        # this coordinator thread; a run remains assigned until evidence arrives.
        compacted=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            phase=read(self.root/'phase.json').get('name') if (self.root/'phase.json').exists() else None
            batch_size=1 if phase in ['confirmation','verification'] else 16
            batches=[(host,[v for v in terminal if v[0]==host][offset:offset+batch_size]) for host in HOSTS for offset in range(0,sum(v[0]==host for v in terminal),batch_size)]
            copies={pool.submit(self.tx.collect_many,h,[v[1] for v in items]):(h,items) for h,items in batches}
            for future in as_completed(copies):
                host,items=copies[future]
                try:future.result()
                except Exception as exc:self.errors[host]=str(exc);continue
                for _,tid,state in items:
                    verified=False
                    if state['state']=='completed':
                        from metaexp.reporting import verify_run
                        try:
                            score=verify_run(self.root,tid,self.bank);verified=True
                            if score.get('model'):self.costs.setdefault(score['model'],[]).append(score.get('elapsed_seconds',300))
                        except Exception as exc:
                            event(self.root/'coordinator.jsonl',event='verification_failed',task=tid,error=str(exc));state={'state':'failed'}
                    self.ledger.finish(tid,state['state'],verified)
                    if verified and host=='pc1':compacted.append({'id':tid,'complete_sha256':sha(self.root/'runs'/tid/'complete.json')})
        compacted+=self.archive_backlog[:32];self.archive_backlog=self.archive_backlog[32:]
        for offset in range(0,len(compacted),32):
            try:self.tx.call('pc1',{'action':'compact','items':compacted[offset:offset+32]},timeout=60)
            except Exception as exc:event(self.root/'scheduler_parallel.jsonl',event='archive_compaction_pending',error=str(exc))
        rows=self.ledger.all();self.queue_counts=Counter()
        for record in rows:
            if record['state']=='assigned':self.queue_counts[(record['host'],json.loads(record['task']).get('device','cpu'))]+=1
        write(self.root/'campaign_status.json',{'state':'running','time':time.time(),'deadline':self.cfg['deadline'],
            'phase':read(self.root/'phase.json') if (self.root/'phase.json').exists() else {},'hosts':{h:s.get('heartbeat') for h,s in self.host_status.items()},'errors':self.errors,
            'counts':dict(Counter(r['state'] for r in rows)),'scheduler':{'revision':self.revision,'cycle_seconds':time.monotonic()-start,
            'queued_or_running':{h:{d:n for (hh,d),n in self.queue_counts.items() if hh==h} for h in HOSTS}}})
    def submit(self,host,recipe,until,mode='cv',tid=None,**extra):
        tid=tid or recipe['id']
        if self.ledger.exists(tid):return False
        device='cuda' if recipe.get('model') in ['cat','xgb'] and host in ['pc2','pc3'] else 'cpu'
        if host=='pc1' and recipe.get('model')=='lgb' and self.cfg['machines'][host].get('opencl') and self.queue_counts[(host,'opencl')]<4:device='opencl'
        if mode in ['nested','ensemble_confirmation'] and host in ['pc2','pc3']:device='cuda'
        task={'id':tid,'mode':mode,'recipe':recipe,'device':device,'until':until,'scheduler_revision':self.revision,**extra}
        self.ledger.insert(task,host);self.tx.call(host,{'action':'push','tasks':[task]});self.queue_counts[(host,device)]+=1
        return True
    def dispatch(self,end):
        records=self.ledger.all();assigned={r['recipe_id'] for r in records}
        for host in HOSTS:
            if host in self.errors:continue
            workers=self.cfg['machines'][host]['workers']
            cpu_samples=[v for key in ['lgb','linear'] for v in self.costs.get(key,[])[-64:]]
            cpu_forecast=max(5,sorted(cpu_samples)[int(.9*(len(cpu_samples)-1))]) if cpu_samples else 60
            cpu_target=max(workers+2,min(workers*3,math.ceil(workers*120/cpu_forecast)))
            gpu_samples=[v for key in ['cat','xgb'] for v in self.costs.get(key,[])[-64:]]
            gpu_forecast=max(5,sorted(gpu_samples)[int(.9*(len(gpu_samples)-1))]) if gpu_samples else 60
            gpu_target=max(2,min(8,math.ceil(120/gpu_forecast)))
            for recipe in self.catalogue['recipes']:
                if recipe['id'] in assigned:continue
                gpu=recipe['model'] in ['cat','xgb']
                if gpu and (host=='pc1' or self.queue_counts[(host,'cuda')]>=gpu_target):continue
                cpu_jobs=self.queue_counts[(host,'cpu')]+self.queue_counts[(host,'opencl')]
                if not gpu and cpu_jobs>=cpu_target:continue
                samples=self.costs.get(recipe['model'],[])[-64:]
                forecast=max(60,sorted(samples)[int(.9*(len(samples)-1))]*1.4) if samples else 300
                if time.time()+forecast>=end:continue
                if self.submit(host,recipe,min(end,time.time()+max(900,forecast*2))):assigned.add(recipe['id'])
        self.retry_delivery();self.flush()
    def run_search(self):
        end=self.phase_end(9.5);write(self.root/'phase.json',{'name':'full_cv_search','until':end})
        controls=[r for r in self.catalogue['recipes'] if r['representation']['id']==self.catalogue['control']]
        for recipe in controls:self.submit('pc3' if recipe['model']=='cat' else 'pc2' if recipe['model']=='xgb' else 'pc1',recipe,end)
        lgb=next(r for r in controls if r['model']=='lgb');self.submit('pc2',lgb,end,tid=lgb['id']+'_cpu')
        self.submit('pc1',{'id':'reference_v3'},self.phase_end(2),mode='reference',tid='reference_v3');self.flush()
        while time.time()<end:
            self.sync();self.dispatch(end)
            rows=self.ledger.all()
            if len({r['recipe_id'] for r in rows})>=len(self.catalogue['recipes'])+1 and not any(r['state']=='assigned' for r in rows):break
            if not (self.root/'reports/eda_summary.json').exists():
                from metaexp.reporting import eda
                eda(self.root,self.bank)
            # Reporting has its own owned process in the same resource group.
            # Deep local queues continue executing while it generates evidence.
            if time.time()-self.last_report>3600 and (self.report_process is None or self.report_process.poll() is not None):
                log=(self.root/'progress_report.log').open('ab')
                code='from pathlib import Path;from metaexp.reporting import report;report(Path.cwd(),final=False)'
                self.report_process=subprocess.Popen([str(self.root/'.venv/bin/python'),'-c',code],cwd=self.root,stdout=log,stderr=log);log.close();self.last_report=time.time()
            time.sleep(2)
        self.wait_phase(self.phase_end(9.55),mode=['cv','reference'])
        if self.report_process is not None and self.report_process.poll() is None:
            self.report_process.terminate()
            try:self.report_process.wait(timeout=8)
            except subprocess.TimeoutExpired:self.report_process.kill();self.report_process.wait()
    def run_ensembles(self):
        self.flush();self.tx.buffering=False
        return super().run_ensembles()
    def wait_phase(self,until,mode=None):
        self.flush()
        while time.time()<until:
            self.sync();self.retry_delivery();self.flush()
            if not any(r['state']=='assigned' and (mode is None or r['mode'] in mode) for r in self.ledger.all()):return
            time.sleep(2)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
    ParallelCoordinator(args.root).run()

if __name__=='__main__':main()
