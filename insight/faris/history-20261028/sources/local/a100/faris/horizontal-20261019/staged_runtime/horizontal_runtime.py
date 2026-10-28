"""Campaign-local finite controller. Generic adaptive search is disabled."""
import argparse,fcntl,importlib.metadata,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from oneshot.common import atomic,read,sha,validate_bundle,check_dependencies,size_mb

def event(root,value):
    value={'time':time.time(),**value};print(json.dumps(value),flush=True)
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')

def source(root):return root/'snapshots'/read(root/'seal.json')['snapshot']/'code'

def save(root,s):
    atomic(root/'state.json',s);atomic(root/'heartbeat.json',{'pid':os.getpid(),'time':time.time()})

def stop_process(p):
    if p.poll() is not None:return
    # Every worker is a new session owned by this controller.
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=4)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait()

def interrupted(root,s):return time.time()>=s['deadline'] or (root/'stop.json').exists()

def command(root,s,args,logname,timeout=1800):
    log=(root/logname).open('ab');p=subprocess.Popen([sys.executable,str(source(root)/'research.py'),*args],stdout=log,stderr=log,start_new_session=True)
    start=time.time();event(root,{'stage_worker':args[0],'pid':p.pid});s['stage_pid']=p.pid;save(root,s)
    try:
        while p.poll() is None:
            if interrupted(root,s) or time.time()-start>timeout:stop_process(p);raise TimeoutError('Stopped or absolute stage deadline')
            save(root,s);time.sleep(2)
        if p.returncode:raise RuntimeError(f'{args[0]} failed; see {logname}')
    finally:stop_process(p);log.close();s.pop('stage_pid',None);save(root,s)

def batch(root,s,jobs,c):
    pending=[];active={}
    for spec in jobs:
        d=root/'fits'/spec['key']
        if (d/'complete.json').exists():
            for name,h in read(d/'complete.json')['files'].items():assert sha(d/name)==h
            s['jobs'][spec['key']]={'status':'complete','directory':str(d.relative_to(root))};continue
        if d.exists():raise RuntimeError('Retain incomplete attempt; explicit investigation required: '+str(d))
        pending.append(spec)
    try:
        while pending or active:
            if interrupted(root,s):raise TimeoutError('Operator stop or 24-hour deadline')
            if psutil.disk_usage(root).free<10*2**30 or size_mb(root)>c['storage_limit_mb']-1024:raise RuntimeError('Storage admission stopped; artifacts retained')
            # Measured host free memory and reserved capacity both restrict concurrency.
            while pending and len(active)<c['max_workers'] and psutil.virtual_memory().available>(5+4*len(active))*2**30:
                spec=pending.pop(0);d=root/'fits'/spec['key'];d.mkdir(parents=True);atomic(d/'spec.json',spec)
                log=(d/'worker.log').open('ab');p=subprocess.Popen([sys.executable,str(source(root)/'research.py'),'worker',str(root),
                    '--spec',str(d/'spec.json'),'--folder',str(d)],stdout=log,stderr=log,start_new_session=True)
                active[spec['key']]={'p':p,'log':log,'start':time.time(),'directory':d,'rss':0,'cpu_seconds':0}
                s['jobs'][spec['key']]={'status':'running','pid':p.pid,'created':psutil.Process(p.pid).create_time(),'directory':str(d.relative_to(root))}
                event(root,{'started_fit':spec['key'],'pid':p.pid,'remaining':len(pending)})
            for key,x in list(active.items()):
                p=x['p'];rc=p.poll()
                if rc is None:
                    try:
                        procs=[psutil.Process(p.pid)];procs+=procs[0].children(recursive=True)
                        rss=sum(q.memory_info().rss for q in procs);cpu=sum(q.cpu_times().user+q.cpu_times().system for q in procs)
                        x['rss']=max(x['rss'],rss);x['cpu_seconds']=max(x['cpu_seconds'],cpu)
                        if rss>5*2**30 or time.time()-x['start']>3600:
                            stop_process(p);rc=p.returncode
                    except psutil.Error:pass
                if rc is not None:
                    x['log'].close();d=x['directory'];atomic(d/'resources.json',{'peak_rss_bytes':x['rss'],'cpu_seconds_observed':x['cpu_seconds'],
                       'wall_seconds':time.time()-x['start'],'exit_code':rc})
                    good=rc==0 and (d/'complete.json').exists();s['jobs'][key]['status']='complete' if good else 'failed';active.pop(key)
                    event(root,{'completed_fit':key,'status':s['jobs'][key]['status'],'complete':sum(j['status']=='complete' for j in s['jobs'].values()),'planned_base_fits':160})
                    if not good:raise RuntimeError('Fit failed: '+key)
            save(root,s);time.sleep(2)
    finally:
        for key,x in active.items():
            stop_process(x['p']);x['log'].close();s['jobs'][key]['status']='partial'
        save(root,s)

def baseline_gate(root,s):
    sys.path.insert(0,str(source(root)));import research as r
    import numpy as np
    c,b,t=r.setup(root);y=t.target.to_numpy(int);out=np.full(len(y),np.nan);fold=np.zeros(len(y),int)
    for f in range(5):
        vi=r.parts(c,f)['validation'];p=np.column_stack([r.loadfit(root,r.key(2026,f,'full',f'champion{i}'),vi) for i in range(5)])
        out[vi]=np.sort(p,axis=1)[:,1:4].mean(1);fold[vi]=f
    scores=r.metrics(y,out);fs=[r.ks(y[fold==f],out[fold==f]) for f in range(5)]
    rng=np.random.default_rng(31027);groups=[np.flatnonzero((fold==f)&(y==cl)) for f in range(5) for cl in [0,1]];bs=[]
    for _ in range(200):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);bs.append(r.ks(y[ix],out[ix]))
    ci=np.quantile(bs,[.025,.975]).tolist()
    ok=bool(np.isfinite(out).all() and .35<scores['ks']<.7 and .5<scores['auc']<1 and .1<scores['logloss']<.6 and min(fs)>.3 and max(fs)-min(fs)<.08 and ci[1]-ci[0]<.04)
    result={'conditional_methodology_gate':'pass' if ok else 'inconclusive','contract_sha':c['contract_sha'],'baseline':scores,'fold_ks':fs,'bootstrap95':ci,
       'required_baseline_rules':{'ks_range':[.35,.7],'auc_above_chance':True,'logloss_range':[.1,.6],'fold_range_max':.08,'CI_width_max':.04},
       'scope':'Admits the frozen small mechanistic pilot panel only','independent_audit_available':False,'snapshot_provenance_certified':False,'broad_search_admitted':False}
    np.savez_compressed(root/'champion_pilot.npz',target=y,prediction=out,fold=fold);atomic(root/'gate_result.json',result)
    if not ok:raise RuntimeError('Baseline gate inconclusive: no further training admitted')
    event(root,{'gate':'conditional_methodology_pass','scope':'finite research panel only','broad_search':False})

def run(root,resume=False):
    root=Path(root).resolve();lock=(root/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=validate_bundle(root);check_dependencies(root)
    assert c.get('finite_research') is True and c['budget_seconds']==86400
    sys.path.insert(0,str(source(root)));import research
    if resume:
        s=read(root/'state.json');assert s['status']!='complete' and time.time()<s['deadline'];assert not (root/'stop.json').exists(),'Remove own STOP request explicitly before resume'
        for key,j in s['jobs'].items():
            if j.get('status')=='running':
                try:
                    q=psutil.Process(j['pid'])
                    if abs(q.create_time()-j['created'])<.01:raise RuntimeError('Uncertain worker still alive; refusing duplicate: '+key)
                except psutil.NoSuchProcess:pass
    else:
        assert not (root/'state.json').exists(),'Existing clock; use resume'
        now=time.time();s={'start':now,'deadline':now+c['budget_seconds'],'phase':'validation','status':'running','jobs':{},'id':c['id']}
    atomic(root/'owner.json',{'pid':os.getpid(),'created':psutil.Process().create_time()});save(root,s)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    if (root/'environment.json').exists():assert read(root/'environment.json')['packages']==packages
    else:atomic(root/'environment.json',{'packages':packages,'python':sys.version,'host':os.uname().nodename})
    event(root,{'phase':'validation','deadline':s['deadline'],'research_hypotheses':3,'broad_search_enabled':False})
    try:
        if not (root/'preflight.json').exists():command(root,s,['preflight',str(root)],'preflight.log',600)
        jobs=research.fit_schedule();pilots=[j for j in jobs if j['seed']==2026 and j['partition']=='full']
        batch(root,s,pilots,c)
        if not (root/'gate_result.json').exists():baseline_gate(root,s)
        assert read(root/'gate_result.json')['conditional_methodology_gate']=='pass'
        s['phase']='focused_research';save(root,s)
        batch(root,s,[j for j in jobs if j not in pilots],c)
        s['phase']='fitting_honest_fusion_heads';save(root,s)
        for seed in [2026,2027]:
            for f in range(5):
                command(root,s,['fuse',str(root),'--seed',str(seed),'--fold',str(f)],f'head_s{seed}_f{f}.log')
                event(root,{'completed_heads':{'seed':seed,'fold':f}})
        s['phase']='reporting';save(root,s);command(root,s,['finish',str(root)],'report.log',7200)
        s.update(status='complete',phase='terminal',finished=time.time());save(root,s);event(root,{'DONE':True,'report':str(root/'REPORT.md')})
    except BaseException as e:
        s.update(status='stopped' if interrupted(root,s) else 'failed',phase='terminal',reason=str(e),finished=time.time());save(root,s)
        atomic(root/'failure.json',{'traceback':traceback.format_exc()})
        (root/'REPORT.md').write_text('# Research stopped\n\n'+str(e)+'\n\nAll available artifacts retained. No partial recipe is promoted. See state.json, worker logs and failure.json.\n')
        event(root,{'DONE':True,'status':s['status'],'reason':str(e)});raise

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','resume','status','stop','report']);p.add_argument('campaign');p.add_argument('--detach',action='store_true');a=p.parse_args();root=Path(a.campaign).resolve()
    if a.command=='status':
        s=read(root/'state.json') if (root/'state.json').exists() else {'status':'prepared'};jobs=s.pop('jobs',{});s['fit_counts']={v:sum(x['status']==v for x in jobs.values()) for v in ['complete','running','failed','partial']};print(json.dumps(s,indent=2));return
    if a.command=='stop':atomic(root/'stop.json',{'requested_at':time.time()});print('Stop requested for this campaign only');return
    if a.command=='report':print((root/'REPORT.md').read_text() if (root/'REPORT.md').exists() else 'Report not produced yet');return
    if a.detach:
        with (root/'controller.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,__file__,a.command,str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print(json.dumps({'launch_requested_pid':child.pid,'live_activity_not_yet_verified':True}));return
    run(root,a.command=='resume')

if __name__=='__main__':main()
