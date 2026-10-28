from oneshot.controller import Controller
from oneshot.common import probe

def test_gpu_queue_does_not_block_cpu(factory,monkeypatch):
    real=probe()
    real['cpus']=4;real['gpus']=[{'id':'0','free_mb':16000,'total_mb':16000,'uuid':'synthetic'}]
    monkeypatch.setattr('oneshot.controller.probe',lambda *a:dict(real))
    monkeypatch.setattr('oneshot.executors.probe',lambda *a:dict(real))
    resources={'cpus':1,'ram_mb':256,'disk_mb':.1,'timeout_seconds':10}
    params={'mode':{'type':'categorical','choices':['sleep']},'seconds':{'type':'categorical','choices':[.7]},'x':{'type':'int','low':0,'high':30}}
    root=factory([{'name':'gpu','hypothesis':'simulated GPU scheduling only','parameters':params,'resources':dict(resources,gpus=1,vram_mb=1000)}, {'name':'cpu','hypothesis':'fill CPU slots behind GPU wait','parameters':params,'resources':resources}],max_workers=4,max_trials=16,budget_seconds=30)
    s=Controller(root).run();jobs=[j for j in s['jobs'].values() if j['status']=='complete']
    assert len(jobs)==16
    cpus=[j for j in jobs if j['family']=='cpu'];gpus=[j for j in jobs if j['family']=='gpu']
    assert any(sum(k['started']<=j['started']<k['finished'] for k in cpus)>=2 for j in cpus)
    assert all(a['finished']<=b['started'] or b['finished']<=a['started'] for i,a in enumerate(gpus) for b in gpus[i+1:])
