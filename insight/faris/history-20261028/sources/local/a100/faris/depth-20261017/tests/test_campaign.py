import json, os, subprocess, sys, time
from pathlib import Path
import pytest
from oneshot.common import read, atomic, validate_bundle, sha
from oneshot.controller import Controller
from oneshot.reporting import report, checked_records
from oneshot.space import pool, canonical, coverage


def cli(root,*args):
    return subprocess.run([sys.executable,str(root/'runtime/campaign.py'),*args,str(root)],capture_output=True,text=True,timeout=30)


def test_real_training_and_protected_confirmation(factory):
    families=[{'name':'ridge','hypothesis':'regularized linear baseline','parameters':{'alpha':{'type':'categorical','choices':[.01,.1]}}},
        {'name':'mlp','hypothesis':'nonlinear representation','parameters':{'hidden':{'type':'categorical','choices':[4,8]}},
         'fidelity':{'max_steps':12,'min_steps':4,'reduction_factor':2,'group':'full'}}]
    root=factory(families,adapter='real',max_trials=4,with_confirm=True,confirmation_top_k=2,budget_seconds=45)
    state=Controller(root).run(); summary=read(root/'reports/summary.json')
    assert state['status']=='finished'
    assert summary['counts']['screen_jobs']==4
    assert summary['counts']['confirmation_jobs']==2
    assert summary['counts']['complete']>=4
    for r in checked_records(root):
        if r['status']=='complete':
            assert r['models'] and r['score']>=0
            assert all((root/r['directory']/m).exists() for m in r['models'])
    assert (root/'reports/report.md').exists()
    assert not list(root.rglob('*submission*'))


def test_dfs_solution(factory):
    root=factory([{'name':'dfs','hypothesis':'exact path enumeration','parameters':{}}],adapter='real',max_trials=1)
    Controller(root).run(); r=checked_records(root)[0]
    assert r['score']==2 and r['models']==['solution.json']


def test_conditional_space_and_pairs(factory):
    f={'name':'toy','hypothesis':'conditional','parameters':{'a':{'type':'categorical','choices':['x','y']},'b':{'type':'categorical','choices':[1,2],'when':{'a':'x'}}}}
    rows,exact=pool(f); assert exact and len(rows)==3
    assert {'a':'y'} in rows
    with pytest.raises(ValueError): canonical(f,{'a':'x'})
    root=factory([f]); Controller(root).run(); result=report(root)
    assert result['coverage']['toy']['pairs_covered']==result['coverage']['toy']['pairs_total']==2
    assert result['counts']['unique_configurations']==3


def test_tamper_rejected(factory):
    root=factory(); Controller(root).run()
    r=checked_records(root)[0]; (root/r['directory']/'model.json').write_text('{}')
    assert checked_records(root)[0]['status']=='invalid'
    assert report(root)['counts']['invalid']==1
    config=read(root/'config.json'); config['metric']='fake'; atomic(root/'config.json',config)
    with pytest.raises(ValueError,match='config changed'): validate_bundle(root)


def test_failure_and_timeout_preserved(factory):
    f={'name':'toy','hypothesis':'failures','parameters':{'mode':{'type':'categorical','choices':['fail','sleep']},'seconds':{'type':'categorical','choices':[2]}},
        'resources':{'cpus':1,'ram_mb':128,'disk_mb':.05,'timeout_seconds':.3}}
    root=factory([f],max_trials=2,cancel_grace_seconds=.1)
    Controller(root).run(); records=checked_records(root)
    assert any(r['status']=='failed' for r in records)
    assert any(r['status']=='timeout' for r in records)
    assert all((root/r['directory']/'stderr.log').exists() for r in records)


def test_pruning_keeps_models(factory):
    f={'name':'toy','hypothesis':'prune weak trials','parameters':{'mode':{'type':'categorical','choices':['prune']},'x':{'type':'categorical','choices':[0,100,200,300]}},
       'fidelity':{'max_steps':8,'min_steps':1,'reduction_factor':2},'anchors':[{'params':{'mode':'prune','x':0}}]}
    root=factory([f],max_trials=4,max_workers=1); Controller(root).run()
    pruned=[r for r in checked_records(root) if r['status']=='pruned']
    assert pruned and all(r['models'] for r in pruned)


def test_resource_and_disk_admission(factory):
    root=factory(ram_limit_mb=1,max_trials=1); Controller(root).run()
    assert read(root/'state.json')['admission_failures'][0]['reason']=='cpu_ram_or_gpu_capacity'
    root=factory(storage_limit_mb=.001,max_trials=1); Controller(root).run()
    assert not checked_records(root)
    assert read(root/'state.json')['reason']=='storage_budget'


def test_input_and_split_changes_rejected(factory):
    root=factory(); c=read(root/'config.json'); Path(c['inputs']['train']).write_text('{}')
    with pytest.raises(ValueError,match='input files changed'): validate_bundle(root)
    root=factory(); c=read(root/'config.json'); Path(c['protocols']['screen']['split']).write_text('{}')
    with pytest.raises(ValueError,match='frozen artifact'): validate_bundle(root)


def test_detach_stop_resume_and_exclusion(factory):
    f={'name':'toy','hypothesis':'resume','parameters':{'mode':{'type':'categorical','choices':['resume']}}}
    root=factory([f],max_trials=1,budget_seconds=30,cancel_grace_seconds=.1)
    launch=cli(root,'run','--detach'); assert launch.returncode==0,launch.stderr
    state=read(root/'state.json'); deadline=state['deadline']
    duplicate=cli(root,'run'); assert duplicate.returncode != 0
    for _ in range(100):
        if list(root.glob('runs/*/*/checkpoint.json')): break
        time.sleep(.03)
    assert cli(root,'stop').returncode==0
    for _ in range(200):
        if read(root/'state.json')['status']=='stopped': break
        time.sleep(.03)
    result=cli(root,'resume'); assert result.returncode==0,result.stderr
    state=read(root/'state.json'); assert state['deadline']==deadline
    assert list(root.glob('runs/*/attempt_002/resumed.json'))
    assert read(root/'reports/summary.json')['counts']['retries']==1


def test_nonfinite_score_rejected(factory):
    root=factory()
    # Mutate test adapter before re-bundling rather than violating a sealed campaign.
    from oneshot.common import bundle
    parent=root.parent.parent; code=parent/'source/adapter.py'
    code.write_text(code.read_text().replace('return {"score": json.loads(ctx.path(result["validation"][0]).read_text())["score"]}', 'return {"score": float("nan")}'))
    fresh=parent/'campaigns/nonfinite'; bundle(parent/'input-config.json',parent/'source',fresh)
    Controller(fresh).run(); assert all(r['status']=='failed' for r in checked_records(fresh))


def test_no_external_runtime_clients():
    from conftest import SKILL
    text='\n'.join(p.read_text() for p in (SKILL/'scripts').rglob('*.py'))
    for token in ('import openai','import anthropic','import requests','import httpx','kaggle competitions','wandb.init'):
        assert token not in text

@pytest.mark.stress
def test_thousand_jobs(factory):
    root=factory([{'name':'toy','hypothesis':'orchestration scale only','parameters':{'x':{'type':'int','low':0,'high':999}}}],
                 max_trials=1000,max_workers=8,budget_seconds=900,storage_limit_mb=512,poll_seconds=.01)
    Controller(root).run(); s=read(root/'reports/summary.json')
    assert s['counts']['complete']==1000
    assert s['counts']['unique_configurations']==1000
    assert len(list(root.glob('runs/*/*/model.json')))==1000
    assert s['counts']['retries']==0


def test_abrupt_controller_loss_and_checkpoint_recovery(factory):
    import signal
    f={'name':'toy','hypothesis':'crash recovery','parameters':{'mode':{'type':'categorical','choices':['resume']}}}
    root=factory([f],max_trials=1,budget_seconds=30,orphan_timeout_seconds=.5,cancel_grace_seconds=.1)
    launch=cli(root,'run','--detach'); assert launch.returncode==0,launch.stderr
    owner=read(root/'owner.json'); deadline=read(root/'state.json')['deadline']
    for _ in range(100):
        if list(root.glob('runs/*/*/checkpoint.json')): break
        time.sleep(.02)
    os.kill(owner['pid'],signal.SIGKILL)
    time.sleep(1.2)
    status=cli(root,'status'); assert json.loads(status.stdout)['status']=='controller_lost'
    result=cli(root,'resume'); assert result.returncode==0,result.stderr
    assert read(root/'state.json')['deadline']==deadline
    assert list(root.glob('runs/*/attempt_002/resumed.json'))
    assert read(root/'reports/summary.json')['counts']['complete']==1


def test_verify_and_replay(factory,tmp_path):
    root=factory(max_trials=1); Controller(root).run(); job=checked_records(root)[0]['job_id']
    cmd=[sys.executable,str(root/'runtime/campaign.py')]
    v=subprocess.run(cmd+['verify',str(root),job],capture_output=True,text=True,timeout=15)
    assert v.returncode==0,v.stderr
    dest=tmp_path/'replay'
    r=subprocess.run(cmd+['replay',str(root),job,'--output',str(dest)],capture_output=True,text=True,timeout=15)
    assert r.returncode==0,r.stderr
    assert read(dest/'outcome.json')['status']=='complete'
    assert len(checked_records(root))==1


def test_mixed_protocol_record_rejected(factory):
    root=factory(max_trials=1); Controller(root).run()
    path=next((root/'records').glob('*.json')); value=read(path); value['split_sha']='wrong'; atomic(path,value)
    assert checked_records(root)[0]['status']=='invalid'
    assert not report(root)['scoreboards']


def test_dependency_drift_rejected(factory):
    from oneshot.common import bundle
    root=factory(); parent=root.parent.parent; lock=parent/'source/requirements.lock'
    lock.write_text('optuna==0.0.0\n')
    fresh=parent/'campaigns/drift'; bundle(parent/'input-config.json',parent/'source',fresh)
    with pytest.raises(ValueError,match='dependency drift'): Controller(fresh)


def test_multi_gpu_admission_simulated(factory,monkeypatch):
    root=factory(max_trials=1)
    controller=Controller(root)
    host={'cpus':8,'ram_available_mb':16000,'disk_free_mb':100000,'gpus':[{'id':'0','free_mb':10000},{'id':'1','free_mb':10000}]}
    controller.hosts={'local':host}
    monkeypatch.setattr('oneshot.controller.probe',lambda *a:host)
    job={'resources':{'cpus':2,'ram_mb':100,'disk_mb':1,'gpus':2,'vram_mb':4000}}
    allocation,reason=controller.admit(job)
    assert reason is None and allocation['gpu_ids']==['0','1']
    controller.lock.close()


def test_insufficient_time_is_not_complete_coverage(factory):
    root=factory(budget_seconds=.2,max_trials=1)
    Controller(root).run(); s=read(root/'reports/summary.json')
    assert s['counts']['complete']==0
    assert s['counts']['budget_skipped'] in (0,1)
    assert s['reason'] in ('deadline','search_exhausted')
    assert s['untested']['not_completed']==3
    with pytest.raises(ValueError,match='finished'): Controller(root,resume=True)
