import importlib.metadata, json, shutil, sys
from pathlib import Path
import pytest
SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / 'toolkit'))
from oneshot.common import atomic, bundle

TOY = '''import json, time

def run_trial(ctx):
    mode = ctx.params.get("mode", "normal")
    if mode == "fail": raise ValueError("deliberate adapter failure")
    if mode == "sleep": time.sleep(ctx.params.get("seconds", 2))
    if mode == "memory":
        value = bytearray(80 * 1024 * 1024); time.sleep(.5)
    if mode == "resume":
        from pathlib import Path
        old = Path(ctx.resume_from) / "checkpoint.json" if ctx.resume_from else None
        if old and old.exists(): ctx.save_json("resumed.json", {"resumed": True})
        ctx.save_json("checkpoint.json", {"step": 1}); ctx.model("checkpoint.json")
        for i in range(50): ctx.check_stop(); time.sleep(.05)
    ctx.save_json("model.json", {"x": ctx.params.get("x", 0)})
    ctx.model("model.json")
    if mode == "prune":
        for step in range(1, ctx.max_steps + 1):
            ctx.report(step, float(ctx.params.get("x", 0))); time.sleep(.03)
    ctx.save_json("validation.json", {"score": float(ctx.params.get("x", 0))})
    return {"validation": ["validation.json"]}

def evaluate(ctx, result):
    return {"score": json.loads(ctx.path(result["validation"][0]).read_text())["score"]}
'''

@pytest.fixture
def factory(tmp_path):
    def create(families=None, adapter='toy', **overrides):
        index = len(list(tmp_path.iterdir())); root = tmp_path / str(index); root.mkdir()
        source = root / 'source'; source.mkdir()
        (source / 'adapter.py').write_text(TOY if adapter == 'toy' else (SKILL / 'assets/example_adapter.py').read_text())
        versions = {p: importlib.metadata.version(p) for p in ('optuna', 'psutil', 'numpy')}
        (source / 'requirements.lock').write_text(''.join(f'{p}=={v}\n' for p,v in versions.items()))
        data = {'x': [[i / 120, (i % 7) / 7] for i in range(120)], 'y': [i / 60 + (i % 7) / 14 for i in range(120)],
            'time': list(range(120)), 'group': [i // 10 for i in range(120)],
            'graph': {'a': [['b', 1], ['c', 5]], 'b': [['c', 1]], 'c': []}, 'start': 'a', 'goal': 'c'}
        atomic(root / 'train.json', data)
        atomic(root / 'split.json', {'train': list(range(80)), 'valid': list(range(80,100)), 'kind': 'temporal'})
        atomic(root / 'confirm.json', {'train': list(range(100)), 'valid': list(range(100,120)), 'kind': 'group'})
        evidence = {}
        for key in ('research','eda','hypotheses','applicability','rules'):
            p = root / (key + '.md'); p.write_text('Synthetic test fixture; no competition or external data. '+key); evidence[key]=str(p)
        resources = {'cpus': 1, 'ram_mb': 256, 'disk_mb': .05, 'timeout_seconds': 20}
        if families is None:
            families = [{'name':'toy','hypothesis':'test distinct configurations','parameters':{'x':{'type':'categorical','choices':[1,2,3]}},'resources':resources}]
        else:
            families = [{**f,'resources':f.get('resources',resources)} for f in families]
        cfg = {'version':1,'id':f'fixture_{index}','competition_root':str(root),'adapter':'adapter.py','metric':'loss','direction':'minimize',
            'inputs':{'train':str(root/'train.json')},'protocols':{'screen':{'split':str(root/'split.json')}},
            'families':families,'brief':{'description':'Synthetic integration fixture','rules':evidence['rules'],'compute_access':'local','external_data_policy':'none','pretrained_policy':'none'},
            'evidence':{k:v for k,v in evidence.items() if k != 'rules'},'budget_seconds':30,'storage_limit_mb':256,
            'max_trials':3,'max_workers':2,'seeds':[42],'poll_seconds':.03,'max_retries':1,'disk_headroom_mb':1}
        cfg.update(overrides)
        if cfg.pop('with_confirm',False): cfg['protocols']['confirm']={'split':str(root/'confirm.json')}
        atomic(root/'input-config.json', cfg)
        campaign = root / 'campaigns' / cfg['id']; bundle(root/'input-config.json',source,campaign)
        return campaign
    return create
