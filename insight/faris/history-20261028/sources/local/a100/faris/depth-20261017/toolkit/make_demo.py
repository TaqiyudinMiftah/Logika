#!/usr/bin/env python3
"""Prepare an explicitly synthetic, self-contained example campaign; never launch it."""
import argparse, importlib.metadata, json, sys
from pathlib import Path
from oneshot.common import atomic, bundle

def dependency_pins(names):
    from packaging.requirements import Requirement
    found = {}
    def visit(name):
        d = importlib.metadata.distribution(name); key = d.metadata['Name']
        if key in found: return
        found[key] = d.version
        for text in d.requires or []:
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate({'extra': ''}): visit(req.name)
    for name in names: visit(name)
    return ''.join(f'{k}=={v}\n' for k,v in sorted(found.items()))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True); p.add_argument('--name',default='oneshot-demo'); p.add_argument('--budget-seconds',type=float,default=60); p.add_argument('--graph',action='store_true'); a=p.parse_args()
    root=Path(a.root).resolve(strict=True); work=root/a.name
    work.mkdir(); source=work/'source'; source.mkdir()
    example=Path(__file__).resolve().parents[1]/'assets/example_adapter.py'
    (source/'adapter.py').write_text(example.read_text()); (source/'requirements.lock').write_text(dependency_pins(['optuna','psutil','numpy']))
    data={'x':[[i/120,(i%7)/7] for i in range(120)],'y':[i/60+(i%7)/14 for i in range(120)],'time':list(range(120)),
        'graph':{'a':[['b',1],['c',5]],'b':[['c',1]],'c':[]},'start':'a','goal':'c'}
    atomic(work/'data.json',data)
    atomic(work/'screen.json',{'train':list(range(80)),'valid':list(range(80,100)),'kind':'temporal'})
    atomic(work/'confirm.json',{'train':list(range(100)),'valid':list(range(100,120)),'kind':'temporal'})
    evidence={}
    for name in ('research','eda','hypotheses','applicability','rules'):
        path=work/(name+'.md'); path.write_text('# Synthetic demonstration\n\nThis is a toolkit smoke-test fixture, not competition research or evidence.\n'+name+'\n'); evidence[name]=str(path)
    resources={'cpus':1,'ram_mb':256,'disk_mb':2,'timeout_seconds':20}
    families=[{'name':'ridge','hypothesis':'linear baseline sensitivity','parameters':{'alpha':{'type':'categorical','choices':[.01,.1,1.]}},'resources':resources},
        {'name':'mlp','hypothesis':'small nonlinear model','parameters':{'hidden':{'type':'categorical','choices':[4,8]},'rate':{'type':'float','low':.01,'high':.1,'log':True}},'resources':resources,
         'fidelity':{'max_steps':20,'min_steps':5,'reduction_factor':2,'group':'full'}}]
    if a.graph: families=[{'name':'dfs','hypothesis':'exact feasible path search','parameters':{},'resources':resources}]
    c={'version':1,'id':a.name,'competition_root':str(root),'adapter':'adapter.py','metric':'path_cost' if a.graph else 'MSE','direction':'minimize',
        'inputs':{'train':str(work/'data.json')},'protocols':{'screen':{'split':str(work/'screen.json')},'confirm':{'split':str(work/'confirm.json')}},
        'families':families,'brief':{'description':'Synthetic toolkit example','rules':evidence['rules'],'compute_access':'local','external_data_policy':'none','pretrained_policy':'none'},
        'evidence':{k:v for k,v in evidence.items() if k!='rules'},'budget_seconds':a.budget_seconds,'storage_limit_mb':256,'max_trials':8,'max_workers':2,'seeds':[42],'confirmation_seeds':[43,44],'confirmation_top_k':2}
    atomic(work/'input.json',c); output=bundle(work/'input.json',source,root/'campaigns'/a.name)
    print(json.dumps({'prepared':str(output),'launch':[sys.executable,str(output/'runtime/campaign.py'),'run',str(output),'--detach']},indent=2))

if __name__=='__main__': main()
