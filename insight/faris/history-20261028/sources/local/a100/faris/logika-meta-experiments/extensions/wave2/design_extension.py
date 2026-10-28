"""Additional, semantically deduplicated feature recipes; no outcome labels read."""
import copy
import itertools
from pathlib import Path
import random
from collections import Counter
from metaexp.common import FAMILIES, MODELS, identity, read, write, sha
from metaexp.design import DEFAULT, factor_space, effective, valid
from metaexp.features import Bank


class Resolver:
    def __init__(self, manifest):
        self.bank=object.__new__(Bank);self.bank.manifest=manifest
        self.bank.names=manifest['columns'];self.bank.lookup={n:i for i,n in enumerate(self.bank.names)}
        self.cache={}

    def signature(self, family, config):
        c=config
        if c['missing']=='group' and 'profile' not in c['sources']:raise ValueError('Group imputation requires profile')
        keys=['view','sources','window','half_life','threshold','mature','representation','append','interactions']
        key=identity([family,{k:c[k] for k in keys}])
        if key not in self.cache:
            self.cache[key]=self.bank.columns({'family':family,'config':c})
        names,cats=self.cache[key]
        if not names:raise ValueError('Numeric probe requires at least one numeric input')
        preprocessing={k:c[k] for k in ['missing','indicator','clip','transform','scale','encoding','weight','cohort_weight','reducer','dimensions','select','feature_count','correlation','spline']}
        if not cats:preprocessing['encoding']='unused'
        if c['reducer']=='none':preprocessing['dimensions']=0
        else:
            preprocessing['append_reduction']=c['append']
            preprocessing['scale']='robust' if c['scale']=='robust' else 'standard'
            if c['reducer']=='pls':preprocessing['dimensions']=min(c['dimensions'],8)
        if c['select']!='correlation':preprocessing['correlation']=None
        name_set=set(names)
        priors=[p for p in self.bank.manifest['priors'] if not p.get('legacy') and p['k'] in name_set and p['n'] in name_set]
        if priors:
            preprocessing['prior_strength']=c['strength']
            preprocessing['prior_output']=c['representation'] if c['representation'] in ['rates','posterior'] else 'posterior_and_uncertainty'
        if c['category_normalize'] and any(self.bank.manifest['info'][n].get('price') for n in names):preprocessing['category_normalize']=True
        return identity({'columns':names,'categoricals':cats,'preprocessing':preprocessing}),len(names)


def design(root,parent,target=4096):
    root=Path(root);parent=Path(parent);old=read(parent/'catalogue.json')
    resolver=Resolver(read(parent/'bank/manifest.json'));seen=set();old_signatures=set()
    for rep in old['representations']:
        try:old_signatures.add(resolver.signature(rep['family'],rep['config'])[0])
        except ValueError:pass
    seen.update(old_signatures);rng=random.Random(2026101602);selected={f:[] for f in FAMILIES};rejected=Counter();spaces={}

    def accept(family,c,reason,parent_id=None):
        c=effective(family,copy.deepcopy(c))
        if not valid(family,c):return False
        try:sig,count=resolver.signature(family,c)
        except ValueError:rejected['invalid_or_empty']+=1;return False
        if sig in seen:rejected['previous_or_equivalent']+=1;return False
        seen.add(sig);changes={k:v for k,v in c.items() if v!=DEFAULT[k]}
        rep={'id':'w2f_'+sig[:16],'family':family,'config':c,'hypothesis':reason,'changes':changes,
             'semantic_signature':sig,'resolved_numeric_columns':count,'wave':2,'matched_parent':parent_id,'is_control':False}
        selected[family].append(rep);return True

    for family in FAMILIES:
        space=factor_space(family)
        if family not in ['sources','preprocessing','reduction']:
            space.update({'view':['v3','extended'],'missing':['median','zero','group'],'indicator':[False,True],
                          'clip':[0,.005,.01],'transform':['identity','signed_log','rank'],'encoding':['onehot','frequency','target'],
                          'select':['none','gain','correlation'],'feature_count':[64,128,256,'all']})
        spaces[family]=space
        # New source-view combinations are finite; enumerate them before all else.
        if family=='sources':
            for mask,view in itertools.product(range(1,128),['v3','base','extended']):
                c=copy.deepcopy(DEFAULT);c['sources']=[s for j,s in enumerate(DEFAULT['sources']) if mask&(1<<j)];c['view']=view
                accept(family,c,'Uncovered source-mask and feature-view contrast',old['control'])
            continue
        quota=target//8;original=[r for r in old['representations'] if r['family']==family]
        # Development-independent parents: single-factor, pairwise and diverse old configurations.
        parents=original[:8]+original[8::max(1,len(original)//8)][:8]
        for rep in parents:
            for key,levels in space.items():
                for value in levels:
                    if key=='source_mask':continue
                    c=copy.deepcopy(rep['config']);c[key]=value
                    accept(family,c,'Matched single-change contrast: '+key,rep['id'])
                    if len(selected[family])>=quota//4:break
                if len(selected[family])>=quota//4:break
            if len(selected[family])>=quota//4:break
        pairs=[]
        for a,b in itertools.combinations(space,2):
            for x,y in itertools.product(space[a],space[b]):pairs.append({a:x,b:y})
        rng.shuffle(pairs)
        for change in pairs:
            c=copy.deepcopy(DEFAULT);c.update(change);accept(family,c,'Previously uncovered pair: '+','.join(change),old['control'])
            if len(selected[family])>=quota*3//5:break
        for _ in range(40000):
            if len(selected[family])>=quota:break
            c=copy.deepcopy(DEFAULT);c.update({k:rng.choice(v) for k,v in space.items()})
            accept(family,c,'Seeded higher-order combination')
    # Reallocate the finite source family's unused quota across the other families.
    cursor=0
    while sum(map(len,selected.values()))<target and cursor<100000:
        family=FAMILIES[1+cursor%7];space=spaces[family];c=copy.deepcopy(DEFAULT)
        c.update({k:rng.choice(v) for k,v in space.items()});accept(family,c,'Seeded higher-order coverage completion');cursor+=1
    representations=[next(r for r in old['representations'] if r['id']==old['control'])]
    for i in range(max(map(len,selected.values()))):
        for family in FAMILIES:
            if i<len(selected[family]):representations.append(selected[family][i])
    recipes=[r for r in old['recipes'] if r['representation']['id']==old['control']]
    for rep in representations[1:]:
        for model in MODELS:
            recipe={'kind':'pipeline','representation':rep,'model':model,'seed':2026,'control':old['control']}
            recipe['id']='w2p_'+identity(recipe)[:20];recipes.append(recipe)
    coverage={}
    for family,space in spaces.items():
        reps=selected[family];coverage[family]={'registered':len(reps),'factor_levels':space,'uncovered_pairs':{}}
        for a,b in itertools.combinations(space,2):
            if a=='source_mask' or b=='source_mask':continue
            actual={(str(r['config'].get(a)),str(r['config'].get(b))) for r in reps}
            missing=[list(v) for v in itertools.product(space[a],space[b]) if tuple(map(str,v)) not in actual]
            coverage[family]['uncovered_pairs'][a+'|'+b]=missing
    catalogue={'representations':representations,'recipes':recipes,'coverage':coverage,'control':old['control'],
               'limits':{'representations':len(representations),'ensembles':8192},'wave':2,
               'parent_catalogue_sha256':sha(parent/'catalogue.json'),'new_representations':len(representations)-1,
               'new_pipeline_recipes':len(recipes)-4,'repeated_control_recipes':4,
               'deduplication':{'original_semantic_signatures':len(old_signatures),'rejected':dict(rejected)},
               'confirmation_policy':'Development-only extension after wave-one confirmation. No new confirmation access or nomination changes.'}
    assert len(recipes)==len({r['id'] for r in recipes})
    assert all(r['semantic_signature'] not in old_signatures for r in representations[1:])
    write(root/'catalogue.json',catalogue)
    write(root/'design_summary.json',{k:catalogue[k] for k in ['wave','new_representations','new_pipeline_recipes','repeated_control_recipes','deduplication','parent_catalogue_sha256','confirmation_policy']})
    return catalogue

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--target',type=int,default=4096);a=p.parse_args()
    c=design(Path.cwd(),a.parent,a.target);print({k:c[k] for k in ['new_representations','new_pipeline_recipes','repeated_control_recipes']})
