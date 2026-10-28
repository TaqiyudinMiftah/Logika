"""A finite, deterministic catalogue. Recipe counts never include folds or retries."""
from __future__ import annotations
import copy
import itertools
import random
from .common import FAMILIES,SOURCES,MODELS,identity,write,read

DEFAULT={'view':'v3','sources':SOURCES,'window':'all','half_life':90,'threshold':30,'strength':6,
         'mature':True,'representation':'raw','missing':'median','indicator':False,'clip':0,
         'transform':'identity','scale':'none','encoding':'onehot','weight':6,'cohort_weight':False,
         'reducer':'none','dimensions':64,'select':'none','feature_count':'all','correlation':0.95,
         'category_normalize':False,'interactions':False,'spline':False,'append':True}

def factor_space(family):
    windows=[30,60,90,180,365,'all']
    if family=='sources':return {'source_mask':list(range(1,128)),'view':['v3','base','extended']}
    if family=='repayment':return {'window':windows,'half_life':[30,90,180,365],'threshold':[1,7,30,60,90],
        'strength':[0,2,6,12,24,48],'mature':[False,True],'representation':['raw','rates','posterior','uncertainty','loan_weighted','decay']}
    if family=='bureau':return {'window':windows,'half_life':[30,90,180,365],
        'representation':['raw','robust','provider','disagreement','latest','decay'],'append':[False,True]}
    if family=='account':return {'window':windows,'representation':['raw','robust','trend','volatility','persistence','coverage'],'append':[False,True],'clip':[0,.01]}
    if family=='shopping':return {'window':windows,'representation':['raw','prices','composition','payment','cancelled','seasonality'],
        'category_normalize':[False,True],'append':[False,True],'clip':[0,.01]}
    if family=='cross_source':return {'window':windows,'representation':['raw','affordability','contracts','income_gap','ratios','products'],
        'interactions':[False,True],'append':[False,True]}
    if family=='preprocessing':return {'missing':['median','zero','group'],'indicator':[False,True],
        'clip':[0,0.005,0.01],'transform':['identity','signed_log','yeo','quantile_normal','rank'],
        'scale':['none','standard','robust'],'encoding':['onehot','ordinal','frequency','target'],
        'weight':[1,3,6],'cohort_weight':[False,True],'spline':[False,True]}
    return {'reducer':['pca','group_pca','svd','ica','projection','nmf','pls','cluster','none'],
        'dimensions':[8,16,32,64,128],'select':['none','gain','mi','variance','correlation'],
        'feature_count':[32,64,128,256,'all'],'correlation':[0.90,0.95,0.99],'append':[False,True]}

def valid(family,c):
    if family=='reduction' and c['reducer']=='none' and c['select']=='none':return False
    if c['encoding']=='target' and c['spline']:return False
    return True

def effective(family,c):
    if family=='reduction':
        if c['reducer']=='none':c['dimensions']=64;c['append']=True
        if c['feature_count']!='all':c['dimensions']=min(c['dimensions'],int(c['feature_count']))
        if c['select']!='correlation':c['correlation']=.95
    if family=='repayment':
        if c['representation'] not in ['raw','decay']:c['half_life']=90
        if c['representation'] in ['rates','loan_weighted','decay']:c['strength']=0
        if c['representation']=='decay':c['mature']=False
    if family=='bureau' and c['representation'] not in ['raw','decay']:c['half_life']=90
    if family=='shopping' and c['representation'] not in ['raw','prices']:c['category_normalize']=False
    if family=='cross_source' and c['representation']=='raw':c['interactions']=False
    return c

def design(root,n=1024):
    root=root.resolve(); rng=random.Random(20261016)
    representations=[]; seen=set(); coverage={}
    for family in FAMILIES:
        space=factor_space(family);quota=n//8+(FAMILIES.index(family)<n%8)
        candidates=[]
        # Deliberately include the untouched representation as a common matched control.
        candidates.append({})
        for key,levels in space.items():
            for v in levels:candidates.append({key:v})
        keys=list(space)
        pairs=[]
        for a,b in itertools.combinations(keys,2):
            for va,vb in itertools.product(space[a],space[b]):pairs.append({a:va,b:vb})
        rng.shuffle(pairs);candidates.extend(pairs)
        # Bounded higher-order sampling fills each family, with complete effective configs.
        for _ in range(30000):candidates.append({k:rng.choice(v) for k,v in space.items()})
        selected=[]
        for changes in candidates:
            c=copy.deepcopy(DEFAULT);c.update(changes)
            if 'source_mask' in c:
                mask=c.pop('source_mask');c['sources']=[s for i,s in enumerate(SOURCES) if mask&(1<<i)]
            c=effective(family,c)
            if not valid(family,c):continue
            semantics=family if family in ['repayment','bureau','account','shopping','cross_source'] else 'generic'
            sig=identity({'family':semantics,'config':c})
            if sig in seen:continue
            seen.add(sig);r={'id':'f_'+sig[:16],'family':family,'config':c,'hypothesis':f'{family}: '+str(changes),
                            'changes':changes,'is_control':len(representations)==0}
            representations.append(r);selected.append(r)
            if len(selected)==quota:break
        coverage[family]={'registered':len(selected),'factor_levels':space,'one_factor_coverage':{},'pairwise_coverage':{}}
        for k,levels in space.items():
            vals={str(r['changes'].get(k,DEFAULT.get(k))) for r in selected}
            coverage[family]['one_factor_coverage'][k]={'tested_levels':sorted(vals),'uncovered':[v for v in levels if str(v) not in vals]}
        for a,b in itertools.combinations(space,2):
            actual={(str(r['changes'].get(a,DEFAULT.get(a))),str(r['changes'].get(b,DEFAULT.get(b)))) for r in selected}
            possible=list(itertools.product(space[a],space[b]));missing=[list(v) for v in possible if tuple(map(str,v)) not in actual]
            coverage[family]['pairwise_coverage'][a+'|'+b]={'possible':len(possible),'covered':len(possible)-len(missing),'uncovered':missing}
    control=representations[0]['id'];recipes=[]
    # Round robin ensures initial batches touch every research family and probe model.
    for j in range(max(sum(r['family']==f for r in representations) for f in FAMILIES)):
        for family in FAMILIES:
            reps=[r for r in representations if r['family']==family]
            if j>=len(reps):continue
            r=reps[j]
            for model in MODELS:
                x={'kind':'pipeline','representation':r,'model':model,'seed':2026,'control':control}
                x['id']='p_'+identity(x)[:20];recipes.append(x)
    write(root/'catalogue.json',{'representations':representations,'recipes':recipes,'coverage':coverage,
                                'limits':{'representations':n,'ensembles':8192},'control':control})
    return recipes
