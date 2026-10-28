"""Check preservation and label-boundary invariants before model search."""
from pathlib import Path
import hashlib,json,tempfile
from types import SimpleNamespace
import numpy as np
import materialize as m
import finalize as f
R=Path(__file__).resolve().parents[1]
assert f.ks([0,0,1,1],[0,0,1,1])==1
assert f.ks([0,0,1,1],[.5,.5,.5,.5])==0
assert f.ks([0,0,1,1],[1,1,0,0])==1
base=np.r_[0.,1.,np.linspace(.01,.99,19998)]
assert np.array_equal(f.shifted(base,np.ones(20000),0),base)
rng=np.random.default_rng(4826);y=np.r_[np.zeros(4500,int),np.ones(1500,int)];rng.shuffle(y)
raw=rng.uniform(.2,.8,20000);direction=np.r_[.5*(y-.25),rng.normal(0,.02,14000)]
from sklearn.model_selection import train_test_split
_,choice=train_test_split(np.arange(6000),test_size=.5,stratify=y,random_state=261026)
plan={'primary':{'weights':[0,.1,.2,.4]}}
a=f.choose_weight(y,raw,direction,choice,plan,0)
assert a[0]>0,'Strong synthetic improvement did not select a positive weight'
direction[6000:]=rng.normal(0,100,14000)
b=f.choose_weight(y,raw,direction,choice,plan,0)
assert a==b,'Outer predictions affected weight selection'
with tempfile.TemporaryDirectory(prefix='logika-augment90-check-') as temp:
    t=Path(temp);(t/'selection').mkdir();(t/'replay_reference').mkdir();old=m.R;m.R=t
    ids=[f'base{i}' for i in range(90)]
    (t/'selection/fold0.json').write_text(json.dumps({'baseline_features':ids,'groups':{'test':['extra']}}))
    (t/'replay_reference/START.json').write_text(json.dumps({'columns':ids[:89]}))
    values={fid:np.full(70000,i//2,dtype=float) for i,fid in enumerate(ids)}
    values['extra']=np.arange(70000,dtype=float)
    fake=SimpleNamespace(fold=0,ctx=np.arange(50000),specs={k:k for k in values},compute=lambda key:values[key])
    control,_=m.materialize(fake,'baseline90');augmented,_=m.materialize(fake,'test');replay,_=m.materialize(fake,'replay89')
    assert list(control)==ids and list(augmented)[:90]==ids and len(augmented.columns)==91 and len(replay.columns)==89
    assert np.array_equal(control.to_numpy(),augmented.iloc[:,:90].to_numpy())
    m.R=old
record={'status':'passed','known_KS_cases':True,'weight_zero_exact_baseline':True,
        'outer_prediction_perturbation_changes_weight_selection':False,
        'all90_columns_including_constants_and_duplicates_preserved':True,
        'augmentation_leaves_all90_values_and_order_unchanged':True,'replay89_exception_scoped_to_pilot':True,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'source/materialize.py',R/'source/finalize.py']}}
(R/'LOGIC_GATE.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
