"""Verify selection is invariant to outer predictions and rejects row corruption."""
from pathlib import Path
import hashlib,importlib.util,json,tempfile
import numpy as np

R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('synthesis_under_test',R/'source/synthesize.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
rng=np.random.default_rng(7261026)
with tempfile.TemporaryDirectory(prefix='logika-selection-check-') as temp:
    t=Path(temp);s.R=t
    plan=json.loads((R/'PLAN.json').read_text())
    (t/'PLAN.json').write_text(json.dumps(plan))
    packs=[{'id':p} for p in ['baseline90','p1','p2']]
    (t/'CATALOG.json').write_text(json.dumps({'packs':packs,'pack_count':len(packs)}))
    for fold in range(5):
        q=t/f'packets/fold{fold}';q.mkdir(parents=True)
        y=np.r_[np.zeros(4500,int),np.ones(1500,int)];rng.shuffle(y)
        np.savez(q/'roles.npz',correction=np.arange(6000),validation=np.arange(6000,20000),y_correction=y)
        raw=np.r_[np.clip(.1+.5*y+rng.normal(0,.25,6000),0,1),rng.uniform(0,1,14000)]
        np.savez(q/'baseline.npz',raw=raw)
        for i,p in enumerate(packs):
            dest=t/f'runs/small2/{p["id"]}/fold{fold}';dest.mkdir(parents=True)
            pred=np.r_[np.clip(.1+.5*y+rng.normal(0,.1+.15*i,6000),0,1),rng.uniform(0,1,14000)]
            np.savez(dest/'prediction.npz',raw=pred,rows=np.arange(20000))
            (dest/'RESULT.json').write_text(json.dumps({'status':'complete','prediction_sha256':s.sha(dest/'prediction.npz')}))
    s.select();first=json.loads((t/'selection/SELECTION.json').read_text())
    (t/'selection').rename(t/'first_selection')
    for dest in (t/'runs').glob('small2/*/fold*'):
        z=np.load(dest/'prediction.npz');pred=z['raw'].copy();pred[6000:]=1-pred[6000:]
        np.savez(dest/'prediction.npz',raw=pred,rows=z['rows'])
        (dest/'RESULT.json').write_text(json.dumps({'status':'complete','prediction_sha256':s.sha(dest/'prediction.npz')}))
    s.select();second=json.loads((t/'selection/SELECTION.json').read_text())
    for a,b in zip(first['selection'],second['selection']):
        for key in ['ranked_packs','selected_members','selected_weight','choice_KS','confirmation_shortlist','trials']:
            assert a[key]==b[key],key
        fold=a['fold'];raw=np.load(t/f'selection/fold{fold}_screened.npz')['raw']
        base=np.load(t/f'packets/fold{fold}/baseline.npz')['raw']
        average=np.mean([s.load(k,fold) for k in b['selected_members']],axis=0)
        assert np.allclose(raw,(1-b['selected_weight'])*base+b['selected_weight']*average)
    dest=t/'runs/small2/p1/fold0';z=np.load(dest/'prediction.npz')
    np.savez(dest/'prediction.npz',raw=z['raw'],rows=z['rows'][::-1])
    (dest/'RESULT.json').write_text(json.dumps({'status':'complete','prediction_sha256':s.sha(dest/'prediction.npz')}))
    rejected=False
    try:s.load('p1',0)
    except AssertionError:rejected=True
    assert rejected
record={'status':'passed','synthetic_only':True,'outer_prediction_perturbation_changes_selection':False,
        'row_corruption_rejected':True,'mixture_recomputed_exactly':True,
        'source_sha256':hashlib.sha256((R/'source/synthesize.py').read_bytes()).hexdigest()}
(R/'SYNTHESIS_GATE.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
