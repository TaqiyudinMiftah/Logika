"""Replay all five H016 pipelines from retained fitted bundles; no fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import joblib
import numpy as np
import baseline as b
from robustness import R, inputs, stages, read
from spline_direct_ks import predict

destination=R/'spline_direct_ks/PIPELINE_REPLAY_AUDIT.json'
assert not destination.exists()
plan=read(R/'H016.json')
for name,digest in plan['source_sha256'].items():assert b.sha(R/'source'/name)==digest
assert b.sha(R/'ROBUSTNESS_OOF.npz')==plan['reference_sha256']
reference=np.load(R/'ROBUSTNESS_OOF.npz')
scored=np.load(R/'spline_direct_ks/OOF.npz')
records=[]
for f in range(5):
    original,ix,q,_,_,_,raw,frame,mature,models=inputs(f)
    query,table=stages(q,frame,mature,'periodicity')
    root=R/f'spline_direct_ks/fold{f}'
    complete=read(root/'complete.json')
    for name,digest in complete['files'].items():assert b.sha(root/name)==digest
    bundle=joblib.load(root/'spline_update.joblib')
    assert bundle['training_ID_hash']==b.ids_hash(original,ix['train'])
    assert bundle['validation_ID_hash']==b.ids_hash(original,ix['validation'])
    assert not len(np.intersect1d(ix['train'],ix['validation']))
    old_spline=b.predict_spline(models['periodicity'][1],table.iloc[ix['validation']])
    new_spline=predict(bundle,table.iloc[ix['validation']])
    saved=np.load(root/'validation.npz')
    assert np.array_equal(saved['spline_delta'],new_spline-old_spline)
    preds={}
    for mode in ('original','fresh_kumo_seed'):
        if mode=='original':head=models['periodicity'][0];base=raw
        else:
            head=joblib.load(R/f'robustness/fold{f}/periodicity__fresh_kumo_seed.joblib')
            base=np.load(R/f'robustness/fold{f}/raw_seed_repeat.npz')['prediction']
        corrected=b.corrected_probability(head,query.iloc[6000:],base[6000:])[0]
        old_pipeline=.75*corrected+.25*old_spline
        old_error=float(np.max(abs(old_pipeline-reference[f'periodicity__{mode}'][ix['validation']])))
        assert old_error<2e-14
        preds[mode]=.75*corrected+.25*new_spline
    preds['two_seed_mean']=.5*(preds['original']+preds['fresh_kumo_seed'])
    errors={mode:float(np.max(abs(p-scored[mode][ix['validation']])))for mode,p in preds.items()}
    assert max(errors.values())<2e-14
    records.append(dict(fold=f,branch_delta_bitwise_equal=True,pipeline_max_errors=errors,
                        train_validation_disjoint=True,training_rows=56000,validation_rows=14000))
b.dump(dict(status='passed',records=records,validation_rows_total=70000,
            hypothesis_sha256=b.sha(R/'H016.json'),source_sha256=b.sha(__file__),
            OOF_sha256=b.sha(R/'spline_direct_ks/OOF.npz'),
            qualification='Artifact replay only; not independent outcome confirmation.'),destination)
print('All five retained H016 pipeline bundles replayed successfully.',flush=True)
