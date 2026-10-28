"""Fit all predeclared combiners inside each outer-training partition."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from common import R,load,dump,sha,utc,check_seal
from gating import calibrate,calibrated,GateEncoder,fit_gate,predict_gate


def completed(task_id):
    root=R/'runs'/task_id
    for attempt in sorted(root.glob('attempt*'),key=lambda p:int(p.name[7:])):
        if (attempt/'RESULT.json').exists():
            result=json.loads((attempt/'RESULT.json').read_text())
            start=json.loads((attempt/'START.json').read_text())
            assert result['status']=='complete' and result['weights_saved'] is False
            assert sha(attempt/'prediction.npz')==result['prediction_sha256']
            assert start['contract_sha256']==load('CLOCK.json')['contract_sha256']
            return attempt
    return None


def task_id(family,view,seed,fold):
    return f'{family}__{view}__s{seed}__f{fold}'


def raw_predictions(family,view,seed,fold):
    families=load('CONFIG.json')['base_families'] if family=='hybrid' else [family]
    values=[];paths=[]
    expected=np.load(R/f'packets/fold{fold}/roles.npz')
    rows=np.r_[expected['correction'],expected['validation']]
    for f in families:
        path=completed(task_id(f,view,seed,fold))
        if path is None:return None,[]
        record=np.load(path/'prediction.npz')
        assert np.array_equal(record['rows'],rows)
        values.append(record['raw']);paths.append(str(path.relative_to(R)))
    return np.mean(values,axis=0),paths


def compose(family,condition,seed,fold):
    root=R/f'composed/{family}/{condition}/s{seed}/fold{fold}'
    if(root/'RESULT.json').exists():
        previous=json.loads((root/'RESULT.json').read_text())
        assert sha(root/'prediction.npz')==previous['prediction_sha256']
        for path,digest in previous['source_predictions_hashes'].items():assert sha(R/path/'prediction.npz')==digest
        return root
    if root.exists():
        failures=R/'failures';failures.mkdir(exist_ok=True)
        root.rename(failures/f'compose_{family}_{condition}_{seed}_{fold}_{__import__("time").time_ns()}')
    config=load('CONFIG.json');packet=R/f'packets/fold{fold}';roles=np.load(packet/'roles.npz')
    anchor=np.load(packet/'anchor.npz');yc=roles['y_correction'];assignment=anchor['inner_fold']
    meta=[anchor['correction_oof']];query=[anchor['validation']];inputs=[]
    domains=['union'] if condition=='wide' else config['domains']
    for domain in domains:
        view='expanded_union' if condition=='wide' else condition+'_'+domain
        raw,paths=raw_predictions(family,view,seed,fold)
        if raw is None:return None
        inner=np.full(6000,np.nan)
        for k in range(3):
            training=np.flatnonzero(assignment!=k);heldout=np.flatnonzero(assignment==k)
            theta=calibrate(raw[training],yc[training]);inner[heldout]=calibrated(theta,raw[heldout])
        full=calibrate(raw[:6000],yc)
        meta.append(inner);query.append(calibrated(full,raw[6000:]));inputs+=paths
    meta=np.column_stack(meta);query=np.column_stack(query)
    assert np.isfinite(meta).all() and np.isfinite(query).all()
    gate=pd.read_parquet(packet/'gate.parquet');training=gate.iloc[roles['correction']].copy();validation=gate.iloc[roles['validation']].copy()
    if config['gate_include_scores']:
        for i in range(meta.shape[1]):
            training[f'expert_logit_{i}']=logit(np.clip(meta[:,i],1e-6,1-1e-6))
            validation[f'expert_logit_{i}']=logit(np.clip(query[:,i],1e-6,1-1e-6))
    encoder=GateEncoder().fit(training);zt=encoder.transform(training);zq=encoder.transform(validation)
    prior=[.9,.1] if condition=='wide' else config['gate_prior']
    beta,audit=fit_gate(zt,meta,yc,prior,config['gate_penalty'])
    gated,weights=predict_gate(zq,query,beta,prior);fixed=query@np.asarray(prior)
    root.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(root/'prediction.npz',rows=roles['validation'],fixed=fixed,gated=gated,
                        routing_probabilities=weights,expert_probabilities=query)
    dump(dict(status='complete',family=family,condition=condition,seed=seed,fold=fold,
              source_inputs=inputs,source_predictions_hashes={p:sha(R/p/'prediction.npz')for p in inputs},
              prediction_sha256=sha(root/'prediction.npz'),gate=audit,
              preprocessing_recipe=encoder.recipe(),calibration='3-fold heldout correction-row monotone logistic; final calibration fits all6k',
              anchor_crossfit='Head4000 and spline54000 exclude each2000 correction rows; frozen Kumo context50000 excludes all6000.',
              gate_training_rows=6000,outer_labels_loaded=False,weights_saved=False,
              mean_routing_probabilities=weights.mean(0).tolist(),UTC=utc()),root/'RESULT.json')
    return root


def main():
    check_seal();config=load('CONFIG.json');complete=[];missing=[]
    for family in config['base_families']+['hybrid']:
        for condition in (['existing','expanded','wide'] if family in ['lightgbm','catboost'] else ['existing','expanded']):
            for seed in config['seeds']:
                for fold in range(5):
                    root=compose(family,condition,seed,fold)
                    key=[family,condition,seed,fold]
                    (complete if root is not None else missing).append(key)
    # Freeze scientific identities before any new outer candidate score is computed.
    dump(dict(UTC=utc(),config_sha256=sha(R/'CONFIG.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),
              contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),primary=config['primary'],
              all_models_predeclared=True,complete=complete,incomplete=missing,
              selection_after_outer_scores=False,independent_audit_available=False,
              source_manifests={str(f):sha(R/f'packets/fold{f}/MANIFEST.json')for f in range(5)},
              composed_results={str(p.relative_to(R)):sha(p)for p in sorted((R/'composed').rglob('RESULT.json'))}),R/'FROZEN.json')
    print(json.dumps(dict(stage='combiners_frozen',complete=len(complete),incomplete=len(missing))),flush=True)


if __name__=='__main__':
    main()
