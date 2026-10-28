"""Predeclared secondary control: learned global versus borrower-dependent weights.

The primary model, acceptance rule and original clock are unchanged. This control
is fitted from the same held-out correction predictions, before its outer scores
are read. Learned coefficients are never serialized.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import R,load,dump,sha,metrics,utc,check_seal
from compose import raw_predictions
from gating import calibrate,calibrated,fit_gate,predict_gate
from evaluate import paired


def main():
    clock=check_seal();assert time.time()<clock['deadline_epoch']-60
    config=load('CONFIG.json');plan=load('ADDITIONAL_DIAGNOSTIC_PLAN.json')
    assert plan['primary_model_change_allowed']is False
    assert sha(Path(__file__))==plan['source_sha256']
    output=R/'diagnostics/constant_gate';output.mkdir(parents=True,exist_ok=True)
    if(output/'RESULT.json').exists():return
    gathered={}
    for seed in config['seeds']:
        for fold in range(5):
            values=[];inputs=[]
            for domain in config['domains']:
                raw,paths=raw_predictions('kumo','expanded_'+domain,seed,fold)
                if raw is None:
                    dump(dict(status='INCOMPLETE_INPUTS',primary_changed=False,UTC=utc()),output/'RESULT.json');return
                values.append(raw);inputs+=paths
            gathered[(seed,fold)]=(values,inputs)
    seeds=[];audit=[]
    for seed in config['seeds']:
        result=np.full(70000,np.nan)
        for fold in range(5):
            assert time.time()<clock['deadline_epoch']-60
            packet=R/f'packets/fold{fold}';roles=np.load(packet/'roles.npz');anchor=np.load(packet/'anchor.npz')
            y=roles['y_correction'];assignment=anchor['inner_fold'];meta=[anchor['correction_oof']];query=[anchor['validation']]
            raw_values,paths=gathered[(seed,fold)]
            for raw in raw_values:
                inner=np.full(6000,np.nan)
                for k in range(3):
                    train=assignment!=k;heldout=assignment==k
                    theta=calibrate(raw[:6000][train],y[train]);inner[heldout]=calibrated(theta,raw[:6000][heldout])
                theta=calibrate(raw[:6000],y);meta.append(inner);query.append(calibrated(theta,raw[6000:]))
            meta=np.column_stack(meta);query=np.column_stack(query)
            cached=np.load(R/f'composed/kumo/expanded/s{seed}/fold{fold}/prediction.npz')['expert_probabilities']
            component_error=float(np.max(abs(query-cached)))
            assert component_error<1e-12,component_error
            beta,fit=fit_gate(np.ones((6000,1)),meta,y,config['gate_prior'],config['gate_penalty'])
            prediction,_=predict_gate(np.ones((14000,1)),query,beta,config['gate_prior'])
            result[roles['validation']]=prediction
            audit.append(dict(seed=seed,fold=fold,fit=fit,correction_rows=6000,outer_labels_loaded=False,
                              component_probability_replay_max_abs=component_error,
                              source_predictions={p:sha(R/p/'prediction.npz')for p in paths}))
        assert np.isfinite(result).all();seeds.append(result)
    prediction=np.mean(seeds,axis=0)
    np.savez_compressed(output/'prediction.npz',global_gate=prediction,seed0=seeds[0],seed1=seeds[1])
    dump(dict(UTC=utc(),source_sha256=sha(Path(__file__)),plan_sha256=sha(R/'ADDITIONAL_DIAGNOSTIC_PLAN.json'),
              primary_config_sha256=sha(R/'CONFIG.json'),clock_sha256=sha(R/'CLOCK.json'),
              prediction_sha256=sha(output/'prediction.npz'),fits=audit,
              all_diagnostic_models_frozen_before_diagnostic_outer_scoring=True,
              coefficients_saved=False),output/'FROZEN.json')
    y=np.load(R/'evaluation_inputs/target.npy');fold=np.load(R/'evaluation_inputs/baseline.npz')['fold']
    existing=np.load(R/'evaluation/predictions.npz');conditional=existing[config['primary']]
    fixed=existing['kumo__expanded__fixed']
    contrast=paired(y,conditional,prediction,fold,2000)
    dump(dict(status='COMPLETE',UTC=utc(),global_weight_metrics=metrics(y,prediction),
              conditional_minus_learned_global=contrast,global_minus_fixed=paired(y,prediction,fixed,fold,2000),
              conditional_routing_benefit_supported_in_reused_development=contrast['percentile95'][0]>0,
              primary_model_changed=False,selection_or_promotion_based_on_diagnostic=False,
              independence='Previously exposed development; no independent audit or public-score inference.',
              frozen_sha256=sha(output/'FROZEN.json')),output/'RESULT.json')


if __name__=='__main__':
    main()
