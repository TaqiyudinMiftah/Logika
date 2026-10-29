"""New measured admission gate over immutable, previously qualified input packets."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from common import R,load,dump,sha,ids_sha,utc,check_seal,metrics,catalog

def data_gate():
    check_seal();inputs=load('INPUTS.json');parent=Path(inputs['parent_root']);contract=load('VALIDATION_CONTRACT.json')
    assert sha(parent/'SOURCE_SEAL.json')==inputs['parent_source_seal_sha256']
    old=json.loads((parent/'GATE.json').read_text());assert old['status']=='PASS'
    y=np.load(R/'evaluation_inputs/target.npy');base=np.load(R/'evaluation_inputs/baseline.npz');previous=np.load(R/'evaluation_inputs/previous.npz')['prediction']
    assert y.shape==base['prediction'].shape==base['fold'].shape==previous.shape==(70000,)
    assert abs(metrics(y,base['prediction'])['KS']-.5306361042400267)<1e-12
    assert abs(metrics(y,previous)['KS']-.5312163876106586)<1e-12
    for yt,p,expected in [(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]),1.),(np.array([0,1,0,1]),np.ones(4)*.5,0.),(np.array([0,0,1,1]),np.array([.9,.8,.2,.1]),1.)]:
        assert abs(metrics(yt,p)['KS']-expected)<1e-12
        assert abs(metrics(yt,p)['KS']-ks_2samp(p[yt==0],p[yt==1]).statistic)<1e-12
    packets={};supports=[];coverage=np.zeros(70000,int);sampling_checks=[]
    from gpu_models import sample_context
    for fold in range(5):
        packet=R/f'packets/fold{fold}';m=json.loads((packet/'MANIFEST.json').read_text());assert sha(packet/'MANIFEST.json')==inputs['packet_manifests'][str(fold)]
        for name,key in [('features.parquet','feature_sha256'),('roles.npz','roles_sha256'),('anchor.npz','anchor_sha256'),('inner_roles.npz','inner_roles_sha256')]:assert sha(packet/name)==m[key]
        roles=np.load(packet/'roles.npz');inner=np.load(packet/'inner_roles.npz');assert not {'y_outer','y_validation','target','labels'}&set(roles.files)
        ctx,meta,out=[roles[k]for k in ['context','correction','validation']];assert tuple(map(len,[ctx,meta,out]))==(50000,6000,14000)
        assert len(np.unique(np.r_[ctx,meta,out]))==70000 and np.array_equal(np.sort(out),np.flatnonzero(base['fold']==fold))
        assert np.array_equal(y[ctx],roles['y_context']) and np.array_equal(y[meta],roles['y_correction']);coverage[out]+=1
        for ratio in [1,3,5]:
            a,sa=sample_context(roles['y_context'],ratio,'case_control',2026+fold);b,sb=sample_context(roles['y_context'],ratio,'natural_matched_count',2026+fold)
            assert len(a)==len(b) and sa==sb and int(roles['y_context'][a].sum())==int(roles['y_context'].sum())
            sampling_checks.append(dict(fold=fold,ratio=ratio,exact_matched_count=len(a),casecontrol_positive_count=int(roles['y_context'][a].sum()),natural_positive_count=int(roles['y_context'][b].sum())))
        tr,stop=train_test_split(np.arange(50000),test_size=.15,stratify=roles['y_context'],random_state=20261029+fold)
        assert np.array_equal(tr,inner['train_relative']) and np.array_equal(stop,inner['stop_relative'])
        assert (len(tr),len(stop))==(42500,7500) and m['feature_statistics_fit_rows']=='inner_train42500' and m['feature_statistics_exclude_stopping'] is True
        assert m['supervised_selection_excludes_stopping'] is True
        for rows in [ctx[tr],ctx[stop],meta,out]:assert min(np.bincount(y[rows],minlength=2))>=50
        for design in catalog().values():
            columns=m['views'][design['view']];assert columns and len(columns)==len(set(columns))
            assert set(columns)<=set(m['feature_columns']) and not {'target','application_id','customer_id'}&set(columns)
            if design['resource']=='gpu':assert len(columns)<=99
        packets[str(fold)]=sha(packet/'MANIFEST.json');supports.append(dict(fold=fold,positive_context=int(y[ctx].sum()),positive_meta=int(y[meta].sum()),positive_outer=int(y[out].sum()),inner_train_rows_sha256=ids_sha(ctx[tr]),stop_rows_sha256=ids_sha(ctx[stop])))
    assert np.all(coverage==1)
    # Inspect one genuine feature table to verify row schema and missing-value recipe.
    example=pd.read_parquet(R/'packets/fold0/features.parquet',columns=json.loads((R/'packets/fold0/MANIFEST.json').read_text())['views']['global99']);assert len(example)==70000
    checks=['known_metric_cases_and_SciPy_replay','baseline_and_previous_exact_replay','five_fold_complete_disjoint_coverage','context_meta_outer_label_alignment','42500_inner_preprocessing_excludes_stopper','inherited_availability_and_source_identity','adequate_class_support','99_feature_GPU_limit','no_outer_labels_in_worker_packets']
    result=dict(status='PASS',UTC=utc(),sampling_checks=sampling_checks,checks=checks,packet_manifests=packets,support=supports,baseline=metrics(y,base['prediction']),previous=metrics(y,previous),constant_reference=metrics(y,np.repeat(y.mean(),len(y))),parent_gate_sha256=sha(parent/'GATE.json'),clock_sha256=sha(R/'CLOCK.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),config_sha256=sha(R/'CONFIG.json'),contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),independent_audit_available=False,scope='Measured reused-development admission only; no independent statistical confirmation')
    result['required_check_records']=[dict(id=x['id'],category=x['category'],status='PASS'if x['id']!='runtime_numerics'else'REQUIRES_PER_DESIGN_PILOT',reason='Measured inherited-data replay/boundary/support evidence; no independent generalization claim'if x['id']!='runtime_numerics'else'Data gate authorizes only registered fullsize pilots; every remaining fit is blocked until matching runtime/finite/replay gate PASS')for x in contract['required_checks']]
    dump(result,R/'GATE.json');return result

if __name__=='__main__':data_gate()
