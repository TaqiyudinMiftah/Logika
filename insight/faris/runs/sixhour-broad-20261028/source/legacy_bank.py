"""Build source-backed domain views and honest correction-row anchor predictions.

Only context labels select expert inputs. Correction labels cross-fit the anchor
head; each anchor spline excludes the correction cases it predicts. Outer labels
are copied to a separate terminal-evaluator file, never worker packets.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import gc
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from common import R, dump, sha, load, ids_sha, metrics, check_seal, utc
sys.path.insert(0, str(R / 'source/vendor'))
import baseline as b
import feature_head
from feature_formulas import Features, ratio
from spline_models import fixed_spline_fit


class Bank(Features):
    def __init__(self, fold, context):
        paths = load('PATHS.json')
        old = Path(paths['horizontal'])
        packet=old/f'packets/fold{fold}'
        original_manifest=json.loads((packet/'manifest.json').read_text())
        for name,digest in original_manifest['files'].items():
            assert sha(packet/name)==digest,(fold,name)
        self.manifest = load('BANK_MANIFEST.json')
        self.lookup = {n:i for i,n in enumerate(self.manifest['columns'])}
        self.numeric = np.load(Path(paths['bank'])/'numeric.npy', mmap_mode='r')
        self.cats = pd.read_parquet(Path(paths['bank'])/'categorical.parquet').iloc[:70000]
        self.current = pd.read_parquet(packet/'current.parquet')
        self.extra = pd.read_parquet(old/'extra.parquet')
        self.ctx = context
        self.cache, self.prior_cache = {}, {}
        self.legacy = {p['name']:p for p in self.manifest['priors'] if p.get('legacy')}
        self.priors = {}
        for p in self.manifest['priors']:
            self.priors[p['name']] = (p, False)
            if p.get('sd'):
                self.priors[p['sd']] = (p, True)

    def base(self, name):
        if name in self.cache:
            return self.cache[name]
        if name in self.current:
            value = self.current[name].to_numpy()
        elif name in self.priors:
            p, uncertainty = self.priors[name]
            k, n = self.base(p['k']), self.base(p['n'])
            mu = self.cohort(p['k'], p['n'])
            strength = p.get('strength',24)
            value = ratio(np.nan_to_num(k)+strength*mu, np.nan_to_num(n)+strength)
            if uncertainty:
                value = np.sqrt(np.maximum(value*(1-value)/(np.nan_to_num(n)+strength+1),0))
        else:
            return super().base(name)
        self.cache[name] = value
        return value

    def compute(self, spec):
        if spec['op'] == 'monthly_spend_ratio':
            return (ratio(self.base(spec['inputs'][0]), self.base(spec['inputs'][1])) *
                    (30/spec['window_days'])).astype(np.float32)
        return super().compute(spec)


def anchor(fold, roles, bank, original):
    from threadpoolctl import threadpool_limits
    paths = load('PATHS.json')
    exp = Path(paths['exploration'])
    old = Path(paths['horizontal'])/f'packets/fold{fold}'
    reference = np.load(old/'baseline.npz')
    columns = load('ANCHOR_SOURCE.json')['head_columns']
    ctx, correction, validation = [roles[k] for k in ['context','correction','validation']]
    yc = roles['y_correction']
    raw = reference['raw']
    head_frame = bank.current[columns]
    full = feature_head.fit(head_frame.iloc[correction].assign(target=yc),raw[:6000],columns,.003,.5)
    replay = .75*b.corrected_probability(full,head_frame.iloc[validation],raw[6000:])[0]+.25*reference['spline']
    error = float(np.max(abs(replay-reference['corrected'])))
    assert error < 3e-12, ('Matched anchor head replay',fold,error)
    metadata=json.loads((exp/f'ranking-pipeline-20261020/input/artifacts/fold{fold}/spline_metadata.json').read_text())
    spline_columns=metadata['recipe']['spec']['columns']
    prior_root=exp/f'fold-local-priors-20261021/fold{fold}'
    assert sha(prior_root/'features.parquet')==json.loads((prior_root/'complete.json').read_text())['feature_table_sha256']
    frame = pd.read_parquet(exp/f'fold-local-priors-20261021/fold{fold}/features.parquet',
                            columns=list(dict.fromkeys(['application_id']+spline_columns)))
    frame = frame.set_index('application_id').loc[original.application_id].reset_index()
    mature = pd.read_parquet(exp/f'repayment-severity-fe-20261021/fold{fold}/features.parquet').set_index('application_id')
    for threshold in [1,30]:
        frame[f're_late_{threshold}_posterior'] = mature.loc[original.application_id,f'matched_beta_late_{threshold}'].to_numpy(float)
    training_rows=np.r_[ctx,correction]
    spline=fixed_spline_fit(frame.iloc[training_rows].assign(target=original.target.to_numpy(int)[training_rows]),
                            spline_columns,b.prepare)
    full_sp = b.predict_spline(spline,frame.iloc[validation])
    sp_error = float(np.max(abs(full_sp-reference['spline'])))
    assert sp_error < 3e-10, ('Matched anchor spline replay',fold,sp_error)
    prediction = np.full(6000,np.nan)
    assignment = np.full(6000,-1,dtype=int)
    records = []
    splitter = StratifiedKFold(n_splits=3,shuffle=True,random_state=20261028+fold)
    for inner,(training,heldout) in enumerate(splitter.split(np.zeros(6000),yc)):
        head = feature_head.fit(head_frame.iloc[correction[training]].assign(target=yc[training]),
                                raw[training],columns,.003,.5)
        h = b.corrected_probability(head,head_frame.iloc[correction[heldout]],raw[heldout])[0]
        spline_rows = np.r_[ctx,correction[training]]
        assert len(spline_rows)==54000 and not np.intersect1d(spline_rows,correction[heldout]).size
        inner_spline=fixed_spline_fit(frame.iloc[spline_rows].assign(target=original.target.to_numpy(int)[spline_rows]),
                                      spline_columns,b.prepare)
        sp=b.predict_spline(inner_spline,frame.iloc[correction[heldout]])
        prediction[heldout] = .75*h+.25*sp
        assignment[heldout] = inner
        records.append(dict(inner=inner,head_rows=len(training),spline_rows=len(spline_rows),
                            heldout_rows=len(heldout),heldout_global_rows_sha256=ids_sha(correction[heldout]),
                            spline_training_rows_sha256=ids_sha(spline_rows),overlap=0,
                            knot_fitting_excludes_heldout=True,weights_saved=False,
                            spline_parameters=inner_spline['resolved_parameters']))
        del inner_spline
        gc.collect()
    assert np.isfinite(prediction).all() and (assignment>=0).all()
    return prediction,replay,assignment,dict(head_replay_max_abs=error,spline_replay_max_abs=sp_error,
                      head_columns=columns,spline_columns=spline['columns'],crossfit=records,
                      reference='Audited fold-local maturity anchor, not an assertion of byte-identical public V25')


def prepare_fold(fold, original):
    import lightgbm as lgb
    clock = check_seal(); assert time.time()<clock['admission_end_epoch']
    paths = load('PATHS.json');old=Path(paths['horizontal'])/f'packets/fold{fold}'
    references=load('INPUT_REFERENCES.json')
    assert sha(old/'roles.npz')==references['roles'][str(fold)]
    assert sha(old/'manifest.json')==references['packet_manifest_sha256'][str(fold)]
    destination=R/f'packets/fold{fold}'
    if(destination/'MANIFEST.json').exists():
        done=json.loads((destination/'MANIFEST.json').read_text())
        for name,key in [('features.parquet','feature_sha256'),('gate.parquet','gate_sha256'),
                         ('roles.npz','roles_sha256'),('anchor.npz','anchor_sha256')]:
            assert sha(destination/name)==done[key]
        assert done['source_schema_sha256']==sha(R/'FEATURE_SCHEMA.json')
        print(json.dumps(dict(stage='packet_reconciled',fold=fold)),flush=True)
        return
    if destination.exists():
        failures=R/'failures';failures.mkdir(exist_ok=True)
        destination.rename(failures/f'preparation_fold{fold}_{time.time_ns()}')
    roles=dict(np.load(old/'roles.npz'))
    ctx,correction,validation=[roles[k]for k in ['context','correction','validation']]
    assert [len(ctx),len(correction),len(validation)]==[50000,6000,14000]
    assert len(np.unique(np.r_[ctx,correction,validation]))==70000
    assert np.array_equal(original.target.to_numpy(int)[ctx],roles['y_context'])
    assert np.array_equal(original.target.to_numpy(int)[correction],roles['y_correction'])
    assert min(np.bincount(roles['y_context'].astype(int)))>1000 and min(np.bincount(roles['y_correction'].astype(int)))>100
    bank=Bank(fold,ctx);schema=load('FEATURE_SCHEMA.json');specs={s['id']:s for s in schema['specs']}
    destination.mkdir(parents=True,exist_ok=False)
    meta,base,assignment,anchor_audit=anchor(fold,roles,bank,original)
    np.savez_compressed(destination/'anchor.npz',correction_oof=meta,validation=base,inner_fold=assignment)
    dump(anchor_audit,destination/'ANCHOR_AUDIT.json')
    views={};saved={};selection=[]
    for domain,candidates in schema['domains'].items():
        ordered=list(dict.fromkeys(schema['core']+candidates));values={};dropped={};seen={}
        for fid in ordered:
            value=bank.compute(specs[fid]);s=pd.Series(value[ctx])
            if s.nunique(dropna=False)<=1:
                dropped[fid]='context_constant';continue
            fingerprint=__import__('hashlib').sha256(pd.util.hash_pandas_object(s,index=False).values.tobytes()).hexdigest()
            if fingerprint in seen:
                dropped[fid]='context_duplicate_of_'+seen[fingerprint]
                if fid in schema['core']:values[fid]=value
                continue
            seen[fingerprint]=fid;values[fid]=value
        frame=pd.DataFrame(values)
        numeric=frame.select_dtypes(include=[np.number]).columns
        frame[numeric]=frame[numeric].replace([np.inf,-np.inf],np.nan)
        train=frame.iloc[ctx].copy()
        categories=train.select_dtypes(include=['object','string','category']).columns.tolist()
        for c in categories:train[c]=pd.Categorical(train[c].astype('string').fillna('MISSING'))
        selector=lgb.LGBMClassifier(n_estimators=200,num_leaves=15,learning_rate=.04,max_depth=5,
                    min_child_samples=150,reg_lambda=10,colsample_bytree=.8,random_state=20261028+fold,
                    n_jobs=2,verbosity=-1,importance_type='gain')
        started=time.time();selector.fit(train,roles['y_context'])
        importance=dict(zip(frame.columns,selector.feature_importances_.tolist()))
        core=[c for c in schema['core']if c in frame]
        ranked=sorted([c for c in frame if c not in core],key=lambda c:(-importance[c],c))
        expanded=core+ranked[:99-len(core)]
        existing=[c for c in schema['existing_views'][domain]if c in frame]
        assert 0<len(existing)<=99 and len(expanded)<=99
        views['existing_'+domain]=existing;views['expanded_'+domain]=expanded
        for fid in set(existing+expanded):saved[fid]=frame[fid].to_numpy()
        selection.append(dict(domain=domain,candidate_columns=len(frame.columns),dropped=dropped,
                   selected_expanded=expanded,existing_columns=existing,selected_gain={c:importance[c]for c in expanded},
                   context_rows_sha256=ids_sha(ctx),correction_labels_used=False,outer_labels_used=False,
                   selector_parameters=selector.get_params(),seconds=time.time()-started,weights_saved=False))
        del selector,train,frame,values
        bank.cache={};gc.collect()
    views['expanded_union']=list(dict.fromkeys(c for name,columns in views.items()
                                              if name.startswith('expanded_') for c in columns))
    frame=pd.DataFrame(saved);frame.to_parquet(destination/'features.parquet',index=False)
    gate_names=load('CONFIG.json')['gate_columns'];gates=pd.DataFrame({c:bank.base(c)for c in gate_names})
    gates.to_parquet(destination/'gate.parquet',index=False)
    np.savez_compressed(destination/'roles.npz',**roles)
    dump(dict(fold=fold,views=views,selection=selection,feature_count_union=len(frame.columns),
              feature_sha256=sha(destination/'features.parquet'),gate_sha256=sha(destination/'gate.parquet'),
              roles_sha256=sha(destination/'roles.npz'),anchor_sha256=sha(destination/'anchor.npz'),
              anchor_audit_sha256=sha(destination/'ANCHOR_AUDIT.json'),source_schema_sha256=sha(R/'FEATURE_SCHEMA.json'),
              context_rows_sha256=ids_sha(ctx),correction_rows_sha256=ids_sha(correction),validation_rows_sha256=ids_sha(validation),
              outer_labels_in_worker_packet=False,weights_saved=False),destination/'MANIFEST.json')
    print(json.dumps(dict(stage='packet_complete',fold=fold,union=len(frame.columns),
                         views={k:len(v)for k,v in views.items()})),flush=True)


def main():
    paths=load('PATHS.json');bank_root=Path(paths['bank']);bank_manifest=load('BANK_MANIFEST.json')
    hashes={}
    for name in ['numeric.npy','categorical.parquet','metadata.parquet']:
        assert sha(bank_root/name)==bank_manifest['files'][name],name
        hashes[name]=bank_manifest['files'][name]
    references=load('INPUT_REFERENCES.json')
    assert sha(Path(references['target_source_file']))==references['target_source_sha256']
    original=pd.read_parquet(Path(paths['exploration'])/'ranking-pipeline-20261020/input/data/train_features.parquet',
                              columns=['application_id','customer_id','target'])
    metadata=pd.read_parquet(bank_root/'metadata.parquet')
    assert original.application_id.tolist()==metadata.application_id.iloc[:70000].tolist()
    assert original.application_id.is_unique and original.customer_id.is_unique
    (R/'evaluation_inputs').mkdir(exist_ok=True)
    np.save(R/'evaluation_inputs/target.npy',original.target.to_numpy(int))
    np.save(R/'evaluation_inputs/application_id.npy',original.application_id.to_numpy(str))
    for fold in range(5):
        prepare_fold(fold,original)
    p=np.full(70000,np.nan);folds=np.full(70000,-1,dtype=int)
    for f in range(5):
        rows=np.load(R/f'packets/fold{f}/roles.npz')['validation']
        p[rows]=np.load(R/f'packets/fold{f}/anchor.npz')['validation'];folds[rows]=f
    assert np.isfinite(p).all() and np.all(folds>=0)
    score=metrics(original.target,p);assert abs(score['KS']-.5306361042400267)<1e-10,score
    np.savez_compressed(R/'evaluation_inputs/baseline.npz',prediction=p,fold=folds)
    dump(dict(status='PASS',source_bank_hashes=hashes,baseline=score,public_V25_KS=.56822,
              reference_policy='Matched audited maturity baseline; public V25 is a separate exact artifact.',
              roles_disjoint=True,unique_customers=True,outer_predictions_cover_all_rows_once=True,
              baseline_recomputed=True,terminal_targets_sha256=sha(R/'evaluation_inputs/target.npy'),
              source_schema_sha256=sha(R/'FEATURE_SCHEMA.json'),
              packet_manifests={str(f):sha(R/f'packets/fold{f}/MANIFEST.json')for f in range(5)},
              independent_audit_available=False,snapshot_semantics='Inherited historical PIT completeness assumption; no new certification.'),R/'PREPARATION_AUDIT.json')


if __name__=='__main__':
    main()
