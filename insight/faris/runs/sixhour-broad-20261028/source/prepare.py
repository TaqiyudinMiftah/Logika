"""Build full actuarial bank and compact GPU views inside the new campaign clock.

The frozen parent supplies only row roles, honest anchor predictions and gate
inputs. It is never modified. Every new selector, cohort prior and PCA basis fits
the fixed42500 inner-training rows, excluding7500 stopping rows and every query.
"""
import os
for _key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_key] = '2'
from pathlib import Path
import gc
import hashlib
import json
import shutil
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from common import R, load, dump, sha, ids_sha, utc, check_seal, metrics
from legacy_bank import Bank
from features_actuarial import expand, definitions as actuarial_definitions


def _admission(clock):
    assert time.time() < min(clock['admission_end_epoch'], clock['deadline_epoch']), 'Original campaign admission closed'


def _fingerprint(series, rows):
    selected = series.iloc[rows]
    if pd.api.types.is_numeric_dtype(series):
        value = selected.to_numpy(dtype='<f4', na_value=np.nan).copy()
        value[~np.isfinite(value)] = np.nan
        value[value == 0] = 0
        raw = value.tobytes()
    else:
        raw = pd.util.hash_pandas_object(selected.astype('string').fillna('__MISSING__'), index=False).to_numpy(dtype='<u8').tobytes()
    return hashlib.sha256(raw).hexdigest()


def _eligible(frame, inner_rows, core):
    """Rank only informative, nonaliased columns; retain full bank separately."""
    ordered = list(dict.fromkeys([c for c in core if c in frame] + list(frame)))
    seen, keep, dropped = {}, [], {}
    for column in ordered:
        s = frame[column].iloc[inner_rows]
        if s.nunique(dropna=False) <= 1:
            dropped[column] = 'constant_or_all_missing_in_inner_training'
            continue
        digest = _fingerprint(frame[column], inner_rows)
        if digest in seen:
            dropped[column] = 'exact_inner_training_alias_of:' + seen[digest]
            continue
        seen[digest] = column
        keep.append(column)
    return keep, dropped


def _selector(frame, rows, labels, columns, seed):
    import lightgbm as lgb
    values = frame.loc[:, columns].iloc[rows].copy()
    assert len(values)==len(labels)==len(rows)
    assert set(np.unique(labels))=={0,1}
    categorical = [c for c in values if not pd.api.types.is_numeric_dtype(values[c])]
    dictionaries = {}
    for column in categorical:
        text = values[column].astype('string').fillna('__MISSING__')
        vocabulary = sorted(text.unique().tolist())
        dictionaries[column] = vocabulary
        values[column] = pd.Categorical(text, categories=vocabulary)
    parameters = dict(n_estimators=200, num_leaves=15, max_depth=5, learning_rate=.04,
                      min_child_samples=150, reg_lambda=15, colsample_bytree=.85,
                      random_state=seed, n_jobs=2, verbosity=-1, importance_type='gain')
    start = time.time()
    model = lgb.LGBMClassifier(**parameters)
    model.fit(values, labels)
    gain = dict(zip(columns, np.asarray(model.feature_importances_, float).tolist()))
    ranked = sorted(columns, key=lambda c: (-gain[c], c))
    audit = dict(parameters=parameters, seconds=time.time()-start,
                 training_rows=len(rows), training_rows_sha256=ids_sha(rows),
                 training_labels_sha256=ids_sha(labels), categorical_vocabulary=dictionaries,
                 correction_labels_used=False, stopping_labels_used=False,
                 outer_labels_used=False, fitted_weights_saved=False)
    del model, values
    gc.collect()
    return ranked, gain, audit


def _redundancy_view(frame, ranked, inner, seed):
    """A fixed small covariate sample bounds the Spearman pruning cost."""
    pool = ranked[:512]
    numeric = [c for c in pool if pd.api.types.is_numeric_dtype(frame[c])]
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(inner, min(5000, len(inner)), replace=False))
    # scipy/pandas compute correlations only on this inner covariate sample.
    correlation = frame.loc[:, numeric].iloc[sample].corr(method='spearman', min_periods=50).abs()
    keep, skipped = [], {}
    for column in pool:
        conflicts = [previous for previous in keep
                     if column in correlation.columns and previous in correlation.columns
                     and np.isfinite(correlation.loc[column, previous])
                     and correlation.loc[column, previous] > .98]
        if conflicts:
            skipped[column] = conflicts[0]
        else:
            keep.append(column)
        if len(keep) == 99:
            break
    assert 0 < len(keep) <= 99
    return keep, dict(method='absolute pairwise Spearman, higher gain retained', threshold=.98,
                      candidate_pool=pool, covariate_sample_rows=len(sample),
                      covariate_sample_sha256=ids_sha(sample), min_pairwise_support=50,
                      skipped_correlated_with=skipped, target_labels_used=False,
                      query_rows_used=False)


def _pca_view(frame, ranked, core, inner, seed):
    selected = [c for c in ranked[:256] if c not in core and pd.api.types.is_numeric_dtype(frame[c])]
    assert selected, 'No numeric candidates for PCA'
    # float64 transform arithmetic avoids overflow before storing float32 packets.
    values = frame.loc[:, selected].to_numpy(dtype=float, na_value=np.nan)
    values[~np.isfinite(values)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(values[inner], axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    filled = np.where(np.isfinite(values), values, median)
    mean = filled[inner].mean(0)
    scale = np.maximum(filled[inner].std(0), 1e-6)
    standardized = (filled-mean)/scale
    assert np.isfinite(standardized).all()
    components = min(64, len(selected), len(inner)-1)
    model = PCA(n_components=components, svd_solver='randomized', iterated_power=3,
                random_state=seed, whiten=False)
    model.fit(standardized[inner])
    transformed = model.transform(standardized).astype('float32')
    names = [f'pca6_{i:03d}' for i in range(components)]
    output = pd.DataFrame(transformed, columns=names, index=frame.index)
    assert np.isfinite(output.to_numpy()).all()
    recipe = dict(input_columns=selected, training_rows=len(inner), training_rows_sha256=ids_sha(inner),
                  imputation='inner-training median', standardization='inner-training mean/std; minimumSD1e-6',
                  imputation_medians=median.tolist(), means=mean.tolist(), scales=scale.tolist(),
                  n_components=components, svd_solver='randomized', iterated_power=3,
                  random_state=seed, whiten=False,
                  components_sha256=hashlib.sha256(np.ascontiguousarray(model.components_).tobytes()).hexdigest(),
                  explained_variance_ratio=model.explained_variance_ratio_.tolist(),
                  components_saved=False, target_labels_used=False, query_rows_used=False)
    del values, filled, standardized, model
    gc.collect()
    return output, names+[c for c in core if c in frame], recipe


def _verify_completed(destination, expected_source):
    manifest = json.loads((destination/'MANIFEST.json').read_text())
    assert manifest['preparation_source_sha256'] == expected_source
    for name, key in [('features.parquet','feature_sha256'), ('all_features.parquet','full_feature_sha256'),
                      ('roles.npz','roles_sha256'), ('inner_roles.npz','inner_roles_sha256'),
                      ('anchor.npz','anchor_sha256'), ('gate.parquet','gate_sha256')]:
        assert sha(destination/name) == manifest[key], name
    return manifest


def prepare_fold(fold, parent, clock):
    _admission(clock)
    destination = R/f'packets/fold{fold}'
    if (destination/'MANIFEST.json').exists():
        manifest = _verify_completed(destination, sha(__file__))
        print(json.dumps(dict(stage='packet_reconciled', fold=fold,
                              full_features=len(manifest['full_feature_columns']))), flush=True)
        return manifest
    if destination.exists():
        failures = R/'failures';failures.mkdir(exist_ok=True)
        destination.rename(failures/f'preparation_fold{fold}_{time.time_ns()}')
    destination.mkdir(parents=True)
    old = parent/f'packets/fold{fold}'
    old_manifest = json.loads((old/'MANIFEST.json').read_text())
    for name,key in [('roles.npz','roles_sha256'), ('anchor.npz','anchor_sha256'), ('gate.parquet','gate_sha256')]:
        assert sha(old/name) == old_manifest[key], (fold,name)
        shutil.copy2(old/name,destination/name)
    if (old/'ANCHOR_AUDIT.json').exists():
        shutil.copy2(old/'ANCHOR_AUDIT.json',destination/'PARENT_ANCHOR_AUDIT.json')
    role = np.load(destination/'roles.npz')
    context, correction, validation = [role[k] for k in ['context','correction','validation']]
    assert (len(context),len(correction),len(validation)) == (50000,6000,14000)
    assert len(np.unique(np.r_[context,correction,validation])) == 70000
    assert 'y_outer' not in role.files and 'y_validation' not in role.files
    assert role['y_context'].shape == (50000,)
    split_seed = 20261029+fold
    train_relative, stop_relative = train_test_split(np.arange(len(context)), test_size=.15,
        stratify=role['y_context'], random_state=split_seed)
    inner, stopping = context[train_relative], context[stop_relative]
    assert len(inner)==42500 and len(stopping)==7500
    assert not np.intersect1d(inner,stopping).size
    np.savez_compressed(destination/'inner_roles.npz', inner_train=inner, inner_stop=stopping,
        train_relative=train_relative, stop_relative=stop_relative,
        y_inner=role['y_context'][train_relative], y_stop=role['y_context'][stop_relative])
    schema = load('FEATURE_SCHEMA.json')
    bank = Bank(fold,inner)
    bank.ctx = inner
    # Bank.base checks current columns before its prior map. Remove named cached
    # cohort-prior columns so this new inner-training boundary actually controls
    # their sufficient statistics instead of reusing a previous context's values.
    removed_prior_columns = sorted(set(bank.current.columns)&set(bank.priors))
    bank.current = bank.current.drop(columns=removed_prior_columns)
    specs = list(schema['specs'])
    ids = [spec['id'] for spec in specs]
    assert len(set(ids))==len(ids)
    assert not {'target','application_id','customer_id'}&set(ids)
    values = {}
    for index,spec in enumerate(specs):
        if index%64==0:
            _admission(clock)
        value = np.asarray(bank.compute(spec))
        assert value.shape==(70000,), (spec['id'],value.shape)
        if pd.api.types.is_numeric_dtype(value.dtype):
            value = np.array(value,dtype='float32',copy=True)
            value[~np.isfinite(value)] = np.nan
        values[spec['id']] = value
    base = pd.DataFrame(values)
    del values
    # New risk representations fit only historical covariate priors, never y.
    additions, expansion_audit = expand(base,schema,inner)
    combined = pd.concat([base,additions],axis=1)
    assert combined.columns.is_unique and len(combined)==70000
    core = [c for c in schema['core'] if c in combined]
    base_eligible,base_dropped = _eligible(base,inner,core)
    augmented_eligible,augmented_dropped = _eligible(combined,inner,core)
    _admission(clock)
    base_rank,base_gain,base_selector = _selector(base,inner,role['y_context'][train_relative],base_eligible,split_seed)
    _admission(clock)
    augmented_rank,augmented_gain,augmented_selector = _selector(combined,inner,role['y_context'][train_relative],augmented_eligible,split_seed)
    views = {f'global{size}':base_rank[:size] for size in [32,48,64,80,99,128,256,512]}
    views.update({f'augmented{size}':augmented_rank[:size] for size in [99,256,512]})
    views['core99'] = list(dict.fromkeys(core+[c for c in base_rank if c not in core]))[:99]
    force_actuarial = core+list(additions)
    views['actuarial99'] = list(dict.fromkeys(force_actuarial+augmented_rank))[:99]
    _admission(clock)
    views['redundant99'],redundancy = _redundancy_view(combined,augmented_rank,inner,split_seed)
    _admission(clock)
    projected,pca_columns,pca_recipe = _pca_view(combined,base_rank,core,inner,split_seed)
    views['pca64_core'] = pca_columns
    # all_features is lab-only; it deliberately retains every computed source
    # definition. Only selector candidates discard constants/aliases.
    full_columns = list(combined)
    views['all'] = full_columns
    assert len(views['pca64_core'])<=84
    for name,columns in views.items():
        assert len(columns)>0 and len(columns)==len(set(columns)),name
        if name.endswith('99') or name in ['global32','global48','global64','global80','pca64_core']:
            assert len(columns)<=99,name
    gpu_columns = list(dict.fromkeys(c for name,columns in views.items() if name!='all' for c in columns))
    pca_frame = pd.concat([combined,projected],axis=1)
    gpu = pca_frame.loc[:,gpu_columns]
    assert not {'target','application_id','customer_id'}&set(gpu.columns)
    combined.to_parquet(destination/'all_features.parquet',index=False)
    gpu.to_parquet(destination/'features.parquet',index=False)
    manifest = dict(fold=fold, views=views, feature_columns=gpu_columns,
        full_feature_columns=full_columns, feature_sha256=sha(destination/'features.parquet'),
        full_feature_sha256=sha(destination/'all_features.parquet'),
        roles_sha256=sha(destination/'roles.npz'), inner_roles_sha256=sha(destination/'inner_roles.npz'),
        anchor_sha256=sha(destination/'anchor.npz'), gate_sha256=sha(destination/'gate.parquet'),
        preparation_source_sha256=sha(__file__), source_schema_sha256=sha(R/'FEATURE_SCHEMA.json'),
        source_bank_manifest_sha256=sha(R/'BANK_MANIFEST.json'),
        parent_manifest_sha256=sha(old/'MANIFEST.json'), parent_feature_values_reused=False,
        parent_anchor_reused=True, parent_gate_covariates_reused=True,
        sourcefeaturecounts=dict(definitions=len(specs), base_saved=len(base.columns),
            base_ranking_eligible=len(base_eligible), actuarial_candidates=len(actuarial_definitions()),
            actuarial_retained=len(additions.columns), all_saved=len(full_columns),
            GPU_saved=len(gpu_columns), PCA_components=len(projected.columns)),
        inner_split_seed=split_seed, inner_train_rows=len(inner), stopping_rows=len(stopping),
        inner_train_rows_sha256=ids_sha(inner), stopping_rows_sha256=ids_sha(stopping),
        inner_train_labels_sha256=ids_sha(role['y_context'][train_relative]),
        stopping_labels_sha256=ids_sha(role['y_context'][stop_relative]),
        context_rows_sha256=ids_sha(context), correction_rows_sha256=ids_sha(correction),
        validation_rows_sha256=ids_sha(validation),
        selector_parameters={'base':base_selector,'augmented':augmented_selector},
        base_gain=base_gain, augmented_gain=augmented_gain,
        ranking_dropped={'base':base_dropped,'augmented':augmented_dropped},
        removed_cached_cohort_prior_columns=removed_prior_columns,
        book_expansionaudit=expansion_audit, redundancy=redundancy, PCA_recipe=pca_recipe,
        feature_statistics_fit_rows='inner_train42500',
        feature_statistics_exclude_stopping=True, supervised_selection_excludes_stopping=True,
        current_target_labels_used_only_for_inner_selectors=True,
        correction_or_outer_labels_used_for_feature_selection=False,
        outer_labels_in_worker_packet=False, fitted_selector_or_PCA_weights_saved=False,
        original_clock_sha256=sha(R/'CLOCK.json'), UTC=utc())
    # The completion marker is atomic and written last. Resume never recomputes
    # a completed fold; any interrupted folder is preserved as failed evidence.
    dump(manifest,destination/'MANIFEST.json')
    print(json.dumps(dict(stage='packet_complete',fold=fold,
        sourcefeaturecounts=manifest['sourcefeaturecounts'],views={k:len(v)for k,v in views.items()})),flush=True)
    del bank,base,combined,additions,projected,pca_frame,gpu
    gc.collect()
    return manifest


def main():
    clock = check_seal()
    _admission(clock)
    inputs = load('INPUTS.json')
    parent = Path(inputs['parent_root'])
    assert parent.is_dir() and parent.resolve()!=R.resolve()
    bank_root = Path(load('PATHS.json')['bank'])
    bank_manifest = load('BANK_MANIFEST.json')
    verified = {}
    for name in ['numeric.npy','categorical.parquet','metadata.parquet']:
        assert sha(bank_root/name)==bank_manifest['files'][name],name
        verified[name]=bank_manifest['files'][name]
    output = R/'evaluation_inputs';output.mkdir(exist_ok=True)
    for name in ['target.npy','baseline.npz']:
        source = parent/'evaluation_inputs'/name
        assert source.exists()
        destination = output/name
        if destination.exists():
            assert sha(destination)==sha(source),name
        else:
            shutil.copy2(source,destination)
    target = np.load(output/'target.npy')
    baseline = np.load(output/'baseline.npz')
    assert target.shape==(70000,) and baseline['prediction'].shape==(70000,)
    baseline_metrics = metrics(target,baseline['prediction'])
    assert abs(baseline_metrics['KS']-.5306361042400267)<1e-12
    manifests = {}
    for fold in range(5):
        m = prepare_fold(fold,parent,clock)
        role = np.load(R/f'packets/fold{fold}/roles.npz')
        assert np.array_equal(target[role['context']],role['y_context'])
        assert np.array_equal(target[role['correction']],role['y_correction'])
        assert set(role['validation'])==set(np.flatnonzero(baseline['fold']==fold))
        anchor = np.load(R/f'packets/fold{fold}/anchor.npz')
        assert np.max(abs(anchor['validation']-baseline['prediction'][role['validation']]))<1e-12
        manifests[str(fold)]=sha(R/f'packets/fold{fold}/MANIFEST.json')
    dump(dict(status='PASS',UTC=utc(),source_bank_hashes=verified,baseline=baseline_metrics,
        baseline_identity='Inherited exact maturity-adjusted fold-local anchor; publicV25 is a separate artifact.',
        packet_manifests=manifests,source_schema_sha256=sha(R/'FEATURE_SCHEMA.json'),
        current_clock_sha256=sha(R/'CLOCK.json'),source_bank_manifest_sha256=sha(R/'BANK_MANIFEST.json'),
        target_sha256=sha(output/'target.npy'),baseline_sha256=sha(output/'baseline.npz'),
        roles_disjoint=True,outer_rows_cover70000_once=True,
        inner_training42500_excludes7500_stoppers=True,
        features_and_PCA_fit_only_inner_training=True,outer_labels_in_worker_packets=False,
        independent_audit_available=False,limitation='All70000 rows are reused development; historical upstream capture completeness remains inherited.'),
        R/'PREPARATION_AUDIT.json')


if __name__=='__main__':
    main()
