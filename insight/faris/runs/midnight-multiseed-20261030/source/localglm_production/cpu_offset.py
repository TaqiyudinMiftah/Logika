"""Base-level Bernoulli GLM plus a fixed-offset tree correction.

This is a tree transfer of the CANN additive-linear-predictor principle, not a
reproduction of its neural architecture. The offset is learned ONLY on permitted
base-training rows. It is never an in-sample anchor prediction imported from the
meta/outer evaluation. For stopping the GLM and residual both use the inner
42,500 rows, while their length is chosen on 7,500 held-out context rows. Both
are then refitted on all 50,000 permitted context labels, retaining the inner
encoder. The GLM is frozen during residual boosting. No fitted state is saved.
"""
import gc
import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits


SOURCES = {
    'cann': 'https://doi.org/10.1017/asb.2018.42',
    'bernoulli_extension': 'https://www.mdpi.com/1999-4893/16/2/99',
    'dataset': 'https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Dataset.html',
    'margin_composition': 'https://github.com/lightgbm-org/LightGBM/issues/5219',
    'glm': 'https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html',
}


def _ids(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _ks(y, score):
    y, score = np.asarray(y, dtype=np.int64), np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind='stable')
    end = np.flatnonzero(np.r_[score[order][1:] != score[order][:-1], True])
    a = np.cumsum(y[order]) / np.sum(y)
    b = np.cumsum(1 - y[order]) / np.sum(1 - y)
    return float(np.max(np.abs(a[end] - b[end])))


def _check(end):
    if time.time() >= end:
        raise TimeoutError('GLM-offset CPU modelling deadline reached')


def _callback(end):
    def callback(_env):
        _check(end)
    callback.before_iteration = True
    callback.order = 0
    return callback


class FrozenInput:
    """Two representations of identical columns; learn all encoding on inner only."""
    def fit(self, frame, rows):
        self.columns = list(frame.columns)
        self.numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical = [c for c in frame if c not in self.numeric]
        self.names = {c: 'f' + str(i) for i, c in enumerate(self.columns)}
        a = self._numeric(frame, rows)
        count = np.isfinite(a).sum(axis=0)
        self.mean = np.divide(np.nansum(a, axis=0), count,
                              out=np.zeros(len(self.numeric)), where=count > 0)
        variance = np.divide(np.nansum((a - self.mean) ** 2, axis=0), count,
                                  out=np.zeros(len(self.numeric)), where=count > 0)
        self.scale = np.sqrt(variance)
        self.scale[~np.isfinite(self.scale) | (self.scale < 1e-8)] = 1.
        self.vocabulary = {
            c: sorted(frame[c].iloc[rows].astype('string').fillna('__MISSING__').unique().tolist())
            for c in self.categorical
        }
        self.onehot = None
        if self.categorical:
            self.onehot = OneHotEncoder(handle_unknown='infrequent_if_exist',
                                       max_categories=64, sparse_output=True,
                                       dtype=np.float64)
            self.onehot.fit(self._categorical(frame, rows))
        self.fit_rows = np.asarray(rows, dtype=np.int64)
        return self

    def _numeric(self, frame, rows):
        a = frame.iloc[rows][self.numeric].to_numpy(dtype=np.float64, na_value=np.nan)
        a[~np.isfinite(a)] = np.nan
        return a

    def _categorical(self, frame, rows):
        return frame.iloc[rows][self.categorical].astype('string').fillna('__MISSING__').astype(str)

    def glm(self, frame, rows):
        a = self._numeric(frame, rows)
        missing = ~np.isfinite(a)
        a = np.clip((a - self.mean) / self.scale, -10., 10.)
        a[missing] = 0.
        parts = [sparse.csr_matrix(a), sparse.csr_matrix(missing.astype(np.float64))]
        if self.onehot is not None:
            parts.append(self.onehot.transform(self._categorical(frame, rows)))
        return sparse.hstack(parts, format='csr')

    def trees(self, frame, rows):
        value = frame.iloc[rows][self.columns].copy()
        for i, c in enumerate(self.numeric):
            a = pd.to_numeric(value[c], errors='coerce').to_numpy(dtype=np.float64)
            a[~np.isfinite(a)] = np.nan
            value[c] = ((a - self.mean[i]) / self.scale[i]).astype(np.float32)
        for c in self.categorical:
            value[c] = pd.Categorical(value[c].astype('string').fillna('__MISSING__'),
                                      categories=self.vocabulary[c])
        return value.rename(columns=self.names)

    def recipe(self):
        return {
            'fit_rows': len(self.fit_rows), 'fit_rows_sha256': _ids(self.fit_rows),
            'numeric_columns': self.numeric, 'categorical_columns': self.categorical,
            'numeric_mean_sha256': _hash(self.mean), 'numeric_scale_sha256': _hash(self.scale),
            'category_dictionary_sha256': hashlib.sha256(json.dumps(self.vocabulary, sort_keys=True).encode()).hexdigest(),
            'glm_numeric': 'inner mean/SD, clip standardized values to [-10,10], missing=0 plus missing indicators',
            'glm_categorical': 'inner-fitted one-hot, max64 categories per column, infrequent-if-exist unknown handling',
            'tree_numeric': 'inner mean/SD float32, no clipping, NaN preserved',
            'tree_categorical': 'inner sorted vocabulary; unseen becomes missing',
            'context_encoder_refit': False,
        }


def _glm(x, y, regularization_C, end):
    _check(end)
    model = LogisticRegression(C=float(regularization_C), penalty='l2', solver='lbfgs',
                               fit_intercept=True, max_iter=800, tol=1e-4,
                               class_weight=None, random_state=0)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x, y)
    _check(end)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if convergence or int(np.max(model.n_iter_)) >= 800:
        raise RuntimeError('GLM convergence gate failed; no silent approximate-offset acceptance: ' + repr(convergence))
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    return model, dict(C=float(regularization_C), penalty='l2', solver='lbfgs',
                       fit_intercept=True, max_iter=800, tol=1e-4,
                       class_weight=None, selected_iterations=int(np.max(model.n_iter_)),
                       convergence_warning=False)


def _tree_parameters(spec, seed):
    leaves = int(spec.get('num_leaves', 7))
    assert leaves in {7, 31}
    return dict(objective='binary', metric='None', boosting_type='gbdt',
                learning_rate=.04, num_leaves=leaves, max_depth=3 if leaves == 7 else 5,
                min_data_in_leaf=200, lambda_l1=.5, lambda_l2=20.,
                feature_fraction=.9, bagging_fraction=.9, bagging_freq=1,
                max_bin=63, min_sum_hessian_in_leaf=.001,
                boost_from_average=False, deterministic=True, force_col_wise=True,
                seed=int(seed), feature_fraction_seed=int(seed), bagging_seed=int(seed),
                data_random_seed=int(seed), num_threads=2, verbosity=-1)


def train_predict(frame, ctx, y, query, spec, seed, device='cpu'):
    """Fit one declared GLM/control/offset design and return query-only probabilities."""
    import lightgbm as lgb
    started = time.time()
    assert str(device) == 'cpu'
    end = min(float(spec.get('deadline_epoch', started + 1200)),
              started + float(spec.get('max_seconds', 1200)))
    ctx, query = np.asarray(ctx, dtype=np.int64), np.asarray(query, dtype=np.int64)
    y = np.asarray(y, dtype=np.int32)
    assert len(ctx) == len(y) == 50000 and len(query) == 20000
    assert len(np.unique(ctx)) == 50000 and not len(np.intersect1d(ctx, query))
    assert frame.columns.is_unique and set(np.unique(y)) == {0, 1}
    assert not {'target', 'application_id', 'customer_id'} & set(frame.columns)
    inner, stop = train_test_split(np.arange(len(ctx)), test_size=.15, stratify=y,
                                  random_state=int(spec['inner_split_seed']))
    assert (len(inner), len(stop)) == (42500, 7500)
    encoder = FrozenInput().fit(frame, ctx[inner])
    recipe = encoder.recipe()
    mode = spec.get('offset_mode', 'glm')
    assert mode in {'glm', 'intercept', 'glm_only'}
    C = float(spec.get('glm_C', .1))
    assert C in {.01, .1}
    glm_audits = {}
    if mode in {'glm', 'glm_only'}:
        x_inner = encoder.glm(frame, ctx[inner])
        initial_glm, glm_audits['inner'] = _glm(x_inner, y[inner], C, end)
        inner_offset = initial_glm.decision_function(x_inner)
        stop_offset = initial_glm.decision_function(encoder.glm(frame, ctx[stop]))
        del x_inner, initial_glm
        gc.collect()
    else:
        initial_intercept = float(logit(float(np.mean(y[inner]))))
        inner_offset = np.full(len(inner), initial_intercept)
        stop_offset = np.full(len(stop), initial_intercept)
    params = _tree_parameters(spec, seed)
    max_rounds, patience = 600, 75
    selected_rounds = 0
    if mode != 'glm_only':
        train = lgb.Dataset(encoder.trees(frame, ctx[inner]), label=y[inner], init_score=inner_offset)
        stopping = lgb.Dataset(encoder.trees(frame, ctx[stop]), label=y[stop],
                              init_score=stop_offset, reference=train)
        def metric(prediction, dataset):
            # Built-in binary objective supplies transformed probabilities,
            # including dataset init_score. Never manually add the offset twice.
            return 'ks', _ks(dataset.get_label(), prediction), True
        pilot = lgb.train(params, train, num_boost_round=max_rounds,
                          valid_sets=[stopping], valid_names=['held_context_stopper'],
                          feval=metric, callbacks=[_callback(end),
                                                  lgb.early_stopping(patience, verbose=False)])
        selected_rounds = max(1, int(pilot.best_iteration or max_rounds))
        # Independently verify Dataset init_score composition at chosen length;
        # the public predictor itself contains ONLY residual tree contributions.
        raw = pilot.predict(encoder.trees(frame, ctx[stop]), raw_score=True,
                            num_iteration=selected_rounds)
        stop_probability = expit(stop_offset + raw)
        stop_metric = float(pilot.best_score['held_context_stopper']['ks'])
        assert abs(_ks(y[stop], stop_probability) - stop_metric) <= 1e-10, 'Offset stopping-score composition failed'
        del train, stopping, pilot
        gc.collect()
    _check(end)
    if mode in {'glm', 'glm_only'}:
        x_context = encoder.glm(frame, ctx)
        full_glm, glm_audits['full_context'] = _glm(x_context, y, C, end)
        context_offset = full_glm.decision_function(x_context)
        query_matrix = encoder.glm(frame, query)
        query_offset = full_glm.decision_function(query_matrix)
        offset_replay = np.r_[full_glm.decision_function(query_matrix[:64]),
                              full_glm.decision_function(query_matrix[64:128])]
        assert np.max(np.abs(query_offset[:128] - offset_replay)) <= 1e-8
        del x_context, query_matrix, full_glm
        gc.collect()
    else:
        full_intercept = float(logit(float(np.mean(y))))
        context_offset = np.full(len(ctx), full_intercept)
        query_offset = np.full(len(query), full_intercept)
    prediction = expit(query_offset)
    replay = expit(query_offset[:128])
    if mode != 'glm_only':
        train = lgb.Dataset(encoder.trees(frame, ctx), label=y, init_score=context_offset)
        model = lgb.train(params, train, num_boost_round=selected_rounds,
                          callbacks=[_callback(end)])
        assert int(model.current_iteration()) == selected_rounds, 'Incomplete residual refit'
        query_values = encoder.trees(frame, query)
        residual = model.predict(query_values, raw_score=True)
        replay_residual = np.r_[model.predict(query_values.iloc[:64], raw_score=True),
                                model.predict(query_values.iloc[64:128], raw_score=True)]
        prediction = expit(query_offset + residual)
        replay = expit(query_offset[:128] + replay_residual)
        del train, model, query_values
    error = float(np.max(np.abs(prediction[:128] - replay)))
    assert error <= 5e-6, 'CPU fixed-offset batch replay failed'
    assert prediction.shape == (20000,) and np.isfinite(prediction).all()
    assert np.all((prediction >= 0) & (prediction <= 1))
    _check(end)
    audit = {
        'model_family': 'base_level_glm_fixed_logit_offset_lightgbm',
        'scientific_status': 'CANN principle transferred to residual trees; not exact neural CANN reproduction',
        'offset_mode': mode, 'formula': 'sigmoid(frozen_glm_logit(x)+residual_tree_margin(x))',
        'offset_fit_protocol': 'inner-only for stopping; full50k for final refit; GLM frozen during each residual fit',
        'offset_training_predictions': 'permitted training-row fitted GLM logits; intentionally in-sample training inputs, never outer/meta labels',
        'glm_parameters': glm_audits, 'parameters': params,
        'maximum_iterations': max_rounds, 'selected_iterations': selected_rounds,
        'stopping_metric': 'ks', 'patience': patience,
        'full_refit_finished_iterations': selected_rounds,
        'context_rows': len(ctx), 'inner_train_rows': len(inner), 'stopping_rows': len(stop),
        'inner_train_rows_sha256': _ids(ctx[inner]), 'stopping_rows_sha256': _ids(ctx[stop]),
        'refit_context_rows': len(ctx), 'final_training_rows': len(ctx),
        'inner_split_seed': int(spec['inner_split_seed']), 'seed': int(seed),
        'preprocessing_refit_on_context': False, 'inner_preprocessor': recipe,
        'full_preprocessor': recipe, 'preprocessing_protocol': 'inner-only encoder frozen for full-context refit',
        'batch_replay_max_abs_probability_error': error,
        'context_offset_sha256': _hash(context_offset), 'query_offset_sha256': _hash(query_offset),
        'query_labels_used': False, 'correction_used_for_early_stopping': False,
        'fitted_weights_saved': False, 'KV_caches_saved': False,
        'execution_backend': 'CPU', 'deadline_epoch': end, 'wall_seconds': time.time() - started,
        'sources': list(SOURCES.values()),
    }
    del encoder, context_offset, query_offset
    gc.collect()
    return np.asarray(prediction, dtype=np.float64), audit
