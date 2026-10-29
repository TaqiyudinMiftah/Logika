"""Preregistered CPU objectives, piecewise-linear trees and ordered boosting.

The API deliberately accepts context labels only. A single encoder is fitted on
the common 42,500 inner rows, then frozen for stopping and the 50,000-row model
refit. No estimator, coefficient, fitted encoder or model dump is serialized.
Focal loss is a transfer hypothesis from dense detection, not a proven credit-KS
method. Its exact nonconvex Hessian is floored at a declared 1e-6 for boosting.
"""
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split


SOURCES = {
    'focal': 'https://arxiv.org/abs/1708.02002v2',
    'lgb_custom': 'https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.train.html',
    'linear_tree': 'https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#linear_tree',
    'linear_paper': 'https://arxiv.org/abs/1802.05640v3',
    'catboost': 'https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html',
    'cat_parameters': 'https://catboost.ai/docs/en/references/training-parameters/common',
    'dart': 'https://proceedings.mlr.press/v38/korlakaivinayak15.html',
    'goss': 'https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html',
}


def _ids(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def ks(y, prediction):
    """Two-sample empirical KS with tied scores accumulated as one threshold."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    assert set(np.unique(y)) == {0, 1} and np.isfinite(prediction).all()
    order = np.argsort(prediction, kind='stable')
    ends = np.flatnonzero(np.r_[prediction[order][1:] != prediction[order][:-1], True])
    positive = np.cumsum(y[order]) / np.sum(y)
    negative = np.cumsum(1 - y[order]) / np.sum(1 - y)
    return float(np.max(np.abs(positive[ends] - negative[ends])))


def focal_loss_derivatives(raw, labels, gamma, positive_weight=1., normalizer=1.):
    """Return per-case focal loss and exact first/second raw-margin derivatives.

    L=-a_y*(1-sigmoid(s*z))**gamma*log(sigmoid(s*z)), s=2*y-1.
    Returned Hessians are exact and may be negative; the boosting wrapper, rather
    than this differentiable function, explicitly applies the registered floor.
    gamma=0 reduces exactly to weighted Bernoulli negative log likelihood.
    """
    raw, labels = np.asarray(raw, dtype=np.float64), np.asarray(labels)
    assert gamma >= 0 and positive_weight > 0 and normalizer > 0
    sign = 2. * labels - 1.
    margin = sign * raw
    t, q = expit(margin), expit(-margin)
    log_t = -np.logaddexp(0., -margin)
    weight = np.where(labels == 1, float(positive_weight), 1.) / float(normalizer)
    power = q ** gamma
    loss = -weight * power * log_t
    gradient = weight * sign * power * (gamma * t * log_t - q)
    hessian = weight * t * power * (gamma * log_t * (q - gamma * t) + (2. * gamma + 1.) * q)
    assert np.isfinite(loss).all() and np.isfinite(gradient).all() and np.isfinite(hessian).all()
    return loss, gradient, hessian


class FrozenTreeInput:
    """The same inner-only preprocessing is used by all new matched controls."""
    def fit(self, frame, rows):
        self.columns = list(frame.columns)
        self.numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical = [c for c in frame if c not in self.numeric]
        self.vocabulary = {
            c: sorted(frame[c].iloc[rows].astype('string').fillna('__MISSING__').unique().tolist())
            for c in self.categorical
        }
        a = frame.iloc[rows][self.numeric].to_numpy(dtype=np.float64, na_value=np.nan)
        a[~np.isfinite(a)] = np.nan
        count = np.isfinite(a).sum(axis=0)
        self.mean = np.divide(np.nansum(a, axis=0), count, out=np.zeros(len(self.numeric)), where=count > 0)
        centered = a - self.mean
        variance = np.divide(np.nansum(centered ** 2, axis=0), count,
                             out=np.zeros(len(self.numeric)), where=count > 0)
        self.scale = np.sqrt(variance)
        self.scale[~np.isfinite(self.scale) | (self.scale < 1e-8)] = 1.
        self.fit_rows = np.asarray(rows, dtype=np.int64)
        self.generic = {c: 'f' + str(i) for i, c in enumerate(self.columns)}
        return self

    def transform(self, frame, rows, kind):
        result = frame.iloc[rows][self.columns].copy()
        for i, c in enumerate(self.numeric):
            value = pd.to_numeric(result[c], errors='coerce').to_numpy(dtype=np.float64)
            value[~np.isfinite(value)] = np.nan
            # Missing values remain NaN, as required for linear-tree handling.
            result[c] = ((value - self.mean[i]) / self.scale[i]).astype(np.float32)
        for c in self.categorical:
            value = result[c].astype('string').fillna('__MISSING__')
            if kind == 'catboost':
                result[c] = value.where(value.isin(self.vocabulary[c]), '__UNSEEN__').astype(str)
            else:
                result[c] = pd.Categorical(value, categories=self.vocabulary[c])
        return result.rename(columns=self.generic)

    def recipe(self):
        return {
            'fit_rows': len(self.fit_rows), 'fit_rows_sha256': _ids(self.fit_rows),
            'numeric_columns': self.numeric, 'categorical_columns': self.categorical,
            'numeric': 'float64 fit-row mean/SD then float32; NaN remains NaN; constant/all-missing SD=1',
            'numeric_mean_sha256': _array_hash(self.mean), 'numeric_scale_sha256': _array_hash(self.scale),
            'category_dictionary_sha256': hashlib.sha256(json.dumps(self.vocabulary, sort_keys=True).encode()).hexdigest(),
            'categorical_cardinalities': {c: len(v) for c, v in self.vocabulary.items()},
            'categorical': 'sorted inner-only vocabulary; unseen is missing in LGB and __UNSEEN__ in CatBoost',
            'full_context_encoder_refit': False,
        }


def _deadline(spec, started):
    end = float(spec.get('deadline_epoch', started + float(spec.get('max_seconds', 1800.))))
    end = min(end, started + float(spec.get('max_seconds', 1800.)))
    if time.time() >= end:
        raise TimeoutError('CPU model deadline reached; no admission or partial-output substitution')
    return end


def _lgb_deadline(end):
    def callback(_env):
        if time.time() >= end:
            raise TimeoutError('CPU LightGBM execution cutoff reached')
    callback.order = 0
    callback.before_iteration = True
    return callback


class FocalObjective:
    def __init__(self, gamma, positive_weight, labels, floor=1e-6):
        self.gamma, self.positive_weight = float(gamma), float(positive_weight)
        self.normalizer = float(np.mean(np.where(np.asarray(labels) == 1, positive_weight, 1.)))
        self.floor = float(floor)
        self.calls = self.points = self.floored = 0
        self.minimum_hessian = np.inf

    def __deepcopy__(self, _memo):
        # LightGBM copies the parameter dictionary. Keep this stateless callable's
        # diagnostic counters attached to the exact objective used by training.
        return self

    def __call__(self, raw, matrix):
        _, gradient, hessian = focal_loss_derivatives(raw, matrix.get_label(), self.gamma,
                                                     self.positive_weight, self.normalizer)
        self.calls += 1
        self.points += len(hessian)
        self.floored += int(np.sum(hessian < self.floor))
        self.minimum_hessian = min(self.minimum_hessian, float(np.min(hessian)))
        return gradient, np.maximum(hessian, self.floor)

    def recipe(self):
        return dict(gamma=self.gamma, positive_class_factor=self.positive_weight,
                    weight_mean_normalizer=self.normalizer, hessian_floor=self.floor,
                    objective_calls=self.calls, evaluated_case_hessians=self.points,
                    floored_hessians=self.floored,
                    floored_fraction=self.floored / max(1, self.points),
                    minimum_exact_hessian=float(self.minimum_hessian) if self.calls else None,
                    initial_margin=0., initial_margin_uses_prevalence=False,
                    interpretation='sigmoid margin is an uncalibrated discrimination score for focal/class-weighted fits')


def _lightgbm(encoder, frame, ctx, y, query, inner, stop, spec, seed, end):
    import lightgbm as lgb
    parameters = dict(learning_rate=.035, num_leaves=31, max_depth=7,
                      min_data_in_leaf=150, lambda_l2=25., lambda_l1=.1,
                      bagging_fraction=.85, bagging_freq=1, feature_fraction=.85,
                      max_bin=63, num_threads=2, verbosity=-1,
                      seed=int(seed), deterministic=True, force_col_wise=True,
                      metric='None', boosting='gbdt', boost_from_average=False,
                      linear_tree=False, tree_learner='serial')
    parameters.update(spec.get('parameters', {}))
    parameters.update(num_threads=2, seed=int(seed), deterministic=True, metric='None',
                      force_col_wise=True, device_type='cpu', tree_learner='serial')
    rounds = int(parameters.pop('n_estimators', spec.get('max_iterations', 1800)))
    gamma = float(spec.get('focal_gamma', 0.))
    factor = float(spec.get('positive_weight', 1.))
    objective = spec.get('objective_kind', 'focal_family')
    if bool(parameters.get('linear_tree')) and objective != 'binary_native':
        raise ValueError('Linear-leaf factorial uses registered native binary objective only')
    stopping = spec.get('stopping_metric', 'ks')
    assert stopping in {'ks', 'logloss'}
    if objective == 'binary_native':
        assert gamma == 0 and factor == 1
        pilot_objective = full_objective = None
        parameters['objective'] = 'binary'
    else:
        assert objective == 'focal_family'
        pilot_objective = FocalObjective(gamma, factor, y[inner], spec.get('hessian_floor', 1e-6))
        parameters['objective'] = pilot_objective
        full_objective = None
    train_x = encoder.transform(frame, ctx[inner], 'lightgbm')
    stop_x = encoder.transform(frame, ctx[stop], 'lightgbm')
    train = lgb.Dataset(train_x, label=y[inner], free_raw_data=False)
    validation = lgb.Dataset(stop_x, label=y[stop], reference=train, free_raw_data=False)
    def evaluation(prediction, matrix):
        truth = matrix.get_label()
        if stopping == 'ks':
            return 'ks', ks(truth, prediction), True
        p = np.asarray(prediction) if objective == 'binary_native' else expit(prediction)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return 'logloss', float(-np.mean(truth * np.log(p) + (1-truth) * np.log1p(-p))), False
    fixed_iterations = bool(spec.get('fixed_iterations', False))
    if parameters.get('boosting') == 'dart' and not fixed_iterations:
        raise ValueError('DART requires registered fixed length; early stopping is not supported')
    callbacks = [_lgb_deadline(end)]
    if not fixed_iterations:
        callbacks.append(lgb.early_stopping(int(spec.get('patience', 100)), first_metric_only=True, verbose=False))
    model = lgb.train(parameters, train, num_boost_round=rounds, valid_sets=[validation],
                      feval=evaluation, callbacks=callbacks)
    iterations = rounds if fixed_iterations else max(1, int(model.best_iteration or rounds))
    pilot_audit = pilot_objective.recipe() if pilot_objective is not None else {'objective': 'native binary', 'initial_margin': 0.}
    del model, train, validation, train_x, stop_x
    gc.collect()
    if time.time() >= end:
        raise TimeoutError('CPU LightGBM cutoff before full-context refit')
    if objective != 'binary_native':
        full_objective = FocalObjective(gamma, factor, y, spec.get('hessian_floor', 1e-6))
        parameters['objective'] = full_objective
    train = lgb.Dataset(encoder.transform(frame, ctx, 'lightgbm'), label=y, free_raw_data=False)
    model = lgb.train(parameters, train, num_boost_round=iterations, callbacks=[_lgb_deadline(end)])
    values = encoder.transform(frame, query, 'lightgbm')
    raw = model.predict(values, raw_score=True, num_threads=2)
    replay = np.r_[model.predict(values.iloc[:64], raw_score=True, num_threads=2),
                   model.predict(values.iloc[64:128], raw_score=True, num_threads=2)]
    replay_error = float(np.max(np.abs(raw[:128] - replay)))
    assert replay_error <= 5e-6, 'CPU LightGBM batch replay failed'
    prediction = expit(np.asarray(raw, dtype=np.float64))
    serializable = {k: v for k, v in parameters.items() if k != 'objective'}
    serializable['objective'] = 'binary' if objective == 'binary_native' else 'declared focal custom objective'
    audit = dict(parameters=serializable, objective_kind=objective, stopping_metric=stopping,
                 maximum_iterations=rounds, selected_iterations=iterations,
                 fixed_registered_iterations=fixed_iterations,
                 stopping_rows_used_to_select_iterations=not fixed_iterations,
                 pilot_objective=pilot_audit,
                 full_objective=full_objective.recipe() if full_objective is not None else {'objective': 'native binary', 'initial_margin': 0.},
                 full_refit_finished_iterations=int(model.current_iteration()),
                 batch_replay_max_abs_margin_error=replay_error,
                 linear_leaf_numeric_only=bool(parameters.get('linear_tree')),
                 sources=[SOURCES['lgb_custom'], SOURCES['focal']] if objective != 'binary_native'
                         else [SOURCES['linear_tree'], SOURCES['linear_paper']])
    del model, train, encoder, values
    gc.collect()
    return prediction, audit


class CatKS:
    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        return ks(np.asarray(target), np.asarray(approxes[0])), 1.

    def get_final_error(self, error, weight):
        return error


class CatDeadline:
    def __init__(self, end):
        self.end, self.expired = float(end), False

    def after_iteration(self, _info):
        if time.time() >= self.end:
            self.expired = True
            return False
        return True


def _catboost(encoder, frame, ctx, y, query, inner, stop, spec, seed, end):
    from catboost import CatBoostClassifier
    parameters = dict(iterations=1120, depth=6, learning_rate=.035,
                      l2_leaf_reg=30., loss_function='Logloss',
                      bootstrap_type='Bernoulli', subsample=.85, border_count=64,
                      random_strength=1., boosting_type='Plain', grow_policy='SymmetricTree',
                      boost_from_average=False, random_seed=int(seed), thread_count=2,
                      verbose=False, allow_writing_files=False, task_type='CPU')
    parameters.update(spec.get('parameters', {}))
    parameters.update(random_seed=int(seed), thread_count=2, verbose=False,
                      allow_writing_files=False, task_type='CPU')
    assert parameters['loss_function'] == 'Logloss'
    assert parameters['boosting_type'] in {'Plain', 'Ordered'}
    assert parameters['grow_policy'] in {'SymmetricTree', 'Depthwise', 'Lossguide'}
    if parameters['boosting_type'] == 'Ordered' and parameters['grow_policy'] != 'SymmetricTree':
        raise ValueError('Ordered CatBoost is preregistered only with symmetric trees')
    stopping = spec.get('stopping_metric', 'ks')
    assert stopping in {'ks', 'auc', 'logloss'}
    parameters['eval_metric'] = CatKS() if stopping == 'ks' else ('AUC' if stopping == 'auc' else 'Logloss')
    guard = CatDeadline(end)
    pilot = CatBoostClassifier(**parameters)
    categorical = [encoder.generic[c] for c in encoder.categorical]
    pilot.fit(encoder.transform(frame, ctx[inner], 'catboost'), y[inner],
              cat_features=categorical,
              eval_set=(encoder.transform(frame, ctx[stop], 'catboost'), y[stop]),
              early_stopping_rounds=int(spec.get('patience', 100)), use_best_model=True,
              callbacks=[guard])
    if guard.expired or time.time() >= end:
        raise TimeoutError('CPU CatBoost stopping fit reached execution cutoff')
    best = pilot.get_best_iteration()
    iterations = max(1, int(best + 1 if best >= 0 else parameters['iterations']))
    maximum = int(parameters['iterations'])
    del pilot
    gc.collect()
    parameters['iterations'] = iterations
    parameters['eval_metric'] = 'Logloss'
    guard = CatDeadline(end)
    model = CatBoostClassifier(**parameters)
    model.fit(encoder.transform(frame, ctx, 'catboost'), y, cat_features=categorical,
              callbacks=[guard])
    if guard.expired or time.time() >= end:
        raise TimeoutError('CPU CatBoost fixed-length refit reached execution cutoff')
    assert model.tree_count_ == iterations, 'Interrupted CatBoost refit is not a completed fit'
    values = encoder.transform(frame, query, 'catboost')
    prediction = model.predict_proba(values)[:, 1]
    replay = np.r_[model.predict_proba(values.iloc[:64])[:, 1],
                   model.predict_proba(values.iloc[64:128])[:, 1]]
    replay_error = float(np.max(np.abs(prediction[:128] - replay)))
    assert replay_error <= 5e-6, 'CPU CatBoost batch replay failed'
    return prediction, dict(parameters=parameters, stopping_metric=stopping,
                            maximum_iterations=maximum, selected_iterations=iterations,
                            full_refit_finished_iterations=int(model.tree_count_),
                            batch_replay_max_abs_probability_error=replay_error,
                            sources=[SOURCES['catboost'], SOURCES['cat_parameters']])


def train_predict(frame, ctx, y, query, spec, seed, device='cpu'):
    """Select length on 42,500/7,500 context rows; refit 50k, predict 20k."""
    started = time.time()
    assert str(device) == 'cpu', 'No silent backend change'
    end = _deadline(spec, started)
    ctx, query = np.asarray(ctx, dtype=np.int64), np.asarray(query, dtype=np.int64)
    y = np.asarray(y, dtype=np.int32)
    assert len(ctx) == len(y) == 50000 and len(query) == 20000
    assert len(np.unique(ctx)) == 50000 and not len(np.intersect1d(ctx, query))
    assert set(np.unique(y)) == {0, 1}
    assert frame.columns.is_unique and not {'target', 'application_id', 'customer_id'} & set(frame.columns)
    split_seed = int(spec['inner_split_seed'])
    inner, stop = train_test_split(np.arange(len(ctx)), test_size=.15, stratify=y, random_state=split_seed)
    assert (len(inner), len(stop)) == (42500, 7500)
    encoder = FrozenTreeInput().fit(frame, ctx[inner])
    recipe = encoder.recipe()
    family = spec.get('family', spec.get('kind'))
    if family in {'lightgbm', 'lgbm', 'lgb_focal', 'lgb_linear'}:
        prediction, audit = _lightgbm(encoder, frame, ctx, y, query, inner, stop, spec, seed, end)
    elif family in {'catboost', 'catboost_cpu'}:
        prediction, audit = _catboost(encoder, frame, ctx, y, query, inner, stop, spec, seed, end)
    else:
        raise ValueError('Unsupported CPU modelling family: ' + str(family))
    if time.time() >= end:
        raise TimeoutError('CPU model execution cutoff before audited completion')
    prediction = np.asarray(prediction, dtype=np.float64)
    assert prediction.shape == (len(query),) and np.isfinite(prediction).all()
    assert np.all((prediction >= 0) & (prediction <= 1))
    audit.update(seed=int(seed), inner_split_seed=split_seed, context_rows=len(ctx),
                 inner_train_rows=len(inner), stopping_rows=len(stop),
                 inner_train_rows_sha256=_ids(ctx[inner]), stopping_rows_sha256=_ids(ctx[stop]),
                 refit_context_rows=len(ctx), preprocessing_refit_on_context=False,
                 inner_preprocessor=recipe, full_preprocessor=recipe,
                 preprocessing_protocol='v2 inner-only encoder frozen for full-context refit',
                 query_labels_used=False, correction_used_for_early_stopping=False,
                 fitted_weights_saved=False, KV_caches_saved=False,
                 execution_backend='CPU', deadline_epoch=end, wall_seconds=time.time()-started)
    return prediction, audit
