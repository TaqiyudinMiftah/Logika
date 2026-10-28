"""Context-only classical, actuarial-link and pairwise-ranking experiments.

No query labels are accepted. Inner encoders fit only the common 85% context
training split; after stopping, encoders and a fixed-length model refit on the
entire context. Fitted estimators and coefficients stay in RAM and are discarded.

The complementary-log-log model is a Bernoulli GLM for a single observed horizon,
not a survival model. XGBoost's pairwise loss is an explicit ranking adaptation;
the returned sigmoid margin is an uncalibrated score, not a default probability.
"""
import gc
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

SOURCES = {
    'actuarial': 'https://link.springer.com/book/10.1007/978-3-031-12409-9',
    'ks_motivation': 'https://doi.org/10.1016/j.csda.2018.10.004',
    'xgb_ranking': 'https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html',
    'xgb_categorical': 'https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/categorical.html',
    'lgb_stopping': 'https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.early_stopping.html',
}


def _ids(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _weights(y, mode):
    if mode in [None, 'none']:
        return None
    positive = float(np.mean(y))
    assert 0 < positive < 1
    if mode == 'balanced':
        value = np.where(y == 1, .5 / positive, .5 / (1 - positive))
    elif mode == 'sqrt':
        value = np.where(y == 1, np.sqrt((1 - positive) / positive), 1.)
    else:
        raise ValueError('Unknown class-weight recipe: ' + str(mode))
    return value / np.mean(value)


def _ks(y, prediction):
    order = np.argsort(prediction, kind='stable')
    ends = np.flatnonzero(np.r_[prediction[order][1:] != prediction[order][:-1], True])
    positive = np.cumsum(y[order]) / np.sum(y)
    negative = np.cumsum(1 - y[order]) / np.sum(1 - y)
    return float(np.max(np.abs(positive[ends] - negative[ends])))


class TreeInput:
    """Fit explicit category dictionaries; unseen query levels never enter them."""
    def fit(self, frame, rows):
        self.columns = list(frame.columns)
        self.numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical = [c for c in frame if c not in self.numeric]
        self.vocabulary = {
            c: sorted(frame[c].iloc[rows].astype('string').fillna('__MISSING__').unique().tolist())
            for c in self.categorical
        }
        self.fit_rows = np.asarray(rows)
        return self

    def transform(self, frame, rows, kind):
        result = frame.iloc[rows][self.columns].copy()
        for c in self.numeric:
            result[c] = pd.to_numeric(result[c], errors='coerce').astype('float32')
            result[c] = result[c].replace([np.inf, -np.inf], np.nan)
        for c in self.categorical:
            values = result[c].astype('string').fillna('__MISSING__')
            if kind == 'catboost':
                result[c] = values.where(values.isin(self.vocabulary[c]), '__UNSEEN__').astype(str)
            else:
                result[c] = pd.Categorical(values, categories=self.vocabulary[c])
        return result

    def xgboost(self, frame, rows, label=None, weight=None):
        import xgboost as xgb
        transformed = self.transform(frame, rows, 'xgboost')
        values = np.empty((len(rows), len(self.columns)), dtype=np.float32)
        feature_types = []
        for i, c in enumerate(self.columns):
            if c in self.vocabulary:
                a = transformed[c].cat.codes.to_numpy(dtype=np.float32)
                a[a < 0] = np.nan
                feature_types.append('c')
            else:
                a = transformed[c].to_numpy(dtype=np.float32)
                feature_types.append('q')
            values[:, i] = a
        # Generic names avoid XGBoost's reserved-character constraints; the exact
        # ordered source column names are retained by the worker's START record.
        return xgb.DMatrix(values, label=label, weight=weight,
                           feature_names=['c' + str(i) for i in range(len(self.columns))],
                           feature_types=feature_types, enable_categorical=True,
                           nthread=2)

    def recipe(self):
        encoded = json.dumps(self.vocabulary, sort_keys=True, ensure_ascii=False).encode()
        return dict(numeric_columns=self.numeric, categorical_columns=self.categorical,
                    categorical_cardinalities={c: len(v) for c, v in self.vocabulary.items()},
                    category_dictionary_sha256=hashlib.sha256(encoded).hexdigest(),
                    fit_rows=len(self.fit_rows), fit_rows_sha256=_ids(self.fit_rows),
                    numeric='float32; infinities become missing',
                    categorical='sorted fit-row vocabulary only; unseen becomes missing for LGB/XGB or __UNSEEN__ for CatBoost')


def _lightgbm(frame, ctx, y, query, inner, stop, spec, seed):
    import lightgbm as lgb
    parameters = dict(n_estimators=1400, num_leaves=31, max_depth=7,
                      learning_rate=.035, min_child_samples=150,
                      reg_lambda=20., reg_alpha=.1, subsample=.85,
                      subsample_freq=1, colsample_bytree=.85, max_bin=63,
                      random_state=seed, n_jobs=2, verbosity=-1,
                      objective='binary')
    parameters.update(spec.get('parameters', {}))
    parameters.update(random_state=seed, n_jobs=2, verbosity=-1)
    stopping = spec.get('stopping_metric', 'logloss')
    parameters['metric'] = 'None' if stopping == 'ks' else ('auc' if stopping == 'auc' else 'binary_logloss')
    parameters.pop('class_weight', None)
    mode = spec.get('class_balance', 'none')
    encoder = TreeInput().fit(frame, ctx[inner])
    pilot = lgb.LGBMClassifier(**parameters)
    kwargs = dict(eval_set=[(encoder.transform(frame, ctx[stop], 'lightgbm'), y[stop])],
                  sample_weight=_weights(y[inner], mode),
                  callbacks=[lgb.early_stopping(int(spec.get('patience', 80)), first_metric_only=True, verbose=False)])
    if stopping == 'ks':
        kwargs['eval_metric'] = lambda truth, pred: ('ks', _ks(np.asarray(truth), np.asarray(pred)), True)
    pilot.fit(encoder.transform(frame, ctx[inner], 'lightgbm'), y[inner], **kwargs)
    iterations = max(1, int(pilot.best_iteration_ or parameters['n_estimators']))
    pilot_recipe = encoder.recipe()
    del pilot, encoder
    gc.collect()
    parameters['n_estimators'] = iterations
    full = TreeInput().fit(frame, ctx)
    model = lgb.LGBMClassifier(**parameters)
    model.fit(full.transform(frame, ctx, 'lightgbm'), y, sample_weight=_weights(y, mode))
    prediction = model.predict_proba(full.transform(frame, query, 'lightgbm'))[:, 1]
    return prediction, dict(parameters=parameters, selected_iterations=iterations,
                            class_balance=mode, stopping_metric=stopping,
                            inner_preprocessor=pilot_recipe, full_preprocessor=full.recipe(),
                            execution_backend='CPU', sources=[SOURCES['lgb_stopping']])


def _catboost(frame, ctx, y, query, inner, stop, spec, seed, device):
    from catboost import CatBoostClassifier
    gpu = str(device).startswith('cuda') and bool(spec.get('allow_gpu', True))
    parameters = dict(iterations=1300, depth=6, learning_rate=.04, l2_leaf_reg=15.,
                      loss_function='Logloss', eval_metric='Logloss',
                      bootstrap_type='Bernoulli', subsample=.85, border_count=64,
                      random_seed=seed, thread_count=2, verbose=False,
                      allow_writing_files=False, task_type='GPU' if gpu else 'CPU')
    parameters.update(spec.get('parameters', {}))
    parameters.update(random_seed=seed, thread_count=2, verbose=False,
                      allow_writing_files=False, task_type='GPU' if gpu else 'CPU')
    parameters.pop('class_weights', None)
    parameters.pop('auto_class_weights', None)
    if gpu:
        parameters['devices'] = '0'
        # rsm is not supported for GPU Logloss; keep a declared compatible config.
        if parameters.get('rsm', 1.) != 1.:
            raise ValueError('CatBoost GPU Logloss does not support feature subsampling rsm')
    mode = spec.get('class_balance', 'none')
    encoder = TreeInput().fit(frame, ctx[inner])
    pilot = CatBoostClassifier(**parameters)
    pilot.fit(encoder.transform(frame, ctx[inner], 'catboost'), y[inner],
              sample_weight=_weights(y[inner], mode), cat_features=encoder.categorical,
              eval_set=(encoder.transform(frame, ctx[stop], 'catboost'), y[stop]),
              early_stopping_rounds=int(spec.get('patience', 80)), use_best_model=True)
    best = pilot.get_best_iteration()
    iterations = max(1, int(best + 1 if best >= 0 else parameters['iterations']))
    pilot_recipe = encoder.recipe()
    del pilot, encoder
    gc.collect()
    parameters['iterations'] = iterations
    full = TreeInput().fit(frame, ctx)
    model = CatBoostClassifier(**parameters)
    model.fit(full.transform(frame, ctx, 'catboost'), y,
              sample_weight=_weights(y, mode), cat_features=full.categorical)
    prediction = model.predict_proba(full.transform(frame, query, 'catboost'))[:, 1]
    return prediction, dict(parameters=parameters, selected_iterations=iterations,
                            class_balance=mode, stopping_metric=parameters['eval_metric'],
                            inner_preprocessor=pilot_recipe, full_preprocessor=full.recipe(),
                            execution_backend='CUDA' if gpu else 'CPU',
                            sources=[SOURCES['actuarial']])


def _xgboost(frame, ctx, y, query, inner, stop, spec, seed, device):
    import xgboost as xgb
    ranking = spec.get('objective', 'binary') in ['pairwise', 'rank:pairwise']
    mode = spec.get('class_balance', 'none')
    if ranking and mode != 'none':
        raise ValueError('Ranking has group weights, not per-case binary class weights')
    parameters = dict(objective='rank:pairwise' if ranking else 'binary:logistic',
                      tree_method='hist', device='cuda' if str(device).startswith('cuda') else 'cpu',
                      max_depth=5, eta=.035, min_child_weight=20., reg_lambda=20.,
                      reg_alpha=.1, subsample=.85, colsample_bytree=.85,
                      max_bin=64, seed=seed, nthread=2, verbosity=0,
                      eval_metric='auc' if ranking else 'logloss')
    parameters.update(spec.get('parameters', {}))
    parameters.update(seed=seed, nthread=2, verbosity=0,
                      device='cuda' if str(device).startswith('cuda') else 'cpu', tree_method='hist')
    if ranking:
        parameters.update(objective='rank:pairwise', lambdarank_pair_method='mean',
                          lambdarank_num_pair_per_sample=int(spec.get('pairs_per_sample', 4)),
                          base_score=0.)
    rounds = int(spec.get('max_iterations', parameters.pop('n_estimators', 1300)))
    stopping = spec.get('stopping_metric', 'auc' if ranking else 'logloss')
    if ranking and stopping not in ['auc', 'ks']:
        raise ValueError('Pairwise ranking stopping must use auc or ks, not a probability loss')
    parameters['disable_default_eval_metric'] = int(stopping == 'ks')
    if stopping != 'ks':
        parameters['eval_metric'] = 'auc' if stopping == 'auc' else 'logloss'
    encoder = TreeInput().fit(frame, ctx[inner])
    train = encoder.xgboost(frame, ctx[inner], y[inner], _weights(y[inner], mode))
    validation = encoder.xgboost(frame, ctx[stop], y[stop])
    if ranking:
        train.set_group(np.array([len(inner)], dtype=np.uint32))
        validation.set_group(np.array([len(stop)], dtype=np.uint32))
    options = {}
    if stopping == 'ks':
        options.update(custom_metric=lambda pred, matrix: ('ks', _ks(matrix.get_label(), np.asarray(pred))), maximize=True)
    pilot = xgb.train(parameters, train, num_boost_round=rounds,
                      evals=[(validation, 'inner_stop')],
                      early_stopping_rounds=int(spec.get('patience', 80)),
                      verbose_eval=False, **options)
    iterations = int(pilot.best_iteration) + 1
    pilot_recipe = encoder.recipe()
    del pilot, train, validation, encoder
    gc.collect()
    full = TreeInput().fit(frame, ctx)
    train = full.xgboost(frame, ctx, y, _weights(y, mode))
    if ranking:
        train.set_group(np.array([len(ctx)], dtype=np.uint32))
    model = xgb.train(parameters, train, num_boost_round=iterations)
    prediction = model.predict(full.xgboost(frame, query))
    if ranking:
        prediction = expit(prediction.astype(float))
    return prediction, dict(parameters=parameters, selected_iterations=iterations,
                            class_balance=mode, stopping_metric=stopping,
                            inner_preprocessor=pilot_recipe, full_preprocessor=full.recipe(),
                            execution_backend=parameters['device'],
                            ranking_grouping='one complete context group; default label is binary relevance' if ranking else None,
                            prediction_scale='uncalibrated sigmoid of ranking margin' if ranking else 'binary logistic probability',
                            sources=[SOURCES['xgb_categorical'], SOURCES['xgb_ranking']] if ranking else [SOURCES['xgb_categorical']])


class GLMInput:
    """Sparse additive spline or linear design, learned only on its fit rows."""
    def __init__(self, spline=False, knots=4, degree=2, category_limit=32):
        self.use_spline, self.knots, self.degree = spline, knots, degree
        self.category_limit = category_limit

    def fit(self, frame, rows):
        self.numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical = [c for c in frame if c not in self.numeric]
        self.fit_rows = np.asarray(rows)
        values = frame.iloc[rows][self.numeric].to_numpy(dtype=float, copy=True, na_value=np.nan)
        values[~np.isfinite(values)] = np.nan
        self.median = np.nanmedian(values, axis=0)
        self.median = np.nan_to_num(self.median, nan=0.)
        values = np.where(np.isfinite(values), values, self.median)
        self.center = np.median(values, axis=0)
        self.scale = np.maximum(np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0), .01)
        self.spline = None
        normalized = np.clip((values - self.center) / self.scale, -8., 8.)
        self.spline_numeric_index = np.flatnonzero(np.ptp(normalized, axis=0) > 1e-12)
        if self.use_spline and len(self.spline_numeric_index):
            self.spline = SplineTransformer(n_knots=self.knots, degree=self.degree,
                                           knots='uniform', extrapolation='constant',
                                           include_bias=False, sparse_output=True).fit(normalized[:, self.spline_numeric_index])
        self.vocabulary = {}
        for c in self.categorical:
            counts = frame[c].iloc[rows].astype('string').fillna('__MISSING__').value_counts()
            # Stable ties keep the recipe reproducible across seeds/backends.
            ranked = sorted(counts.index.tolist(), key=lambda value: (-int(counts[value]), str(value)))
            self.vocabulary[c] = ranked[:self.category_limit]
        self.onehot = None
        if self.categorical:
            self.onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=True,
                                        dtype=np.float64).fit(self._cats(frame, rows))
        return self

    def _cats(self, frame, rows):
        return np.column_stack([
            frame[c].iloc[rows].astype('string').fillna('__MISSING__')
            .where(lambda s: s.isin(self.vocabulary[c]), '__OTHER__').astype(str).to_numpy()
            for c in self.categorical
        ])

    def transform(self, frame, rows):
        values = frame.iloc[rows][self.numeric].to_numpy(dtype=float, na_value=np.nan)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        normalized = np.clip((filled - self.center) / self.scale, -8., 8.)
        parts = [sparse.csr_matrix(np.ones((len(rows), 1)))]
        if self.numeric:
            numeric_design = (self.spline.transform(normalized[:, self.spline_numeric_index]) if self.spline is not None
                              else sparse.csr_matrix((len(rows), 0))) if self.use_spline else normalized
            parts.extend([sparse.csr_matrix(numeric_design),
                          sparse.csr_matrix(missing.astype(float))])
        if self.categorical:
            parts.append(self.onehot.transform(self._cats(frame, rows)))
        return sparse.hstack(parts, format='csr')

    def recipe(self):
        return dict(numeric_columns=self.numeric, categorical_columns=self.categorical,
                    numeric='fit-row median/IQR, clipped[-8,8], additive missing indicators',
                    median_sha256=_array_hash(self.median), center_sha256=_array_hash(self.center),
                    scale_sha256=_array_hash(self.scale), spline=self.use_spline,
                    spline_knots=self.knots, spline_degree=self.degree,
                    spline_knots_recipe='uniform fit-row normalized range; fit-row constants have no spline basis',
                    spline_numeric_columns=[self.numeric[i] for i in self.spline_numeric_index],
                    category_limit=self.category_limit,
                    category_vocabulary_sha256=hashlib.sha256(json.dumps(self.vocabulary, sort_keys=True).encode()).hexdigest(),
                    fit_rows=len(self.fit_rows), fit_rows_sha256=_ids(self.fit_rows))


def _bernoulli(theta, x, y, ridge, link):
    eta = np.asarray(x @ theta).ravel()
    if link == 'logit':
        loss = np.mean(np.logaddexp(0., eta) - y * eta)
        derivative = expit(eta) - y
    elif link == 'cloglog':
        clipped = np.clip(eta, -25., 8.)
        intensity = np.exp(clipped)
        positive_loss = -np.log(-np.expm1(-intensity))
        loss = np.mean(y * positive_loss + (1 - y) * intensity)
        ratio = np.zeros_like(intensity)
        moderate = intensity < 50.
        ratio[moderate] = intensity[moderate] / np.expm1(intensity[moderate])
        derivative = ((1 - y) * intensity - y * ratio) * ((eta >= -25.) & (eta <= 8.))
    else:
        raise ValueError('Unsupported Bernoulli link: ' + str(link))
    regularized = theta.copy()
    regularized[0] = 0.
    loss += .5 * ridge * np.dot(regularized, regularized)
    gradient = np.asarray(x.T @ derivative).ravel() / len(y) + ridge * regularized
    return float(loss), gradient


def _glm_probability(x, theta, link):
    eta = np.asarray(x @ theta).ravel()
    return expit(eta) if link == 'logit' else -np.expm1(-np.exp(np.clip(eta, -25., 8.)))


def _nll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _glm(frame, ctx, y, query, inner, stop, spec):
    if len(frame.columns) > int(spec.get('max_input_columns', 128)):
        raise ValueError('GLM sparse spline experiment is declared only for at most128 columns')
    ridge = float(spec.get('ridge', spec.get('parameters', {}).get('ridge', .03)))
    link = spec.get('link', 'logit')
    blocks = int(spec.get('optimization_blocks', 8))
    block_iterations = int(spec.get('iterations_per_block', 25))
    patience = int(spec.get('glm_patience', 2))
    encoder_args = dict(spline=bool(spec.get('spline', False)), knots=int(spec.get('knots', 4)), degree=2,
                        category_limit=int(spec.get('category_limit', 32)))
    encoder = GLMInput(**encoder_args).fit(frame, ctx[inner])
    x, validation = encoder.transform(frame, ctx[inner]), encoder.transform(frame, ctx[stop])
    prevalence = float(np.clip(np.mean(y[inner]), 1e-5, 1 - 1e-5))
    theta = np.zeros(x.shape[1])
    theta[0] = np.log(prevalence / (1 - prevalence)) if link == 'logit' else np.log(-np.log1p(-prevalence))
    best, selected_blocks, stale, history = np.inf, 1, 0, []
    for block in range(1, blocks + 1):
        fit = minimize(_bernoulli, theta, args=(x, y[inner], ridge, link), jac=True,
                       method='L-BFGS-B', options=dict(maxiter=block_iterations, ftol=1e-9, gtol=1e-6))
        assert np.isfinite(fit.fun) and np.isfinite(fit.x).all()
        theta = fit.x
        score = _nll(y[stop], _glm_probability(validation, theta, link))
        history.append(dict(block=block, solver_iterations=int(fit.nit), stop_logloss=score))
        if score < best - 1e-7:
            best, selected_blocks, stale = score, block, 0
        else:
            stale += 1
        if stale >= patience or (fit.success and fit.nit < block_iterations):
            break
    pilot_recipe = encoder.recipe()
    del encoder, x, validation, theta, fit
    gc.collect()
    full = GLMInput(**encoder_args).fit(frame, ctx)
    x = full.transform(frame, ctx)
    prevalence = float(np.clip(np.mean(y), 1e-5, 1 - 1e-5))
    theta = np.zeros(x.shape[1])
    theta[0] = np.log(prevalence / (1 - prevalence)) if link == 'logit' else np.log(-np.log1p(-prevalence))
    refit_history = []
    for block in range(selected_blocks):
        fit = minimize(_bernoulli, theta, args=(x, y, ridge, link), jac=True,
                       method='L-BFGS-B', options=dict(maxiter=block_iterations, ftol=1e-9, gtol=1e-6))
        assert np.isfinite(fit.fun) and np.isfinite(fit.x).all()
        theta = fit.x
        refit_history.append(dict(solver_iterations=int(fit.nit), objective=float(fit.fun), success=bool(fit.success)))
    prediction = _glm_probability(full.transform(frame, query), theta, link)
    return prediction, dict(parameters=dict(link=link, ridge=ridge, **encoder_args,
                                            iterations_per_block=block_iterations,
                                            selected_optimization_blocks=selected_blocks),
                            pilot_inner_stopping=history, refit_optimization=refit_history,
                            selected_iterations=selected_blocks * block_iterations,
                            additive_design_columns=int(x.shape[1]),
                            inner_preprocessor=pilot_recipe, full_preprocessor=full.recipe(),
                            execution_backend='CPU', sources=[SOURCES['actuarial']],
                            survival_times_or_severity_labels_invented=False,
                            glm_link_interpretation='Bernoulli response at the supplied fixed horizon')


def train_predict(frame, ctx, y, query, spec, seed, device='cpu'):
    """Fit one declared model and return compact held-out query predictions."""
    started = time.monotonic()
    ctx, query = np.asarray(ctx, dtype=np.int64), np.asarray(query, dtype=np.int64)
    y = np.asarray(y, dtype=np.int32)
    assert len(ctx) == len(y) and len(np.unique(ctx)) == len(ctx)
    assert len(np.intersect1d(ctx, query)) == 0 and set(np.unique(y)) == {0, 1}
    assert frame.columns.is_unique and not {'target', 'application_id', 'customer_id'} & set(frame.columns)
    inner, stop = train_test_split(np.arange(len(ctx)), test_size=.15, stratify=y,
                                   random_state=int(spec.get('inner_split_seed', 20261029)))
    family = spec.get('family', spec.get('kind'))
    if family in ['lightgbm', 'lgbm']:
        prediction, audit = _lightgbm(frame, ctx, y, query, inner, stop, spec, seed)
    elif family == 'catboost':
        prediction, audit = _catboost(frame, ctx, y, query, inner, stop, spec, seed, device)
    elif family in ['xgboost', 'xgb', 'xgboost_pairwise']:
        if family == 'xgboost_pairwise':
            spec = {**spec, 'objective': 'pairwise'}
        prediction, audit = _xgboost(frame, ctx, y, query, inner, stop, spec, seed, device)
    elif family in ['glm', 'actuarial_glm']:
        prediction, audit = _glm(frame, ctx, y, query, inner, stop, spec)
    else:
        raise ValueError('Unsupported classical family: ' + str(family))
    prediction = np.asarray(prediction, dtype=float)
    assert prediction.shape == (len(query),) and np.isfinite(prediction).all()
    assert np.all((prediction >= 0) & (prediction <= 1))
    audit.update(seed=int(seed), inner_split_seed=int(spec.get('inner_split_seed', 20261029)),
                 context_rows=len(ctx), inner_train_rows=len(inner), stopping_rows=len(stop),
                 inner_train_rows_sha256=_ids(ctx[inner]), stopping_rows_sha256=_ids(ctx[stop]),
                 refit_context_rows=len(ctx), preprocessing_refit_on_context=True,
                 query_labels_used=False, correction_used_for_early_stopping=False,
                 fitted_weights_saved=False, KV_caches_saved=False,
                 wall_seconds=time.monotonic() - started)
    return prediction, audit
