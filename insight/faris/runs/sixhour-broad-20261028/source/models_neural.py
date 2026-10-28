"""Context-only neural experiments with held-out stopping and full-context refit.

TabM uses the official tabm==0.0.3 package. ResNet/MLP and preprocessing are
adapted from the sealed 27 October Logika architecture experiment, whose module
SHA256 was 4747e0c36e3347a505e0a2a7bf0401dc385bfc867ebd1910214e762860be8bba.
The KS loss is a task-specific smooth CDF-gap surrogate inspired by Fang and
Chen (2019); it is not their DMKS algorithm or a claimed paper reproduction.

Only ``y`` associated with ``ctx`` is accepted. No meta/outer targets are inputs.
Fitted state stays in memory and is discarded; this module performs no file I/O.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch import nn


DEPENDENCY_PINS = {
    'torch': '2.7.1+cu126', 'tabm': '0.0.3', 'numpy': '2.2.6',
    'pandas': '2.3.3', 'scikit-learn': '1.7.2', 'rtdl-num-embeddings': '0.0.12',
}
SOURCE_REFERENCES = [
    {'id': 'TABM', 'url': 'https://arxiv.org/abs/2410.24210v3',
     'implementation': 'https://github.com/yandex-research/tabm',
     'use': 'Official TabM.make, per-member BCE, mean member probabilities.'},
    {'id': 'RESNET', 'url': 'https://arxiv.org/abs/2106.11959',
     'use': 'Task-specific residual MLP and context-only numerical transforms.'},
    {'id': 'DMKS_MOTIVATION', 'url': 'https://doi.org/10.1016/j.csda.2018.10.004',
     'use': 'Objective mismatch motivates a distinct smooth KS surrogate; not exact DMKS.'},
]


def _ids_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def _check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Neural fit execution limit reached; incomplete refit is not a completed model.')


def _ks(y, probability):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    assert np.isfinite(p).all() and set(np.unique(y)) == {0, 1}
    order = np.argsort(p, kind='stable')
    ends = np.flatnonzero(np.r_[p[order][1:] != p[order][:-1], True])
    delta = np.cumsum(((y == 1) / np.sum(y == 1) - (y == 0) / np.sum(y == 0))[order])
    return float(np.max(np.abs(delta[ends])))


class Preprocessor:
    """Every learned transform is fitted on explicitly supplied training rows."""
    def fit(self, frame, rows, seed):
        rows = np.asarray(rows, dtype=np.int64)
        self.fit_rows_sha256 = _ids_sha(rows)
        self.fit_rows_count = len(rows)
        self.numeric = list(frame.select_dtypes(include=[np.number, 'bool']).columns)
        self.categorical = [name for name in frame if name not in self.numeric]
        self.median = np.empty(0, dtype=np.float64)
        self.quantile = None
        if self.numeric:
            values = frame[self.numeric].iloc[rows].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
            values[~np.isfinite(values)] = np.nan
            # Avoid all-missing warnings without using held-out feature values.
            self.median = np.asarray([
                float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.
                for column in values.T
            ])
            filled = np.where(np.isfinite(values), values, self.median)
            self.quantile = QuantileTransformer(
                n_quantiles=min(256, len(rows)), output_distribution='normal',
                subsample=None, random_state=seed,
            ).fit(filled)
        self.vocabulary = {
            name: {value: index + 1 for index, value in enumerate(sorted(
                frame[name].iloc[rows].astype('string').fillna('__MISSING__').unique().tolist()
            ))}
            for name in self.categorical
        }
        self.cards = [len(self.vocabulary[name]) + 1 for name in self.categorical]
        return self

    def transform(self, frame):
        if self.numeric:
            values = frame[self.numeric].to_numpy(dtype=np.float64, na_value=np.nan)
            missing = ~np.isfinite(values)
            z = self.quantile.transform(np.where(missing, self.median, values))
            z = np.column_stack([np.clip(z, -5., 5.), missing.astype(np.float64)]).astype(np.float32)
        else:
            z = np.empty((len(frame), 0), dtype=np.float32)
        category = np.column_stack([
            frame[name].astype('string').fillna('__MISSING__').map(self.vocabulary[name])
            .fillna(0).to_numpy(dtype=np.int64)
            for name in self.categorical
        ]) if self.categorical else np.empty((len(frame), 0), dtype=np.int64)
        assert np.isfinite(z).all()
        return z, category

    def recipe(self):
        return {
            'numeric_columns': self.numeric, 'categorical_columns': self.categorical,
            'fit_rows': self.fit_rows_count, 'fit_rows_sha256': self.fit_rows_sha256,
            'numeric_transform': 'Training-only median; 256-quantile Gaussian mapping; clip[-5,5]; missing masks.',
            'categorical_transform': 'Training-only sorted vocabulary; unknown index0.',
            'categorical_cardinalities': self.cards,
            # Exact statistics are regenerable from the retained rows/source/seed.
            # Keep compact identities, avoiding case-bearing vocabularies in JSON.
            'numeric_median_sha256': hashlib.sha256(np.asarray(self.median, dtype='<f8').tobytes()).hexdigest(),
            'quantile_grid_sha256': None if self.quantile is None else hashlib.sha256(
                np.asarray(self.quantile.quantiles_, dtype='<f8').tobytes()).hexdigest(),
            'vocabulary_sha256': hashlib.sha256(json.dumps(self.vocabulary, sort_keys=True).encode()).hexdigest(),
            'fitted_preprocessor_serialized': False,
        }


class FlatInput(nn.Module):
    def __init__(self, numeric_count, cards, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, embedding_dim) for card in cards])
        self.width = numeric_count + embedding_dim * len(cards)

    def forward(self, numeric, categorical):
        return torch.cat([numeric] + [module(categorical[:, index])
                                     for index, module in enumerate(self.embeddings)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * width, width), nn.Dropout(dropout),
        )

    def forward(self, value):
        return value + self.block(value)


class DenseModel(nn.Module):
    def __init__(self, numeric_count, cards, spec):
        super().__init__()
        self.inputs = FlatInput(numeric_count, cards, spec['cat_embedding_dim'])
        width = spec['width']
        if spec['kind'] == 'resnet':
            layers = [nn.Linear(self.inputs.width, width)]
            layers += [ResidualBlock(width, spec['dropout']) for _ in range(spec['blocks'])]
            layers += [nn.LayerNorm(width), nn.ReLU()]
        elif spec['kind'] == 'mlp':
            layers = []
            incoming = self.inputs.width
            for _ in range(spec['blocks']):
                layers += [nn.Linear(incoming, width), nn.ReLU(), nn.Dropout(spec['dropout'])]
                incoming = width
        else:
            raise ValueError('Unsupported dense architecture: ' + spec['kind'])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, numeric, categorical):
        return self.head(self.backbone(self.inputs(numeric, categorical)))


class OfficialTabM(nn.Module):
    def __init__(self, numeric_count, cards, spec):
        super().__init__()
        import tabm
        if importlib.metadata.version('tabm') != '0.0.3':
            raise RuntimeError('This sealed implementation requires tabm==0.0.3.')
        self.model = tabm.TabM.make(
            n_num_features=numeric_count, cat_cardinalities=cards, d_out=1,
            n_blocks=spec['blocks'], d_block=spec['width'], dropout=spec['dropout'],
            k=spec['ensemble_members'], arch_type='tabm',
        )

    def forward(self, numeric, categorical):
        return self.model(numeric, categorical if categorical.shape[1] else None).squeeze(-1)


def _model(numeric_count, cards, spec):
    if spec['kind'] == 'tabm':
        return OfficialTabM(numeric_count, cards, spec)
    return DenseModel(numeric_count, cards, spec)


def threshold_grid(prevalence, device):
    # Choosing the scale from training prevalence does not use query labels.
    multipliers = np.asarray([.35, .6, 1., 1.5, 2.2, 3.2])
    return torch.tensor(np.clip(prevalence * multipliers, .015, .75), dtype=torch.float32, device=device)


def training_loss(logits, targets, spec, thresholds):
    if logits.ndim == 1:
        logits = logits[:, None]
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets[:, None].expand_as(logits))
    loss = bce
    positive, negative = targets > .5, targets <= .5
    if positive.any() and negative.any():
        if spec['loss'] == 'smooth_ks':
            probability = torch.sigmoid(logits).mean(dim=1)
            soft_above = torch.sigmoid((probability[:, None] - thresholds[None, :]) / spec['ks_bandwidth'])
            separation = soft_above[positive].mean(dim=0) - soft_above[negative].mean(dim=0)
            temperature = spec['ks_softmax_temperature']
            smooth_max = temperature * (torch.logsumexp(separation / temperature, dim=0) - math.log(len(thresholds)))
            loss = loss - spec['ks_weight'] * smooth_max
        elif spec['loss'] == 'bce_pair':
            a = torch.nonzero(positive, as_tuple=True)[0]
            b = torch.nonzero(negative, as_tuple=True)[0]
            a = a[torch.randint(len(a), (min(1024, len(targets) * 2),), device=logits.device)]
            b = b[torch.randint(len(b), (len(a),), device=logits.device)]
            loss = loss + spec['pair_weight'] * nn.functional.softplus(-(logits[a] - logits[b])).mean()
    return loss


def _predict(model, numeric, categorical, rows, batch_size, deadline):
    model.eval()
    output = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            _check_time(deadline)
            batch = rows[offset:offset + batch_size]
            value = model(numeric[batch], categorical[batch])
            if value.ndim == 1:
                value = value[:, None]
            output.append(torch.sigmoid(value).mean(dim=1).float().cpu().numpy())
    return np.concatenate(output)


def _fit(model, numeric, categorical, rows, labels, spec, seed, deadline,
         stop_rows=None, stop_labels=None, fixed_epochs=None):
    rng = np.random.default_rng(seed)
    targets = torch.tensor(labels, dtype=torch.float32, device=numeric.device)
    thresholds = threshold_grid(float(np.mean(labels)), numeric.device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or any(key in name for key in ['embedding', 'numeric_weight', 'numeric_bias', 'cat_bias', 'cls']):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': spec['weight_decay']},
        {'params': no_decay, 'weight_decay': 0.},
    ], lr=spec['lr'])
    best_score, best_epoch, stale = -np.inf, 0, 0
    trace = []
    maximum = spec['max_epochs'] if fixed_epochs is None else fixed_epochs
    for epoch in range(1, maximum + 1):
        _check_time(deadline)
        model.train()
        order = rng.permutation(len(rows))
        total = 0.
        for offset in range(0, len(order), spec['batch_size']):
            _check_time(deadline)
            indices = order[offset:offset + spec['batch_size']]
            value = model(numeric[rows[indices]], categorical[rows[indices]])
            loss = training_loss(value, targets[indices], spec, thresholds)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite neural objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        item = {'epoch': epoch, 'training_objective': total / len(rows)}
        if stop_rows is not None:
            prediction = _predict(model, numeric, categorical, stop_rows, spec['predict_batch_size'], deadline)
            score = _ks(stop_labels, prediction)
            item['context_inner_stop_KS'] = score
            if score > best_score + spec['min_delta']:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        trace.append(item)
        if stop_rows is not None and epoch >= spec['min_epochs'] and stale >= spec['patience']:
            break
    if fixed_epochs is not None:
        assert len(trace) == fixed_epochs
        return fixed_epochs, trace, thresholds.detach().cpu().tolist()
    assert best_epoch > 0
    return best_epoch, trace, thresholds.detach().cpu().tolist()


def _spec(spec):
    defaults = dict(
        kind='resnet', width=128, blocks=3, dropout=.15, cat_embedding_dim=8,
        ensemble_members=16, loss='bce', lr=.001, weight_decay=.0001,
        batch_size=1024, predict_batch_size=1024, max_epochs=60, min_epochs=8,
        patience=8, min_delta=1e-6, early_stop_fraction=.15, inner_split_seed=20261029,
        gradient_clip=5., max_seconds=900., threads=2,
        pair_weight=.1, ks_weight=.15, ks_bandwidth=.03, ks_softmax_temperature=.02,
    )
    defaults.update(spec)
    assert defaults['kind'] in {'tabm', 'resnet', 'mlp'}
    assert defaults['loss'] in {'bce', 'bce_pair', 'smooth_ks'}
    assert 0 < defaults['early_stop_fraction'] < .5
    assert 1 <= defaults['min_epochs'] <= defaults['max_epochs'] <= 100
    assert defaults['max_seconds'] > 0
    assert defaults['ks_bandwidth'] > 0 and defaults['ks_softmax_temperature'] > 0
    return defaults


def train_predict(frame, ctx, y, query, spec, seed, device='cuda'):
    """Train one selected-column model and return predictions plus compact audit."""
    started = time.monotonic()
    spec = _spec(spec)
    deadline = started + spec['max_seconds']
    ctx, query = np.asarray(ctx, np.int64), np.asarray(query, np.int64)
    y = np.asarray(y, np.int8)
    assert len(ctx) == len(y) and len(ctx) > 20 and len(query) > 0
    assert len(np.unique(ctx)) == len(ctx) and len(np.unique(query)) == len(query)
    assert not np.intersect1d(ctx, query).size
    assert np.min(np.r_[ctx, query]) >= 0 and np.max(np.r_[ctx, query]) < len(frame)
    assert set(np.unique(y)) == {0, 1}
    assert isinstance(frame, pd.DataFrame) and frame.columns.is_unique
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available; no silent CPU fallback.')
    torch.set_num_threads(spec['threads'])
    if device.startswith('cuda'):
        # Explicit new-campaign execution choice: use A100 TF32 tensor cores.
        # Model parameters/loss remain float32; no GPU bitwise claim is made.
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    inner, stopping = train_test_split(
        np.arange(len(ctx)), test_size=spec['early_stop_fraction'], stratify=y,
        random_state=spec['inner_split_seed'],
    )
    train_rows, stop_rows = ctx[inner], ctx[stopping]
    assert not np.intersect1d(train_rows, stop_rows).size
    pre = Preprocessor().fit(frame, train_rows, seed)
    numeric, categorical = pre.transform(frame)
    inner_recipe = pre.recipe()
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    numeric_tensor = torch.as_tensor(numeric, device=device)
    categorical_tensor = torch.as_tensor(categorical, device=device)
    model = _model(numeric.shape[1], pre.cards, spec).to(device)
    chosen, inner_trace, inner_thresholds = _fit(
        model, numeric_tensor, categorical_tensor, train_rows, y[inner], spec,
        seed, deadline, stop_rows, y[stopping],
    )
    del model, numeric_tensor, categorical_tensor, numeric, categorical, pre
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    _check_time(deadline)
    pre = Preprocessor().fit(frame, ctx, seed)
    numeric, categorical = pre.transform(frame)
    numeric_tensor = torch.as_tensor(numeric, device=device)
    categorical_tensor = torch.as_tensor(categorical, device=device)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    model = _model(numeric.shape[1], pre.cards, spec).to(device)
    _, refit_trace, refit_thresholds = _fit(
        model, numeric_tensor, categorical_tensor, ctx, y, spec, seed, deadline,
        fixed_epochs=chosen,
    )
    output = _predict(model, numeric_tensor, categorical_tensor, query, spec['predict_batch_size'], deadline)
    replay = _predict(model, numeric_tensor, categorical_tensor, query[:128], spec['predict_batch_size'], deadline)
    replay_error = float(np.max(np.abs(output[:len(replay)] - replay)))
    assert replay_error <= 5e-6, replay_error
    assert output.shape == (len(query),) and np.isfinite(output).all()
    assert np.all((output >= 0) & (output <= 1))
    versions = {}
    for package in ['torch', 'tabm', 'numpy', 'pandas', 'scikit-learn']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not_installed; unused for this model'
    audit = {
        'parameters': spec, 'seed': int(seed), 'device': device, 'seconds': time.monotonic() - started,
        'selected_epochs': int(chosen), 'inner_train_rows': len(train_rows),
        'inner_stop_rows': len(stop_rows), 'refit_context_rows': len(ctx),
        'inner_train_rows_sha256': _ids_sha(train_rows), 'stopping_rows_sha256': _ids_sha(stop_rows),
        'context_rows_sha256': _ids_sha(ctx), 'prediction_rows_sha256': _ids_sha(query),
        'inner_preprocessing': inner_recipe, 'full_context_preprocessing': pre.recipe(),
        'early_stopping_metric': 'Exact tied binary KS; internal context holdout only.',
        'inner_split_seed': int(spec['inner_split_seed']),
        'inner_trace': inner_trace, 'refit_trace': refit_trace,
        'KS_surrogate': {
            'used': spec['loss'] == 'smooth_ks',
            'method': 'BCE minus weighted smooth maximum of class-conditional sigmoid CDF gaps.',
            'threshold_scale': 'Training prevalence times fixed multipliers [.35,.6,1,1.5,2.2,3.2], clipped [.015,.75].',
            'inner_training_thresholds': inner_thresholds, 'full_context_thresholds': refit_thresholds,
            'not_exact_DMKS_reproduction': True,
        },
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'precision': {
            'model_parameters': 'float32', 'AMP_used': False,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'CUDA_TF32_matmul_allowed': bool(torch.backends.cuda.matmul.allow_tf32) if device.startswith('cuda') else False,
            'CUDA_cuDNN_TF32_allowed': bool(torch.backends.cudnn.allow_tf32) if device.startswith('cuda') else False,
            'bitwise_backend_equivalence_claimed': False,
        },
        'peak_CUDA_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        'prediction_repeat_max_abs': replay_error, 'versions': versions,
        'outer_labels_loaded': False, 'correction_used_for_early_stopping': False,
        'stopping_preprocessing_fit_excludes_stopping_rows': True,
        'fitted_weights_saved': False, 'weights_saved': False, 'KV_cache_saved': False,
        'source_references': SOURCE_REFERENCES,
        'limitation': 'Seeded GPU execution is not claimed bitwise identical across hardware/backends.',
    }
    del model, numeric_tensor, categorical_tensor
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return output.astype(np.float64), audit


def synthetic_checks():
    """Numerical/boundary checks with synthetic data; no competition rows loaded."""
    rng = np.random.default_rng(261028)
    frame = pd.DataFrame({'a': rng.normal(size=300), 'b': rng.normal(size=300),
                          'category': ['x', 'y', 'z'] * 100})
    frame.loc[::11, 'b'] = np.nan
    training = np.arange(200)
    first = Preprocessor().fit(frame, training, 11)
    changed = frame.copy()
    changed.loc[200:, 'a'] = 1e99
    changed.loc[200:, 'category'] = 'query_only_category'
    second = Preprocessor().fit(changed, training, 11)
    assert first.recipe() == second.recipe()
    x, c = first.transform(changed)
    assert np.isfinite(x).all() and np.all(c[200:, 0] == 0)
    finite = {}
    for loss in ['bce', 'bce_pair', 'smooth_ks']:
        torch.manual_seed(111)
        logits = torch.randn(256, 4, requires_grad=True)
        target = torch.tensor([0., 1.] * 128)
        value = training_loss(logits, target, _spec({'loss': loss}), threshold_grid(.15, 'cpu'))
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(logits.grad).all()
        finite[loss] = float(value.detach())
    assert abs(_ks(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) - 1.) < 1e-12
    assert abs(_ks(np.array([0, 1, 0, 1]), np.ones(4)) - 0.) < 1e-12
    return {'preprocessing_excludes_changed_query': True, 'unknown_categories_mapped_to_zero': True,
            'losses_and_gradients_finite': finite, 'known_case_KS_passed': True,
            'competition_rows_loaded': False, 'weights_or_KV_serialized': False}
