"""Prospective Bernoulli offset adaptations of DCNv2 and random Fourier kernels.

These are credit-risk adaptations, not reproductions of recommendation benchmarks.
All model and encoder state remains in RAM. References and exact grids are sealed
in the campaign configuration before full-size pilots are admitted.
"""
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_name] = '2'

import gc
import math
import time
import numpy as np
from scipy.special import logit
from sklearn.model_selection import train_test_split
import torch
from cpu_offset import FrozenInput, _glm, _ids, _hash, _ks, _check


class CrossLayer(torch.nn.Module):
    """Paper equations 2--4: low-rank cross with optional nonlinear expert gates."""
    def __init__(self, features, rank, experts):
        super().__init__()
        self.down = torch.nn.ModuleList([torch.nn.Linear(features, rank, bias=False) for _ in range(experts)])
        self.up = torch.nn.ModuleList([torch.nn.Linear(rank, features, bias=True) for _ in range(experts)])
        self.middle = torch.nn.ModuleList([torch.nn.Linear(rank, rank, bias=False) for _ in range(experts)]) if experts>1 else torch.nn.ModuleList()
        self.gate = torch.nn.Linear(features, experts, bias=False) if experts > 1 else None
        for layer in self.up:
            torch.nn.init.zeros_(layer.bias)

    def forward(self, original, current):
        if self.gate is None:
            return current + original * self.up[0](self.down[0](current))
        weights = torch.softmax(self.gate(current), dim=-1)
        crossed = torch.zeros_like(current)
        for index, (down, middle, up) in enumerate(zip(self.down, self.middle, self.up)):
            value = up(torch.tanh(middle(torch.tanh(down(current)))))
            crossed = crossed + weights[:, index:index + 1] * original * value
        return current + crossed


class Network(torch.nn.Module):
    def __init__(self, features, spec, seed):
        super().__init__()
        kind = spec['kind']
        self.kind = kind
        width = int(spec.get('width', 128))
        self.intercept = torch.nn.Parameter(torch.zeros(()))
        if kind == 'rff':
            count = int(spec['rff_components'])
            generator = torch.Generator(device='cpu').manual_seed(int(seed) + 314159)
            gamma = float(spec['gamma_multiplier']) / features
            self.register_buffer('frequencies', torch.randn(2048, features, generator=generator).T[:, :count] * math.sqrt(2 * gamma))
            self.register_buffer('phases', torch.rand(2048, generator=generator)[:count] * (2 * math.pi))
            self.head = torch.nn.Linear(count, 1, bias=False)
            self.feature_scale = math.sqrt(2. / count)
        else:
            self.body = torch.nn.Sequential(
                torch.nn.Linear(features, width), torch.nn.GELU(), torch.nn.Dropout(.1),
                torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Dropout(.1))
            if kind.startswith('cross_'):
                experts = 1 if kind == 'cross_linear' else 4
                self.cross = torch.nn.ModuleList([
                    CrossLayer(features, int(spec['rank']), experts)
                    for _ in range(int(spec['cross_layers']))])
                self.head = torch.nn.Linear(features + width, 1, bias=False)
            elif kind == 'localglm':
                self.head = torch.nn.Linear(width, features)
            else:
                assert kind == 'mlp'
                self.head = torch.nn.Linear(width, 1, bias=False)
        torch.nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if self.kind == 'rff':
            z = x if getattr(self, 'precomputed', False) else torch.cos(x @ self.frequencies + self.phases) * self.feature_scale
            return self.intercept + self.head(z).squeeze(-1)
        hidden = self.body(x)
        if self.kind == 'localglm':
            return self.intercept + (self.head(hidden) * x).sum(dim=-1)
        if self.kind.startswith('cross_'):
            crossed = x
            for layer in self.cross:
                crossed = layer(x, crossed)
            hidden = torch.cat([hidden, crossed], dim=-1)
        return self.intercept + self.head(hidden).squeeze(-1)


def build(features, spec, seed, device):
    torch.manual_seed(int(seed))
    if device == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    return Network(features, spec, seed).to(device)


def dense(encoder, frame, rows, device):
    array = encoder.glm(frame, rows).toarray().astype(np.float32)
    assert np.isfinite(array).all()
    return torch.from_numpy(array).to(device)


def project_fixed(net, x):
    # Fixed RFF basis: compute once in RAM, never serialize a representation cache.
    with torch.no_grad():
        return torch.cat([torch.cos(chunk @ net.frequencies + net.phases) * net.feature_scale for chunk in x.split(1024)], dim=0)


def probabilities(net, x, offsets, batch=1024):
    net.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            value = torch.sigmoid(offsets[start:start + batch] + net(x[start:start + batch]))
            values.append(value.cpu().numpy().astype(np.float64))
    return np.concatenate(values)


def epochs(net, x, y, offsets, spec, seed, end, number, stop=None):
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(spec['learning_rate']), weight_decay=.01)
    generator = torch.Generator(device='cpu').manual_seed(int(seed) + 271828)
    penalty = float(spec['residual_penalty'])
    best_epoch, stale, done = 0, 0, 0
    best_score = _ks(stop[1], probabilities(net, stop[0], stop[2])) if stop else None
    for epoch in range(1, number + 1):
        _check(end)
        net.train()
        for index in torch.randperm(len(x), generator=generator).split(512):
            _check(end)
            index = index.to(x.device)
            optimizer.zero_grad(set_to_none=True)
            residual = net(x[index])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(offsets[index] + residual, y[index])
            loss = loss + penalty * residual.square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        done = epoch
        if stop:
            score = _ks(stop[1], probabilities(net, stop[0], stop[2]))
            if score > best_score + 1e-12:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if stale >= 12:
                break
    del optimizer
    return dict(epochs_completed=done, selected_epochs=best_epoch if stop else done,
                best_stopping_KS=best_score)


def train_predict(frame, ctx, y, query, spec, seed, device='cpu'):
    started = time.time()
    assert device in {'cpu', 'cuda'}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda':
        assert torch.cuda.is_available() and os.environ.get('NVIDIA_TF32_OVERRIDE') == '0'
    if device == 'cpu':
        assert not torch.cuda.is_initialized(), 'CPU owner must not create a GPU context'
    end = min(float(spec['deadline_epoch']), started + float(spec['max_seconds']))
    ctx, query, y = np.asarray(ctx, np.int64), np.asarray(query, np.int64), np.asarray(y, np.int64)
    assert (len(ctx), len(query), len(y)) == (50000, 20000, 50000)
    assert not np.intersect1d(ctx, query).size
    inner, stop = train_test_split(np.arange(50000), test_size=.15, stratify=y,
                                  random_state=int(spec['inner_split_seed']))
    encoder = FrozenInput().fit(frame, ctx[inner])
    recipe = encoder.recipe()
    mode = spec['offset_mode']
    assert mode in {'glm', 'intercept'}
    glm_audits = {}
    if mode == 'glm':
        matrix = encoder.glm(frame, ctx[inner])
        glm, glm_audits['inner'] = _glm(matrix, y[inner], float(spec['glm_C']), end)
        train_offset = glm.decision_function(matrix)
        stop_offset = glm.decision_function(encoder.glm(frame, ctx[stop]))
        del matrix, glm
    else:
        value = float(logit(y[inner].mean()))
        train_offset, stop_offset = np.full(len(inner), value), np.full(len(stop), value)
    x, xs = dense(encoder, frame, ctx[inner], device), dense(encoder, frame, ctx[stop], device)
    of = torch.tensor(train_offset, dtype=torch.float32, device=device)
    ofs = torch.tensor(stop_offset, dtype=torch.float32, device=device)
    labels = torch.tensor(y[inner], dtype=torch.float32, device=device)
    net = build(x.shape[1], spec, seed, device)
    if spec['kind'] == 'rff':
        x, xs = project_fixed(net, x), project_fixed(net, xs)
        net.precomputed = True
    pilot = epochs(net, x, labels, of, spec, seed, end, 120, stop=(xs, y[stop], ofs))
    selected = int(pilot['selected_epochs'])
    del x, xs, of, ofs, labels, net
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    _check(end)
    if mode == 'glm':
        matrix = encoder.glm(frame, ctx)
        glm, glm_audits['context'] = _glm(matrix, y, float(spec['glm_C']), end)
        context_offset = glm.decision_function(matrix)
        query_offset = glm.decision_function(encoder.glm(frame, query))
        del matrix, glm
    else:
        value = float(logit(y.mean()))
        context_offset, query_offset = np.full(50000, value), np.full(20000, value)
    x = dense(encoder, frame, ctx, device)
    labels = torch.tensor(y, dtype=torch.float32, device=device)
    of = torch.tensor(context_offset, dtype=torch.float32, device=device)
    net = build(x.shape[1], spec, seed, device)
    if spec['kind'] == 'rff':
        x = project_fixed(net, x)
        net.precomputed = True
    full = epochs(net, x, labels, of, spec, seed, end, selected)
    assert full['epochs_completed'] == selected
    del x, labels, of
    gc.collect()
    xq = dense(encoder, frame, query, device)
    oq = torch.tensor(query_offset, dtype=torch.float32, device=device)
    embedding_error = None
    if spec['kind'] == 'rff':
        embedding = project_fixed(net, xq)
        alternate = torch.cat([project_fixed(net, xq[:64]), project_fixed(net, xq[64:128])])
        embedding_error = float(torch.max(torch.abs(embedding[:128]-alternate)).cpu())
        assert embedding_error <= 5e-6
        xq = embedding
    probability = probabilities(net, xq, oq)
    replay = np.r_[probabilities(net, xq[:64], oq[:64], 64),
                   probabilities(net, xq[64:128], oq[64:128], 64)]
    error = float(np.max(np.abs(probability[:128] - replay)))
    assert error <= 5e-6
    assert probability.shape == (20000,) and np.isfinite(probability).all()
    assert np.all((probability >= 0) & (probability <= 1))
    _check(end)
    audit = dict(
        model_family='Bernoulli_' + spec['kind'], offset_mode=mode,
        trainable_parameter_count=sum(p.numel() for p in net.parameters() if p.requires_grad),
        CPU_GPU_context_created=torch.cuda.is_initialized() if device=='cpu' else None,
        model_spec=spec, learning_rate=float(spec['learning_rate']), batch_size=512,
        pilot=pilot, full_refit=full, selected_epochs=selected, glm_parameters=glm_audits,
        inner_preprocessor=recipe, full_preprocessor=recipe, preprocessing_refit_on_context=False,
        context_rows=50000, inner_train_rows=42500, stopping_rows=7500, final_training_rows=50000,
        inner_train_rows_sha256=_ids(ctx[inner]), stopping_rows_sha256=_ids(ctx[stop]),
        batch_replay_max_abs_probability_error=error, query_labels_used=False,
        correction_used_for_early_stopping=False, full_refit_fresh_initialization=True,
        model_initialization_seed=int(seed), shuffle_seed=int(seed) + 271828,
        RFF_seed=int(seed) + 314159 if spec['kind'] == 'rff' else None,
        RFF_nested_max_components=2048 if spec['kind']=='rff' else None,
        RFF_embedding_batch_replay_error=embedding_error,
        fixed_RFF_projection_precomputed_only_in_RAM=spec['kind']=='rff',
        execution_backend=device + ' float32; TF32 disabled;2 CPUthreads',
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        model_RNG='CUDA dropout/CPU initialization' if device == 'cuda' else 'CPU',
        context_offset_sha256=_hash(context_offset), query_offset_sha256=_hash(query_offset),
        fitted_weights_saved=False, KV_caches_saved=False, wall_seconds=time.time() - started)
    del net, xq, oq, encoder
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return probability, audit
