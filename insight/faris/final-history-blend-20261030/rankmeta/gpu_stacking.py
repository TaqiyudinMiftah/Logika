"""Separate prospective pairwise-ranking meta stacking on the two idle RTXs."""
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import time
import traceback
import numpy as np
import torch
from torch import nn
from scipy.special import logit
from sklearn.model_selection import train_test_split

R = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(p)


def designs():
    return [dict(bank=b, width=w, penalty=p, lr=.001, rank_weight=rw, id=f'rankmeta_{b}_w{w}_p{p}_rw{rw}')
            for b, w, p, rw in itertools.product(('all', 'family'), (0, 32), (.01, .1), (.5, 1.))]


def fit_one(spec, seed, fold, deadline):
    assert time.time() < deadline - 30
    output = R / 'gpu_results' / spec['id'] / f's{seed}' / f'fold{fold}'
    if output.exists():
        return None
    output.mkdir(parents=True)
    started = time.time()
    frozen = json.loads((R / 'BANK_FROZEN.json').read_text())
    name = f'fold{fold}.npy' if spec['bank'] == 'all' else f'family{fold}.npy'
    assert sha(R / 'bank' / name) == frozen['bank_hashes'][str(fold)][name]
    z = np.load(R / f'bank/meta{fold}.npz')
    raw = np.load(R / 'bank' / name, mmap_mode='r')
    anchor = np.asarray(z['anchor'], np.float64)
    offset = logit(np.clip(anchor, 1e-6, 1 - 1e-6)).astype(np.float32)
    x = logit(np.clip(raw.astype(np.float32), 1e-6, 1 - 1e-6)) - offset[:, None]
    y = np.asarray(z['y_meta'], np.float32)
    train, stop = train_test_split(np.arange(6000), test_size=2000, random_state=20261030 + fold, stratify=y)
    mean = x[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x[train].std(axis=0, dtype=np.float64), .01).astype(np.float32)
    x = np.clip((x - mean) / scale, -10, 10)
    device = torch.device('cuda:0')
    xx = torch.as_tensor(x, dtype=torch.float32, device=device)
    oo = torch.as_tensor(offset, dtype=torch.float32, device=device)
    yy = torch.as_tensor(y, dtype=torch.float32, device=device)
    tr = torch.as_tensor(train, device=device)
    va = torch.as_tensor(stop, device=device)
    def model():
        torch.manual_seed(seed + fold)
        torch.cuda.manual_seed_all(seed + fold)
        if spec['width']:
            m = nn.Sequential(nn.Linear(x.shape[1], spec['width']), nn.SiLU(), nn.Linear(spec['width'], 1)).to(device)
        else:
            m = nn.Sequential(nn.Linear(x.shape[1], 1)).to(device)
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
        return m
    def update(m, optimizer, rows):
        optimizer.zero_grad(set_to_none=True)
        correction = m(xx[rows]).squeeze(-1)
        score = oo[rows] + correction
        positive = torch.nonzero(yy[rows] > .5).flatten()
        negative = torch.nonzero(yy[rows] < .5).flatten()
        pi = positive[torch.randint(len(positive), (4096,), device=device)]
        ni = negative[torch.randint(len(negative), (4096,), device=device)]
        ranking = nn.functional.softplus(-(score[pi] - score[ni])).mean()
        loss = nn.functional.binary_cross_entropy_with_logits(score, yy[rows]) + spec['rank_weight'] * ranking + spec['penalty'] * correction.square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 10.)
        optimizer.step()
    m = model()
    optimizer = torch.optim.AdamW(m.parameters(), lr=spec['lr'], weight_decay=0.)
    best, best_epoch, stale = float('inf'), 1, 0
    for epoch in range(1, 101):
        assert time.time() < deadline - 20
        update(m, optimizer, tr)
        with torch.no_grad():
            loss = float(nn.functional.binary_cross_entropy_with_logits(oo[va] + m(xx[va]).squeeze(-1), yy[va]).item())
        if loss < best - 1e-7:
            best, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
        if stale >= 12:
            break
    del m, optimizer
    m = model()
    optimizer = torch.optim.AdamW(m.parameters(), lr=spec['lr'], weight_decay=0.)
    allmeta = torch.arange(6000, device=device)
    for epoch in range(best_epoch):
        assert time.time() < deadline - 20
        update(m, optimizer, allmeta)
    with torch.no_grad():
        prediction = torch.sigmoid(oo[6000:] + m(xx[6000:]).squeeze(-1)).cpu().numpy()
        replay = torch.cat([torch.sigmoid(oo[k:min(k + 997, 20000)] + m(xx[k:k + 997]).squeeze(-1)) for k in range(6000, 20000, 997)]).cpu().numpy()
    error = float(np.max(abs(prediction - replay)))
    assert error <= 5e-6 and np.isfinite(prediction).all()
    np.savez_compressed(output / 'prediction.npz', rows=z['rows'], prediction=prediction)
    dump(output / 'FROZEN.json', dict(spec=spec, seed=seed, fold=fold, seconds=time.time() - started,
         best_inner_epoch=best_epoch, inner_meta_rows=4000, stopping_meta_rows=2000, final_meta_refit_rows=6000,
         outer_labels_loaded=False, encoder_rows=4000, encoder_excludes_meta_stopping=True,
         prediction_sha256=sha(output / 'prediction.npz'), base_bank_sha256=frozen['bank_hashes'][str(fold)][name],
         source_sha256=sha(__file__), replay_max_abs=error, fitted_weights_saved=False, KV_cache_saved=False,
         backend='CUDAfloat32;TF32disabled;CUDA RNG;two base seeds already averaged', GPU=torch.cuda.get_device_name(0)))
    del m, optimizer, xx, oo, yy, tr, va
    torch.cuda.empty_cache()
    return time.time() - started


def main(index):
    plan = json.loads((R / 'GPU_PLAN.json').read_text())
    assert (R / 'GPU_PREPUBLICATION.json').exists()
    assigned = [spec for i, spec in enumerate(designs()) if i % 2 == index]
    for spec in assigned:
        pilot = fit_one(spec, 2026, 0, plan['model_deadline_epoch'])
        if pilot is None:
            continue
        feasible = time.time() + pilot * 9 * 1.5 + 120 < plan['model_deadline_epoch']
        dump(R / 'gpu_gates' / (spec['id'] + '.json'), dict(status='PASS' if feasible else 'FAIL_RESOURCE_FEASIBILITY',
             pilot_seconds=pilot, safety_factor=1.5, finish_margin_seconds=120, scientific_conclusion_from_resource_failure=False))
        if not feasible:
            continue
        for seed, fold in itertools.product((2026, 7731), range(5)):
            if seed == 2026 and fold == 0:
                continue
            fit_one(spec, seed, fold, plan['model_deadline_epoch'])
        print(json.dumps(dict(recipe=spec['id'], status='complete10fits')), flush=True)
    dump(R / f'GPU_OWNER_FINAL_{index}.json', dict(epoch=time.time(), owner_index=index, source_sha256=sha(__file__)))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--owner-index', type=int, choices=(0, 1), required=True)
    args = a.parse_args()
    try:
        main(args.owner_index)
    except BaseException:
        folder = R / 'private'
        folder.mkdir(exist_ok=True)
        (folder / f'GPU_FAILURE_{args.owner_index}.log').write_text(traceback.format_exc())
        raise
