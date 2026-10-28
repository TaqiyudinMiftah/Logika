"""Runnable examples: ridge regression, a one-hidden-layer neural net, and DFS.

This illustrates the adapter contract. Generate task-specific implementations for
actual competitions; it is not a universal training pipeline.
"""
import json, pickle

def run_trial(ctx):
    data = json.loads(open(ctx.inputs['train']).read()); split = json.loads(ctx.split.read_text())
    if ctx.family == 'dfs':
        graph = data['graph']; start = data['start']; goal = data['goal']; best = [float('inf'), None]
        def visit(node, path, cost):
            ctx.check_stop()
            if node == goal:
                if cost < best[0]: best[:] = [cost, path]
                return
            for nxt, weight in graph.get(node, []):
                if nxt not in path: visit(nxt, path + [nxt], cost + weight)
        visit(start, [start], 0)
        if best[1] is None: raise ValueError('no feasible solution')
        ctx.save_json('solution.json', {'path': best[1]}); ctx.model('solution.json')
        ctx.save_json('validation.json', {'path': best[1]})
        return {'validation': ['validation.json'], 'folds': 1}
    import numpy as np
    x = np.asarray(data['x'], dtype=float); y = np.asarray(data['y'], dtype=float)
    train = np.asarray(split['train']); valid = np.asarray(split['valid'])
    if set(train) & set(valid): raise ValueError('train/validation overlap')
    if split.get('kind') == 'temporal' and max(data['time'][i] for i in train) >= min(data['time'][i] for i in valid): raise ValueError('temporal leakage')
    if split.get('kind') == 'group' and {data['group'][i] for i in train} & {data['group'][i] for i in valid}: raise ValueError('group leakage')
    mean = x[train].mean(0); std = x[train].std(0); std[std == 0] = 1
    tx = (x[train] - mean) / std; vx = (x[valid] - mean) / std
    if ctx.family == 'ridge':
        alpha = ctx.params.get('alpha', 1.)
        xx = np.column_stack([np.ones(len(tx)), tx]); vv = np.column_stack([np.ones(len(vx)), vx])
        coef = np.linalg.solve(xx.T @ xx + alpha * np.eye(xx.shape[1]), xx.T @ y[train]); pred = vv @ coef
        model = {'kind': 'ridge', 'coef': coef, 'mean': mean, 'std': std}
    elif ctx.family == 'mlp':
        rng = np.random.default_rng(ctx.seed); hidden = ctx.params.get('hidden', 8); rate = ctx.params.get('rate', .02)
        w = rng.normal(0, .2, (tx.shape[1], hidden)); b = np.zeros(hidden); v = rng.normal(0, .2, hidden); bias = 0.
        begin = 0
        if ctx.resume_from:
            from pathlib import Path
            checkpoint = Path(ctx.resume_from) / 'checkpoint.pkl'
            if checkpoint.exists():
                saved = pickle.loads(checkpoint.read_bytes()); w, b, v, bias, begin = [saved[k] for k in ('w', 'b', 'v', 'bias', 'step')]
        for step in range(begin, ctx.max_steps):
            h = np.tanh(tx @ w + b); error = h @ v + bias - y[train]; grad = 2 * error / len(tx)
            dw = tx.T @ ((grad[:, None] * v) * (1 - h * h)); db = ((grad[:, None] * v) * (1 - h * h)).sum(0)
            v -= rate * (h.T @ grad); bias -= rate * grad.sum(); w -= rate * dw; b -= rate * db
            saved = {'w': w, 'b': b, 'v': v, 'bias': bias, 'step': step + 1}
            ctx.path('checkpoint.pkl').write_bytes(pickle.dumps(saved)); ctx.model('checkpoint.pkl')
            pred = np.tanh(vx @ w + b) @ v + bias
            ctx.report(step + 1, float(np.mean((pred - y[valid]) ** 2)))
        model = {'kind': 'mlp', **saved, 'mean': mean, 'std': std}
    else: raise ValueError(f'unknown example family {ctx.family}')
    ctx.path('model.pkl').write_bytes(pickle.dumps(model)); ctx.model('model.pkl')
    ctx.save_json('validation.json', {'indices': valid.tolist(), 'predictions': pred.tolist()})
    return {'validation': ['validation.json'], 'folds': 1}

def evaluate(ctx, result):
    data = json.loads(open(ctx.inputs['train']).read()); saved = json.loads(ctx.path(result['validation'][0]).read_text())
    if ctx.family == 'dfs':
        path = saved['path']
        if path[0] != data['start'] or path[-1] != data['goal'] or len(set(path)) != len(path): raise ValueError('invalid path')
        score = 0
        for a, b in zip(path, path[1:]):
            edge = [w for nxt, w in data['graph'][a] if nxt == b]
            if not edge: raise ValueError('non-edge in solution')
            score += min(edge)
        return {'score': score, 'per_fold': [score]}
    split = json.loads(ctx.split.read_text())
    if saved['indices'] != split['valid']: raise ValueError('prediction IDs differ from frozen split')
    if len(saved['predictions']) != len(split['valid']): raise ValueError('prediction count mismatch')
    score = sum((float(p) - data['y'][i]) ** 2 for i, p in zip(saved['indices'], saved['predictions'])) / len(saved['indices'])
    return {'score': score, 'per_fold': [score]}
