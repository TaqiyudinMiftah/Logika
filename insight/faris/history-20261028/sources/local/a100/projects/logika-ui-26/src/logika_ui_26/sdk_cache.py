from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import types

PINNED_FIT_SHA256 = 'cd88a83747c8fcca8ee4cc8a14b31190236adb0c079aa3efbf4e1368f4cc3eef'


def cuda_fit_ast(source: str) -> ast.Module:
    source = textwrap.dedent(source)
    if hashlib.sha256(source.encode()).hexdigest() != PINNED_FIT_SHA256:
        raise ValueError('Pinned SDK fit source differs, cache adaptation refused')
    tree = ast.parse(source)
    condition = ast.dump(ast.parse('x.is_cuda and len(contexts) > 1', mode='eval').body)
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.If) and ast.dump(node.test) == condition]
    if len(matches) != 1:
        raise ValueError('Expected exactly one pinned CPU cache offload branch')
    matches[0].test = ast.copy_location(ast.Constant(value=False), matches[0].test)
    return ast.fix_missing_locations(tree)


def keep_fit_cache_on_cuda(model) -> dict:
    original = model.fit.__func__
    source = textwrap.dedent(inspect.getsource(original))
    tree = cuda_fit_ast(source)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<logika_cuda_cache_fit>', 'exec'), namespace)
    model.fit = types.MethodType(namespace['fit'], model)
    return {'storage': 'cuda', 'native_fit_sha256': PINNED_FIT_SHA256,
            'adaptation': 'Disable only the pinned per-estimator CPU offload branch'}


def cache_measurement(model) -> dict:
    cache = model._cache
    tensors = [tensor for number in range(cache['num_batches'])
               for tensor in cache[number]._tensors()]
    if not tensors or any(tensor.device.type != 'cuda' for tensor in tensors):
        raise ValueError('Estimator context cache escaped CUDA')
    return {'estimator_batches': cache['num_batches'], 'cached_tensors': len(tensors),
            'cache_tensor_bytes': sum(tensor.numel() * tensor.element_size() for tensor in tensors)}


def cache_digest(model) -> str:
    import torch

    digest = hashlib.sha256()
    cache = model._cache
    for number in range(cache['num_batches']):
        for tensor in cache[number]._tensors():
            digest.update(str((number, tuple(tensor.shape), str(tensor.dtype))).encode())
            flat = tensor.detach().reshape(-1).contiguous().view(torch.uint8).reshape(-1)
            for chunk in flat.split(8 * 1024 * 1024):
                digest.update(chunk.cpu().numpy().tobytes())
    return digest.hexdigest()
