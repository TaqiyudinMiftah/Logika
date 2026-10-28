"""Pinned Kumo cache adaptation reused from the A100 LOGIKA project.

Source: Projects/logika-ui-26/src/logika_ui_26/sdk_cache.py on Faris.
Retains estimator caches on CUDA without changing the model or preprocessing.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import types

PINNED_FIT_SHA256 = "cd88a83747c8fcca8ee4cc8a14b31190236adb0c079aa3efbf4e1368f4cc3eef"


def keep_fit_cache_on_cuda(model):
    original = model.fit.__func__
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != PINNED_FIT_SHA256:
        raise ValueError("Pinned SDK fit source differs; CUDA cache adaptation refused")
    tree = ast.parse(source)
    condition = ast.dump(ast.parse("x.is_cuda and len(contexts) > 1", mode="eval").body)
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.If) and ast.dump(node.test) == condition]
    if len(matches) != 1:
        raise ValueError("Expected exactly one pinned CPU cache offload branch")
    matches[0].test = ast.copy_location(ast.Constant(value=False), matches[0].test)
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<logika_cuda_cache_fit>", "exec"), namespace)
    model.fit = types.MethodType(namespace["fit"], model)
    return {"storage": "cuda", "native_fit_sha256": PINNED_FIT_SHA256}
