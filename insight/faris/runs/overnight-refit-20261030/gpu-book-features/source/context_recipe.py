"""Context-subsample Kumo experiments, without changing the pinned predictor.

This adapter varies only which permitted context rows are supplied and how
independently sampled contexts are averaged. Feature construction, feature
selection, checkpoint, tensorization and inference remain the historical recipe.
No fit, cache or parameter tensor is written to disk by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np


CHECKPOINT_SHA256 = {
    "large": "cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613",
    "medium": "70d6aee8fd13416300afead71a5145a40edb790952e2417d87f27b3f7e57793d",
}
ENGINE_SHA256 = "f05ccade0bfead7dae94d80965f6d99b15a7a7c0fe5abe6173d9f1499aa74697"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _check_deadline(spec):
    end = spec.get("model_deadline_epoch")
    if end is not None and time.time() >= float(end):
        raise TimeoutError("Original model deadline reached; context bag not admitted.")


def _checkpoint(spec, size):
    """Resolve a private physical path while enforcing the public weight identity."""
    expected = CHECKPOINT_SHA256[size]
    supplied_digest = spec.get("checkpoint_sha256", expected)
    assert supplied_digest == expected, "Checkpoint differs from the registered v1.0.1 recipe."
    path = spec.get("checkpoint_path", spec.get("checkpoint_" + size))
    if path is None:
        root = Path(__file__).resolve().parents[1]
        environment = root / "ENVIRONMENT.json"
        if environment.exists():
            path = json.loads(environment.read_text()).get("checkpoint_" + size)
    if not path:
        raise ValueError("Physical checkpoint_path must be declared in the runtime spec/ENVIRONMENT.")
    physical = Path(path).resolve()
    assert physical.is_relative_to(Path.home().resolve()), "Only actual user-home checkpoints are allowed."
    assert physical.is_file() and _sha(physical) == expected, "Pinned checkpoint verification failed."
    return str(physical), expected


def sample_context(y, count, seed, bag):
    """Return unique allowed-context offsets; no query covariates or labels enter.

    Whole-context control preserves the historical row order and model RNG seed.
    Smaller contexts use independent deterministic stratified draws without
    replacement inside each bag. Overlap between different bags is intentional.
    """
    y = np.asarray(y, dtype=np.int8)
    n = len(y)
    assert y.ndim == 1 and set(np.unique(y)) == {0, 1}
    assert count in {10000, 25000, 50000} and 0 < count <= n
    assert isinstance(bag, int) and 0 <= bag < 4
    draw_seed = int(seed) + 49999 + 32452843 * bag
    if count == n:
        result = np.arange(n, dtype=np.int64)
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=count, random_state=draw_seed)
        result, _ = next(splitter.split(np.zeros((n, 1), dtype=np.int8), y))
        # Sorting removes accidental context-order variation as a second factor.
        result = np.sort(np.asarray(result, dtype=np.int64))
    assert len(result) == count and len(np.unique(result)) == count
    assert result.min() >= 0 and result.max() < n
    assert set(np.unique(y[result])) == {0, 1}
    return result, draw_seed


def train_predict(frame, ctx, y, query, spec, seed, device="cuda"):
    """Historical worker interface: 50k labels, fixed features, 20k unlabeled query.

    Root worker is responsible for sealed packet hashes and model deadline unit.
    `spec.model_deadline_epoch` additionally prevents admitting a later bag after
    the same immutable deadline. No scientific recipe is shortened to fit time.
    """
    started = time.time()
    assert device == "cuda", "CPU Kumo is prohibited; no silent CPU fallback."
    import torch

    assert torch.cuda.is_available(), "Admitted CUDA backend is unavailable."
    ctx = np.asarray(ctx, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    y = np.asarray(y, dtype=np.int8)
    assert len(ctx) == len(y) == 50000 and len(query) == 20000
    assert len(np.unique(ctx)) == len(ctx) and len(np.unique(query)) == len(query)
    assert not np.intersect1d(ctx, query).size, "Context/query row boundary violation."
    assert len(frame) == 70000 and min(ctx.min(), query.min()) >= 0
    assert max(ctx.max(), query.max()) < len(frame)
    columns = list(frame.columns)
    assert 1 <= len(columns) <= 99 and len(set(columns)) == len(columns)
    assert not {"target", "application_id", "customer_id", "split", "y_outer", "y_validation"} & set(columns)
    size = spec["size"]
    count = int(spec["context_count"])
    bags = int(spec["context_bags"])
    estimators = int(spec["estimators"])
    assert size in CHECKPOINT_SHA256 and bags in {1, 4} and estimators in {2, 8}
    assert (count, bags, estimators) in {
        (50000, 1, 2), (25000, 1, 2), (25000, 1, 8),
        (25000, 4, 2), (10000, 4, 2),
    }, "Unregistered context-sensitivity recipe."
    assert spec.get("engine", "concat_context_vocab") == "concat_context_vocab"
    assert int(spec.get("prediction_batch", 256)) == 256
    assert float(spec.get("memory_fraction", .7)) == .7
    assert spec.get("aggregation", "probability_mean") == "probability_mean"
    assert spec.get("context_sampling", "stratified_without_replacement_per_bag") == "stratified_without_replacement_per_bag"
    checkpoint, digest = _checkpoint(spec, size)
    engine_path = Path(__file__).with_name("gpu_kumo_engine.py")
    assert _sha(engine_path) == ENGINE_SHA256, "Historical inference engine was modified."
    from gpu_kumo_engine import predict

    sums = np.zeros(len(query), dtype=np.float64)
    sums_square = np.zeros(len(query), dtype=np.float64)
    records, draws = [], []
    for bag in range(bags):
        _check_deadline(spec)
        selected, draw_seed = sample_context(y, count, int(seed), bag)
        physical_context = ctx[selected]
        # Bag zero keeps the historical model RNG when the context is the same.
        inference_seed = int(seed) + 104729 * bag
        print(json.dumps({"stage": "context_bag_start", "bag": bag, "bags": bags,
                          "context_rows": count, "model_seed": inference_seed,
                          "context_draw_seed": draw_seed}), flush=True)
        prediction, audit = predict(
            frame.iloc[physical_context], frame.iloc[query], columns, y[selected],
            checkpoint, inference_seed, engine="concat_context_vocab",
            estimators=estimators, pred_batch=256, mem_fraction=.7, size=size,
            expected_sha=digest,
        )
        prediction = np.asarray(prediction, dtype=np.float64)
        assert prediction.shape == (len(query),) and np.isfinite(prediction).all()
        assert np.all((prediction >= 0) & (prediction <= 1))
        assert audit["first_batch_replay_error"] < 1e-7
        sums += prediction
        sums_square += prediction**2
        records.append({
            "bag": bag, "context_draw_seed": draw_seed, "model_seed": inference_seed,
            "context_rows": count, "class_counts": np.bincount(y[selected], minlength=2).tolist(),
            "context_rows_sha256": _array_sha(physical_context, "<i8"),
            "context_offsets_sha256": _array_sha(selected, "<i8"),
            "context_labels_sha256": _array_sha(y[selected], "i1"),
            "prediction_array_sha256": _array_sha(prediction, "<f8"),
            "engine_audit": audit,
        })
        draws.append(selected)
        print(json.dumps({"stage": "context_bag_complete", "bag": bag,
                          "seconds": audit["seconds"],
                          "same_cache_replay_error": audit["first_batch_replay_error"]}), flush=True)
        _check_deadline(spec)
    prediction = sums / bags
    overlap = [[int(np.intersect1d(a, b).size) for b in draws] for a in draws]
    variance = np.maximum(0., sums_square / bags - prediction**2)
    return prediction, {
        "family": "kumo_context", "size": size, "context_bags": bags,
        "sampled_context_rows_per_bag": count, "available_context_rows": len(ctx),
        "estimators_per_bag": estimators, "total_internal_estimators": bags * estimators,
        "sampling": "Independent stratified without replacement per bag; sorted offsets; overlap allowed.",
        "aggregation": "Unweighted arithmetic mean of probabilities; no learned bag weights.",
        "bag_context_overlap_rows": overlap,
        "mean_between_bag_prediction_variance": float(variance.mean()),
        "max_between_bag_prediction_variance": float(variance.max()),
        "bags": records, "seconds": time.time() - started,
        "checkpoint_sha256": digest, "vendor_inference_source_sha256": ENGINE_SHA256,
        "execution_backend": "CUDA AMP float16", "generator": "CUDA",
        "context_selection_uses_query_labels": False,
        "context_selection_uses_query_covariates": False,
        "query_labels_used_for_early_stopping": False,
        "correction_labels_loaded": False, "outer_labels_loaded": False,
        "feature_selection_changed": False, "fitted_weights_saved": False,
        "KV_caches_saved": False,
        "same_cache_replay_tolerance": 1e-7,
        "replay_scope": "Exact original first-query-batch same-cache replay; not a different-batch-size equivalence claim.",
    }


# Both exported names intentionally use the old packet-worker interface.
fit_predict = train_predict
