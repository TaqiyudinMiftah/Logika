from __future__ import annotations

import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from logika_ui_26.experiment_io import sha256_file, write_json
from logika_ui_26.kumo import SDK_REVISION


def main(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    if "worker_sha256" in request and sha256_file(Path(__file__)) != request["worker_sha256"]:
        raise ValueError("Worker source differs from the experiment snapshot")
    config = request["config"]
    output = Path(request["output"])
    context_ids = request["context_ids"]
    validation_ids = request["validation_ids"]
    test_ids = request["test_ids"]
    if set(context_ids) & (set(request["outer_validation_ids"]) | set(test_ids)):
        raise ValueError("Context contains validation or test rows")
    if (
        len(set(context_ids)) != config["context_rows"]
        or len(context_ids) != config["context_rows"]
    ):
        raise ValueError("Context count differs from explicit configuration")
    if {"target", "application_id", "customer_id", "decision_datetime"} & set(request["features"]):
        raise ValueError("Labels and identifiers must not be model features")
    if sha256_file(Path(request["parquet"])) != request["parquet_sha256"]:
        raise ValueError("Feature cache checksum differs")
    columns = request["features"]
    if config["feature_limit"] is not None:
        selection = json.loads((output / "selection.json").read_text())
        columns = selection["features"]
        if len(columns) != config["feature_limit"] or not set(columns).issubset(request["features"]):
            raise ValueError("Invalid selected feature recipe")
        if selection["fitting_ids"] != request["development_ids"] or selection["inner_validation_ids"] != request["inner_validation_ids"]:
            raise ValueError("Selection partition differs from request")
    frame = pd.read_parquet(request["parquet"], columns=["application_id", "target", *columns]).set_index("application_id")
    labels = frame.loc[context_ids, "target"].to_numpy(dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Context must contain both classes")
    frame = frame.drop(columns="target")
    for column in columns:
        if column in request["categorical_features"]:
            frame[column] = frame[column].astype("string").fillna("__MISSING__")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="raise").replace(
                [np.inf, -np.inf], np.nan
            )
    import torch
    from huggingface_hub import hf_hub_download
    from sdm import TableTensor
    from sdm.models import KumoTabular

    source = importlib.metadata.distribution("structured-data-models").read_text("direct_url.json")
    if not source or json.loads(source).get("vcs_info", {}).get("commit_id") != SDK_REVISION:
        raise RuntimeError("SDK revision differs. Install the locked kumo extra")
    if not torch.cuda.is_available():
        raise RuntimeError("Kumo configuration requires CUDA")
    torch.cuda.set_device(config["gpu_index"])
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured GPU does not support BF16")
    free, total = torch.cuda.mem_get_info()
    budget = config["gpu_budget_gib"] * 1024**3
    if free < budget + 1024**3:
        raise RuntimeError("Shared GPU needs the configured budget plus 1 GiB free headroom")
    torch.cuda.set_per_process_memory_fraction(min(budget / total, 1), config["gpu_index"])
    torch.set_num_threads(config["threads"])
    seed = config["seed"] + request["fold"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    assets = {}
    print("Downloading pinned foundation assets", flush=True)
    if config["backend"] == "kumo":
        repository = "nvidia/Kumo-Tabular"
        files = ("config.json", f"{config['size']}/classifier.pt")
    else:
        repository = "jingang/TabICL"
        files = ("tabicl-classifier-v2-20260212.ckpt",)
    for name in files:
        asset = Path(hf_hub_download(repository, name, revision=config["pretrained_revision"]))
        assets[name] = sha256_file(asset)
    provenance = {
        "backend": "KumoTabular" if config["backend"] == "kumo" else "TabICLv2",
        "size": config["size"] if config["backend"] == "kumo" else None,
        "sdk_revision": SDK_REVISION,
        "checkpoint_revision": config["pretrained_revision"],
        "asset_sha256": assets,
        "context_rows": len(context_ids),
        "estimators": config["estimators"],
        "precision": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "seed": seed,
        "cuda_runtime": torch.version.cuda,
        "gradient_updates": 0,
        "features": columns,
        "context_positive_fraction": float(labels.mean()),
        "outer_training_positive_fraction": request["outer_training_prior"],
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "structured-data-models",
                "huggingface-hub",
                "pyarrow",
                "numpy",
            )
        },
    }
    write_json(output / "model.json", provenance)
    print("Loading foundation model", flush=True)
    if config["backend"] == "kumo":
        if config["pretrained_revision"] != "v1.0.1":
            raise ValueError("The pinned Kumo SDK expects weights v1.0.1")
        model = KumoTabular(task="classification", size=config["size"], pretrained=True, device="cuda")
    else:
        from unittest.mock import patch

        from sdm.models import TabICLv2

        with patch("sdm.models.tabiclv2.model.download_checkpoint", return_value=asset):
            model = TabICLv2(task="classification", pretrained=True, device="cuda")
    if config.get("cache_storage", "cpu") == "cuda":
        from logika_ui_26.sdk_cache import keep_fit_cache_on_cuda

        provenance["cache_adaptation"] = keep_fit_cache_on_cuda(model)
    model.eval()
    stypes = {
        name: "categorical" if name in request["categorical_features"] else "numerical"
        for name in columns
    }

    def tensor(ids: list):
        arrow = pa.Table.from_pandas(frame.loc[ids, columns], preserve_index=False)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64 if config["backend"] == "kumo" else torch.float32)
        try:
            return TableTensor.from_arrow(arrow, stypes=stypes, device="cuda")
        finally:
            torch.set_default_dtype(previous)

    x = tensor(context_ids)
    y = TableTensor.from_arrow(
        pa.table({"target": pa.array(labels, type=pa.int64())}),
        stypes={"target": "categorical"},
        device="cuda",
    )
    started = time.monotonic()
    print(
        f"Fitting frozen context: {len(context_ids)} rows, {len(columns)} features, {config['estimators']} estimators",
        flush=True,
    )
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.fit(x, y, num_estimators=config["estimators"], estimator_batch_size=1)
    torch.cuda.synchronize()
    if config.get("cache_storage", "cpu") == "cuda":
        from logika_ui_26.sdk_cache import cache_digest, cache_measurement

        provenance["cache_adaptation"].update(cache_measurement(model))
        provenance["cache_adaptation"]["before_queries_sha256"] = cache_digest(model)
    provenance["fit_seconds"] = time.monotonic() - started
    print(f"Context ready in {provenance['fit_seconds']:.1f}s", flush=True)
    del x, y
    torch.cuda.empty_cache()
    provenance["query_batches"] = {}
    for phase, ids in (("validation", validation_ids), ("test", test_ids)):
        probabilities = []
        batch_size = config["query_batch_size"]
        offset = 0
        reductions = []
        started = time.monotonic()
        while offset < len(ids):
            selected = ids[offset : offset + batch_size]
            query = None
            try:
                query = tensor(selected)
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model.predict(query)
                if "1" not in prediction.column_names:
                    raise ValueError("Model output lacks the class-1 probability column")
                probability = prediction[["1"]].numerical.squeeze(-1).float().cpu().numpy()
                if config["positive_fraction"] is not None:
                    original_prior = request["outer_training_prior"]
                    sampled_prior = float(labels.mean())
                    positive = probability * original_prior * (1 - sampled_prior)
                    negative = (1 - probability) * (1 - original_prior) * sampled_prior
                    probability = positive / (positive + negative)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                del query
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                reductions.append({"offset": offset, "batch_size": batch_size})
                print(f"Query OOM: retry {phase} at batch size {batch_size}", flush=True)
                continue
            if probability.shape != (len(selected),):
                raise ValueError("Unexpected output probability shape")
            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError("Invalid output probabilities")
            probabilities.extend(probability.tolist())
            offset += len(selected)
            del query, prediction
            if offset % 2048 == 0 or offset == len(ids):
                print(f"{phase}: {offset}/{len(ids)}", flush=True)
        provenance["query_batches"][phase] = {
            "final_batch_size": batch_size,
            "reductions": reductions,
            "seconds": time.monotonic() - started,
        }
        pd.DataFrame({"application_id": ids, "default_probability": probabilities}).to_csv(
            output / f"{phase}.csv", index=False
        )
    if config.get("cache_storage", "cpu") == "cuda":
        after = cache_digest(model)
        if after != provenance["cache_adaptation"]["before_queries_sha256"]:
            raise ValueError("Query inference mutated the resident estimator cache")
        provenance["cache_adaptation"]["after_queries_sha256"] = after
    provenance.update(
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    write_json(output / "model.json", provenance)
    model.clear()
    print("Fold complete, exiting worker to release CUDA and pinned CPU caches", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main(Path(sys.argv[1]))
