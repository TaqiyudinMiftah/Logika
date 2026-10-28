"""Replace the v3 learners with frozen Kumo Large on the same 388 features.

The controller runs one fold at a time and waits for shared A100 capacity.
Every fold records its context IDs, checkpoint hashes and validation predictions.
Historical submissions are not overwritten. No Kaggle calls are made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SDK_REVISION = "329a707bbcf9ca147aa45a63fef8b2029f4ee37b"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def cgroup_headroom():
    root = Path("/sys/fs/cgroup")
    if not (root / "memory.max").exists():
        raise RuntimeError("Cannot establish shared cgroup headroom on this host")
    limit = (root / "memory.max").read_text().strip()
    if limit == "max":
        import psutil
        return psutil.virtual_memory().available
    current = int((root / "memory.current").read_text())
    stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
    # Only count inactive file cache as reclaimable; do not count anonymous RAM.
    reclaimable = int(stats.get("inactive_file", 0))
    return max(0, int(limit) - current + reclaimable)


def gpu_free_gib():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return float(result.stdout.strip()) / 1024


def wait_capacity(output, config):
    while True:
        ram = cgroup_headroom() / 1024**3
        gpu = gpu_free_gib()
        if ram >= config["minimum_headroom_gib"] and gpu >= config["gpu_budget_gib"] + 1:
            return
        write_json(output / "status.json", {
            "state": "waiting_capacity", "headroom_gib": ram, "free_gpu_gib": gpu,
            "required_ram_gib": config["minimum_headroom_gib"],
            "required_gpu_gib": config["gpu_budget_gib"] + 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        time.sleep(20)


def scores(y, probability):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve, log_loss
    fpr, tpr, _ = roc_curve(y, probability, drop_intermediate=False)
    return {"ks": float(np.max(np.abs(tpr - fpr))),
            "directional_ks": float(np.max(tpr - fpr)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "log_loss": float(log_loss(y, probability))}


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    config = request["config"]
    output = Path(request["output"])
    manifest = json.loads(Path(request["manifest"]).read_text())
    if sha256(request["features"]) != manifest["feature_sha256"]:
        raise ValueError("Feature checksum differs")
    if sha256(__file__) != request["worker_sha256"]:
        raise ValueError("Worker source differs")
    if sha256(Path(__file__).with_name("kumo_cuda_cache.py")) != request["cache_source_sha256"]:
        raise ValueError("CUDA cache helper differs")
    context_ids = request["context_ids"]
    if len(context_ids) != config["context_rows"] or len(set(context_ids)) != len(context_ids):
        raise ValueError("Context size differs from declared recipe")
    if set(context_ids) & (set(request["validation_ids"]) | set(request["test_ids"])):
        raise ValueError("Validation or test rows entered labeled context")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import torch
    from huggingface_hub import hf_hub_download
    from sdm import TableTensor
    from sdm.models import KumoTabular
    from kumo_cuda_cache import keep_fit_cache_on_cuda
    direct = importlib.metadata.distribution("structured-data-models").read_text("direct_url.json")
    if not direct or json.loads(direct).get("vcs_info", {}).get("commit_id") != SDK_REVISION:
        raise RuntimeError("Install the pinned A100 structured-data-models revision")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < (config["gpu_budget_gib"] + 1) * 1024**3:
        raise RuntimeError("Shared GPU capacity changed before model load")
    torch.cuda.set_per_process_memory_fraction(config["gpu_budget_gib"] * 1024**3 / total)
    torch.set_num_threads(4)
    seed = config["fold_seed"] + request["fold"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    assets = {}
    for name in ["config.json", "large/classifier.pt"]:
        path = hf_hub_download("nvidia/Kumo-Tabular", name, revision="v1.0.1", local_files_only=True)
        assets[name] = sha256(path)
    columns = manifest["identity"]["features"]
    if len(columns) != 388 or set(columns) & {"target", "split", "application_id", "customer_id", "decision_datetime", "application_datetime"}:
        raise ValueError("The exact 388-feature v3 view is required")
    frame = pd.read_parquet(request["features"], columns=["application_id", "target"] + columns).set_index("application_id")
    categorical = manifest["categorical_features"]
    stypes = {name: "categorical" if name in categorical else "numerical" for name in columns}
    for name in categorical:
        frame[name] = frame[name].astype("string").fillna("__MISSING__")

    def tensor(ids):
        arrow = pa.Table.from_pandas(frame.loc[ids, columns], preserve_index=False)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return TableTensor.from_arrow(arrow, stypes=stypes, device="cuda")
        finally:
            torch.set_default_dtype(previous)

    labels = frame.loc[context_ids, "target"].to_numpy(dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both classes are required in the labeled context")
    model = KumoTabular(task="classification", size="large", pretrained=True, device="cuda")
    cache = keep_fit_cache_on_cuda(model)
    model.eval()
    x = tensor(context_ids)
    y = TableTensor.from_arrow(pa.table({"target": pa.array(labels, type=pa.int64())}),
                              stypes={"target": "categorical"}, device="cuda")
    print(f"GPU {torch.cuda.get_device_name()}; fold {request['fold']}; context {len(labels)}; features {len(columns)}", flush=True)
    started = time.monotonic()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.fit(x, y, num_estimators=config["estimators"], estimator_batch_size=1,
                  generator=torch.Generator(device="cuda").manual_seed(seed))
    torch.cuda.synchronize()
    del x, y
    print(f"Context fit completed in {time.monotonic()-started:.1f}s", flush=True)
    for phase in ["validation", "test"]:
        ids = request[phase + "_ids"]
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(ids), config["query_batch_size"]):
                query = tensor(ids[start:start + config["query_batch_size"]])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model.predict(query)
                if "1" not in predicted.column_names:
                    raise ValueError("Missing named class-1 output")
                predictions.append(predicted[["1"]].numerical.squeeze(-1).float().cpu().numpy().copy())
                del query, predicted
        probability = np.concatenate(predictions).astype(float)
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError("Invalid model probabilities")
        result = pd.DataFrame({"application_id": ids, "default_probability": probability})
        if phase == "validation":
            result["target"] = frame.loc[ids, "target"].astype(int).to_numpy()
        result.to_csv(output / f"{phase}.csv", index=False, float_format="%.12f")
        print(f"Completed {phase}: {len(ids)} queries", flush=True)
    provenance = {
        "gpu": torch.cuda.get_device_name(), "checkpoint_repository": "nvidia/Kumo-Tabular",
        "checkpoint_revision": "v1.0.1", "asset_sha256": assets,
        "sdk_revision": SDK_REVISION, "gradient_updates": 0, "cache": cache,
        "context_rows": len(context_ids), "feature_count": len(columns),
        "seed": seed, "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "structured-data-models", "pandas", "pyarrow", "numpy"]},
    }
    write_json(output / "model.json", provenance)
    write_json(output / "complete.json", {"hashes": {name: sha256(output / name) for name in ["request.json", "validation.csv", "test.csv", "model.json"]}})


def supervise(folder, output, config):
    import psutil
    with (folder / "worker.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-u", __file__, "--worker", str(folder / "request.json")], stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        peak = 0
        started = time.monotonic()
        try:
            while child.poll() is None:
                try:
                    owned = [process] + process.children(recursive=True)
                except psutil.NoSuchProcess:
                    owned = []
                rss = 0
                for item in owned:
                    try:
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                write_json(output / "status.json", {"state": "running", "fold": folder.name,
                           "worker_pid": child.pid,
                           "rss_bytes": rss, "elapsed_seconds": time.monotonic() - started})
                if rss > config["process_budget_gib"] * 1024**3:
                    raise RuntimeError("Owned worker RSS exceeded the declared budget")
                if cgroup_headroom() < 0.25 * 1024**3:
                    raise RuntimeError("Shared cgroup headroom fell below 0.25 GiB")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"Fold worker failed with exit code {child.returncode}; inspect {folder / 'worker.log'}")
        except BaseException:
            for owned in process.children(recursive=True) if process.is_running() else []:
                try:
                    owned.terminate()
                except psutil.NoSuchProcess:
                    pass
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            write_json(folder / "resources.json", {"peak_worker_rss_bytes": peak,
                       "elapsed_seconds": time.monotonic() - started, "returncode": child.poll()})


def run(args):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, train_test_split
    manifest_path = args.features.with_name("manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    if sha256(args.features) != manifest["feature_sha256"]:
        raise ValueError("Feature artifact checksum differs")
    frame = pd.read_parquet(args.features, columns=["application_id", "target", "split", "decision_datetime"])
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    sample = pd.read_csv(args.sample)
    if not sample.application_id.is_unique or set(sample.application_id) != set(test.application_id):
        raise ValueError("Sample IDs differ from test")
    config = {"model": "Kumo Large", "folds": 10, "fold_seed": 9173,
              "context_rows": 16384, "estimators": 2, "query_batch_size": 256,
              "gpu_budget_gib": 24, "process_budget_gib": 4,
              "minimum_headroom_gib": 4.5, "feature_count": 388,
              "precision": "bfloat16", "gradient_updates": 0,
              "pilot": args.pilot,
              "context_sampling": "target-by-decision-month stratified outer-training subset"}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "feature_sha256": manifest["feature_sha256"],
                "manifest_sha256": sha256(manifest_path), "sample_sha256": sha256(args.sample),
                "source_sha256": sha256(__file__),
                "cache_source_sha256": sha256(Path(__file__).with_name("kumo_cuda_cache.py"))}
    if (output / "identity.json").exists():
        if json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Refusing to resume against changed source, inputs or configuration")
    else:
        write_json(output / "identity.json", identity)
        import shutil
        shutil.copyfile(__file__, output / "train_kumo_large.py")
        shutil.copyfile(Path(__file__).with_name("kumo_cuda_cache.py"), output / "kumo_cuda_cache.py")
    y = train.target.astype(int)
    strata = y.astype(str) + "_" + pd.to_datetime(train.decision_datetime).dt.strftime("%Y-%m")
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=9173).split(train, y))
    oof = np.full(len(train), np.nan)
    test_predictions, fold_metrics, assets = [], [], []
    for number, (ti, vi) in enumerate(folds, 1):
        if args.pilot and number > 1:
            break
        folder = output / f"fold_{number:02d}"
        folder.mkdir(exist_ok=True)
        _, context = train_test_split(ti, test_size=config["context_rows"], stratify=strata.iloc[ti], random_state=9173 + number)
        request = {"fold": number, "config": config, "features": str(args.features.resolve()),
                   "manifest": str(manifest_path), "output": str(folder),
                   "context_ids": train.application_id.iloc[context].tolist(),
                   "validation_ids": train.application_id.iloc[vi[:512] if args.pilot else vi].tolist(),
                   "test_ids": test.application_id.iloc[:128].tolist() if args.pilot else test.application_id.tolist(),
                   "worker_sha256": identity["source_sha256"],
                   "cache_source_sha256": identity["cache_source_sha256"]}
        if (folder / "request.json").exists():
            if json.loads((folder / "request.json").read_text()) != request:
                raise ValueError("Saved fold assignment or request differs")
        else:
            write_json(folder / "request.json", request)
        if not (folder / "complete.json").exists():
            wait_capacity(output, config)
            print(f"Starting fold {number}/10", flush=True)
            supervise(folder, output, config)
        complete = json.loads((folder / "complete.json").read_text())
        for name, expected in complete["hashes"].items():
            if sha256(folder / name) != expected:
                raise ValueError(f"Corrupt fold artifact: {folder / name}")
        validation = pd.read_csv(folder / "validation.csv")
        predicted = pd.read_csv(folder / "test.csv")
        if validation.application_id.tolist() != request["validation_ids"] or predicted.application_id.tolist() != request["test_ids"]:
            raise ValueError("Prediction IDs differ from fold plan")
        if args.pilot:
            summary = {"state": "pilot_completed", "pilot": True,
                       "validation_queries": len(validation), "test_queries": len(predicted),
                       "config": config, "model": json.loads((folder / "model.json").read_text()),
                       "full_cv_score": None, "submission": None}
            write_json(output / "summary.json", summary)
            write_json(output / "status.json", {"state": "pilot_completed"})
            print(json.dumps(summary), flush=True)
            return
        if not np.array_equal(validation.target.to_numpy(), y.iloc[vi].to_numpy()):
            raise ValueError("Validation labels differ")
        assets.append(json.loads((folder / "model.json").read_text())["asset_sha256"])
        if any(value != assets[0] for value in assets):
            raise ValueError("Pretrained checkpoint changed between folds")
        oof[vi] = validation.default_probability
        test_predictions.append(predicted.default_probability.to_numpy())
        fold_metrics.append({"fold": number, **scores(y.iloc[vi], oof[vi])})
        write_json(output / "fold_metrics.json", fold_metrics)
        print(json.dumps(fold_metrics[-1]), flush=True)
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete OOF predictions")
    pd.DataFrame({"application_id": train.application_id, "target": y, "default_probability": oof}).to_csv(output / "oof.csv", index=False, float_format="%.12f")
    averaged = pd.Series(np.mean(test_predictions, axis=0), index=test.application_id)
    submission = sample[["application_id"]].copy()
    submission["default_probability"] = submission.application_id.map(averaged)
    p = submission.default_probability.to_numpy()
    if len(submission) != 30000 or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid submission")
    path = output / "submission_kumo_large_logika_v3.csv"
    submission.to_csv(path, index=False, float_format="%.12f")
    summary = {"state": "completed", "config": config, "pooled_oof": scores(y, oof),
               "mean_fold_ks": float(np.mean([m["ks"] for m in fold_metrics])),
               "sd_fold_ks": float(np.std([m["ks"] for m in fold_metrics], ddof=1)),
               "submission_sha256": sha256(path), "oof_sha256": sha256(output / "oof.csv"),
               "checkpoint_sha256": assets[0], "submission_rows": len(submission),
               "historical_v3_oof_ks": 0.5190088512771253,
               "caveat": "Historical v3 feature selection reused training labels. Kumo context uses 16384 labels per fold; v3's new trees used 63000. Historical v3 also blends a 3-fold component. No new public score is known."}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"state": "completed", "submission": str(path)})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--pilot", action="store_true", help="Same context/model; 512 validation and 128 test queries, no CV score or submission")
    parser.add_argument("--features", type=Path, default=ROOT / "data/kumo_v3/features.parquet")
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.sample is None or args.output is None:
        parser.error("--sample and --output are required for a controller run")
    try:
        run(args)
    except BaseException as error:
        if args.output.is_dir():
            write_json(args.output / "status.json", {"state": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
