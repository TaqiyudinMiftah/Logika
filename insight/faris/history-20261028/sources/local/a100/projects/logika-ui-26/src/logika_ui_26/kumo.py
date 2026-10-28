from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split

from logika_ui_26.experiment_io import (
    cached_features,
    digest_json,
    runtime_info,
    sha256_file,
    write_json,
)
from logika_ui_26.experiments import build_folds, classification_metrics
from logika_ui_26.features import CATEGORICAL_FEATURES

SDK_REVISION = "329a707bbcf9ca147aa45a63fef8b2029f4ee37b"
CHECKPOINT_REVISION = "v1.0.1"
DEFAULTS = {
    "name": "kumo",
    "size": "large",
    "context_rows": 16384,
    "estimators": 2,
    "query_batch_size": 256,
    "seed": 2026,
    "n_splits": 5,
    "threads": 4,
    "gpu_index": 0,
    "gpu_budget_gib": 8,
    "process_budget_gib": 4,
    "memory_limit": "768MB",
    "exclude_features": [],
    "exclude_prefixes": [],
    "feature_view": "base",
    "feature_limit": None,
    "backend": "kumo",
    "pretrained_revision": "v1.0.1",
    "positive_fraction": None,
    "cache_storage": "cpu",
    "nested_context_rows": None,
}


def load_config(path: Path) -> dict:
    supplied = json.loads(path.read_text())
    if set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown Kumo configuration fields")
    config = {**DEFAULTS, **supplied}
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["name"]):
        raise ValueError("Invalid run name")
    if config["size"] not in {"small", "medium", "large"}:
        raise ValueError("Kumo size must be small, medium or large")
    if config["cache_storage"] not in {"cpu", "cuda"}:
        raise ValueError("Unknown estimator cache storage")
    if config["cache_storage"] == "cuda" and config["backend"] != "kumo":
        raise ValueError("CUDA cache adaptation is validated only for Kumo")
    if config["backend"] not in {"kumo", "tabiclv2"}:
        raise ValueError("Unknown foundation backend")
    for key in ("context_rows", "estimators", "query_batch_size", "threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 2 <= config["n_splits"] <= 10 or config["gpu_index"] < 0:
        raise ValueError("Invalid fold count or GPU index")
    if min(config["gpu_budget_gib"], config["process_budget_gib"]) <= 0:
        raise ValueError("Memory budgets must be positive")
    if config["feature_limit"] is not None and config["feature_limit"] < 1:
        raise ValueError("Feature limit must be positive")
    base = config["nested_context_rows"]
    if base is not None and (not isinstance(base, int) or not 1 <= base < config["context_rows"] or config["positive_fraction"] is not None):
        raise ValueError("Nested natural-prevalence context requires a smaller positive base size")
    if config["positive_fraction"] is not None and not 0 < config["positive_fraction"] < 1:
        raise ValueError("Context positive fraction must be strictly between zero and one")
    return config


def context_indices(training: pd.DataFrame, outer: np.ndarray, rows: int, seed: int,
                    positive_fraction: float | None = None) -> np.ndarray:
    if rows > len(outer):
        raise ValueError("Requested context exceeds the outer training fold")
    if positive_fraction is not None:
        counts = {1: round(rows * positive_fraction)}
        counts[0] = rows - counts[1]
        selected = []
        months = pd.to_datetime(training.decision_datetime).dt.strftime("%Y-%m")
        for label in (0, 1):
            eligible = outer[training.iloc[outer].target.to_numpy() == label]
            if counts[label] > len(eligible):
                raise ValueError("The requested class quota exceeds available labeled rows")
            if counts[label] == len(eligible):
                part = eligible
            else:
                part, _ = train_test_split(eligible, train_size=counts[label],
                                          stratify=months.iloc[eligible], random_state=seed + label)
            selected.extend(part.tolist())
        return np.sort(np.asarray(selected, dtype=int))
    if rows == len(outer):
        return np.sort(outer)
    strata = (
        training.target.astype(int).astype(str)
        + "_"
        + pd.to_datetime(training.decision_datetime).dt.strftime("%Y-%m")
    )
    selected, _ = train_test_split(
        outer,
        train_size=rows,
        stratify=strata.iloc[outer],
        random_state=seed,
    )
    return np.sort(selected)


def extended_context_indices(training, outer, rows, base_rows, seed):
    original = context_indices(training, outer, base_rows, seed)
    remaining = np.setdiff1d(outer, original)
    extra = context_indices(training, remaining, rows - base_rows, seed + 10000)
    return np.sort(np.concatenate((original, extra)))


def supervise(request: Path, root: Path, budget_gib: float,
              module: str = "logika_ui_26.kumo_worker") -> dict:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["HF_HOME"] = str(root / "artifacts/hf_cache")
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["HF_HUB_DISABLE_XET"] = "1"
    peaks = {"rss": 0, "pss": 0, "uss": 0}
    started = time.monotonic()
    controller = psutil.Process()
    stage = "selection" if module.endswith("foundation_selection") else "worker"
    log_path = request.parent / f"{stage}.log"
    memory_path = request.parent / f"{stage}_memory.jsonl"
    summary = {"budget_metric": "combined_rss", "budget_gib": budget_gib, "state": "running"}
    events_file = Path("/sys/fs/cgroup/memory.events")
    if events_file.is_file():
        summary["cgroup_events_before"] = dict(line.split() for line in events_file.read_text().splitlines())
    while True:
        maximum = Path("/sys/fs/cgroup/memory.max")
        stats_file = Path("/sys/fs/cgroup/memory.stat")
        if not maximum.is_file() or not stats_file.is_file() or maximum.read_text().strip() == "max":
            break
        values = dict(line.split() for line in stats_file.read_text().splitlines())
        available = int(maximum.read_text()) - int(values["anon"]) - int(values["kernel"])
        required = budget_gib * 1024**3 - controller.memory_info().rss + 0.35 * 1024**3
        if available >= required:
            break
        write_json(request.parent / f"{stage}_capacity.json", {
            "state": "waiting_capacity", "reclaimable_headroom_bytes": available,
            "required_bytes": required, "controller_pid": controller.pid,
            "controller_create_time": controller.create_time(),
        })
        print(f"Waiting for host capacity before {stage}: {available / 1024**3:.2f} GiB headroom", flush=True)
        time.sleep(10)
    with log_path.open("w") as log, memory_path.open("w") as memory_log:
        worker = subprocess.Popen(
            [sys.executable, "-m", module, str(request)],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        summary["worker_pid"] = worker.pid
        try:
            while worker.poll() is None:
                totals = {"rss": 0, "pss": 0, "uss": 0}
                details = []
                for process in [controller, *controller.children(recursive=True)]:
                    try:
                        values = {"pid": process.pid, "rss": process.memory_info().rss}
                        try:
                            full = process.memory_full_info()
                            values.update(pss=getattr(full, "pss", None), uss=getattr(full, "uss", None))
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            values.update(pss=None, uss=None)
                        for metric in totals:
                            totals[metric] += values.get(metric) or 0
                        details.append(values)
                    except psutil.NoSuchProcess:
                        pass
                for metric, value in totals.items():
                    peaks[metric] = max(peaks[metric], value)
                snapshot = {"elapsed_seconds": time.monotonic() - started,
                            "combined_bytes": totals, "processes": details}
                cgroup = Path("/sys/fs/cgroup/memory.current")
                if cgroup.is_file():
                    snapshot["cgroup_current_bytes"] = int(cgroup.read_text())
                stats_file = Path("/sys/fs/cgroup/memory.stat")
                maximum = Path("/sys/fs/cgroup/memory.max")
                if stats_file.is_file() and maximum.is_file() and maximum.read_text().strip() != "max":
                    values = dict(line.split() for line in stats_file.read_text().splitlines())
                    headroom = int(maximum.read_text()) - int(values["anon"]) - int(values["kernel"])
                    snapshot["reclaimable_host_headroom_bytes"] = headroom
                memory_log.write(json.dumps(snapshot) + "\n")
                memory_log.flush()
                if snapshot.get("reclaimable_host_headroom_bytes", 1024**3) < 0.25 * 1024**3:
                    raise MemoryError(f"Account host RAM reserve dropped below 0.25 GiB. See {memory_path}")
                if totals["rss"] > budget_gib * 1024**3:
                    raise MemoryError(f"Own process RSS exceeded {budget_gib} GiB. See {memory_path}")
                time.sleep(1)
            if worker.returncode:
                raise RuntimeError(f"Kumo worker exited {worker.returncode}. See {log_path}")
            summary["state"] = "completed"
        except Exception as error:
            summary.update(state="failed", error=str(error))
            raise
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            summary.update({f"combined_peak_{metric}_bytes": value for metric, value in peaks.items()})
            summary.update(elapsed_seconds=time.monotonic() - started, worker_returncode=worker.returncode)
            if events_file.is_file():
                summary["cgroup_events_after"] = dict(line.split() for line in events_file.read_text().splitlines())
            write_json(request.parent / f"{stage}_memory_summary.json", summary)
    return summary


def _validate_prediction(path: Path, ids: list) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    if predictions.columns.tolist() != ["application_id", "default_probability"]:
        raise ValueError("Invalid prediction columns")
    if predictions.application_id.tolist() != ids:
        raise ValueError("Prediction application order differs from request")
    p = predictions.default_probability.to_numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid class-1 probabilities")
    return predictions


def _run_kumo(
    config_path: Path, root: Path, pilot: bool = False, resume: Path | None = None
) -> dict:
    root = root.resolve()
    config = load_config(config_path)
    frame, cache = cached_features(
        root / "data/raw", root / "data/processed/feature_cache", config["memory_limit"],
        config["feature_view"], columns=["application_id", "target"],
    )
    training = (
        pd.read_csv(root / "data/raw/train_applications.csv")
        .sort_values("application_id")
        .reset_index(drop=True)
    )
    test = (
        pd.read_csv(root / "data/raw/test_applications.csv")
        .sort_values("application_id")
        .reset_index(drop=True)
    )
    if not test.application_id.is_unique or set(training.customer_id) & set(test.customer_id):
        raise ValueError("Duplicate test applications or overlapping train/test borrowers")
    by_id = frame.set_index("application_id")
    if not by_id.index.is_unique or set(by_id.index) != set(training.application_id) | set(
        test.application_id
    ):
        raise ValueError("Invalid feature application IDs")
    if not np.array_equal(
        by_id.loc[training.application_id, "target"].to_numpy(), training.target.to_numpy()
    ):
        raise ValueError("Cached labels differ from raw labels")
    if by_id.loc[test.application_id, "target"].notna().any():
        raise ValueError("Test features contain labels")
    features = [
        c
        for c in cache["columns"]
        if c
        not in {
            "application_id",
            "customer_id",
            "decision_datetime",
            "target",
            *config["exclude_features"],
        }
        and not c.startswith(tuple(config["exclude_prefixes"]))
    ]
    if not features:
        raise ValueError("No features selected")
    folds = build_folds(training, config["seed"], config["n_splits"], 0.2)
    plan = []
    for fold in folds[:1] if pilot else folds:
        number = fold["fold"]
        if config["nested_context_rows"] is None:
            context = context_indices(
                training, fold["outer_training"], config["context_rows"], config["seed"] + number,
                config["positive_fraction"],
            )
        else:
            context = extended_context_indices(training, fold["outer_training"],
                config["context_rows"], config["nested_context_rows"], config["seed"] + number)
        query = fold["outer_validation"][:512] if pilot else fold["outer_validation"]
        plan.append(
            {
                "fold": number,
                "context_ids": training.iloc[context].application_id.tolist(),
                "validation_ids": training.iloc[query].application_id.tolist(),
                "outer_validation_ids": training.iloc[
                    fold["outer_validation"]
                ].application_id.tolist(),
                "outer_training_ids": training.iloc[fold["outer_training"]].application_id.tolist(),
                "development_ids": training.iloc[fold["development"]].application_id.tolist(),
                "inner_validation_ids": training.iloc[fold["inner_validation"]].application_id.tolist(),
                "outer_training_prior": float(training.iloc[fold["outer_training"]].target.mean()),
                "test_ids": test.application_id.tolist()[:128]
                if pilot
                else test.application_id.tolist(),
            }
        )
    source_names = [
        "src/logika_ui_26/kumo.py",
        "src/logika_ui_26/kumo_worker.py",
        "src/logika_ui_26/experiments.py",
        "src/logika_ui_26/experiment_io.py",
        "src/logika_ui_26/features.py",
        "src/logika_ui_26/credit_features.py",
        "src/logika_ui_26/foundation_selection.py",
        "src/logika_ui_26/sdk_cache.py",
        "src/logika_ui_26/selection_reuse.py",
        "src/logika_ui_26/gpu_runtime.py",
        "scripts/run_kumo.py",
        "pyproject.toml",
        "uv.lock",
    ]
    identity = {
        "config": config,
        "pilot": pilot,
        "plan_hash": digest_json(plan),
        "feature_hash": cache["parquet_sha256"],
        "feature_fingerprint": cache["fingerprint"],
        "features": features,
        "sdk_revision": SDK_REVISION,
        "checkpoint_revision": config["pretrained_revision"],
        "source_hashes": {name: sha256_file(root / name) for name in source_names},
    }
    if resume:
        output = resume.resolve()
        if json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Source, features, configuration or fold plan changed since this run")
    else:
        suffix = "-pilot" if pilot else ""
        run_id = (
            config["name"]
            + suffix
            + "-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:6]
        )
        output = root / "artifacts/experiments" / run_id
        output.mkdir(parents=True, exist_ok=False)
        for name, data in (
            ("identity", identity),
            ("config", config),
            ("plan", plan),
            ("runtime", runtime_info()),
            ("feature_cache", cache),
        ):
            write_json(output / f"{name}.json", data)
        for name in source_names:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
    print(f"Kumo run: {output}", flush=True)
    del frame, by_id
    gc.collect()
    import pyarrow as pa

    pa.default_memory_pool().release_unused()
    try:
        import ctypes

        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass
    oof = []
    test_predictions = []
    metrics = []
    assets = None
    try:
        for fold in plan:
            folder = output / f"fold_{fold['fold']}"
            folder.mkdir(exist_ok=True)
            write_json(
                output / "status.json", {"state": "running", "fold": fold["fold"], "pilot": pilot}
            )
            request = {
                **fold,
                "config": config,
                "features": features,
                "categorical_features": [c for c in features if c in CATEGORICAL_FEATURES],
                "parquet": cache["path"],
                "parquet_sha256": cache["parquet_sha256"],
                "output": str(folder),
                "worker_sha256": identity["source_hashes"]["src/logika_ui_26/kumo_worker.py"],
            }
            complete = folder / "complete.json"
            if complete.exists():
                saved = json.loads(complete.read_text())
                if saved["request_hash"] != digest_json(request):
                    raise ValueError("Completed fold request changed")
                for filename, digest in saved["hashes"].items():
                    if sha256_file(folder / filename) != digest:
                        raise ValueError("Completed fold checksum differs")
            else:
                write_json(folder / "request.json", request)
                print(
                    f"Fold {fold['fold']}: {len(fold['context_ids'])} context, {len(fold['validation_ids'])} validation, {len(fold['test_ids'])} test",
                    flush=True,
                )
                if config["feature_limit"] is not None:
                    from logika_ui_26.selection_reuse import reuse_verified_selection

                    selection_resources = reuse_verified_selection(request, folder, root)
                    if selection_resources is None:
                        selection_resources = supervise(folder / "request.json", root, config["process_budget_gib"], "logika_ui_26.foundation_selection")
                    else:
                        print(f"Reusing verified inner selection from {selection_resources['origin']}", flush=True)
                    write_json(folder / "selection_resources.json", selection_resources)
                resources = supervise(folder / "request.json", root, config["process_budget_gib"])
                write_json(folder / "resources.json", resources)
                artifact_names = ["validation.csv", "test.csv", "model.json", "resources.json"]
                if config["feature_limit"] is not None:
                    artifact_names.extend(["selection.json", "selection_model.txt", "selection_resources.json"])
                saved = {
                    "request_hash": digest_json(request),
                    "hashes": {
                        filename: sha256_file(folder / filename)
                        for filename in artifact_names
                    },
                }
                write_json(complete, saved)
            provenance = json.loads((folder / "model.json").read_text())
            if assets is not None and assets != provenance["asset_sha256"]:
                raise ValueError("Pretrained assets changed across folds")
            assets = provenance["asset_sha256"]
            validation = _validate_prediction(folder / "validation.csv", fold["validation_ids"])
            predicted_test = _validate_prediction(folder / "test.csv", fold["test_ids"])
            y = (
                training.set_index("application_id")
                .loc[fold["validation_ids"], "target"]
                .to_numpy()
            )
            if not pilot:
                score = classification_metrics(y, validation.default_probability.to_numpy())
                metrics.append({"fold": fold["fold"], **score})
                print(f"Fold {fold['fold']} KS={score['ks']:.6f}", flush=True)
            oof.append(validation.assign(target=y, fold=fold["fold"]))
            test_predictions.append(predicted_test.default_probability.to_numpy())
        combined = pd.concat(oof).sort_values("application_id")
        combined.to_csv(output / "oof.csv", index=False)
        result = {
            "state": "complete",
            "pilot": pilot,
            "run_directory": str(output),
            "folds": metrics,
        }
        if not pilot:
            if combined.application_id.tolist() != training.application_id.tolist():
                raise ValueError("OOF predictions do not cover the full training set")
            submission = pd.DataFrame(
                {
                    "application_id": plan[0]["test_ids"],
                    "default_probability": np.mean(test_predictions, axis=0),
                }
            )
            sample = pd.read_csv(root / "data/raw/sample_submission.csv")
            submission = (
                submission.set_index("application_id").loc[sample.application_id].reset_index()
            )
            submission.to_csv(output / "submission.csv", index=False)
            result.update(
                pooled=classification_metrics(
                    combined.target.to_numpy(), combined.default_probability.to_numpy()
                ),
                mean_fold_ks=float(np.mean([m["ks"] for m in metrics])),
                std_fold_ks=float(np.std([m["ks"] for m in metrics], ddof=1)),
            )
        write_json(output / "summary.json", result)
        write_json(output / "status.json", result)
        return result
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": str(error), "pilot": pilot})
        raise


def run_kumo(config_path: Path, root: Path, pilot: bool = False, resume: Path | None = None) -> dict:
    from logika_ui_26.gpu_runtime import gpu_slot

    with gpu_slot(root):
        return _run_kumo(config_path, root, pilot, resume)
