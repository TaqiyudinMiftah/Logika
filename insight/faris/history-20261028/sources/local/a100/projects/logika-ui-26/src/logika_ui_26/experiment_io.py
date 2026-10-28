from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

from logika_ui_26.features import CATEGORICAL_FEATURES, build_feature_frame

SOURCE_FILES = (
    "train_applications.csv", "test_applications.csv", "customer_profiles.csv",
    "previous_applications.csv", "bureau_observations.csv", "account_behavior.csv",
    "ecommerce_transactions.csv", "DATA_DICTIONARY.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_info() -> dict:
    versions = {}
    for package in (
        "duckdb", "lightgbm", "catboost", "pandas", "numpy", "scikit-learn", "pyarrow",
        "torch", "structured-data-models", "huggingface-hub", "psutil", "xgboost-cpu",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader"], text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    memory_limits = {}
    for location in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        path = Path(location)
        if path.is_file():
            memory_limits[location] = path.read_text().strip()
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": versions, "gpu": gpu, "memory_limits": memory_limits,
    }


def cached_features(data_dir: Path, cache_dir: Path, memory_limit: str,
                    feature_view: str = "base",
                    columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    print("Checking source files and the feature cache", flush=True)
    source = {
        "files": {name: sha256_file(data_dir / name) for name in SOURCE_FILES},
        "feature_builder": sha256_file(Path(__file__).with_name("features.py")),
        "duckdb": importlib.metadata.version("duckdb"),
        "pandas": importlib.metadata.version("pandas"),
    }
    if feature_view not in {"base", "credit_v2", "smooth_v3", "credit_v4"}:
        raise ValueError("Unknown feature view")
    if feature_view != "base":
        source["feature_view"] = feature_view
        source["files"]["repayment_history.csv"] = sha256_file(data_dir / "repayment_history.csv")
        source["credit_builder"] = sha256_file(Path(__file__).with_name("credit_features.py"))
        if feature_view == "credit_v4":
            source["extension_builder"] = sha256_file(Path(__file__).with_name("credit_extensions.py"))
    fingerprint = digest_json(source)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{fingerprint}.parquet"
    metadata_path = parquet_path.with_suffix(".json")
    if parquet_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source"] != source or sha256_file(parquet_path) != metadata["parquet_sha256"]:
            raise ValueError("Feature cache provenance or checksum differs")
        print(f"Reusing feature cache {fingerprint[:12]}", flush=True)
        frame = pd.read_parquet(parquet_path, columns=columns)
    else:
        print(f"Building feature cache {fingerprint[:12]}", flush=True)
        temporary = parquet_path.with_suffix(".building.parquet")
        credit_metadata = {}
        if feature_view == "base":
            frame = build_feature_frame(data_dir, temporary, memory_limit=memory_limit, threads=1)
        elif feature_view == "credit_v4":
            from logika_ui_26.credit_extensions import build_extensions

            original, parent_metadata = cached_features(data_dir, cache_dir, memory_limit, "credit_v2")
            frame, additions = build_extensions(data_dir, original, temporary, memory_limit)
            credit_metadata = {**parent_metadata, **additions,
                               "families": {**parent_metadata["families"], **additions["families"]}}
            for key in ("path", "fingerprint", "source", "parquet_sha256", "rows", "columns"):
                credit_metadata.pop(key, None)
        else:
            from logika_ui_26.credit_features import build_credit_frame

            original, _ = cached_features(data_dir, cache_dir, memory_limit)
            frame, credit_metadata = build_credit_frame(
                data_dir, original, temporary, memory_limit, smooth=feature_view == "smooth_v3",
            )
        temporary.replace(parquet_path)
        metadata = {
            "source": source, "fingerprint": fingerprint, "rows": len(frame),
            "columns": frame.columns.tolist(), "parquet_sha256": sha256_file(parquet_path),
            **credit_metadata,
        }
        write_json(metadata_path, metadata)
    if columns is not None:
        frame = frame[columns].copy()
    for column in CATEGORICAL_FEATURES:
        if column not in frame:
            continue
        frame[column] = frame[column].astype("category")
    for column in frame.columns:
        if column not in {"application_id", "target", "decision_datetime", *CATEGORICAL_FEATURES}:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float32")
    if "decision_datetime" in frame:
        frame["decision_datetime"] = pd.to_datetime(frame["decision_datetime"])
    return frame, {**metadata, "path": str(parquet_path.resolve())}
