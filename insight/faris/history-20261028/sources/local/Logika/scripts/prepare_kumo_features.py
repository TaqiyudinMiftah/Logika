"""Build the exact v3 feature view for a replacement Kumo learner.

Runs the existing, unchanged aggregators in an isolated workspace. This command
does not train a model, alter a historical submission, or contact Kaggle.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDERS = ["analyze", "extend_features", "build_features_v3"]
RAW_NAMES = [
    "train_applications.csv", "test_applications.csv", "customer_profiles.csv",
    "previous_applications.csv", "repayment_history.csv", "account_behavior.csv",
    "bureau_observations.csv", "ecommerce_transactions.csv",
]
FORBIDDEN = {
    "application_id", "customer_id", "target", "split",
    "application_datetime", "decision_datetime",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_builder(name: str, workspace: Path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = workspace
    module.RAW = workspace / "data/raw"
    module.PROCESSED = workspace / "data/processed"
    module.OUT = workspace / "artifacts"
    module.TABLES = workspace / "artifacts/tables"
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT.parent / "dataset")
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/kumo_v3")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    sample = args.sample
    if sample is None:
        candidates = sorted(raw.glob("sample_submission*.csv"))
        if len(candidates) != 1:
            raise ValueError("Pass --sample when the raw folder has zero or multiple samples")
        sample = candidates[0]
    sample = sample.resolve()
    sources = {name: raw / name for name in RAW_NAMES}
    sources["sample_submission.csv"] = sample
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    recipe_path = ROOT / "artifacts/optimization/tenfold/locked_recipe.json"
    recipe = json.loads(recipe_path.read_text())
    feature_views = [d["columns"] for d in recipe["definitions"].values()]
    if not feature_views or any(view != feature_views[0] for view in feature_views):
        raise ValueError("The v3 components no longer share a single feature view")
    columns = feature_views[0]
    if len(columns) != len(set(columns)) or set(columns) & FORBIDDEN:
        raise ValueError("Duplicate columns or identifiers/labels in model inputs")
    identity = {
        "raw_sha256": {name: sha256(path) for name, path in sources.items()},
        "builder_sha256": {
            name: sha256(ROOT / "scripts" / f"{name}.py") for name in BUILDERS
        },
        "recipe_sha256": sha256(recipe_path),
        "features": columns,
    }
    output = args.output.resolve()
    parquet = output / "features.parquet"
    manifest_path = output / "manifest.json"
    if parquet.exists() and not args.rebuild:
        if not manifest_path.is_file():
            raise ValueError("Existing features lack a manifest; inspect them or pass --rebuild")
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity or previous["feature_sha256"] != sha256(parquet):
            raise ValueError("Inputs, source, recipe, or cache changed; pass --rebuild explicitly")
        print(f"Verified cached features: {parquet}", flush=True)
        return
    workspace = output / "workspace"
    raw_links = workspace / "data/raw"
    raw_links.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        link = raw_links / name
        if link.is_symlink():
            if link.resolve() == source:
                continue
            if not args.rebuild:
                raise ValueError(f"Raw input link differs: {link}")
            link.unlink()
        elif link.exists():
            raise ValueError(f"Refusing to replace a non-symlink input: {link}")
        link.symlink_to(source)
    (workspace / "artifacts/optimization").mkdir(parents=True, exist_ok=True)
    for name in BUILDERS:
        print(f"Running unchanged {name}.py", flush=True)
        module = load_builder(name, workspace)
        if name == "analyze":
            built = module.build_features()
            del built
        else:
            module.main()
        del module
        gc.collect()

    import numpy as np
    import pandas as pd
    import pyarrow

    metadata = ["application_id", "customer_id", "split", "target", "decision_datetime"]
    frame = pd.read_parquet(
        workspace / "data/processed/application_features_v3.parquet",
        columns=metadata + columns,
    )
    if not frame.application_id.is_unique or not frame.customer_id.is_unique:
        raise ValueError("Selected application/customer IDs are not unique")
    for split in ["train", "test"]:
        original = pd.read_csv(sources[f"{split}_applications.csv"])
        selected = frame.loc[frame.split.eq(split)].reset_index(drop=True)
        if not selected.application_id.equals(original.application_id):
            raise ValueError(f"{split} IDs or row order changed during aggregation")
        if split == "train" and not np.array_equal(selected.target, original.target):
            raise ValueError("Training labels changed during aggregation")
        if split == "test" and not selected.target.isna().all():
            raise ValueError("Test rows unexpectedly contain labels")
    if set(frame.split) != {"train", "test"}:
        raise ValueError("Unexpected split membership")
    categorical = frame[columns].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    frame.to_parquet(parquet, index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "feature_sha256": sha256(parquet),
        "rows": len(frame), "training_rows": int(frame.split.eq("train").sum()),
        "test_rows": int(frame.split.eq("test").sum()),
        "feature_count": len(columns), "categorical_features": categorical,
        "metadata_columns_excluded_from_model": metadata,
        "sample_path": str(sample), "raw_paths": {k: str(v) for k, v in sources.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "validation_caveat": "Historical v3 feature selection reused training labels. "
                             "Historical repayment priors use all training histories, without targets.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["rows", "training_rows", "test_rows", "feature_count", "feature_sha256"]}), flush=True)
    print(f"Saved Kumo inputs: {parquet}", flush=True)


if __name__ == "__main__":
    main()
