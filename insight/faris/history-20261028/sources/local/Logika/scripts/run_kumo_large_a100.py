"""Build v3 inputs, run a same-recipe GPU pilot, then run Kumo Large CV."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from train_kumo_large import cgroup_headroom, wait_capacity, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / "artifacts/kumo_large_v3"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "launcher.json", {
        "pid": os.getpid(), "proc_start_ticks": Path("/proc/self/stat").read_text().split()[21],
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
        "raw_dir": str(args.raw_dir.resolve()), "hf_home": str(args.hf_home.resolve()),
    })
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    if not (args.raw_dir / "train_applications.csv").is_file() or not args.sample.is_file():
        raise FileNotFoundError("The existing A100 competition raw data or sample is missing")
    feature_dir = ROOT / "data/kumo_v3"
    steps = [
        ("features", ["prepare_kumo_features.py", "--raw-dir", str(args.raw_dir.resolve()),
                      "--sample", str(args.sample.resolve())]),
        ("pilot", ["train_kumo_large.py", "--pilot", "--features", str(feature_dir / "features.parquet"),
                   "--sample", str(args.sample.resolve()), "--output", str(output / "pilot")]),
        ("full", ["train_kumo_large.py", "--features", str(feature_dir / "features.parquet"),
                  "--sample", str(args.sample.resolve()), "--output", str(output / "full")]),
    ]
    import psutil
    for stage, arguments in steps:
        wait_capacity(output, {"minimum_headroom_gib": 4.5, "gpu_budget_gib": 24})
        print(f"Starting {stage}", flush=True)
        with (output / f"{stage}.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / arguments[0]), *arguments[1:]], stdout=log, stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid)
            peak = 0
            try:
                while child.poll() is None:
                    rss = 0
                    try:
                        owned_processes = [process] + process.children(recursive=True)
                    except psutil.NoSuchProcess:
                        owned_processes = []
                    for owned in owned_processes:
                        try:
                            rss += owned.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, rss)
                    write_json(output / "status.json", {"state": "running", "stage": stage,
                               "pid": child.pid, "rss_bytes": rss, "peak_rss_bytes": peak})
                    if cgroup_headroom() < 0.25 * 1024**3 or rss > 4.5 * 1024**3:
                        raise RuntimeError(f"Resource budget exceeded during {stage}")
                    time.sleep(3)
                if child.returncode:
                    raise RuntimeError(f"{stage} exited {child.returncode}; inspect {output / (stage + '.log')}")
            except BaseException:
                if process.is_running():
                    for owned in process.children(recursive=True):
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
                write_json(output / f"{stage}_resources.json", {"peak_rss_bytes": peak, "returncode": child.poll()})
    held = ROOT / "submissions_held"
    held.mkdir(exist_ok=True)
    source = output / "full/submission_kumo_large_logika_v3.csv"
    destination = held / source.name
    shutil.copyfile(source, destination)
    write_json(output / "status.json", {"state": "completed", "submission": str(destination)})
    print(f"Completed: {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        output = ROOT / "artifacts/kumo_large_v3"
        if output.is_dir():
            write_json(output / "status.json", {"state": "failed", "error": str(error)})
        raise
