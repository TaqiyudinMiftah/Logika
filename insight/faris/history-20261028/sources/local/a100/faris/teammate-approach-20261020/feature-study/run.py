"""Detached finite CPU study; only owns its own worker process groups."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import fcntl
import importlib.metadata
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "source"))
import baseline as b


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    lock = (ROOT / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((ROOT / "plan.json").read_text())
    seal = json.loads((ROOT / "seal.json").read_text())
    for name, digest in seal["files"].items():
        assert b.sha(ROOT / name) == digest, name
    for name, digest in seal["inputs"].items():
        assert b.sha(name) == digest, name
    packages = {name: importlib.metadata.version(name) for name in plan["versions"]}
    assert packages == plan["versions"]
    b.dump(packages, ROOT / "environment.json")
    assert not (ROOT / "state.json").exists(), "Do not reset the original attempt"
    state = dict(status="training", start=time.time(), completed=[], active={})
    state["deadline"] = state["start"] + 86400
    b.dump(state, ROOT / "state.json")
    pending = list(range(5)); active = {}
    try:
        assert json.loads((ROOT / "preflight.json").read_text())["status"] == "passed"
        while pending or active:
            assert time.time() < state["deadline"] and not (ROOT / "STOP").exists()
            while pending and len(active) < 2:
                fold = pending.pop(0)
                log = (ROOT / f"fold{fold}.log").open("ab")
                child = subprocess.Popen(
                    [sys.executable, str(ROOT.parent / "source/attribution.py"), "fold",
                     "--root", str(ROOT), "--fold", str(fold)],
                    stdout=log, stderr=log, start_new_session=True,
                )
                active[fold] = child, log
                state["active"][str(fold)] = child.pid
                print(json.dumps(dict(started_fold=fold, pid=child.pid)), flush=True)
            for fold, (child, log) in list(active.items()):
                if child.poll() is None:
                    continue
                log.close(); del active[fold]; state["active"].pop(str(fold))
                assert child.returncode == 0, f"Fold {fold} failed; see fold{fold}.log"
                assert (ROOT / f"folds/fold{fold}/complete.json").exists()
                state["completed"].append(fold)
                print(json.dumps(dict(completed_fold=fold)), flush=True)
            b.dump(state, ROOT / "state.json")
            time.sleep(3)
        state["status"] = "reporting"; b.dump(state, ROOT / "state.json")
        child = subprocess.Popen(
            [sys.executable, str(ROOT.parent / "source/attribution.py"), "finish", "--root", str(ROOT)],
            start_new_session=True,
        )
        try:
            while child.poll() is None:
                assert time.time() < state["deadline"] and not (ROOT / "STOP").exists()
                time.sleep(3)
            assert child.returncode == 0, "Assessment failed"
        finally:
            stop_child(child)
        state["status"] = "complete"
        state["result"] = json.loads((ROOT / "result.json").read_text())["status"]
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()
        for child, log in active.values():
            stop_child(child); log.close()
        (ROOT / "FAILED.txt").write_text(state["error"])
        raise
    finally:
        state["finished"] = time.time()
        state["active"] = {}
        b.dump(state, ROOT / "state.json")
        print(json.dumps(dict(DONE=True, status=state["status"])), flush=True)


if __name__ == "__main__":
    main()
