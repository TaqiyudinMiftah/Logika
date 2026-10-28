"""Small remote bootstrap; launch only this task's isolated, verified source."""
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

root = Path("/home/fariselhakim/Faris/projects/logika-2026/kumo-large-v3-20261015")
record = root / "bootstrap.json"
try:
    if record.exists():
        previous = json.loads(record.read_text())
        pid = previous.get("pid")
        stat = Path("/proc/{}/stat".format(pid))
        if stat.exists() and stat.read_text().split()[21] == previous.get("proc_start_ticks"):
            raise RuntimeError("The recorded launcher is already live; refusing duplicate launch")
        raise RuntimeError("A previous launch record exists; inspect it before restarting")
    manifest = json.loads((root / "code_manifest.json").read_text())
    archive = root / "code.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Uploaded source archive differs")
    with tarfile.open(str(archive), "r:gz") as tar:
        if {member.name for member in tar.getmembers()} != set(manifest["files"]):
            raise ValueError("Unexpected source archive members")
        if any(not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts for member in tar.getmembers()):
            raise ValueError("Unsafe source archive member")
        tar.extractall(str(root))
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Extracted source differs: " + name)
    environment = Path("/home/fariselhakim/gipta/projects/logika-ui-26")
    python = environment / ".venv/bin/python"
    raw = environment / "data/raw"
    hf_home = environment / "artifacts/hf_cache"
    if not python.is_file() or not raw.is_dir() or not hf_home.is_dir():
        raise FileNotFoundError("The existing A100 environment, raw data, or checkpoint cache is missing")
    command = [str(python), "-u", str(root / "scripts/run_kumo_large_a100.py"),
               "--raw-dir", str(raw), "--sample", str(raw / "sample_submission.csv"),
               "--hf-home", str(hf_home)]
    with (root / "launcher.log").open("ab") as log:
        child = subprocess.Popen(command, cwd=str(root), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path("/proc/{}/stat".format(child.pid)).read_text().split()[21]
    record.write_text(json.dumps({"state": "launched", "pid": child.pid,
                                  "proc_start_ticks": ticks, "command": command,
                                  "archive_sha256": manifest["sha256"],
                                  "launched_epoch": time.time()}, indent=2) + "\n")
except Exception as error:
    (root / "bootstrap_error.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
    raise
