"""A fixed absolute ceiling survives service restarts and missing state files."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
hard = float(os.environ['LOGIKA_HARD_DEADLINE'])
try:
    hard = min(hard, json.loads((Path(args.root) / 'CLOCK.json').read_text())['deadline_epoch'])
except (OSError, ValueError, KeyError):
    if time.time() < hard - 30:
        raise
remaining = hard - time.time() - 30
if remaining > 0:
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    process = subprocess.Popen(command)
    try:
        raise SystemExit(process.wait(timeout=remaining))
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
