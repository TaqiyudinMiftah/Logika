"""Run the official Kaggle CLI with a private credential; never log the token."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CREDENTIAL=Path('/home/parasite/.config/logika-2026/kaggle-api-token-20261025')
CLI=Path('/home/parasite/.cache/logika-2026/kaggle-cli-env/bin/kaggle')

parser=argparse.ArgumentParser()
parser.add_argument('--record',required=True)
parser.add_argument('args',nargs=argparse.REMAINDER)
options=parser.parse_args()
arguments=options.args
if arguments and arguments[0]=='--':arguments=arguments[1:]
assert arguments and CREDENTIAL.is_file() and CLI.is_file()
destination=ROOT/'kaggle_receipts'/options.record
destination.parent.mkdir(parents=True,exist_ok=True)
assert not destination.exists(), 'Preserve earlier receipts; use a distinct record name'
assert CREDENTIAL.stat().st_mode & 0o077 == 0
token=CREDENTIAL.read_text().strip()
assert token.startswith('KGAT_') and not any(c.isspace()for c in token)
env=os.environ.copy()
env['KAGGLE_API_TOKEN']=token
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
result=subprocess.run([str(CLI),*arguments],env=env,text=True,capture_output=True)
stdout=result.stdout.replace(token,'[REDACTED]')
stderr=result.stderr.replace(token,'[REDACTED]')
destination.write_text(json.dumps({'started_UTC':started,'finished_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'argv':[str(CLI),*arguments],'exit_code':result.returncode,
                                 'stdout':stdout,'stderr':stderr,'credential_included':False},indent=2)+'\n')
sys.stdout.write(stdout)
sys.stderr.write(stderr)
raise SystemExit(result.returncode)
