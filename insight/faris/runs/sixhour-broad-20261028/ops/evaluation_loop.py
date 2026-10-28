"""Periodic development scoring; leave15minutes before terminal selection."""
from pathlib import Path
import subprocess,json,time
R=Path(__file__).resolve().parents[1];C=json.loads((R/'CLOCK.json').read_text())
while time.time()<C['model_deadline_epoch']-900:
    with(R/'logs/evaluation_blocking.log').open('ab')as log:
        subprocess.run([str(R/'env311/bin/python'),str(R/'ops/evaluate_blocking.py')],stdout=log,stderr=subprocess.STDOUT,timeout=900)
    time.sleep(180)
