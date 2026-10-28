"""Dependency waiter for this one bounded run; no fitting or scheduling."""
from pathlib import Path
import json,time,subprocess
R=Path(__file__).resolve().parents[1]
plan=json.loads((R/'H010.json').read_text())
while not all((R/f'role_bag/outer{o}/member{j}/complete.json').exists()for o in range(5)for j in range(5)):
    assert time.time()<plan['CV_deadline_epoch'],'H010 CV deadline reached before all25members completed'
    time.sleep(20)
subprocess.run(['/mnt/home/logika-faris-exploration/ranking-pipeline-20261020/.venv/bin/python',str(R/'source/role_bag_report.py')],check=True)
