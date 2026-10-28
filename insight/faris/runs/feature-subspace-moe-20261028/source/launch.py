"""Persist the new user-authorized clock once, then hand control to the program."""
import argparse
import json
import subprocess
import sys
import time
from common import R,load,dump,sha,utc,source_seal


def main(hours):
    assert hours==12 and load('CONFIG.json')['user_hours']==12
    assert not(R/'CLOCK.json').exists(),'Clock exists; reconcile and resume; never reset it.'
    assert load('SOURCE_SEAL.json')==source_seal(),'Unsealed source change'
    now=time.time();clock=dict(started_UTC=utc(),started_epoch=now,
       admission_end_epoch=now+7.5*3600,deadline_epoch=now+12*3600,
       user_hours=hours,config_sha256=sha(R/'CONFIG.json'),
       contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),
       previous_campaign_clock_reused=False,preparation_before_first_actual_fit=True,
       accounting='All actual-data qualification, selection, pilots, modelling, calibration, evaluation and any refit share this clock.')
    # Exclusive creation makes accidental repeated launch impossible.
    with(R/'CLOCK.json').open('x')as stream:json.dump(clock,stream,indent=2);stream.write('\n')
    command=['systemd-run','--user','--quiet','--unit=logika-feature-subspace-moe-20261028',
       '-p','WorkingDirectory='+str(R),'-p','MemoryMax=8G','-p','MemorySwapMax=0',
       '-p','RuntimeMaxSec=43200','-p','KillMode=control-group',sys.executable,str(R/'source/coordinator.py')]
    subprocess.run(command,check=True)
    dump(dict(stage='launched_qualification',clock=clock,service='logika-feature-subspace-moe-20261028.service',UTC=utc()),R/'LAUNCH_RECEIPT.json')
    print(json.dumps(clock,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hours',type=float,required=True);a=p.parse_args();main(a.hours)
