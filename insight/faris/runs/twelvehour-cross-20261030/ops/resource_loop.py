"""Hourly read-only Colab checks and deadline cleanup of explicitly owned names."""
from pathlib import Path
import concurrent.futures
import json
import os
import subprocess
import time

R = Path('/home/faris/logika-twelvehour-cross-20261030')
C = json.loads((R / 'CLOCK.json').read_text())
os.environ['PATH'] = '/home/faris/.local/bin:' + os.environ['PATH']
CLI = '/home/faris/.local/bin/colab'


def bounded(args, maximum):
    remaining=C['deadline_epoch']-time.time()-10
    assert remaining>0, 'Original resource-control deadline reached'
    return subprocess.run(args,capture_output=True,text=True,timeout=min(maximum,remaining))

def one(account, closing=False):
    folder = R / f'private/colab/account{account}'
    folder.mkdir(parents=True, exist_ok=True)
    prefix = [CLI, '--account', str(account), '--config', str(folder / 'sessions.json')]
    name = f'logika-cross-a{account}-20261030'
    record = dict(account=account, epoch=time.time(), owned_session=name)
    allocation = folder / 'ALLOCATION_RESPONSE.json'
    owned = allocation.exists() and json.loads(allocation.read_text()).get('returncode') == 0
    if closing and owned and not (folder / 'RELEASED.json').exists():
        response = bounded([*prefix,'stop','--session',name],120)
        record['owned_deadline_stop'] = dict(returncode=response.returncode,
                                           stdout=response.stdout, stderr=response.stderr)
    for operation in ['usage', 'sessions']:
        response=bounded([*prefix,operation],90)
        record[operation] = dict(returncode=response.returncode, stdout=response.stdout,
                                 stderr=response.stderr)
    record['owned_absent_confirmed'] = record['sessions']['returncode'] == 0 and name not in record['sessions']['stdout']
    (folder / 'RESOURCE_LATEST.json').write_text(json.dumps(record, indent=2) + '\n')
    return dict(account=account, owned=owned, owned_absent_confirmed=record['owned_absent_confirmed'])


while time.time() < C['deadline_epoch']:
    try:
        code="from pathlib import Path;import json;r=Path('/mnt/home/faris/logika-twelvehour-cross-20261030');print(json.dumps({n:json.loads((r/n).read_text()) for n in ['LIVE_STATE.json','COMPLETION.json'] if (r/n).exists()}))"
        result=bounded(['ssh','-i','/home/faris/.ssh/id_ed25519_logika_lab','-o','BatchMode=yes','-o','ConnectTimeout=8','faris@100.68.29.126','python3 -c '+__import__('shlex').quote(code)],30)
        if result.returncode==0:
            status=json.loads(result.stdout);(R/'CANONICAL_HEALTH.json').write_text(json.dumps(status,indent=2))
            if 'COMPLETION.json' in status:(R/'COMPLETION.json').write_text(json.dumps(status['COMPLETION.json'],indent=2))
    except Exception:pass
    closing = time.time() >= C['deadline_epoch'] - 180
    values = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=11 if closing else 4) as executor:
        futures = {executor.submit(one, account, closing): account for account in range(1, 12)}
        for future in concurrent.futures.as_completed(futures):
            try:
                values.append(future.result())
            except Exception as exc:
                values.append(dict(account=futures[future], error=type(exc).__name__))
    result = dict(epoch=time.time(), accounts=values, actual_server_checks=True,
                  all_owned_absent=all(x.get('owned_absent_confirmed') for x in values),
                  no_allocation_or_quota_retry=True)
    (R / 'RESOURCE_LATEST.json').write_text(json.dumps(result, indent=2) + '\n')
    if closing or (R / 'COMPLETION.json').exists() and result['all_owned_absent']:
        (R / 'FINAL_RESOURCE_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
        break
    time.sleep(min(3600, max(1, C['deadline_epoch'] - 180 - time.time())))
