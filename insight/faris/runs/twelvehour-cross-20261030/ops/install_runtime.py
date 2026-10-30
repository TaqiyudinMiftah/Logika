"""Launch persistent owned controllers only after all prerequisite gates pass."""
from pathlib import Path
import json
import shlex
import subprocess
import sys
import tarfile

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'source'))
sys.path.insert(0, str(R / 'ops'))
from common import load, sha, check_seal
from coordinator import FILES, EVAL_PY
from service_install import install


def run(args, timeout=120):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)


def main():
    clock = check_seal()
    assert load('GATE.json')['status'] == load('BASELINE_REPLAY_GATE.json')['status'] == 'PASS'
    assert load('PREPUBLICATION.json')['execution_plan_sha256'] == sha(R / 'EXECUTION_PLAN.json')
    plan = load('EXECUTION_PLAN.json')['owners']
    assert load('DISPATCH_SEAL.json')['clock_sha256'] == sha(R / 'CLOCK.json')
    source = R / 'private/SOURCE.tar.gz'
    with tarfile.open(source, 'w:gz', compresslevel=1) as archive:
        for name in [*FILES, 'source', 'ops', 'evidence']:
            path = R / name
            assert path.exists(), name
            archive.add(path, arcname=name,
                        filter=lambda member: None if '__pycache__' in member.name.split('/') else member)
    remote_root = '/home/faris/logika-twelvehour-cross-20261030'
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i',
           '/mnt/home/faris/.ssh/id_ed25519_logika_backup119', 'faris@100.89.211.119']
    run([*ssh, 'mkdir -p ' + remote_root + '/private && chmod 700 ' + remote_root])
    for name in [*FILES, 'source', 'ops', 'evidence']:
        run(['rsync', '-az', '-e', shlex.join(ssh[:-1]), str(R / name), ssh[-1] + ':' + remote_root + '/'], 180)
    run(['rsync', '-az', '-e', shlex.join(ssh[:-1]), str(source), ssh[-1] + ':' + remote_root + '/private/'], 120)
    controllers = []
    for owner in plan:
        if not owner.isdigit():
            continue
        controllers.append((
            'logika-cross-colab-a' + owner + '-20261030',
            ['/usr/bin/python3', '-u', remote_root + '/ops/colab_controller.py', '--account', owner],
            remote_root, '4G', 100))
    controllers.append(('logika-cross-resources-20261030',
                        ['/usr/bin/python3', '-u', remote_root + '/ops/resource_loop.py'],
                        remote_root, '3G', 200))
    code = ("import sys;sys.path.insert(0," + repr(remote_root + '/ops') + ");"
            "from service_install import install;items=" + repr(controllers) + ";"
            "[install(name,command,root,memory=memory,cpu=cpu)for name,command,root,memory,cpu in items]")
    run([*ssh, 'python3 -c ' + shlex.quote(code)], 150)
    install('logika-cross-checkpoint42-20261030',
            [EVAL_PY, '-u', str(R / 'ops/checkpoint_loop.py')], R, memory='4G', cpu=200)
    install('logika-cross-coordinator-20261030',
            [EVAL_PY, '-u', str(R / 'ops/coordinator.py')], R, memory='12G', cpu=200)
    print(json.dumps(dict(status='owned_persistent_units_installed',
                          clock_sha256=sha(R / 'CLOCK.json'),
                          actual_workers_still_require_live_verification=True,
                          Colab_controllers=len(controllers) - 1)))


if __name__ == '__main__':
    main()
