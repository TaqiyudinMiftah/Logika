"""Owned idle-device launch; scientific GPU recipe is separately preregistered."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import time
R = Path(__file__).resolve().parent


def run(args, timeout=180):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)


def install():
    plan = json.loads((R / 'GPU_PLAN.json').read_text())
    for index in (0, 1):
        response = run(['nvidia-smi', '--id=' + str(index), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
        memory, use = map(int, response.stdout.strip().replace(' ', '').split(','))
        assert memory <= 512 and use <= 5, 'Existing GPU activity: no launch'
        name = 'logika-finalgatedmeta-gpu' + str(index) + '-20261030'
        command = ['/mnt/home/faris/logika-submit-best-sub25k-20261029/env_gpu/bin/python', '-u', str(R / 'gpu_stacking.py'), '--owner-index', str(index)]
        folder = Path.home() / '.config/systemd/user'
        unit = '\n'.join(['[Unit]', 'Description=Owned final nonlinear historical blending', '[Service]',
            'Type=simple', 'ExecStart=' + shlex.join(command), 'WorkingDirectory=' + str(R),
            'Environment=CUDA_VISIBLE_DEVICES=' + str(index) + ' NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2',
            'RuntimeMaxSec=' + str(max(1, int(plan['model_deadline_epoch'] - time.time() - 15))),
            'Restart=no', 'KillMode=control-group', 'TimeoutStopSec=10', 'MemoryMax=24G', 'CPUQuota=200%', 'UMask=0077',
            'StandardOutput=append:' + str(R / (name + '.log')), 'StandardError=append:' + str(R / (name + '.log')),
            '[Install]', 'WantedBy=default.target', ''])
        (folder / (name + '.service')).write_text(unit)
        run(['systemctl', '--user', 'daemon-reload'])
        run(['systemctl', '--user', 'enable', '--now', name])
        print(name, 'started verified idle device', index, flush=True)


def deploy():
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-p2003', 'faris@gpu-labkc.ub.ac.id']
    target = '/mnt/home/faris/logika-final-history-blend-20261030/gatedmeta'
    transfer = ['rsync', '-az', '--ignore-existing', '-e', shlex.join(ssh[:-1])]
    names = ('gpu_stacking.py', 'GPU_PLAN.json', 'GPU_PREPUBLICATION.json', 'submissions.py', 'gpu_launch.py')
    run([*transfer, *[str(R / n) for n in names], ssh[-1] + ':' + target + '/'])
    run([*transfer, *[str(p) for p in (R / 'bank').iterdir()], ssh[-1] + ':' + target + '/bank/'], timeout=180)
    q = run([*ssh, 'python3 ' + shlex.quote(target + '/gpu_launch.py') + ' --install'])
    print(q.stdout, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--install', action='store_true')
    args = p.parse_args()
    install() if args.install else deploy()
