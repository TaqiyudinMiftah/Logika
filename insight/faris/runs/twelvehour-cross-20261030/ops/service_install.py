"""Install an owned user service that survives reboot; absolute clocks remain in code."""
from pathlib import Path
import shlex
import json
import subprocess


def install(name, command, working, memory='28G', cpu=800, start=True):
    assert name.startswith('logika-cross-') and all('\n' not in str(x) for x in command)
    folder = Path.home() / '.config/systemd/user'
    folder.mkdir(parents=True, exist_ok=True)
    deadline=json.loads((Path(working)/'CLOCK.json').read_text())['deadline_epoch']
    bounded=['/usr/bin/python3',str(Path(working)/'ops/within_clock.py'),'--root',str(working),'--',*[str(x) for x in command]]
    source = '\n'.join([
        '[Unit]', 'Description=Owned bounded Logika cross/kernel campaign',
        'StartLimitIntervalSec=0', '', '[Service]', 'Type=simple',
        'WorkingDirectory=' + str(working), 'ExecStart=' + shlex.join(bounded),
        'Environment=LOGIKA_HARD_DEADLINE='+str(deadline)+' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8',
        'UMask=0077', 'Nice=10', 'Restart=on-failure', 'RestartSec=60', 'KillMode=control-group', 'TimeoutStopSec=20',
        'MemoryMax=' + memory, 'CPUQuota=' + str(cpu) + '%',
        'StandardOutput=append:' + str(Path(working) / (name + '.log')),
        'StandardError=append:' + str(Path(working) / (name + '.log')),
        '', '[Install]', 'WantedBy=default.target', ''])
    (folder / (name + '.service')).write_text(source)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True, timeout=30)
    subprocess.run(['systemctl', '--user', 'enable', *( ['--now'] if start else []), name + '.service'],
                   check=True, timeout=30, capture_output=True)

