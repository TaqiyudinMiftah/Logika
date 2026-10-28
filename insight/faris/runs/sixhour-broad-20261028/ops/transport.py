"""One authenticated SSH bridge per explicitly allocated session; no auto-new."""
from pathlib import Path
import json
import shlex
import subprocess
import time

LOCAL = Path(__file__).resolve().parents[1]

class Colab:
    def __init__(self, number):
        self.number = number
        self.session = 'logika-sixhour-20261028-' + str(number)
        self.config = LOCAL / f'private/vm{number}/sessions.json'
        self.root = '/root/logika-sixhour-broad-20261028'
        socketdir = Path.home() / '.cache/logika6'
        socketdir.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.prefix = ['colab', '--account', '2', '--config', str(self.config)]
        self.options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
                        '-o', 'StrictHostKeyChecking=accept-new',
                        '-o', 'UserKnownHostsFile=' + str(self.config.parent / 'known_hosts'),
                        '-o', 'HostKeyAlias=' + self.session, '-o', 'ControlMaster=auto',
                        '-o', 'ControlPersist=600', '-o', 'ControlPath=' + str(socketdir / f'vm{number}'),
                        '-o', 'ProxyCommand=' + shlex.join(self.prefix + ['ssh', '--proxy-mode', '--session', self.session])]

    def guard(self):
        assert self.config.exists(), 'Explicit allocation required before SSH'
        assert not (self.config.parent / 'RELEASED.json').exists(), 'Never recreate a released session'
        clock = json.loads((LOCAL / 'CLOCK.json').read_text())
        assert time.time() < clock['deadline_epoch'], 'Original total ceiling passed'

    def ssh(self, command, code=None, timeout=90, check=True):
        self.guard()
        return subprocess.run(['ssh', '-T', *self.options, 'root@colab', command],
                              input=code, text=True, capture_output=True, timeout=timeout, check=check)

    def put(self, local, remote, timeout=1800):
        self.guard()
        return subprocess.run(['scp', '-q', *self.options, str(local), 'root@colab:' + remote],
                              text=True, capture_output=True, timeout=timeout, check=True)

    def get(self, remote, local, timeout=180):
        self.guard()
        return subprocess.run(['scp', '-q', *self.options, 'root@colab:' + remote, str(local)],
                              text=True, capture_output=True, timeout=timeout, check=True)

    def json(self, relative):
        code = 'from pathlib import Path;p=Path(' + repr(self.root + '/' + relative) + ");print(p.read_text() if p.exists() else 'null')"
        return json.loads(self.ssh('python3 -', code=code).stdout)

    def stop(self):
        with (self.config.parent / 'stop.log').open('ab') as log:
            subprocess.run([*self.prefix, 'stop', '--session', self.session], check=True, timeout=120,
                           stdout=log, stderr=subprocess.STDOUT)
        (self.config.parent / 'RELEASED.json').write_text(json.dumps({'epoch': time.time(), 'session': self.session}) + '\n')
