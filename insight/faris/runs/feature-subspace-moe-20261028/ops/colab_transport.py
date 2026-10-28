"""Use one authenticated SSH bridge to the already-created isolated session."""
from pathlib import Path
import shlex
import subprocess

LOCAL = Path(__file__).resolve().parents[1]
PRIVATE = LOCAL / 'colab_private'
SOCKET_DIR = Path.home() / '.cache/logika-moe28'
SOCKET_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
PREFIX = ['colab', '--account', '2', '--config', str(PRIVATE / 'sessions.json')]
SESSION = 'logika-moe-20261028-12h'
REMOTE = '/root/logika-feature-subspace-moe-20261028'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(PRIVATE / 'known_hosts'),
           '-o', 'HostKeyAlias=' + SESSION, '-o', 'ControlMaster=auto',
           '-o', 'ControlPersist=600', '-o', 'ControlPath=' + str(SOCKET_DIR / 'mux'),
           '-o', 'ProxyCommand=' + shlex.join(PREFIX + ['ssh', '--proxy-mode', '--session', SESSION])]

def ssh(command, *, code=None, timeout=90, check=True):
    return subprocess.run(['ssh', '-T', *OPTIONS, 'root@colab', command], input=code,
                          text=True, capture_output=True, timeout=timeout, check=check)

def put(local, remote, timeout=600):
    return subprocess.run(['scp', '-q', *OPTIONS, str(local), 'root@colab:' + remote],
                          text=True, capture_output=True, timeout=timeout, check=True)

def get(remote, local, timeout=300):
    return subprocess.run(['scp', '-q', *OPTIONS, 'root@colab:' + remote, str(local)],
                          text=True, capture_output=True, timeout=timeout, check=True)
