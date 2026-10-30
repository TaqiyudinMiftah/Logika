"""Workstation-only sanitized Git publication; private evidence stays on remotes."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import shutil
import subprocess
import time
import traceback


def run(args, cwd=None, timeout=180):
    if args[0] == 'git':
        args = ['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false', *args[1:]]
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True, timeout=timeout)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redact(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if any(token in key.lower() for token in ['traceback','stderr','stdout','error','exception','log_tail','log_excerpt']) and isinstance(item, (str, list, dict)):
                raw = json.dumps(item, sort_keys=True).encode()
                result[key + '_private_sha256'] = hashlib.sha256(raw).hexdigest()
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def main(local, repo, once=False):
    private = local / 'private/publication'
    private.mkdir(parents=True, exist_ok=True)
    lock = (private / 'publisher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    namespace = 'insight/faris/runs/' + local.name
    destination = repo / namespace
    assert (destination / 'RUN.json').exists(), 'Preregister before ordinary publication'
    helper_spec = importlib.util.spec_from_file_location('public_history_helper', repo / 'tools/publish_experiment_history.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    mirror = private / 'aggregate_mirror'
    mirror.mkdir(exist_ok=True)
    failures = 0
    while True:
        try:
            assert shutil.disk_usage(local).free >= 512 * 1024**2, 'Preserve workstation storage reserve'
            run(['git', 'fetch', 'origin', 'main'], repo, 60)
            head = run(['git', 'rev-parse', 'HEAD'], repo).stdout.strip()
            remote = run(['git', 'rev-parse', 'origin/main'], repo).stdout.strip()
            if head != remote:
                changed = run(['git', 'diff', '--name-only', head + '...' + remote], repo).stdout.splitlines()
                assert not any(path.startswith(namespace + '/') for path in changed), 'Peer changed this namespace; inspect before replacing metadata'
                run(['git', 'merge', '--no-edit', 'origin/main'], repo)
            filters = []
            for folder in ['private', 'packets', 'source', 'ops', 'logs', '__pycache__']:
                filters.extend(['--exclude=' + folder + '/***', '--exclude=**/' + folder + '/***'])
            filters.extend(['--include=*/', '--include=*.json', '--include=*.jsonl', '--include=*.md', '--exclude=*'])
            run(['rsync', '-az', *filters, '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=8',
                 'faris@100.68.29.126:/mnt/home/faris/logika-' + local.name + '/', str(mirror) + '/'])
            exported = []
            for path in [*mirror.rglob('*'), *local.joinpath('ops').glob('*.py')]:
                if not path.is_file():
                    continue
                relative = path.relative_to(local) if path.parent == local / 'ops' else path.relative_to(mirror)
                payload, policy = helper.public_content(path)
                if payload is None:
                    continue
                if path.suffix == '.json':
                    payload = (json.dumps(redact(json.loads(payload)), indent=2, allow_nan=False) + '\n').encode()
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.read_bytes() != payload:
                    target.write_bytes(payload)
                exported.append(str(target.relative_to(repo)))
            state = json.loads((mirror / 'LIVE_STATE.json').read_text()) if (mirror / 'LIVE_STATE.json').exists() else {}
            record = json.loads((destination / 'RUN.json').read_text())
            record.update(status='completed' if (mirror / 'COMPLETION.json').exists() else 'running',
                          completed_fits=state.get('completed_fits', 0), expected_fits=6160,
                          failure_records=state.get('failure_records', 0),
                          scientific_source_commit=json.loads((local / 'PREPUBLICATION.json').read_text())['remote_commit'])
            (destination / 'RUN.json').write_text(json.dumps(record, indent=2) + '\n')
            exported.append(namespace + '/RUN.json')
            scanner = local.parent / 'github-history-20261028/private/scanner/gitleaks'
            scan = private / 'SECRET_SCAN.json'
            run([str(scanner), 'dir', str(destination), '--config', str(repo / '.gitleaks.toml'),
                 '--ignore-gitleaks-allow', '--redact=100', '--no-banner', '--no-color',
                 '--report-format', 'json', '--report-path', str(scan)], timeout=90)
            assert json.loads(scan.read_text()) == []
            for start in range(0, len(exported), 300):
                run(['git', 'add', '-f', '--', *exported[start:start + 300]], repo)
            staged = run(['git', 'diff', '--cached', '--name-only'], repo).stdout.splitlines()
            assert all(path.startswith(namespace + '/') for path in staged)
            if staged:
                run(['git', 'commit', '-m', 'Record cross-network and kernel research progress and outcomes'], repo)
                run(['git', 'fetch', 'origin', 'main'], repo, 60)
                assert subprocess.run(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'], cwd=repo).returncode == 0
                run(['git', 'push', 'origin', 'HEAD:refs/heads/main'], repo, 120)
            head = run(['git', 'rev-parse', 'HEAD'], repo).stdout.strip()
            remote = run(['git', 'ls-remote', 'origin', 'refs/heads/main'], repo, 60).stdout.split()[0]
            assert head == remote
            receipt = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           remote_commit=remote, source_namespace=namespace,
                           completed_fits=state.get('completed_fits', 0), secret_findings=0,
                           private_artifacts_not_copied_to_Git=True,
                           remote_compute_does_not_depend_on_this_publisher=True)
            (private / 'LATEST_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps(receipt), flush=True)
            failures = 0
            if once or (mirror / 'COMPLETION.json').exists():
                return
        except Exception:
            failures += 1
            (private / 'FAILURE_PRIVATE.log').write_text(traceback.format_exc())
            if once or failures >= 5:
                raise
        time.sleep(300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    main(args.local.resolve(), args.repo.resolve(), args.once)
