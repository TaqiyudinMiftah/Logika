"""Workstation-only aggregate publication and private terminal artifact collection.

The lab and Colab workers never receive Git credentials. This operational module
does not import scientific modules, select models, fit anything, or extend clocks.
Initial source/configs must already have been reviewed and published by the root.
"""
import argparse
from artifact_aliases import apply_approved_aliases
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time
import traceback
import shutil
import re

REMOTE_HOST = 'faris@100.68.29.126'
REMOTE_ROOT = '/mnt/home/faris/logika-twelvehour-next-20261029'
NAMESPACE = 'insight/faris/runs/twelvehour-next-20261029'
BLOCKED_PARTS = {
    'private', 'publication_ready', 'INCREMENTAL_EVIDENCE', 'FINAL_EVIDENCE', 'publication_private', 'colab_private', 'research', 'source',
    'evaluation_inputs', 'checkpoints', 'pretrained', 'pretrained-cache',
    'huggingface', 'hf', 'storage', 'env', 'env311', '.venv', '__pycache__',
    'data', 'datasets', 'raw_data', 'raw_logs', 'logs', 'catboost_info',
}
TERMINAL_DIRS = ['runs', 'evaluation', 'failures']  # Full composed cases remain in both lab copies; local retains raw/cohort evidence.


class PublicationConflict(RuntimeError):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args, cwd=None, timeout=180):
    if args and args[0] == 'git':
        # Collection/publication cannot spend the storage reserve on an implicit
        # repack. Objects and peer history remain intact; explicit maintenance
        # can run after the campaign has been collected and audited.
        args = ['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false', *args[1:]]
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True,
                          timeout=timeout, check=True)


def aggregate_filters():
    result = []
    # Exclusions precede the directory include: never traverse private trees.
    for part in sorted(BLOCKED_PARTS):
        result.extend(['--exclude=' + part + '/***', '--exclude=**/' + part + '/***'])
    result.extend(['--include=*/', '--include=*.json', '--include=*.jsonl',
                   '--include=*.md', '--exclude=*'])
    return result


def allowed_relative(relative):
    return (not any(part in BLOCKED_PARTS for part in relative.parts)
            and relative.suffix.lower() in {'.json', '.jsonl', '.md'})


def redact_diagnostics(value):
    """Keep trace identity while withholding potentially case-bearing text."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in {'categorical_vocabulary','vocabularies','category_levels','category_maps','raw_value_preview','sample_cases'}:
                raw = json.dumps(item, sort_keys=True, default=str).encode()
                result[key + '_private_sha256'] = hashlib.sha256(raw).hexdigest()
                result[key + '_policy'] = 'Raw category levels and case previews remain private; recipe identity retained.'
                continue
            diagnostic = any(token in normalized for token in ['error', 'traceback', 'exception', 'stderr', 'stdout', 'log_tail', 'log_excerpt'])
            if diagnostic and isinstance(item, (str, dict, list)):
                raw = item.encode() if isinstance(item, str) else json.dumps(item, sort_keys=True, default=str).encode()
                result[key + '_private_sha256'] = hashlib.sha256(raw).hexdigest()
                result[key + '_policy'] = 'Full diagnostic retained privately; case-bearing text is not published.'
            else:
                result[key] = redact_diagnostics(item)
        return result
    if isinstance(value, list):
        return [redact_diagnostics(item) for item in value]
    return value


def safe_payload(path, public_content):
    content, policy = public_content(path)
    if content is None:
        return None
    if path.suffix == '.json':
        value = redact_diagnostics(json.loads(content))
        return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if path.suffix == '.jsonl':
        lines = []
        for line in content.decode().splitlines():
            if line.strip():
                # An interrupted append is retried on a later snapshot. Never
                # publish malformed raw lines containing unstructured data.
                lines.append(json.dumps(redact_diagnostics(json.loads(line)), allow_nan=False))
        return ('\n'.join(lines) + ('\n' if lines else '')).encode()
    if path.suffix == '.md' and any(token in path.name.lower() for token in ['error', 'failure', 'traceback', 'stderr', 'stdout']):
        return None
    return content


def helper_from_repo(repo):
    path = repo / 'tools/publish_experiment_history.py'
    spec = importlib.util.spec_from_file_location('logika_public_history_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.public_content


def reconcile_peer(repo, private):
    run(['git', 'fetch', 'origin', 'main'], repo)
    head = run(['git', 'rev-parse', 'HEAD'], repo).stdout.strip()
    remote = run(['git', 'rev-parse', 'origin/main'], repo).stdout.strip()
    if head == remote:
        return
    changed = run(['git', 'diff', '--name-only', head + '...' + remote], repo).stdout.splitlines()
    conflicts = [name for name in changed if name == NAMESPACE or name.startswith(NAMESPACE + '/')]
    if conflicts:
        dump(dict(UTC=utc(), local_head=head, remote_head=remote, paths=conflicts,
                  policy='Peer changed this campaign namespace; stop before exporting over it.'), private / 'NAMESPACE_CONFLICT.json')
        raise PublicationConflict('Peer changed campaign namespace; manual review required')
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', head, remote], cwd=repo).returncode == 0
    run(['git', 'merge', '--quiet', '--ff-only', 'origin/main'] if ancestor
        else ['git', 'merge', '--quiet', '--no-edit', 'origin/main'], repo)


def collect_terminal(local, private, completion_hash):
    """Collect only compact predictions/metadata; never raw labels or weights."""
    previous = private / 'COLLECTION_RECEIPT.json'
    manifest_path = private / 'FINAL_PRIVATE_MANIFEST.json'
    if previous.exists() and manifest_path.exists():
        saved = json.loads(previous.read_text())
        if saved.get('campaign_completion_sha256') == completion_hash:
            assert sha(manifest_path) == saved['private_manifest_sha256']
            for relative, identity in json.loads(manifest_path.read_text()).items():
                assert sha(local / 'FINAL_EVIDENCE' / relative) == identity['sha256']
            return saved
    namespaces=json.loads((local/'UNION_PLAN.json').read_text())['component_studies']
    allowed=[str(Path(namespace)/name)for namespace in namespaces for name in TERMINAL_DIRS]
    code = ('from pathlib import Path;import json;r=Path(' + repr(REMOTE_ROOT)
            + ');print(json.dumps([name for name in ' + repr(allowed) + ' if(r/name).is_dir()]))')
    present = json.loads(run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                              REMOTE_HOST, 'python3 -c ' + shlex.quote(code)], timeout=45).stdout)
    destination = local / 'FINAL_EVIDENCE'
    destination.mkdir(exist_ok=True)
    filters = ['--include=*/', '--include=prediction.npz', '--include=predictions.npz',
               '--include=*.json', '--include=*.jsonl', '--include=*.md',
               '--include=*.log', '--exclude=*']
    for name in present:
        assert name in allowed
        target = destination / name
        target.mkdir(parents=True,exist_ok=True)
        # Reuse already collected identical files on this filesystem. Rsync's
        # default temporary-file replacement preserves existing linked copies.
        links = ['--link-dest=' + str((local / 'INCREMENTAL_EVIDENCE' / name).resolve())] \
            if (local / 'INCREMENTAL_EVIDENCE' / name).is_dir() else []
        transfer=['rsync','-az',*links,*filters,REMOTE_HOST+':'+REMOTE_ROOT+'/'+name+'/',str(target)+'/']
        reserve_rsync(transfer,local)
        run(transfer,timeout=300)
    registry=local/'APPROVED_PRESTART_RETRIES.json'
    if registry.exists():shutil.copy2(registry,destination/registry.name)
    apply_approved_aliases(destination,require_predictions=True)
    manifest = {str(path.relative_to(destination)): dict(bytes=path.stat().st_size, sha256=sha(path))
                for path in sorted(destination.rglob('*')) if path.is_file()}
    fit_count, composition_count = 0, 0
    for directory in ['runs', 'composed']:
        for path in destination.rglob('RESULT.json'):
            if directory not in path.relative_to(destination).parts:continue
            result = json.loads(path.read_text())
            if result.get('status') != 'complete':
                continue
            if 'prediction_sha256' in result:
                assert sha(path.parent / 'prediction.npz') == result['prediction_sha256']
            if 'start_sha256' in result:
                assert sha(path.parent / 'START.json') == result['start_sha256']
            if 'frozen_sha256' in result:
                assert sha(path.parent / 'FROZEN.json') == result['frozen_sha256']
            assert result.get('weights_saved', False) is False and result.get('fitted_weights_saved', False) is False
            assert result.get('KV_cache_saved', False) is False
            if directory == 'runs':
                fit_count += 1
            else:
                composition_count += 1
    dump(manifest, private / 'FINAL_PRIVATE_MANIFEST.json')
    receipt = dict(status='COLLECTED_AND_HASH_VERIFIED', UTC=utc(), private_artifacts=len(manifest),
                   private_bytes=sum(item['bytes'] for item in manifest.values()),
                   private_manifest_sha256=sha(private / 'FINAL_PRIVATE_MANIFEST.json'),
                   campaign_completion_sha256=completion_hash,
                   completed_fit_attempts_checked=fit_count,
                   completed_compositions_checked=composition_count,
                   scope='Artifact collection only; root terminal scientific/selection audit is separate.',
                   raw_data_or_case_labels_collected=False, fitted_weights_or_KV_caches_collected=False)
    dump(receipt, private / 'COLLECTION_RECEIPT.json')
    return receipt


def root_audit(mirror):
    for name in ['FINAL_LOCAL_COLLECTION_AUDIT.json', 'ARTIFACT_CONSISTENCY_AUDIT.json', 'TERMINAL_AUDIT.json']:
        path = mirror / name
        if path.exists():
            value = json.loads(path.read_text())
            if str(value.get('status', '')).upper() == 'PASS':
                return dict(path=name, sha256=sha(path), status='PASS')
    return None


def counts(mirror, state):
    complete = len([p for p in (mirror/'runs').glob('*/attempt1/RESULT.json') if json.loads(p.read_text()).get('status')=='complete'])
    config=json.loads((mirror/'CONFIG.json').read_text())if(mirror/'CONFIG.json').exists()else{}
    ledger=mirror/'EXTRA_DESIGNS.jsonl'
    extra=sum(bool(line.strip())for line in ledger.read_text().splitlines())if ledger.exists()else 0
    expected=(len(config.get('designs',{}))+extra)*len(config.get('seeds',[2026,7731]))*5 if config else state.get('expected_fits',990)
    return complete,expected


def union_counts(mirror):
    plan=mirror/'UNION_PLAN.json'
    if not plan.exists():return None
    value=json.loads(plan.read_text());records={}
    for name in value['component_studies']:
        root=mirror if name=='.' else mirror/name
        records[name]=dict(completed_fits=sum(json.loads(p.read_text()).get('status')=='complete'
                           for p in (root/'runs').glob('*/attempt1/RESULT.json')),
                           failure_records=len(list((root/'runs').glob('*/attempt1/FAILURE.json'))),
                           complete=(root/'COMPLETION.json').exists())
    return dict(planned_fit_slots=value['planned_fit_attempts'],
                completed_fits=sum(x['completed_fits']for x in records.values()),
                failure_records=sum(x['failure_records']for x in records.values()),components=records,
                planned_slots_are_not_completion=True)


def reserve_rsync(args,local):
    """Forecast changed bytes without mutation, preserve512MiB after transfer."""
    preview=run([args[0],*args[1:-2],'--dry-run','--stats',*args[-2:]],timeout=180)
    match=re.search(r'Total transferred file size:\s*([\d,]+)',preview.stdout)
    assert match,'No rsync byte forecast; defer rather than assume zero'
    required=int(match.group(1).replace(',',''))
    # The private mirror can be a path-preserving symlink onto another volume.
    # Guard both the checkout volume and the actual rsync destination volume.
    destination=Path(args[-1]).resolve()
    while not destination.exists():destination=destination.parent
    free=min(shutil.disk_usage(local).free,shutil.disk_usage(destination).free)
    if free-required < 512*1024**2:raise RuntimeError('Storage reserve would be breached; canonical artifacts retained')
    return required


def publish(local, repo, once=False):
    private = local / 'private/publication'
    private.mkdir(parents=True, exist_ok=True)
    assert (local/'PREPUBLICATION.json').exists(), 'Root preregistration receipt missing'
    lock = (private / 'publisher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mirror = private / 'remote_aggregate_mirror'
    mirror.mkdir(exist_ok=True)
    destination = repo / NAMESPACE
    assert destination.is_dir(), 'Root must prepublish the campaign before runtime metadata publication'
    public_content = helper_from_repo(repo)
    consecutive_errors = 0
    while True:
        try:
            staged = run(['git', 'diff', '--cached', '--name-only'], repo).stdout.splitlines()
            assert all(name.startswith(NAMESPACE + '/') for name in staged), 'Unrelated staged changes detected; do not publish them'
            # Fetch before modifying the worktree, preserving concurrent peer work.
            reconcile_peer(repo, private)
            transfer=['rsync','-az',*aggregate_filters(),REMOTE_HOST+':'+REMOTE_ROOT+'/',str(mirror)+'/']
            reserve_rsync(transfer,local)
            run(transfer,timeout=180)
            apply_approved_aliases(mirror,require_predictions=False)
            state = json.loads((mirror / 'LIVE_STATE.json').read_text()) if (mirror / 'LIVE_STATE.json').exists() else {'stage': 'launch_pending'}
            clock = json.loads((mirror / 'CLOCK.json').read_text()) if (mirror / 'CLOCK.json').exists() else {}
            stage = str(state.get('stage', state.get('status', 'running')))
            completion = mirror / 'COMPLETION.json'
            terminal = completion.exists()
            if terminal: stage='complete' if (mirror/'ARTIFACT_CONSISTENCY_AUDIT.json').exists() else 'terminal_audit_pending'
            collection=None
            if terminal:
                try:collection=collect_terminal(local,private,sha(completion))
                except RuntimeError as error:
                    if 'Storage reserve would be breached'not in str(error):raise
                    collection=dict(status='LOCAL_COLLECTION_DEFERRED_STORAGE',UTC=utc(),
                        campaign_completion_sha256=sha(completion),private_local_copy_complete=False,
                        canonical_evidence_retained=REMOTE_HOST+':'+REMOTE_ROOT,
                        redundant_checkpoint_root='faris@100.98.169.42:/home/faris/logika-next-checkpoints-20261029',
                        reason='Preserve the 512MiB workstation reserve; do not fabricate a local hash verification.')
                    dump(collection,private/'COLLECTION_DEFERRED.json')
            exported = []
            # Operational repairs are versioned too; scientific sources remain
            # preregistered and sealed. Scan exactly what will be published.
            for path in sorted((local / 'ops').glob('*.py')):
                payload = safe_payload(path, public_content)
                if payload is None:
                    continue
                target = destination / 'ops' / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.read_bytes() != payload:
                    target.write_bytes(payload)
                exported.append(str(target.relative_to(repo)))
            for path in sorted(mirror.rglob('*')):
                if not path.is_file():
                    continue
                relative = path.relative_to(mirror)
                if not allowed_relative(relative):
                    continue
                payload = safe_payload(path, public_content)
                if payload is None:
                    continue
                target = destination / relative
                if not target.exists() or target.read_bytes() != payload:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                exported.append(str(target.relative_to(repo)))
            complete, expected = counts(mirror, state)
            union=union_counts(mirror)
            run_record = destination / 'RUN.json'
            if run_record.exists():
                record = json.loads(run_record.read_text())
                record.update(status=stage, completed_fits=complete,
                              expected_fits=expected, computation_clock_started=bool(clock),
                              current_state_evidence='LIVE_STATE.json', clock=clock)
                if union:record['union_progress']=union
                content = json.dumps(record, indent=2, allow_nan=False) + '\n'
                if run_record.read_text() != content:
                    run_record.write_text(content)
                exported.append(str(run_record.relative_to(repo)))
            if collection:
                target = destination / 'FINAL_COLLECTION_STATUS.json'
                content = json.dumps(collection, indent=2) + '\n'
                if not target.exists() or target.read_text() != content:
                    target.write_text(content)
                exported.append(str(target.relative_to(repo)))
            scanner = local.parent / 'github-history-20261028/private/scanner/gitleaks'
            scan = private / 'SECRET_SCAN.json'
            run([str(scanner), 'dir', str(destination), '--config', str(repo / '.gitleaks.toml'),
                 '--ignore-gitleaks-allow', '--redact=100', '--no-banner', '--no-color',
                 '--report-format', 'json', '--report-path', str(scan)], timeout=180)
            assert json.loads(scan.read_text()) == [], 'Public aggregate evidence requires secret review'
            # Stage exactly the exported aggregate paths, never a whole tree that
            # could contain another user's untracked files or private artifacts.
            if exported:
                paths = sorted(set(exported))
                for start in range(0, len(paths), 300):
                    run(['git', 'add', '-f', '--', *paths[start:start + 300]], repo)
            staged = run(['git', 'diff', '--cached', '--name-only'], repo).stdout.splitlines()
            assert all(name in exported for name in staged), 'Unrelated or non-aggregate staged changes detected'
            if staged:
                body = private / 'commit_message.txt'
                body.write_text('Record twelve-hour CANN and case-control campaign ' + stage + ' evidence\n\n'
                                'Preserve completed, failed and partial experiment metadata.\n'
                                'Raw cases, predictions, weights, caches and credentials stay private.\n')
                run(['git', 'commit', '--quiet', '--file', str(body)], repo)
                # Reconcile once more immediately before a normal non-forced push.
                reconcile_peer(repo, private)
                run(['git', 'push', 'origin', 'HEAD:refs/heads/main'], repo, timeout=180)
            head = run(['git', 'rev-parse', 'HEAD'], repo).stdout.strip()
            verified = run(['git', 'ls-remote', 'origin', 'refs/heads/main'], repo).stdout.split()[0]
            assert verified == head, 'Remote advanced concurrently; preserve and inspect on the next sync'
            audit = root_audit(mirror) if terminal else None
            receipt = dict(UTC=utc(), remote_commit=verified, stage=stage,
                           completed_fits=complete, expected_fits=expected,
                           secret_findings=0, metadata_files=len(list(mirror.rglob('*.json'))),
                           union_progress=union,
                           compact_evidence_collected=bool(collection and collection.get('status')=='COLLECTED_AND_HASH_VERIFIED'), root_terminal_audit=audit)
            dump(receipt, private / 'LATEST_RECEIPT.json')
            print(json.dumps(receipt), flush=True)
            consecutive_errors = 0
            if terminal and collection and collection.get('status')=='COLLECTED_AND_HASH_VERIFIED' and audit:
                dump(dict(receipt, private_manifest_sha256=collection['private_manifest_sha256'],
                          final_evidence_location=str(local / 'FINAL_EVIDENCE')),
                     private / 'FINAL_RECEIPT.json')
                return
            if terminal and collection and not audit:
                dump(dict(status='AWAITING_ROOT_TERMINAL_AUDIT', UTC=utc(),
                          collection_receipt_sha256=sha(private / 'COLLECTION_RECEIPT.json')),
                     private / 'AWAITING_ROOT_AUDIT.json')
            if once:
                return
            if clock and time.time() > clock['deadline_epoch'] + 3600:
                dump(dict(status='NEEDS_ROOT_REVIEW', UTC=utc(), reason='Post-deadline publication reserve exhausted; no final audit receipt fabricated.'), private / 'PUBLICATION_NEEDS_REVIEW.json')
                return
        except BaseException as error:
            consecutive_errors += 1
            (private / 'publication_failure_private.log').write_text(traceback.format_exc())
            dump(dict(UTC=utc(), exception_class=type(error).__name__, consecutive_errors=consecutive_errors,
                      private_trace_sha256=sha(private / 'publication_failure_private.log')),
                 private / 'PUBLICATION_ERROR.json')
            print(json.dumps(dict(publication_error_class=type(error).__name__, consecutive_errors=consecutive_errors,
                                  private_evidence='private/publication/PUBLICATION_ERROR.json')), flush=True)
            if isinstance(error, (PublicationConflict, KeyboardInterrupt, SystemExit)) or once or consecutive_errors >= 5:
                raise
        time.sleep(300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    arguments = parser.parse_args()
    publish(arguments.local.resolve(), arguments.repo.resolve(), arguments.once)
