import datetime as dt
import importlib.machinery
import importlib.util
import os
import signal
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


cleanup = load_script('a100_cleanup_tested', 'a100-cleanup')
terminal = load_script('a100_term_tested', 'a100-term')
NOW = dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)


def kernel(kid, **changes):
    return dict(id=kid, execution_state='idle', connections=0,
                last_activity='2026-10-03T00:00:00Z', **changes)


def process(kid, pid, **changes):
    row = dict(kernel_id=kid, pid=pid, start=str(pid), uid=1113,
               children=[], rss_mib=60, name='python3')
    row.update(changes)
    return row


class SingleKernelTests(unittest.TestCase):
    def test_one_shots_reuse_term_and_keep_it_alive(self):
        client = Mock()
        client.session_kernel.return_value = 'existing-term-id'
        client.session_cwd.return_value = '/remote/project'
        client.run.return_value = (0, '')
        with patch.object(terminal, 'client', return_value=client):
            self.assertEqual(terminal.one_shot('pwd'), 0)
            self.assertEqual(terminal.one_shot('whoami'), 0)
        self.assertEqual(client.session_kernel.call_count, 2)
        client.session_kernel.assert_called_with('term', 'python3')
        self.assertEqual([call.args[0] for call in client.run.call_args_list],
                         ['existing-term-id', 'existing-term-id'])
        self.assertEqual(client.run.call_args.kwargs['cwd'], '/remote/project')
        client.create_kernel.assert_not_called()
        client.delete_kernel.assert_not_called()


class CandidateTests(unittest.TestCase):
    def test_only_owned_old_detached_childless_helpers_are_selected(self):
        kernels = [kernel(kid) for kid in ['term', 'old', 'busy', 'connected', 'notebook',
                                          'unknown', 'children', 'foreign', 'fresh', 'missing']]
        kernels[2]['execution_state'] = 'busy'
        kernels[3]['connections'] = 1
        kernels[8]['last_activity'] = '2026-10-07T23:00:00Z'
        processes = [process(k['id'], i + 100) for i, k in enumerate(kernels[:-1])]
        processes[6]['children'] = [999]
        processes[7]['uid'] = 2222
        tracked = {k['id']: k['id'] for k in kernels if k['id'] != 'unknown'}
        sessions = [{'kernel': {'id': 'notebook'}}]
        candidates = cleanup.idle_helper_candidates(
            kernels, sessions, tracked, {'uid': 1113, 'processes': processes}, 'term', 24, NOW)
        self.assertEqual([c['kernel_id'] for c in candidates], ['old'])

    def test_missing_or_invalid_activity_is_not_eligible(self):
        k = kernel('bad')
        k['last_activity'] = 'invalid'
        self.assertEqual(cleanup.idle_helper_candidates(
            [k], [], {'bad': 'bad'}, {'uid': 1113, 'processes': [process('bad', 100)]},
            'term', 24, NOW), [])

    def test_duplicate_monitors_keep_newest_and_skip_other_users_and_children(self):
        rows = [process(None, 100, name='nvtop', start='100'),
                process(None, 200, name='nvtop', start='200'),
                process(None, 300, name='nvtop', start='300', uid=2222),
                process(None, 400, name='nvtop', start='400', children=[401])]
        candidates = cleanup.duplicate_monitors({'uid': 1113, 'processes': rows})
        self.assertEqual([p['pid'] for p in candidates], [100])


class PreviewTests(unittest.TestCase):
    def test_preview_never_creates_deletes_or_signals(self):
        client = Mock()
        client.list_kernels.return_value = [kernel('term'), kernel('old')]
        client.request.return_value = []
        snapshot = dict(uid=1113, kernel_pid=100, zombies=[], other_zombie_count=0,
                        processes=[process('term', 100), process('old', 200)])
        with patch.object(cleanup, 'load_state', return_value={'sessions': {'term': 'term', 'old': 'old'}}), \
                patch.object(cleanup, 'load_token', return_value='test-token'), \
                patch.object(cleanup, 'JupyterClient', return_value=client), \
                patch.object(cleanup, 'remote_snapshot', return_value=snapshot), \
                patch('builtins.print'):
            self.assertEqual(cleanup.main(['--close-idle-helpers', '--close-duplicate-monitors', '--reap-zombies']), 0)
        client.create_kernel.assert_not_called()
        client.delete_kernel.assert_not_called()
        client.run_python.assert_not_called()

    def test_missing_control_kernel_never_creates_replacement(self):
        client = Mock()
        client.list_kernels.return_value = []
        with patch.object(cleanup, 'load_state', return_value={'sessions': {'term': 'term'}}), \
                patch.object(cleanup, 'load_token', return_value='test-token'), \
                patch.object(cleanup, 'JupyterClient', return_value=client):
            with self.assertRaisesRegex(RuntimeError, 'no kernel was created'):
                cleanup.main([])
        client.create_kernel.assert_not_called()

    def test_kernel_that_becomes_busy_is_not_closed(self):
        client = Mock()
        busy = kernel('old')
        busy['execution_state'] = 'busy'
        client.list_kernels.side_effect = [[kernel('term'), kernel('old')], [kernel('term'), busy]]
        client.request.return_value = []
        snapshot = dict(uid=1113, kernel_pid=100, zombies=[], other_zombie_count=0,
                        processes=[process('term', 100), process('old', 200)])
        with patch.object(cleanup, 'load_state', return_value={'sessions': {'term': 'term', 'old': 'old'}}), \
                patch.object(cleanup, 'load_token', return_value='test-token'), \
                patch.object(cleanup, 'JupyterClient', return_value=client), \
                patch.object(cleanup, 'remote_snapshot', return_value=snapshot), \
                patch('builtins.print'):
            self.assertEqual(cleanup.main(['--close-idle-helpers', '--apply']), 0)
        client.delete_kernel.assert_not_called()


class ProcessActionTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {}
        exec(compile(cleanup.REMOTE_CODE, '<remote-cleanup>', 'exec'), self.namespace)

    def test_reused_pid_is_not_signalled(self):
        self.namespace['_a100_cleanup_snapshot'] = lambda: dict(
            processes=[process(None, 100, name='nvtop', start='new')])
        with patch.object(os, 'kill') as kill:
            result = self.namespace['_a100_cleanup_process_action'](
                dict(kind='monitor', pid=100, start='old'))
        self.assertEqual(result['status'], 'skipped')
        kill.assert_not_called()

    def test_monitor_with_children_is_not_signalled(self):
        self.namespace['_a100_cleanup_snapshot'] = lambda: dict(
            processes=[process(None, 100, name='nvtop', children=[101])])
        with patch.object(os, 'kill') as kill:
            result = self.namespace['_a100_cleanup_process_action'](
                dict(kind='monitor', pid=100, start='100'))
        self.assertEqual(result['status'], 'skipped')
        kill.assert_not_called()

    def test_foreign_process_is_not_signalled(self):
        self.namespace['_a100_cleanup_snapshot'] = lambda: dict(processes=[], zombies=[])
        with patch.object(os, 'kill') as kill:
            result = self.namespace['_a100_cleanup_process_action'](
                dict(kind='zombie-parent', pid=100, start='100'))
        self.assertEqual(result['status'], 'skipped')
        kill.assert_not_called()

    @unittest.skipUnless(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'pidfd unavailable')
    def test_zombie_parent_receives_sigchld_never_a_termination_signal(self):
        parent = process(None, 987654, start='100')
        self.namespace['_a100_cleanup_snapshot'] = lambda: dict(
            processes=[parent], zombies=[dict(pid=987655, ppid=987654)])
        with patch.object(os, 'pidfd_open', return_value=99), patch.object(os, 'close'), \
                patch.object(signal, 'pidfd_send_signal') as send, patch.object(os, 'kill') as kill:
            result = self.namespace['_a100_cleanup_process_action'](
                dict(kind='zombie-parent', pid=987654, start='100'))
        send.assert_called_once_with(99, signal.SIGCHLD)
        kill.assert_not_called()
        self.assertEqual(result['status'], 'parent notified with SIGCHLD')

    @unittest.skipUnless(hasattr(os, 'fork') and Path('/proc/self/stat').exists(), 'Linux fork required')
    def test_real_zombie_child_is_reaped_without_killing_parent(self):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                snapshot = self.namespace['_a100_cleanup_snapshot']()
                if any(z['pid'] == pid for z in snapshot['zombies']):
                    break
                time.sleep(0.01)
            else:
                self.fail('child did not enter zombie state')
            parent = next(p for p in snapshot['processes'] if p['pid'] == os.getpid())
            result = self.namespace['_a100_cleanup_process_action'](
                dict(kind='zombie-parent', pid=os.getpid(), start=parent['start']))
            self.assertEqual(result['status'], 'reaped')
            self.assertIn(pid, result['pids'])
            self.assertFalse(Path(f'/proc/{pid}').exists())
        finally:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


if __name__ == '__main__':
    unittest.main()
