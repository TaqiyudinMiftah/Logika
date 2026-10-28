import os
from pathlib import Path
import runpy
import stat
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jupyter_remote as remote
from connection_profiles import activated, load_profile
from profile_login import LoginFields, write_private_token


class ProfileTests(unittest.TestCase):
    def test_destinations_and_all_caches_are_distinct(self):
        faris, node = load_profile("faris"), load_profile("node380")
        self.assertEqual(faris["base_url"], "https://node480-ai-hub.ub.ac.id/user/fariselhakim/")
        self.assertEqual(node["base_url"], "https://node380-ai-hub.ub.ac.id/user/taqiyudinmiftah/")
        for field in ["sysinfo", "state_path", "manifest_cache", "shell_history", "term_history", "token_env"]:
            self.assertNotEqual(faris[field], node[field], field)
        self.assertEqual(faris["state_path"], remote.STATE_PATH)
        self.assertEqual(faris["sysinfo"], Path(remote.DEFAULT_SYSINFO))

    def test_same_session_name_never_reuses_the_other_nodes_kernel(self):
        faris, node = load_profile("faris"), load_profile("node380")
        original = remote.STATE_PATH
        with tempfile.TemporaryDirectory() as temporary:
            faris["state_path"] = Path(temporary) / "faris.json"
            node["state_path"] = Path(temporary) / "node380.json"
            with activated(faris):
                remote.save_state({"sessions": {"term": "faris-kernel"}, "cwd": {"term": "/faris/project"}})
            with activated(node):
                self.assertEqual(remote.load_state()["sessions"], {})
                remote.save_state({"sessions": {"term": "node380-kernel"}, "cwd": {"term": "/node380/project"}})
            for profile, kernel, cwd in [(faris, "faris-kernel", "/faris/project"), (node, "node380-kernel", "/node380/project")]:
                with activated(profile):
                    client = remote.JupyterClient(profile["base_url"], "test-token")
                    with patch.object(client, "kernel_alive", return_value=True), patch.object(client, "create_kernel") as create:
                        self.assertEqual(client.session_kernel("term", "python3"), kernel)
                        self.assertEqual(client.session_cwd("term"), cwd)
                        create.assert_not_called()
        self.assertEqual(remote.STATE_PATH, original)

    def test_node380_ignores_unscoped_faris_environment_token(self):
        profile = load_profile("node380")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"JUPYTERHUB_API_TOKEN": "[REDACTED_CREDENTIAL]"}, clear=True):
            profile["sysinfo"] = Path(temporary) / "sysinfo.txt"
            with activated(profile):
                with self.assertRaises(FileNotFoundError):
                    remote.load_token(profile["sysinfo"])
                write_private_token(profile["sysinfo"], "node380-file-token")
                self.assertEqual(remote.load_token(profile["sysinfo"]), "node380-file-token")
                with patch.dict(os.environ, {"A100_NODE380_API_TOKEN": "[REDACTED_CREDENTIAL]"}):
                    self.assertEqual(remote.load_token(profile["sysinfo"]), "node380-env-token")
            self.assertEqual(remote.load_token(Path(temporary) / "missing"), "wrong-host-token")

    def test_term_helper_receives_selected_destination_and_history(self):
        profile = load_profile("node380")
        with activated(profile), patch.dict(os.environ, {"A100_NODE380_API_TOKEN": "[REDACTED_CREDENTIAL]"}), patch.object(remote, "JupyterClient") as constructor:
            namespace = runpy.run_path(str(ROOT / "a100-term"), run_name="term_profile_test")
            namespace["client"]()
            constructor.assert_called_once_with(profile["base_url"], "test-token")
            self.assertEqual(namespace["TERM_HISTORY"], profile["term_history"])

    def test_profile_settings_restore_even_on_failure(self):
        before = (remote.DEFAULT_BASE_URL, remote.TOKEN_ENV, remote.MANIFEST_CACHE)
        with self.assertRaises(RuntimeError):
            with activated(load_profile("node380")):
                raise RuntimeError("simulated disconnect")
        self.assertEqual((remote.DEFAULT_BASE_URL, remote.TOKEN_ENV, remote.MANIFEST_CACHE), before)

    def test_credentials_are_private_and_only_contain_the_server_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sysinfo.txt"
            write_private_token(path, "fake-server-token")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(path.read_text(), "JUPYTERHUB_API_TOKEN=fake-server-token\n")

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            load_profile("../faris")

    def test_old_and_new_hub_login_forms(self):
        for suffix in ["", '<input name="_xsrf" value="test-xsrf">']:
            parser = LoginFields()
            parser.feed('<form><input name="username"><input name="password">' + suffix + '</form>')
            self.assertTrue({"username", "password"}.issubset(parser.names))
            self.assertEqual(parser.xsrf, "test-xsrf" if suffix else None)


if __name__ == "__main__":
    unittest.main()
