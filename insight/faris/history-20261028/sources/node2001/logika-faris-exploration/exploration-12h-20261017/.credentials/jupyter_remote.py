#!/usr/bin/env python3
"""Execute shell commands through authenticated Jupyter kernels.

Upgraded client: persistent named sessions (kernel reuse + sticky cwd),
background jobs, chunked push/pull for large files and directories,
parallel run-many, and an interactive shell REPL.

Backward compatible single-shot usage:
    jupyter_remote.py "ls ~"
    jupyter_remote.py --download REMOTE_PATH --output LOCAL_PATH

New usage:
    jupyter_remote.py exec --session work "ls ~"
    jupyter_remote.py shell --session work
    jupyter_remote.py sessions
    jupyter_remote.py session-kill work
    jupyter_remote.py push LOCAL_PATH REMOTE_PATH
    jupyter_remote.py pull REMOTE_PATH LOCAL_PATH
    jupyter_remote.py bg --job train "python train.py"
    jupyter_remote.py poll train
    jupyter_remote.py log train --tail 50
    jupyter_remote.py kill-job train
    jupyter_remote.py run-many --workers 4 "sleep 2; echo a" "echo b"
    jupyter_remote.py bootstrap
    jupyter_remote.py sync-push ./mycode a100-code --delete --workers 4
    jupyter_remote.py sync-pull a100-code ./mycode
    jupyter_remote.py log train --follow
    jupyter_remote.py tunnel --port 8000 --job llm-tunnel
    jupyter_remote.py doctor
"""
# v8: persistent agent deployed ONCE via bootstrap and reused every call
# (no per-call package send -- exec/run_python only send a slim `import a100h`
# line; bulk bytes only cross the wire in sync-push/push for changed chunks).
# sync-push/sync-pull are incremental and default to fast size+mtime comparison
# with a local manifest cache, so large code trees re-sync in seconds without
# re-hashing. Full sha256 only on demand (--full-hash) or for small-file
# verification. Chunked resume (A100_CHUNK_MB, default 2, max 8), parallel
# workers (default 8), resumable .part files with prefix-hash resume check,
# mtime preserved so fast sync converges, default excludes keep
# .venv/.git/node_modules out. Shell is terminal-like with :bg/:poll/:log/
# :jobs/:kill/:kernels/:sessions/:vram plus :session/:push/:pull/:sync/:tunnel/
# :health/:doctor and :par (parallel `;;`-separated cmds on separate session
# kernels so agents never wait sequentially). bg/run-many stay the
# non-blocking primitives for parallel agent workflows. vram/health helpers
# verify model load/unload. Client-only v8: remote agent VER stays 7, no
# redeploy needed, fully backward compatible with v7 state/cache.

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import fnmatch
import threading
import hashlib
import json
import os
import re
import shutil
import ssl
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import websocket


DEFAULT_BASE_URL = "https://node480-ai-hub.ub.ac.id/user/fariselhakim/"
DEFAULT_SYSINFO_NAME = "sysinfo_fariselhakim.txt"
DEFAULT_SYSINFO = str(Path(__file__).resolve().parent / DEFAULT_SYSINFO_NAME)
STATE_PATH = Path.home() / ".cache" / "a100_faris_sessions.json"
TOKEN_ENV = "JUPYTERHUB_API_TOKEN"
SHELL_HISTORY = Path.home() / ".cache" / "a100_shell_history"
TERM_HISTORY = Path.home() / ".cache" / "a100_term_history"
CHUNK_RAW = max(1, min(8, int(os.environ.get("A100_CHUNK_MB", "2")))) * 1024 * 1024
SMALL_FILE_LIMIT = 4 * 1024 * 1024
# Directories/files never worth syncing for code pushes (huge, regenerable).
# Opt out per-call with --no-default-excludes.
DEFAULT_EXCLUDES = [
    ".git", ".venv", "venv", ".tox", "__pycache__", "*.pyc",
    "node_modules", ".cache", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "*.egg-info", "dist", "build",
]
SKIP_DIRS = {".git", ".venv", "venv", ".tox", "__pycache__",
             "node_modules", ".cache", ".mypy_cache", ".pytest_cache",
             ".ruff_cache", "dist", "build"}
EXIT_MARKER_RE = re.compile(r"\n?\[remote exit code: (-?\d+)\]\s*$")
CD_RE = re.compile(r"^cd(?:\s+(.*))?\s*$")
SUBCOMMANDS = {
    "exec", "shell", "sessions", "session-kill", "push", "pull",
    "bg", "poll", "log", "kill-job", "jobs", "run-many",
    "bootstrap", "sync-push", "sync-pull", "tunnel", "doctor", "vram", "health",
}


AGENT_VERSION = "7"
AGENT_CODE = r"""
# Persistent a100 remote helper. Deployed once by bootstrap, imported by later calls.
VER = "7"

import base64
import hashlib
import json
import os
import sys


def wp(path, data_b64, append=False):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    raw = base64.b64decode(data_b64)
    with open(path, "ab" if append else "wb") as handle:
        handle.write(raw)
    return len(raw)


def rp(path, off, num):
    with open(path, "rb") as handle:
        handle.seek(off)
        return base64.b64encode(handle.read(num)).decode()


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fin(tmp, dest, want):
    h = sha(tmp)
    if h != want:
        return "MISMATCH:" + h
    d = os.path.dirname(dest)
    if d:
        os.makedirs(d, exist_ok=True)
    os.replace(tmp, dest)
    return "sha:" + h


def st(path):
    try:
        s = os.stat(path)
        return json.dumps({"size": s.st_size, "mtime": s.st_mtime, "ok": True})
    except OSError as exc:
        return json.dumps({"ok": False, "err": str(exc)})


def manifest(root, fast=False):
    base = os.path.expandvars(os.path.expanduser(root))
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue
            item = {
                "rel": os.path.relpath(full, base),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
            # fast mode: size+mtime only, no content hashing (large trees stay quick)
            item["sha"] = "" if fast else sha(full)
            out.append(item)
    sys.stdout.write(json.dumps(out))


def manifest_fast(root):
    manifest(root, fast=True)


if __name__ == "__main__":
    manifest(sys.argv[1] if len(sys.argv) > 1 else ".")
"""


# Slim per-call import: agent is deployed once, later calls just import it.
# No per-call package send. A VER guard reloads once per kernel if a stale
# in-memory module is cached (e.g. kernel survived an agent upgrade).
def _h_import() -> str:
    # Slim per-call import with a stale-module guard: normally just imports
    # the cached module (fast); reloads once per kernel after an upgrade.
    return ("import sys, os, importlib; sys.path.insert(0, os.path.expanduser('~/.a100-agent')); "
            "import a100h; "
            f"importlib.reload(a100h) if getattr(a100h, 'VER', None) != {AGENT_VERSION!r} else None\n")


def _h(body: str) -> str:
    return _h_import() + body


MANIFEST_CACHE = Path.home() / ".cache" / "a100_manifest.json"


def _load_manifest_cache() -> dict:
    try:
        return json.loads(MANIFEST_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_manifest_cache(cache: dict) -> None:
    try:
        MANIFEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass


def load_token(sysinfo: Path) -> str:
    token = os.environ.get(TOKEN_ENV)
    if token:
        return token
    match = re.search(
        r"^JUPYTERHUB_API_TOKEN=(\S+)\s*$",
        sysinfo.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"JUPYTERHUB_API_TOKEN was not found in {sysinfo}")
    return match.group(1)


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"sessions": {}, "cwd": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JupyterClient:
    RETRIES = 5

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.token = token

    def request(self, method: str, endpoint: str, payload: dict | None = None):
        url = urljoin(self.base_url, endpoint)
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"token {self.token}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(self.RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                raise RuntimeError(
                    f"Jupyter API returned HTTP {exc.code}: {detail[:300]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt == self.RETRIES - 1:
                    raise RuntimeError(
                        f"Could not reach {url} after {self.RETRIES} attempts: {exc.reason}"
                    ) from exc
                time.sleep(2**attempt)
        return json.loads(body) if body else None

    # -- kernels ---------------------------------------------------------
    def list_kernels(self) -> list:
        return self.request("GET", "api/kernels") or []

    def create_kernel(self, kernel_name: str = "python3") -> dict:
        return self.request("POST", "api/kernels", {"name": kernel_name})

    def delete_kernel(self, kernel_id: str) -> None:
        self.request("DELETE", f"api/kernels/{kernel_id}")

    def kernel_alive(self, kernel_id: str) -> bool:
        try:
            self.request("GET", f"api/kernels/{kernel_id}")
            return True
        except RuntimeError:
            return False

    def session_kernel(self, session: str, kernel_name: str = "python3") -> str:
        state = load_state()
        kernel_id = state.get("sessions", {}).get(session)
        if kernel_id and self.kernel_alive(kernel_id):
            return kernel_id
        kernel = self.create_kernel(kernel_name)
        kernel_id = kernel["id"]
        state.setdefault("sessions", {})[session] = kernel_id
        save_state(state)
        return kernel_id

    def forget_session(self, session: str) -> None:
        state = load_state()
        state.get("sessions", {}).pop(session, None)
        state.get("cwd", {}).pop(session, None)
        save_state(state)

    def session_cwd(self, session: str | None) -> str | None:
        if not session:
            return None
        return load_state().get("cwd", {}).get(session)

    def set_session_cwd(self, session: str, cwd: str) -> None:
        state = load_state()
        state.setdefault("cwd", {})[session] = cwd
        save_state(state)

    # -- contents API (small files) --------------------------------------
    @staticmethod
    def _check_relative(remote_path: str) -> str:
        path = Path(remote_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise RuntimeError("remote path must be a relative project path")
        return path.as_posix()

    def download_small(self, remote_path: str, output: Path) -> None:
        rel = self._check_relative(remote_path)
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"refusing to overwrite local output: {output}")
        payload = self.request(
            "GET", "api/contents/" + quote(rel, safe="/") + "?format=base64&type=file")
        if (
            not isinstance(payload, dict)
            or payload.get("type") != "file"
            or payload.get("format") != "base64"
        ):
            raise RuntimeError("remote path is not a downloadable base64 file")
        content = payload.get("content")
        if not isinstance(content, str):
            raise RuntimeError("remote file has no content")
        try:
            data = base64.b64decode("".join(content.split()), validate=True)
        except ValueError as error:
            raise RuntimeError("remote file content is not valid base64") from error
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            with output.open("xb") as handle:
                handle.write(data)
        except OSError as error:
            raise RuntimeError(f"cannot write local output: {output}") from error

    def upload_small(self, local: Path, remote_path: str) -> None:
        rel = self._check_relative(remote_path)
        data = local.read_bytes()
        if len(data) > SMALL_FILE_LIMIT:
            raise RuntimeError("file too large for single-shot upload")
        content = base64.b64encode(data).decode()
        parent = str(Path(rel).parent)
        if parent != ".":
            # create missing levels one at a time; tolerate "already exists"
            prefix = Path()
            for part in Path(parent).parts:
                prefix = prefix / part
                try:
                    self.request("PUT", "api/contents/" + quote(prefix.as_posix(), safe="/"),
                                 {"type": "directory"})
                except RuntimeError as exc:
                    if "HTTP 400" not in str(exc) and "exists" not in str(exc).lower():
                        raise
        self.request(
            "PUT", "api/contents/" + quote(rel, safe="/"),
            {"type": "file", "format": "base64", "content": content},
        )

    # -- kernel execution -------------------------------------------------
    def _open_channel(self, kernel_id: str):
        parsed = urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        session_id = uuid.uuid4().hex
        ws_url = (
            f"{ws_scheme}://{parsed.netloc}{parsed.path}"
            f"api/kernels/{kernel_id}/channels?session_id={session_id}"
        )
        origin = f"{parsed.scheme}://{parsed.netloc}"
        last_error: Exception | None = None
        for attempt in range(self.RETRIES):
            try:
                ws = websocket.create_connection(
                    ws_url,
                    header=[f"Authorization: token {self.token}"],
                    origin=origin,
                    sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                    timeout=30,
                )
                return ws, session_id
            except (websocket.WebSocketAddressException, OSError) as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(
            f"Could not open the Jupyter WebSocket after {self.RETRIES} attempts: {last_error}"
        )

    def run(self, kernel_id: str, command: str, timeout: float = 120.0,
            cwd: str | None = None, quiet: bool = False) -> tuple[int, str]:
        """Run one shell command on a kernel. Returns (exit_code, output)."""
        ws, session_id = self._open_channel(kernel_id)
        message_id = uuid.uuid4().hex
        prelude = ""
        if cwd:
            prelude = f"import os as _os\n_os.chdir({cwd!r})\n"
        code = (
            f"{prelude}import subprocess\n"
            f"_p = subprocess.run({command!r}, shell=True, text=True, "
            "stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n"
            "print(_p.stdout, end='')\n"
            "print(f'\\n[remote exit code: {_p.returncode}]')\n"
        )
        message = {
            "header": {
                "msg_id": message_id,
                "username": "codex",
                "session": session_id,
                "date": dt.datetime.now(dt.timezone.utc).isoformat(),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {"exit_code": "_p.returncode"},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
            "buffers": [],
        }
        exit_code: int | None = None
        idle = False
        pieces: list[str] = []
        deadline = time.monotonic() + timeout
        try:
            ws.send(json.dumps(message))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"timed out after {timeout:g}s waiting for remote command; "
                        "use bg for long-running work"
                    )
                ws.settimeout(remaining)
                try:
                    reply = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException as exc:
                    raise RuntimeError(
                        f"timed out after {timeout:g}s waiting for remote command; "
                        "use bg for long-running work"
                    ) from exc
                if reply.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                msg_type = reply.get("msg_type") or reply.get("header", {}).get("msg_type")
                content = reply.get("content", {})
                if msg_type == "stream":
                    text = content.get("text", "")
                    pieces.append(text)
                    if not quiet:
                        print(text, end="", flush=True)
                elif msg_type == "error":
                    traceback = "\n".join(content.get("traceback", []))
                    pieces.append(traceback)
                    if not quiet:
                        print(traceback, file=sys.stderr)
                elif msg_type == "execute_reply":
                    expression = content.get("user_expressions", {}).get("exit_code", {})
                    value = expression.get("data", {}).get("text/plain")
                    exit_code = int(value) if value is not None else 1
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    idle = True
                if idle and exit_code is not None:
                    break
        finally:
            ws.close()
        output = EXIT_MARKER_RE.sub("", "".join(pieces))
        return exit_code if exit_code is not None else 1, output

    def run_python(self, kernel_id: str, snippet: str, timeout: float = 120.0,
                   quiet: bool = True) -> tuple[int, str]:
        """Run a raw python snippet on a kernel and capture its stdout."""
        ws, session_id = self._open_channel(kernel_id)
        message_id = uuid.uuid4().hex
        message = {
            "header": {
                "msg_id": message_id,
                "username": "codex",
                "session": session_id,
                "date": dt.datetime.now(dt.timezone.utc).isoformat(),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": snippet,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
            "buffers": [],
        }
        idle = False
        replied = False
        ok = False
        pieces: list[str] = []
        deadline = time.monotonic() + timeout
        try:
            ws.send(json.dumps(message))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("timed out waiting for remote python snippet")
                ws.settimeout(remaining)
                try:
                    reply = json.loads(ws.recv())
                except websocket.WebSocketTimeoutException as exc:
                    raise RuntimeError("timed out waiting for remote python snippet") from exc
                if reply.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                msg_type = reply.get("msg_type") or reply.get("header", {}).get("msg_type")
                content = reply.get("content", {})
                if msg_type == "stream":
                    pieces.append(content.get("text", ""))
                elif msg_type == "error":
                    pieces.append("\n".join(content.get("traceback", [])))
                elif msg_type == "execute_reply":
                    replied = True
                    ok = content.get("status") == "ok"
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    idle = True
                if idle and replied:
                    break
        finally:
            ws.close()
        if not ok:
            raise RuntimeError("remote python snippet failed: " + "".join(pieces)[-2000:])
        return 0, "".join(pieces)

    # -- exec with sticky cwd ---------------------------------------------
    def exec_command(self, command: str, session: str | None, kernel_name: str,
                     timeout: float, cwd: str | None, quiet: bool,
                     ephemeral: bool = False) -> int:
        match = CD_RE.match(command.strip()) if session and not ephemeral else None
        if match is not None:
            target = (match.group(1) or "~").strip().strip("'\"")
            base = cwd or self.session_cwd(session) or "~"
            kernel_id = self.session_kernel(session, kernel_name)
            code, out = self.run(
                kernel_id, f"cd {base} >/dev/null 2>&1; cd {target} >/dev/null 2>&1; pwd -P",
                timeout=timeout, quiet=True,
            )
            if code != 0:
                print(f"cd: no such directory: {target}", file=sys.stderr)
                return code
            new_cwd = out.strip().splitlines()[-1].strip()
            self.set_session_cwd(session, new_cwd)
            if not quiet:
                print(new_cwd)
            return 0
        if session and not ephemeral:
            kernel_id = self.session_kernel(session, kernel_name)
            use_cwd = cwd or self.session_cwd(session)
            code, _ = self.run(kernel_id, command, timeout=timeout, cwd=use_cwd, quiet=quiet)
            return code
        kernel = self.create_kernel(kernel_name)
        try:
            code, _ = self.run(kernel["id"], command, timeout=timeout,
                               cwd=cwd, quiet=quiet)
            return code
        finally:
            self.delete_kernel(kernel["id"])

    # -- chunked transfer --------------------------------------------------
    def _transfer_kernel(self, kernel_name: str) -> str:
        return self.session_kernel("__transfer", kernel_name)

    def push_file(self, local: Path, remote_path: str, kernel_name: str) -> None:
        rel = self._check_relative(remote_path)
        size = local.stat().st_size
        if size <= SMALL_FILE_LIMIT:
            self.upload_small(local, rel)
            print(f"push {local} -> {rel} ({size} bytes, contents-api)")
            return
        # Large files go through the resumable agent-based uploader (deployed
        # once via bootstrap, reused every call -- no per-call package send).
        self.ensure_agent(kernel_name)
        kernel_id = self._transfer_kernel(kernel_name)
        before = time.monotonic()
        self._sync_upload_one(self, kernel_id, local, rel, size)
        el = time.monotonic() - before
        print(f"push {local} -> {rel} ({size} bytes, chunked resume sha256 ok, {el:.0f}s)")

    def pull_file(self, remote_path: str, local: Path, kernel_name: str) -> None:
        rel = self._check_relative(remote_path)
        kernel_id = self._transfer_kernel(kernel_name)
        code, out = self.run(
            kernel_id, f"stat -c %s {rel} && sha256sum {rel} | cut -d' ' -f1",
            timeout=60.0, quiet=True,
        )
        if code != 0:
            # fall back to small-file contents API (better error message)
            self.download_small(rel, local)
            return
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        size, want = int(lines[0]), lines[1]
        if size <= SMALL_FILE_LIMIT:
            self.download_small(rel, local)
            print(f"pull {rel} -> {local} ({size} bytes, contents-api)")
            return
        if local.exists() or local.is_symlink():
            raise RuntimeError(f"refusing to overwrite local output: {local}")
        local.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.ensure_agent(kernel_name)
        digest = hashlib.sha256()
        received = 0
        index = 0
        with local.open("xb") as handle:
            while received < size:
                want_n = min(CHUNK_RAW, size - received)
                snippet = (
                    _h(f"import a100h; print(a100h.rp({rel!r}, {received}, {want_n}))\n")
                )
                for attempt in range(5):
                    try:
                        _, text = self.run_python(kernel_id, snippet, timeout=300.0)
                        break
                    except (RuntimeError, OSError):
                        if attempt == 4:
                            raise
                        time.sleep(2 ** attempt)
                raw = base64.b64decode("".join(text.split()), validate=True)
                handle.write(raw)
                digest.update(raw)
                received += len(raw)
                index += 1
                print(f"\rpull chunk {index} ({Path(rel).name})", end="", flush=True)
                if not raw:
                    break
        print()
        if digest.hexdigest() != want:
            raise RuntimeError(f"pull checksum mismatch for {rel}")
        print(f"pull {rel} -> {local} ({size} bytes, chunked sha256 ok)")

    def push(self, local: Path, remote: str, kernel_name: str, workers: int = 8) -> None:
        if local.is_dir():
            # Incremental: only changed files, parallel workers, resumable.
            rc = self.sync_push(local, remote, kernel_name, workers=workers)
            if rc != 0:
                raise RuntimeError("push dir failed (see FAILED lines above)")
            return
        dest = remote
        if dest.endswith("/"):
            dest = dest.rstrip("/") + "/" + local.name
        self.push_file(local, dest, kernel_name)

    def pull(self, remote: str, local: Path, kernel_name: str, workers: int = 8) -> None:
        rel = self._check_relative(remote)
        kernel_id = self._transfer_kernel(kernel_name)
        code, out = self.run(
            kernel_id, f"test -d {rel} && echo DIR || echo FILE",
            timeout=60.0, quiet=True,
        )
        is_dir = out.strip().splitlines()[-1].strip() == "DIR" if code == 0 else False
        if not is_dir:
            self.pull_file(remote, local, kernel_name)
            return
        if local.exists() and not local.is_dir():
            raise RuntimeError(f"refusing to overwrite local output: {local}")
        # Incremental: only changed files, parallel workers.
        rc = self.sync_pull(remote, local, kernel_name, workers=workers)
        if rc != 0:
            raise RuntimeError("pull dir failed (see FAILED lines above)")

    # -- persistent agent (deployed once, reused by every call) -----------
    def ensure_agent(self, kernel_name: str, force: bool = False) -> bool:
        """Deploy ~/.a100-agent/a100h.py once; later calls just import it."""
        state = load_state()
        if not force and state.get("agent_ver") == AGENT_VERSION:
            return True
        # Version check must bypass the kernel's cached a100h module
        # (slim `import a100h` elsewhere reuses the cached module, which is
        # what makes v6 fast -- but upgrades need a reload to see the new VER).
        RELOAD_CHECK = (
            "import sys, os, importlib; sys.path.insert(0, os.path.expanduser('~/.a100-agent')); "
            "importlib.invalidate_caches(); import a100h; importlib.reload(a100h)\n"
            "import a100h; print('agent-ver:' + a100h.VER)\n"
        )
        kernel_id = self._transfer_kernel(kernel_name)
        try:
            _, out = self.run_python(kernel_id, RELOAD_CHECK, timeout=60.0)
            if f"agent-ver:{AGENT_VERSION}" in out:
                state["agent_ver"] = AGENT_VERSION
                save_state(state)
                return True
        except RuntimeError:
            pass
        raw = AGENT_CODE.encode()
        self.run_python(
            kernel_id,
            "import os; os.makedirs(os.path.expanduser('~/.a100-agent'), exist_ok=True)\n"
            "open(os.path.expanduser('~/.a100-agent/a100h.py'), 'wb').write("
            + repr(raw) + ")\nprint('agent-deployed')\n",
            timeout=120.0,
        )
        self.run_python(
            kernel_id,
            "import shutil, os; shutil.rmtree(os.path.expanduser('~/.a100-agent/__pycache__'), ignore_errors=True)\nprint('pycache-cleared')\n",
            timeout=60.0,
        )
        _, out = self.run_python(kernel_id, RELOAD_CHECK, timeout=60.0)
        if f"agent-ver:{AGENT_VERSION}" not in out:
            raise RuntimeError("agent bootstrap verification failed")
        state["agent_ver"] = AGENT_VERSION
        save_state(state)
        # Drop sync kernels so they re-import the new agent (no stale cache).
        try:
            for i in range(8):
                kid = load_state().get("sessions", {}).get(f"__sync-{i}")
                if kid:
                    try:
                        self.delete_kernel(kid)
                    except RuntimeError:
                        pass
                    self.forget_session(f"__sync-{i}")
        except RuntimeError:
            pass
        print(f"agent v{AGENT_VERSION} deployed to ~/.a100-agent/")
        return True

    @staticmethod
    def _wp_snippet(remote_tmp: str, b64: str, append: bool) -> str:
        mode = "True" if append else "False"
        return _h(
            "try:\n"
            "    import a100h as _H\n"
            f"    _H.wp({remote_tmp!r}, {b64!r}, {mode})\n"
            "except Exception:\n"
            "    import base64\n"
            f"    open({remote_tmp!r}, {repr('ab' if append else 'wb')}).write(base64.b64decode({b64!r}))\n"
            "print('chunk-ok')\n"
        )

    # -- incremental sync (only changed files cross the wire) -------------
    @staticmethod
    def _sync_keep(rel: str, excludes: list) -> bool:
        base = rel.rsplit("/", 1)[-1]
        if base.endswith(".part") or ".part-" in base:
            return False
        rel = rel.replace(os.sep, "/")
        for pat in excludes or []:
            pat = pat.strip().strip("/")
            if not pat:
                continue
            if "/" not in pat and "*" not in pat and "?" not in pat and "[" not in pat:
                if rel == pat or rel.startswith(pat + "/"):
                    return False
                continue
            if fnmatch.fnmatch(rel, pat):
                return False
            parts = rel.split("/")
            for i in range(1, len(parts)):
                if fnmatch.fnmatch("/".join(parts[:i]), pat):
                    return False
        return True

    @staticmethod
    def _local_manifest(root: Path, excludes: list, fast: bool = True) -> dict:
        items: list[tuple[Path, str]] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in filenames:
                full = Path(dirpath) / name
                rel = full.relative_to(root).as_posix()
                if JupyterClient._sync_keep(rel, excludes):
                    items.append((full, rel))
        if fast:
            # fast: size+mtime only, no hashing -> large trees stay quick.
            # sha="" matches remote manifest_fast; equality is size+mtime.
            out: dict = {}
            for full, rel in items:
                try:
                    stat = full.stat()
                except OSError:
                    continue
                out[rel] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha": ""}
            return out

        cache = _load_manifest_cache()
        root_key = str(root.resolve())

        def _one(item: tuple[Path, str]):
            full, rel = item
            stat = full.stat()
            key = f"{root_key}\x00{rel}"
            hit = cache.get(key)
            if hit and hit.get("size") == stat.st_size and int(hit.get("mtime", 0)) == int(stat.st_mtime):
                return rel, {"size": stat.st_size, "mtime": stat.st_mtime, "sha": hit["sha"]}
            return rel, {"size": stat.st_size, "mtime": stat.st_mtime,
                         "sha": sha256_file(full)}

        out = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for rel, meta in pool.map(_one, items):
                out[rel] = meta
                cache[f"{root_key}\x00{rel}"] = meta
        _save_manifest_cache(cache)
        return out

    def _remote_manifest(self, remote: str, kernel_id: str, kernel_name: str = "python3", fast: bool = True) -> dict:
        self.ensure_agent(kernel_name)
        call = "a100h.manifest_fast(" if fast else "a100h.manifest("
        _, out = self.run_python(
            kernel_id,
            _h("import a100h; " + call + repr(remote) + ")\n"),
            timeout=600.0,
        )
        start = out.find("[")
        if start < 0:
            raise RuntimeError("remote manifest returned no JSON: " + out[-300:])
        return {item["rel"]: item for item in json.loads(out[start:])}

    @staticmethod
    def _touch_remote(client: "JupyterClient", kernel_id: str, dest: str, mtime: float) -> None:
        try:
            client.run(kernel_id, f"touch -d @{int(mtime)} {dest}", timeout=60.0, quiet=True)
        except RuntimeError:
            pass

    def _sync_upload_one(self, client: "JupyterClient", kernel_id: str,
                         local: Path, dest: str, size: int) -> None:
        mtime = local.stat().st_mtime
        if size <= SMALL_FILE_LIMIT:
            client.upload_small(local, dest)
            self._touch_remote(client, kernel_id, dest, mtime)
            return
        tmp = dest + ".part"
        have = 0
        try:
            _, out = client.run(
                kernel_id, f"stat -c %s {tmp} 2>/dev/null || echo 0",
                timeout=60.0, quiet=True)
            have = int(out.strip().splitlines()[-1].strip() or 0)
        except (RuntimeError, ValueError, IndexError):
            have = 0
        if have > size:
            have = 0
        digest = hashlib.sha256()
        resume = False
        if have and have < size:
            _, hexd = client.run_python(
                kernel_id,
                _h("import a100h; print(a100h.sha(" + repr(tmp) + "))\n"),
                timeout=120.0)
            # verify prefix matches local prefix before resuming (streamed:
            # never hold multi-GB weights fully in memory)
            ph = hashlib.sha256()
            with local.open("rb") as handle:
                left = have
                while left > 0:
                    blk = handle.read(min(4 * 1024 * 1024, left))
                    if not blk:
                        break
                    ph.update(blk)
                    left -= len(blk)
            if ph.hexdigest() == hexd.strip().splitlines()[-1].strip():
                digest = ph
                resume = True
            else:
                have = 0
        if have == size and have > 0:
            # .part already full-size (e.g. chunks all landed, fin never ran):
            # verify it matches the local file, then finalize without re-upload.
            local_sha = sha256_file(local)
            _, hexd = client.run_python(
                kernel_id,
                _h("import a100h; print(a100h.sha(" + repr(tmp) + "))\n"),
                timeout=300.0)
            if hexd.strip().splitlines()[-1].strip() == local_sha:
                _, out = client.run_python(
                    kernel_id,
                    _h("import a100h\n"
                    f"print(a100h.fin({tmp!r}, {dest!r}, {local_sha!r}))\n"),
                    timeout=300.0)
                got = out.strip().splitlines()[-1].strip().replace("sha:", "")
                if got != local_sha:
                    raise RuntimeError(f"checksum mismatch for {dest}")
                self._touch_remote(client, kernel_id, dest, local.stat().st_mtime)
                return
            have = 0
        if have == 0:
            # fresh start: drop any stale .part so chunk 0 truncates.
            client.run(kernel_id, f"rm -f {tmp}", timeout=60.0, quiet=True)
            resume = False
        first = not resume
        sent = 0
        start = time.monotonic()
        with local.open("rb") as handle:
            handle.seek(have)
            while True:
                raw = handle.read(CHUNK_RAW)
                if not raw:
                    break
                digest.update(raw)
                b64 = base64.b64encode(raw).decode()
                for attempt in range(5):
                    try:
                        client.run_python(
                            kernel_id, self._wp_snippet(tmp, b64, not first),
                            timeout=300.0)
                        break
                    except (RuntimeError, OSError):
                        if attempt == 4:
                            raise
                        time.sleep(2 ** attempt)
                first = False
                sent += len(raw)
                el = time.monotonic() - start
                if sent >= CHUNK_RAW and el > 5:
                    rate = (have + sent) / el / 1e6
                    print(f"\r  ... {Path(dest).name}: {(have + sent)/1e6:.0f}/{size/1e6:.0f} MB ({rate:.1f} MB/s)",
                          end="", flush=True)
        want = digest.hexdigest()
        _, out = client.run_python(
            kernel_id,
            _h("import a100h\n"
            f"print(a100h.fin({tmp!r}, {dest!r}, {want!r}))\n"),
            timeout=300.0)
        got = out.strip().splitlines()[-1].strip().replace("sha:", "")
        if got != want:
            raise RuntimeError(
                f"checksum mismatch for {dest}: local={want} remote={got} "
                "(stale .part kept for resume; dest left untouched)")
        self._touch_remote(client, kernel_id, dest, local.stat().st_mtime)

    @staticmethod
    def _effective_excludes(excludes: list | None, use_defaults: bool = True) -> list:
        out = list(DEFAULT_EXCLUDES) if use_defaults else []
        out.extend(excludes or [])
        return out

    def sync_push(self, local: Path, remote: str, kernel_name: str,
                  delete: bool = False, excludes: list | None = None,
                  workers: int = 8, dry_run: bool = False, fast: bool = True,
                  use_defaults: bool = True) -> int:
        excludes = self._effective_excludes(excludes, use_defaults)
        if not local.is_dir():
            raise RuntimeError("sync-push needs a local directory")
        local_man = self._local_manifest(local, excludes, fast=fast)
        kid0 = self._transfer_kernel(kernel_name)
        self.ensure_agent(kernel_name)
        self.run(kid0, f"mkdir -p {remote}", timeout=60.0, quiet=True)
        remote_man = self._remote_manifest(remote, kid0, kernel_name, fast=fast)
        remote_man = {r: m for r, m in remote_man.items()
                      if self._sync_keep(r, excludes)}
        if fast:
            # size+mtime only: skips re-hashing multi-GB weights on both ends.
            uploads = sorted(
                rel for rel, m in local_man.items()
                if rel not in remote_man
                or remote_man[rel].get("size") != m["size"]
                or int(remote_man[rel].get("mtime", 0)) != int(m["mtime"]))
        else:
            uploads = sorted(rel for rel, m in local_man.items()
                             if remote_man.get(rel, {}).get("sha") != m["sha"])
        deletions = sorted(set(remote_man) - set(local_man)) if delete else []
        total_bytes = sum(local_man[r]["size"] for r in uploads)
        print(f"sync-push: {len(local_man)} local, {len(remote_man)} remote, "
              f"{len(uploads)} to upload ({total_bytes/1e6:.1f} MB), "
              f"{len(deletions)} to delete")
        if dry_run:
            for rel in uploads:
                print(f"  + {rel}")
            for rel in deletions:
                print(f"  - {rel}")
            return 0
        if deletions:
            self.run_python(
                kid0,
                "import os\n" + "".join(
                    f"p = os.path.join({remote!r}, {rel!r})\n"
                    "os.remove(p) if os.path.isfile(p) else None\n"
                    for rel in deletions),
                timeout=300.0)
            print(f"deleted {len(deletions)} remote files")
        if not uploads:
            print("sync-push: already up to date")
            return 0
        kids: list[str] = []
        base = JupyterClient(self.base_url, self.token)
        for i in range(min(workers, len(uploads))):
            kids.append(base.session_kernel(f"__sync-{i}", kernel_name))
        lock = threading.Lock()
        done = {"n": 0, "bytes": 0}
        failed: list[str] = []
        start = time.monotonic()

        def _task(idx: int, rel: str) -> None:
            worker = JupyterClient(self.base_url, self.token)
            try:
                self._sync_upload_one(
                    worker, kids[idx % len(kids)], local / rel,
                    remote.rstrip("/") + "/" + rel, local_man[rel]["size"])
                with lock:
                    done["n"] += 1
                    done["bytes"] += local_man[rel]["size"]
                    el = time.monotonic() - start
                    rate = done["bytes"] / el / 1e6 if el > 0 else 0
                    print(f"[{done['n']}/{len(uploads)}] UP {rel} "
                          f"({rate:.1f} MB/s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    failed.append(f"{rel}: {exc}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(kids)) as pool:
            list(pool.map(_task, range(len(uploads)), uploads))
        el = time.monotonic() - start
        print(f"sync-push done: {done['n']}/{len(uploads)} files, "
              f"{done['bytes']/1e6:.1f} MB in {el:.0f}s")
        if failed:
            print("FAILED:", *failed[:10], sep="\n  ")
            return 1
        return 0

    def _sync_download_one(self, client: "JupyterClient", kernel_id: str,
                           rel: str, remote_base: str, meta: dict,
                           dest: Path) -> None:
        src = remote_base.rstrip("/") + "/" + rel
        size = meta["size"]
        if size <= SMALL_FILE_LIMIT:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            client.download_small(src, dest)
        else:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            digest = hashlib.sha256()
            received = 0
            with dest.open("xb") as handle:
                while received < size:
                    want_n = min(CHUNK_RAW, size - received)
                    _, text = client.run_python(
                        kernel_id,
                        _h("import a100h; print(a100h.rp(" + repr(src) + f", {received}, {want_n}))\n"),
                        timeout=300.0)
                    raw = base64.b64decode("".join(text.split()), validate=True)
                    if not raw:
                        break
                    handle.write(raw)
                    digest.update(raw)
                    received += len(raw)
            if meta.get("sha") and digest.hexdigest() != meta["sha"]:
                raise RuntimeError(f"checksum mismatch for {rel}")
        try:
            os.utime(dest, (meta["mtime"], meta["mtime"]))
        except OSError:
            pass

    def sync_pull(self, remote: str, local: Path, kernel_name: str,
                  delete: bool = False, excludes: list | None = None,
                  workers: int = 8, dry_run: bool = False, fast: bool = True,
                  use_defaults: bool = True) -> int:
        excludes = self._effective_excludes(excludes, use_defaults)
        kid0 = self._transfer_kernel(kernel_name)
        self.ensure_agent(kernel_name)
        remote_man = {r: m for r, m in
                      self._remote_manifest(remote, kid0, kernel_name, fast=fast).items()
                      if self._sync_keep(r, excludes)}
        local_man = self._local_manifest(local, excludes, fast=fast) if local.is_dir() else {}
        if fast:
            downloads = sorted(
                rel for rel, m in remote_man.items()
                if rel not in local_man
                or local_man[rel].get("size") != m["size"]
                or int(local_man[rel].get("mtime", 0)) != int(m["mtime"]))
        else:
            downloads = sorted(rel for rel, m in remote_man.items()
                               if local_man.get(rel, {}).get("sha") != m["sha"])
        extra = sorted(set(local_man) - set(remote_man)) if delete else []
        total_bytes = sum(remote_man[r]["size"] for r in downloads)
        print(f"sync-pull: {len(remote_man)} remote, {len(local_man)} local, "
              f"{len(downloads)} to download ({total_bytes/1e6:.1f} MB), "
              f"{len(extra)} extra")
        if dry_run:
            for rel in downloads:
                print(f"  + {rel}")
            for rel in extra:
                print(f"  - {rel}")
            return 0
        for rel in extra:
            victim = local / rel
            if victim.is_file() or victim.is_symlink():
                victim.unlink()
        if not downloads:
            print("sync-pull: already up to date")
            return 0
        local.mkdir(mode=0o700, parents=True, exist_ok=True)
        kids: list[str] = []
        base = JupyterClient(self.base_url, self.token)
        for i in range(min(workers, len(downloads))):
            kids.append(base.session_kernel(f"__sync-{i}", kernel_name))
        lock = threading.Lock()
        done = {"n": 0, "bytes": 0}
        failed: list[str] = []
        start = time.monotonic()

        def _task(idx: int, rel: str) -> None:
            worker = JupyterClient(self.base_url, self.token)
            try:
                self._sync_download_one(
                    worker, kids[idx % len(kids)], rel, remote,
                    remote_man[rel], local / rel)
                with lock:
                    done["n"] += 1
                    done["bytes"] += remote_man[rel]["size"]
                    el = time.monotonic() - start
                    rate = done["bytes"] / el / 1e6 if el > 0 else 0
                    print(f"[{done['n']}/{len(downloads)}] DOWN {rel} "
                          f"({rate:.1f} MB/s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    failed.append(f"{rel}: {exc}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(kids)) as pool:
            list(pool.map(_task, range(len(downloads)), downloads))
        el = time.monotonic() - start
        print(f"sync-pull done: {done['n']}/{len(downloads)} files, "
              f"{done['bytes']/1e6:.1f} MB in {el:.0f}s")
        if failed:
            print("FAILED:", *failed[:10], sep="\n  ")
            return 1
        return 0

    def tunnel(self, port: int, job: str, kernel_name: str) -> str:
        kid = self._transfer_kernel(kernel_name)
        _, out = self.run(
            kid, "test -x $HOME/.a100-agent/cloudflared && echo HAVE_BIN || echo NO_BIN",
            timeout=60.0, quiet=True)
        if "HAVE_BIN" not in out:
            print("downloading cloudflared (one time)...")
            code, _ = self.run(
                kid,
                "mkdir -p $HOME/.a100-agent && "
                "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
                "-o $HOME/.a100-agent/cloudflared && chmod +x $HOME/.a100-agent/cloudflared && echo BIN_OK",
                timeout=600.0)
            if code != 0:
                raise RuntimeError("cloudflared download failed")
        self.bg_start(job, f"$HOME/.a100-agent/cloudflared tunnel --url http://127.0.0.1:{port}",
                      kernel_name, None)
        url = ""
        for _ in range(18):
            time.sleep(5)
            _, out = self.run(
                kid, f"grep -Eo 'https://[A-Za-z0-9.-]+[.]trycloudflare[.]com' $HOME/.a100-jobs/{job}/job.log 2>/dev/null | tail -1",
                timeout=60.0, quiet=True)
            cand = out.strip().splitlines()
            if cand and cand[-1].strip():
                url = cand[-1].strip()
                break
        if not url:
            raise RuntimeError(f"tunnel {job} produced no URL; check: log {job}")
        print(f"tunnel {job} -> {url}")
        return url

    # -- background jobs ----------------------------------------------------
    @staticmethod
    def _job_dir(job: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job):
            raise RuntimeError(f"bad job name: {job}")
        return f"$HOME/.a100-jobs/{job}"

    def bg_start(self, job: str, command: str, kernel_name: str, cwd: str | None) -> None:
        jobdir = self._job_dir(job)
        kernel_id = self._transfer_kernel(kernel_name)
        snippet = (
            "import os, subprocess\n"
            f"_jd = os.path.expandvars({jobdir!r})\n"
            "os.makedirs(_jd, exist_ok=True)\n"
            "_log = open(os.path.join(_jd, 'job.log'), 'ab', buffering=0)\n"
            "_log.write(b'\\n===== cmd: ' + " + repr(command.encode()) + " + b'\\n')\n"
            "_p = subprocess.Popen(['bash', '-c', " + repr(command) + "], "
            "stdout=_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, "
            "start_new_session=True, cwd=os.path.expandvars(" + repr(cwd or "$HOME") + "))\n"
            "open(os.path.join(_jd, 'job.pid'), 'w').write(str(_p.pid))\n"
            "print(_p.pid)\n"
        )
        _, out = self.run_python(kernel_id, snippet, timeout=60.0)
        print(f"bg job {job} started pid={out.strip()} log=~/.a100-jobs/{job}/job.log")

    def bg_poll(self, job: str, kernel_name: str) -> int:
        jobdir = self._job_dir(job)
        kernel_id = self._transfer_kernel(kernel_name)
        code, out = self.run(
            kernel_id,
            f"_pid=$(cat {jobdir}/job.pid 2>/dev/null || echo); "
            f"if [ -n \"$_pid\" ] && kill -0 $_pid 2>/dev/null; then echo RUNNING; "
            f"else echo DONE; fi; tail -n 20 {jobdir}/job.log 2>/dev/null",
            timeout=60.0,
        )
        return code

    def bg_log(self, job: str, tail: int, kernel_name: str, follow: bool = False,
             interval: float = 2.0) -> int:
        jobdir = self._job_dir(job)
        kernel_id = self._transfer_kernel(kernel_name)
        code, _ = self.run(kernel_id, f"tail -n {tail} {jobdir}/job.log", timeout=60.0)
        if not follow:
            return code
        offset = -1
        try:
            _, out = self.run(
                kernel_id, f"stat -c %s {jobdir}/job.log 2>/dev/null || echo 0",
                timeout=60.0, quiet=True)
            offset = int(out.strip().splitlines()[-1].strip() or 0)
        except (RuntimeError, ValueError, IndexError):
            offset = 0
        quiet_rounds = 0
        while True:
            code, out = self.run(
                kernel_id,
                f"_sz=$(stat -c %s {jobdir}/job.log 2>/dev/null || echo 0); "
                f"echo SIZE:$_sz; "
                f"_pid=$(cat {jobdir}/job.pid 2>/dev/null || echo); "
                f'if [ -n "$_pid" ] && kill -0 $_pid 2>/dev/null; then echo STATE:RUNNING; '
                f"else echo STATE:DONE; fi",
                timeout=60.0, quiet=True)
            size, state = offset, "DONE"
            for line in out.splitlines():
                if line.startswith("SIZE:"):
                    try:
                        size = int(line[5:].strip())
                    except ValueError:
                        pass
                elif line.startswith("STATE:"):
                    state = line[6:].strip()
            if size > offset:
                _, chunk = self.run(
                    kernel_id,
                    f"tail -c +{offset + 1} {jobdir}/job.log",
                    timeout=60.0, quiet=True)
                print(chunk, end="", flush=True)
                offset = size
                quiet_rounds = 0
            else:
                quiet_rounds += 1
            if state == "DONE" and quiet_rounds >= 2:
                print(f"\n[job {job} finished]")
                return 0
            time.sleep(max(0.5, interval))

    def bg_kill(self, job: str, kernel_name: str) -> int:
        jobdir = self._job_dir(job)
        kernel_id = self._transfer_kernel(kernel_name)
        code, _ = self.run(
            kernel_id,
            f"_pid=$(cat {jobdir}/job.pid 2>/dev/null || echo); "
            f"[ -n \"$_pid\" ] && kill -- -$_pid 2>/dev/null; "
            f"[ -n \"$_pid\" ] && kill $_pid 2>/dev/null; echo killed",
            timeout=60.0,
        )
        return code

    def bg_jobs(self, kernel_name: str) -> int:
        kernel_id = self._transfer_kernel(kernel_name)
        code, _ = self.run(
            kernel_id,
            "for d in $HOME/.a100-jobs/*/; do "
            "[ -d \"$d\" ] || continue; "
            "j=$(basename \"$d\"); "
            "pid=$(cat \"$d/job.pid\" 2>/dev/null || echo); "
            'if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then st=RUNNING; else st=DONE; fi; '
            'sz=$(stat -c %s \"$d/job.log\" 2>/dev/null || echo 0); '
            'echo \"$st $j pid=$pid log_bytes=$sz\"; '
            "done",
            timeout=60.0,
        )
        return code

    def vram(self, kernel_name: str) -> int:
        return self.exec_command(
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv; "
            "echo '--- server ---'; "
            "(curl -s -m 5 http://127.0.0.1:8899/health || echo NO_SERVER)",
            session=None, kernel_name=kernel_name,
            timeout=60.0, cwd=None, quiet=False, ephemeral=True,
        )

    def health(self, kernel_name: str, url: str = "http://127.0.0.1:8899/health") -> int:
        return self.exec_command(
            f"curl -s -m 10 {url} || echo NO_SERVER",
            session=None, kernel_name=kernel_name,
            timeout=60.0, cwd=None, quiet=False, ephemeral=True,
        )


def build_parser(legacy: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sysinfo", type=Path, default=Path(DEFAULT_SYSINFO))
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--session", default=None)
    parser.add_argument("--cwd", default=None)
    if legacy:
        # legacy single-shot interface (no subparsers: the `command`
        # positional would otherwise collide with subcommand choices)
        parser.add_argument("command", nargs="?", help="shell command to run on the Jupyter node")
        parser.add_argument("--download", metavar="REMOTE_PATH")
        parser.add_argument("--output", type=Path)
        return parser
    # subcommands
    sub = parser.add_subparsers(dest="subcommand")

    def _common(s: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # SUPPRESS defaults: globals given before the subcommand are folded
        # in via set_defaults() in main() and must not be clobbered here.
        s.add_argument("--base-url", default=argparse.SUPPRESS)
        s.add_argument("--sysinfo", type=Path, default=argparse.SUPPRESS)
        s.add_argument("--session", default=argparse.SUPPRESS)
        s.add_argument("--cwd", default=argparse.SUPPRESS)
        s.add_argument("--timeout", type=float, default=argparse.SUPPRESS)
        s.add_argument("--kernel", default=argparse.SUPPRESS)
        return s

    p = _common(sub.add_parser("exec", help="run a command (session-persistent if --session)"))
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    _common(sub.add_parser("shell", help="interactive REPL on a persistent session"))
    sub.add_parser("sessions", help="list kernels and tracked sessions")
    p = sub.add_parser("session-kill", help="delete a session kernel")
    p.add_argument("name")
    p = _common(sub.add_parser("push", help="upload a file or incremental directory"))
    p.add_argument("local", type=Path)
    p.add_argument("remote")
    p.add_argument("--workers", type=int, default=8)
    p = _common(sub.add_parser("pull", help="download a file or incremental directory"))
    p.add_argument("remote")
    p.add_argument("local", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p = _common(sub.add_parser("bg", help="start a detached background job"))
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.add_argument("--job", required=True)
    p = _common(sub.add_parser("poll", help="check a background job + tail its log"))
    p.add_argument("job")
    p = _common(sub.add_parser("log", help="print a background job log tail"))
    p.add_argument("job")
    p.add_argument("--tail", type=int, default=50)
    p.add_argument("--follow", action="store_true", help="tail -f: stream until job ends")
    p.add_argument("--interval", type=float, default=2.0)
    p = _common(sub.add_parser("kill-job", help="kill a background job"))
    p.add_argument("job")
    _common(sub.add_parser("jobs", help="list background jobs (RUNNING/DONE)"))
    p = sub.add_parser("bootstrap", help="deploy the persistent remote helper agent (once)")
    p.add_argument("--force", action="store_true")
    p = _common(sub.add_parser("sync-push", help="incremental dir upload (only changed files)"))
    p.add_argument("local", type=Path)
    p.add_argument("remote")
    p.add_argument("--delete", action="store_true")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--no-default-excludes", action="store_true",
                   help="also sync .venv/.git/node_modules etc.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast", action="store_true",
                   help="legacy: fast mode is now default; kept as no-op")
    p.add_argument("--full-hash", action="store_true",
                   help="force full sha256 compare (slow, for audit)")
    p = _common(sub.add_parser("sync-pull", help="incremental dir download (only changed files)"))
    p.add_argument("remote")
    p.add_argument("local", type=Path)
    p.add_argument("--delete", action="store_true")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--no-default-excludes", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast", action="store_true",
                   help="legacy: fast mode is now default; kept as no-op")
    p.add_argument("--full-hash", action="store_true",
                   help="force full sha256 compare (slow, for audit)")
    p = _common(sub.add_parser("tunnel", help="expose a remote port via cloudflared quick tunnel"))
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--job", required=True)
    _common(sub.add_parser("doctor", help="print remote CUDA/GPU/python/disk diagnostics"))
    _common(sub.add_parser("vram", help="concise nvidia-smi + local server health"))
    p = _common(sub.add_parser("health", help="curl the remote OpenAI server health endpoint"))
    p.add_argument("--url", default="http://127.0.0.1:8899/health")
    p = _common(sub.add_parser("run-many", help="run commands in parallel, each own kernel"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--use-sessions", action=argparse.BooleanOptionalAction, default=True,
                   help="reuse a named session kernel per slot (default True; --no-sessions for fresh kernels)")
    p.add_argument("cmds", nargs="+")
    return parser


def legacy_main(args: argparse.Namespace, client: JupyterClient) -> int:
    if bool(args.command) == bool(args.download):
        print("provide exactly one of COMMAND or --download", file=sys.stderr)
        return 2
    if args.download and args.output is None:
        print("--download requires --output", file=sys.stderr)
        return 2
    if args.output is not None and not args.download:
        print("--output is only valid with --download", file=sys.stderr)
        return 2
    if args.download:
        assert args.output is not None
        client.download_small(args.download, args.output)
        return 0
    assert args.command is not None
    return client.exec_command(
        args.command, session=args.session, kernel_name=args.kernel,
        timeout=args.timeout, cwd=args.cwd, quiet=False,
        ephemeral=args.session is None,
    )


_FLAGS_WITH_VALUE = frozenset({
    "--base-url", "--sysinfo", "--kernel", "--timeout", "--session",
    "--cwd", "--download", "--output",
})


def _find_subcommand(argv: list[str]) -> int | None:
    """Index of the first subcommand token, skipping global flags/values.

    Returns None for legacy single-shot usage (a bare command positional).
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in SUBCOMMANDS:
            return i
        if tok in _FLAGS_WITH_VALUE:
            i += 2
            continue
        if tok.startswith("-") and tok != "-":
            i += 1
            continue
        return None
    return None


def _strip_dashes(tokens: list[str]) -> list[str]:
    # argparse REMAINDER keeps a literal `--` separator; drop it so
    # `prog exec -- echo hi` runs `echo hi` rather than `-- echo hi`.
    return tokens[1:] if tokens[:1] == ["--"] else tokens


def main() -> int:
    argv = sys.argv[1:]
    sub_idx = _find_subcommand(argv)
    if sub_idx is not None:
        parser = build_parser(legacy=False)
        # Globals may appear before the subcommand
        # (e.g. `prog --sysinfo X exec -- cmd`); fold them in as defaults
        # so they survive alongside the subcommand form.
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--base-url", default=DEFAULT_BASE_URL)
        pre.add_argument("--sysinfo", type=Path, default=Path(DEFAULT_SYSINFO))
        pre.add_argument("--kernel", default="python3")
        pre.add_argument("--timeout", type=float, default=120.0)
        pre.add_argument("--session", default=None)
        pre.add_argument("--cwd", default=None)
        known, _ = pre.parse_known_args(argv[:sub_idx])
        parser.set_defaults(
            base_url=known.base_url, sysinfo=known.sysinfo,
            kernel=known.kernel, timeout=known.timeout,
            session=known.session, cwd=known.cwd,
        )
        args = parser.parse_args(argv[sub_idx:])
        client = JupyterClient(args.base_url, load_token(args.sysinfo))
        if args.subcommand == "exec":
            cmd = " ".join(_strip_dashes(args.cmd)).strip()
            if not cmd:
                print("exec requires a command", file=sys.stderr)
                return 2
            return client.exec_command(
                cmd, session=args.session, kernel_name=args.kernel,
                timeout=args.timeout, cwd=args.cwd, quiet=False,
                ephemeral=args.session is None,
            )
        if args.subcommand == "shell":
            session = args.session or "main"
            try:
                import readline as _rl
                _hist = str(SHELL_HISTORY)
                try:
                    _rl.read_history_file(_hist)
                except OSError:
                    pass
                _rl.set_history_length(500)
                import atexit as _ax
                _ax.register(_rl.write_history_file, _hist)
            except ImportError:
                pass
            print(f"a100 shell on session '{session}' (:quit exit, cd works, "
                  ":bg <job> <cmd> detached, :poll <job>, :log <job>, :jobs, :kill <job>, :kernels, :sessions, "
                  ":session <name>, :push/:pull/:sync, :tunnel, :health, :doctor, "
                  ":par <c1> ;; <c2> parallel, :help)")
            while True:
                try:
                    line = input(f"a100:{session}$ ")
                except EOFError:
                    print()
                    return 0
                if line.strip() in {":quit", ":q", "exit"}:
                    return 0
                if not line.strip():
                    continue
                if line.strip() == ":jobs":
                    try:
                        client.bg_jobs(args.kernel)
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip() == ":kernels":
                    try:
                        kernels = client.list_kernels()
                        print(f"{len(kernels)} kernels alive")
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":log "):
                    job = line.strip().split(None, 1)[1]
                    try:
                        client.bg_log(job, 50, args.kernel)
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":bg "):
                    rest = line.strip()[4:].split(None, 1)
                    if len(rest) != 2:
                        print("usage: :bg <job> <cmd>", file=sys.stderr)
                        continue
                    try:
                        client.bg_start(rest[0], rest[1], args.kernel,
                                        client.session_cwd(session))
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":poll "):
                    job = line.strip().split(None, 1)[1]
                    try:
                        client.bg_poll(job, args.kernel)
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":kill "):
                    job = line.strip().split(None, 1)[1]
                    try:
                        client.bg_kill(job, args.kernel)
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip() == ":vram":
                    try:
                        client.vram(args.kernel)
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip() == ":sessions":
                    try:
                        state = load_state()
                        for name, kid in sorted(state.get("sessions", {}).items()):
                            cwd = state.get("cwd", {}).get(name, "-")
                            print(f"{name} kid={kid[:8]} cwd={cwd}")
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":session "):
                    session = line.strip().split(None, 1)[1].strip() or session
                    print(f"session -> {session} cwd={client.session_cwd(session) or '~'}")
                    continue
                if line.strip() in {":help", ":h", ":?"}:
                    print(":quit/:q/exit quit | cd <dir> sticky cwd | :session <name> switch")
                    print(":bg <job> <cmd> detached | :poll <job> | :log <job> | :jobs | :kill <job>")
                    print(":kernels | :sessions | :vram | :health [url] | :doctor | :tunnel <port> <job>")
                    print(":push <local> <remote> | :pull <remote> <local> | :sync <local> <remote>")
                    print(":par <c1> ;; <c2> run cmds in parallel, print as they finish")
                    continue
                if line.strip().startswith(":tunnel "):
                    parts = line.strip().split()
                    if len(parts) != 3:
                        print("usage: :tunnel <port> <job>", file=sys.stderr)
                        continue
                    try:
                        client.tunnel(int(parts[1]), parts[2], args.kernel)
                    except (RuntimeError, ValueError) as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":health"):
                    parts = line.strip().split(None, 1)
                    url = parts[1].strip() if len(parts) > 1 else "http://127.0.0.1:8899/health"
                    try:
                        client.health(args.kernel, url)
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip() == ":doctor":
                    try:
                        client.exec_command(
                            "echo '--- gpu ---'; "
                            "nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv; "
                            "echo '--- torch ---'; "
                            "(~/llm/venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 || python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1); "
                            "echo '--- disk ---'; df -h ~ | tail -n 1; du -sh ~/models 2>/dev/null; "
                            "echo '--- jobs ---'; ls ~/.a100-jobs/ 2>/dev/null",
                            session=None, kernel_name=args.kernel,
                            timeout=args.timeout, cwd=None, quiet=False,
                            ephemeral=True,
                        )
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":push "):
                    parts = line.strip().split(None, 2)
                    if len(parts) != 3:
                        print("usage: :push <local> <remote>", file=sys.stderr)
                        continue
                    try:
                        client.push(Path(parts[1]).expanduser(), parts[2], args.kernel)
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":pull "):
                    parts = line.strip().split(None, 2)
                    if len(parts) != 3:
                        print("usage: :pull <remote> <local>", file=sys.stderr)
                        continue
                    try:
                        client.pull(parts[1], Path(parts[2]).expanduser(), args.kernel)
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":sync "):
                    parts = line.strip().split()
                    if len(parts) < 3:
                        print("usage: :sync <local> <remote> [--delete]", file=sys.stderr)
                        continue
                    try:
                        client.sync_push(Path(parts[1]).expanduser(), parts[2], args.kernel,
                                         delete=("--delete" in parts))
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                if line.strip().startswith(":par "):
                    cmds = [c.strip() for c in line.strip()[5:].split(";;") if c.strip()]
                    if not cmds:
                        print("usage: :par <c1> ;; <c2> [;; <c3> ...]", file=sys.stderr)
                        continue
                    import concurrent.futures as _cf
                    _sess = session
                    def _one(_i_cmd):
                        _i, _c = _i_cmd
                        _w = JupyterClient(client.base_url, client.token)
                        _kid = _w.session_kernel(f"{_sess}-par{_i}", args.kernel)
                        _code, _out = _w.run(_kid, _c, timeout=args.timeout,
                                             cwd=_w.session_cwd(f"{_sess}-par{_i}"), quiet=True)
                        return _i, _c, _code, _out
                    try:
                        with _cf.ThreadPoolExecutor(max_workers=len(cmds)) as _pool:
                            for _i, _c, _code, _out in sorted(_pool.map(_one, enumerate(cmds)), key=lambda r: r[0]):
                                print(f"=== [{_i}] rc={_code} :: {_c} ===")
                                print(_out, end="" if _out.endswith("\n") or not _out else "\n")
                    except RuntimeError as exc:
                        print(f"error: {exc}", file=sys.stderr)
                    continue
                try:
                    client.exec_command(
                        line, session=session, kernel_name=args.kernel,
                        timeout=args.timeout, cwd=args.cwd, quiet=False,
                    )
                except RuntimeError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                except KeyboardInterrupt:
                    print()
            return 0
        if args.subcommand == "sessions":
            kernels = client.list_kernels()
            state = load_state()
            tracked = {kid: name for name, kid in state.get("sessions", {}).items()}
            for kernel in kernels:
                mark = f"session={tracked[kernel['id']]}" if kernel["id"] in tracked else "-"
                print(f"{kernel['id']} name={kernel.get('name')} {mark}")
            return 0
        if args.subcommand == "session-kill":
            state = load_state()
            kernel_id = state.get("sessions", {}).get(args.name, args.name)
            try:
                client.delete_kernel(kernel_id)
            except RuntimeError as exc:
                print(f"warning: {exc}", file=sys.stderr)
            client.forget_session(args.name)
            print(f"killed {args.name}")
            return 0
        if args.subcommand == "push":
            client.push(args.local, args.remote, args.kernel, workers=args.workers)
            return 0
        if args.subcommand == "pull":
            client.pull(args.remote, args.local, args.kernel, workers=args.workers)
            return 0
        if args.subcommand == "bg":
            cmd = " ".join(_strip_dashes(args.cmd)).strip()
            if not cmd:
                print("bg requires a command", file=sys.stderr)
                return 2
            client.bg_start(args.job, cmd, args.kernel, args.cwd or client.session_cwd(args.session))
            return 0
        if args.subcommand == "poll":
            return client.bg_poll(args.job, args.kernel)
        if args.subcommand == "log":
            return client.bg_log(args.job, args.tail, args.kernel,
                                 follow=args.follow, interval=args.interval)
        if args.subcommand == "bootstrap":
            client.ensure_agent(args.kernel, force=args.force)
            return 0
        if args.subcommand == "sync-push":
            fast = not getattr(args, "full_hash", False)
            return client.sync_push(args.local, args.remote, args.kernel,
                                    delete=args.delete, excludes=args.exclude,
                                    workers=args.workers, dry_run=args.dry_run,
                                    fast=fast,
                                    use_defaults=not args.no_default_excludes)
        if args.subcommand == "sync-pull":
            fast = not getattr(args, "full_hash", False)
            return client.sync_pull(args.remote, args.local, args.kernel,
                                    delete=args.delete, excludes=args.exclude,
                                    workers=args.workers, dry_run=args.dry_run,
                                    fast=fast,
                                    use_defaults=not args.no_default_excludes)
        if args.subcommand == "tunnel":
            client.tunnel(args.port, args.job, args.kernel)
            return 0
        if args.subcommand == "doctor":
            return client.exec_command(
                "echo '--- gpu ---'; "
                "nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv; "
                "echo '--- torch ---'; "
                "(~/llm/venv/bin/python -c \'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())\' 2>&1 || python3 -c \'import torch; print(torch.__version__, torch.cuda.is_available())\' 2>&1); "
                "echo '--- disk ---'; df -h ~ | tail -n 1; du -sh ~/models 2>/dev/null; "
                "echo '--- jobs ---'; ls ~/.a100-jobs/ 2>/dev/null",
                session=None, kernel_name=args.kernel,
                timeout=args.timeout, cwd=None, quiet=False,
                ephemeral=True,
            )
        if args.subcommand == "vram":
            return client.vram(args.kernel)
        if args.subcommand == "health":
            return client.health(args.kernel, args.url)
        if args.subcommand == "kill-job":
            return client.bg_kill(args.job, args.kernel)
        if args.subcommand == "jobs":
            return client.bg_jobs(args.kernel)
        if args.subcommand == "run-many":
            results: dict[str, tuple[int, str]] = {}

            def _task(index: int, cmd: str) -> tuple[int, int, str, str]:
                worker = JupyterClient(client.base_url, client.token)
                if args.use_sessions:
                    kid = worker.session_kernel(f"par-{index}", args.kernel)
                    code, out = worker.run(kid, cmd, timeout=args.timeout,
                                           cwd=args.cwd, quiet=True)
                else:
                    kernel = worker.create_kernel(args.kernel)
                    try:
                        code, out = worker.run(kernel["id"], cmd,
                                               timeout=args.timeout,
                                               cwd=args.cwd, quiet=True)
                    finally:
                        worker.delete_kernel(kernel["id"])
                return index, code, cmd, out

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_task, i, cmd) for i, cmd in enumerate(args.cmds)]
                for future in concurrent.futures.as_completed(futures):
                    index, code, cmd, out = future.result()
                    results[cmd] = (code, out)
                    print(f"=== [{index}] rc={code} :: {cmd} ===")
                    print(out, end="" if out.endswith("\n") or not out else "\n")
            failed = sum(1 for code, _ in results.values() if code != 0)
            print(f"run-many: {len(results) - failed}/{len(results)} ok")
            return 1 if failed else 0
        return 2
    parser = build_parser()
    args = parser.parse_args()
    client = JupyterClient(args.base_url, load_token(args.sysinfo))
    return legacy_main(args, client)


if __name__ == "__main__":
    raise SystemExit(main())
