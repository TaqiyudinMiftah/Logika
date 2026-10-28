"""Explicit Jupyter destinations with separate local state and credentials."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


def load_profile(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("Invalid profile name")
    path = ROOT / "profiles" / name / "profile.json"
    data = json.loads(path.read_text())
    if data["name"] != name:
        raise ValueError("Profile name differs from its directory")
    parsed = urlparse(data["base_url"])
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.query
            or parsed.fragment or parsed.path != f"/user/{data['username']}/"):
        raise ValueError("Profile requires an HTTPS Jupyter user URL without credentials")
    for field in ["sysinfo", "state_path", "manifest_cache", "shell_history", "term_history"]:
        value = Path(data[field]).expanduser()
        data[field] = value.resolve() if value.is_absolute() else (path.parent / value).resolve()
    data["hub_url"] = f"{parsed.scheme}://{parsed.netloc}/hub/"
    return data


@contextmanager
def activated(profile):
    import jupyter_remote as remote
    replacements = {
        "DEFAULT_BASE_URL": profile["base_url"],
        "DEFAULT_SYSINFO": str(profile["sysinfo"]),
        "DEFAULT_SYSINFO_NAME": profile["sysinfo"].name,
        "STATE_PATH": profile["state_path"],
        "MANIFEST_CACHE": profile["manifest_cache"],
        "TOKEN_ENV": profile["token_env"],
        "SHELL_HISTORY": profile["shell_history"],
        "TERM_HISTORY": profile["term_history"],
    }
    previous = {key: getattr(remote, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(remote, key, value)
        yield remote
    finally:
        for key, value in previous.items():
            setattr(remote, key, value)
