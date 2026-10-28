#!/usr/bin/env python3
"""Select faris or node380 before using the existing A100 tools.

Examples:
  python profile.py list
  python profile.py node380 login
  python profile.py node380 term 'whoami; hostname'
  python profile.py faris exec --session term -- 'pwd'
  python profile.py node380 sessions
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

from connection_profiles import ROOT, activated, load_profile


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["--help"]:
        print(__doc__)
        return 0
    if arguments == ["list"]:
        for file in sorted((ROOT / "profiles").glob("*/profile.json")):
            profile = load_profile(file.parent.name)
            configured = profile["sysinfo"].is_file() or bool(os.environ.get(profile["token_env"]))
            print(f"{profile['name']}: {profile['base_url']}  credentials={'configured' if configured else 'login needed'}")
        return 0
    profile = load_profile(arguments.pop(0))
    if not arguments:
        arguments = ["term"]
    action = arguments.pop(0)
    if action == "login":
        if arguments:
            raise ValueError("Login takes no command-line password; use the protected prompt")
        from profile_login import login
        login(profile)
        return 0
    with activated(profile) as remote:
        original = sys.argv
        try:
            if action in {"term", "cleanup"}:
                script = ROOT / ("a100-term" if action == "term" else "a100-cleanup")
                sys.argv = [str(script), *arguments]
                namespace = runpy.run_path(str(script), run_name=f"a100_{profile['name']}_{action}")
                return namespace["main"]()
            sys.argv = [str(ROOT / "jupyter_remote.py"), action, *arguments]
            return remote.main()
        finally:
            sys.argv = original


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Profile error: {error}", file=sys.stderr)
        raise SystemExit(1)
