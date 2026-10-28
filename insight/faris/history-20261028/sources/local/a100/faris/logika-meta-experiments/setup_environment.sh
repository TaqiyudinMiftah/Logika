#!/bin/sh
set -eu
cd "$(dirname "$0")"
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_LINK_MODE=copy
.tools/uv venv --python 3.11.13 .venv
.tools/uv pip install --python .venv/bin/python -r requirements.txt
.tools/uv pip freeze --python .venv/bin/python > environment.freeze.txt
