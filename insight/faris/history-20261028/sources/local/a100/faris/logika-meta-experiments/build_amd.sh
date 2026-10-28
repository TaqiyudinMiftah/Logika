#!/bin/sh
set -eu
cd "$(dirname "$0")"
export UV_CACHE_DIR="$PWD/.uv-cache"
export CMAKE_BUILD_PARALLEL_LEVEL=12
mkdir -p .amd/debs .amd/sysroot
cd .amd/debs
apt-get download libboost1.83-dev libboost-filesystem1.83-dev libboost-filesystem1.83.0 libboost-system1.83-dev libboost-system1.83.0 libboost-atomic1.83-dev libboost-atomic1.83.0
for file in ./*.deb; do dpkg-deb -x "$file" ../sysroot; done
cd ../..
export LD_LIBRARY_PATH="$PWD/.amd/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.tools/uv pip install --python .venv/bin/python --reinstall-package lightgbm --no-binary lightgbm --no-deps --config-settings cmake.define.USE_GPU=ON --config-settings "cmake.define.BOOST_ROOT=$PWD/.amd/sysroot/usr" --config-settings "cmake.define.CMAKE_INSTALL_RPATH=$PWD/.amd/sysroot/usr/lib/x86_64-linux-gnu" lightgbm==4.6.0
.venv/bin/python -c 'import numpy as np,lightgbm as l; x=np.random.default_rng(1).normal(size=(512,12));y=(x[:,0]>0).astype(int);m=l.LGBMClassifier(n_estimators=10,device_type="gpu",gpu_platform_id=0,gpu_device_id=0,n_jobs=1,max_bin=63,verbosity=1);m.fit(x,y);print("AMD_SMOKE_PASSED",float(m.predict_proba(x)[:,1].mean()))'
