#!/usr/bin/env bash
# REAP venv on a server. B = the bundle folder (reap_shelf/, bd_samples/, dataset66/, wheels/), default: this script's
# grand-parent. Everything stays inside B (uv cache, Conan home); the system python3.12 is used.
set -euo pipefail
B=${B:-$(cd "$(dirname "$0")/../.." && pwd)}
export UV_CACHE_DIR=$B/.uv-cache UV_PYTHON_INSTALL_DIR=$B/.uv-python CONAN_HOME=$B/.conan2
export CMAKE_BUILD_PARALLEL_LEVEL=8 MAKEFLAGS=-j8
UV=$(command -v uv || echo ~/.local/bin/uv)
cd $B
[ -d reap ] || git clone -q https://github.com/elicited-preconditions/reap
cd reap
git checkout -q d2ba23d
git checkout -q -- . && git apply ../reap_shelf/reap_yadu.patch
git diff --stat
[ -d .venv ] || $UV venv -q --python /usr/bin/python3.12 .venv
$UV pip install -q --python .venv/bin/python $B/wheels/mifrost-0.2.0-cp312-cp312-linux_x86_64.whl
# requirements.txt pins mifrost as a git URL, which uv would build from source (Conan + nauty download, slow and fragile);
# the override points it at the wheel built on the laptop (same commit cd612e9, cp312, x86_64)
echo "mifrost @ file://$B/wheels/mifrost-0.2.0-cp312-cp312-linux_x86_64.whl" > $B/wheels/overrides.txt
# TAMPanda v1.1.0 (commit a7293410) copied from the laptop: large git fetches from GitHub break on some servers
[ -d $B/wheels/TAMPanda ] && echo "tampanda @ file://$B/wheels/TAMPanda" >> $B/wheels/overrides.txt
# torch + its NVIDIA libraries from download.pytorch.org (cu130): from IIIT, files.pythonhosted.org gives ~1 MB/s,
# pytorch.org 10-15 MB/s. The rest then comes from PyPI with torch already satisfied.
$UV pip install -q --python .venv/bin/python torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
UV_TORCH_BACKEND=cu130 $UV pip install -q --python .venv/bin/python -e . -r requirements-tampanda.txt --overrides $B/wheels/overrides.txt
$UV pip install -q --python .venv/bin/python mujoco==3.16.0
.venv/bin/python -c "import torch, mujoco, mifrost, tampanda; print('OK torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'mujoco', mujoco.__version__)"
bash scripts/download_assets.sh confined datasets 2>&1 | grep -v "#" | tail -3
echo SETUP_DONE
