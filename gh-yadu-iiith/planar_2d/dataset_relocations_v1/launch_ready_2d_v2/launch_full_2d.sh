#!/usr/bin/env bash
# Guarded production launch of the full 2D placement-sidecar collection (launch_ready_2d_v1). INERT by default.
# It starts work only when called EXACTLY as
#     ./launch_full_2d.sh --execute-full --confirm <token>
# where <token> equals `python launch_runner.py token <this folder>` for the CURRENT frozen query plan, split manifest, launch
# config and production policy. The guard (token + metadata-only preflight) runs in the foreground first; any refusal exits
# non-zero before a worker exists, a run directory is created or the physical teacher is called. Re-running the same command
# after a pause, drain, cancel or crash resumes the same frozen task list (finished queries are never re-run).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$HERE/../code_v2"
PY="${PLACEMENT_PY:-/home/naren/Desktop/rss3d_env/venv/bin/python}"
if [ "$#" -ne 3 ] || [ "$1" != "--execute-full" ] || [ "$2" != "--confirm" ]; then
  echo "REFUSED: usage: $0 --execute-full --confirm <token>   (token: $PY $CODE/launch_runner.py token $HERE)" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$CODE"
"$PY" launch_runner.py guard "$HERE" --execute-full --confirm "$3"          # foreground checks only; exits non-zero on refusal
LOG="$HERE/../full_2d_v2.launch.log"
nohup "$PY" launch_runner.py full "$HERE" --execute-full --confirm "$3" >> "$LOG" 2>&1 &
echo "LAUNCHED coordinator pid $!  log: $LOG  status: $PY $CODE/launch_runner.py status $HERE/../full_2d_v2"
