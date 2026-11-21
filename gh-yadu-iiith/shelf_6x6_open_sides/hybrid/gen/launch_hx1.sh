#!/usr/bin/env bash
# Start (or resume) fresh K1-K3 root generation "hx1" on the home server. Copy to ~/Documents/RRC/rss_gen/ and run there.
# Same generator, unchanged (code/shelf66_v3/run_shelf66_full.py); see README_homeserver.md §3-5. Safe to re-run: resumes.
set -euo pipefail
G=~/Documents/RRC/rss_gen
OUT=$G/runs/hx1
WORKERS=${WORKERS:-14}
export UV_PYTHON_INSTALL_DIR=$G/.uv/python UV_CACHE_DIR=$G/.uv/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
[ "${FORCE:-0}" = 1 ] && rm -f $OUT/STOP
if pgrep -f -- "run_shelf66_full.py --out $OUT --name" >/dev/null; then echo "already running"; exit 0; fi
if [ -f $OUT/STOP ] && [ "${FORCE:-0}" != 1 ]; then echo "STOP file present (someone asked hx1 to stop); re-run with FORCE=1 to resume"; exit 1; fi
# the server is on a UPS: uncapped, 14 workers crashed it twice (2026-11-21). Refuse to start above a 130 W CPU cap.
PL=$(cat /sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw)
if [ "$PL" -gt 130000000 ]; then echo "CPU power cap is $((PL/1000000)) W; set it first: echo 120000000 | sudo tee /sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw"; exit 1; fi
$G/.venv/bin/python -c "import sys, mujoco, numpy; assert mujoco.__version__=='3.16.0' and numpy.__version__=='2.2.6'; print(sys.version.split()[0], mujoco.__version__, numpy.__version__)"
mkdir -p $OUT
[ -f $OUT/QUOTA.json ] || echo '{"NONE": 0, "0": 0, "1": 115, "2": 115, "3": 115, "4": 0}' > $OUT/QUOTA.json
[ -f $OUT/K4_MAX_RUNNING ] || echo 0 > $OUT/K4_MAX_RUNNING
cd $G/code
nohup nice -n 10 $G/.venv/bin/python shelf66_v3/run_shelf66_full.py --out $OUT --name hx1 --workers $WORKERS >> $OUT/run.log 2>&1 < /dev/null &
echo "started pid $! -> $OUT"
