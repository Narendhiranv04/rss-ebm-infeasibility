#!/bin/bash
# Stage 1 of the class-quota 3D dataset, restarted automatically (resumable); clean stop: touch 3dt_plus_raw/STOP
cd /home/ayush/codebase/bd_samples
export MUJOCO_GL=egl XARM7_XML=/home/ayush/Downloads/mujoco_menagerie/ufactory_xarm7/xarm7.xml PYTHONWARNINGS=ignore
export BD_RSS3D_ROOT=/home/ayush/codebase/rss-ebm-infeasibility/gh-yadu-iiith PYTHONPATH=/home/ayush/codebase/bd_samples/dataset-bd
for n in $(seq 20); do
  python3 run_3dt_plus.py --out 3dt_plus_raw --workers 4 >> 3dt_plus_raw.log 2>&1 && break
  [ -f 3dt_plus_raw/STOP ] && break
  echo "$(date +%FT%T) runner exited; restart #$n" >> 3dt_plus_raw.log; sleep 30
done
