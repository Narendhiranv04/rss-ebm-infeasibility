#!/bin/bash
# V4 chain (user 2026-11-09 ~15:50 IST: everything to gvlab, no Blackwell): wait until GVLAB3 is final, then run GVLAB4 on the same CPUs.
# If GVLAB3 ended on a runner safety stop, resume it once afterwards (packed roots are kept).
cd $HOME/approachb3d_v3/2d_q2_handoff_v2
R=$HOME/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2_gvlab/runs
PY=$HOME/approachb3d_v3/2d_q2_spillover/venv313/bin/python
LOG=$HOME/approachb3d_v3/2d_q2_spillover/spillover4.log
fin() { python3 -c "import json,sys;d=json.load(open(\"$R/$1/status.json\"));sys.exit(0 if d.get(\"final\") else 1)" 2>/dev/null; }
stopped() { python3 -c "import json,sys;d=json.load(open(\"$R/$1/status.json\"));sys.exit(0 if d.get(\"stop\") else 1)" 2>/dev/null; }
echo "$(date -u +%FT%TZ) chain_gvlab4 waiting for GVLAB3" >> $LOG
until fin spillover3_gvlab; do sleep 60; done
sleep 30
echo "$(date -u +%FT%TZ) GVLAB3 final; starting GVLAB4" >> $LOG
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY spill2d_v2.py run GVLAB4 >> $LOG 2>&1
if stopped spillover3_gvlab; then echo "$(date -u +%FT%TZ) GVLAB3 had a safety stop; resuming once" >> $LOG; sleep 30
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY spill2d_v2.py run GVLAB3 >> $HOME/approachb3d_v3/2d_q2_spillover/spillover3.log 2>&1; fi
echo "$(date -u +%FT%TZ) chain_gvlab4 done" >> $LOG
