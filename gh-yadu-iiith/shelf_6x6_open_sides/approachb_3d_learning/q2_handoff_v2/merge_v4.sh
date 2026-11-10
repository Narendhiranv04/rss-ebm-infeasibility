#!/bin/bash
cd $HOME/approachb3d_v3/2d_q2_handoff_v2
G=$HOME/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1
M=$HOME/approachb3d_v3/2d_q2_merge_src
R=$G/launch_ready_2d_v2_gvlab/runs
echo "$(date -u +%FT%TZ) merge V4 start"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $HOME/approachb3d_v3/2d_q2_spillover/venv313/bin/python spill2d_v2.py merge $G/q2_spillover/handoff_v4/HANDOFF_ROOT_OWNER_MANIFEST_V4.json \
  ORIGINAL=$M/ORIGINAL LAPTOP=$M/LAPTOP LAPTOP2=$M/LAPTOP2 LAPTOP3=$M/LAPTOP3 \
  GVLAB=$R/spillover_gvlab GVLAB2=$R/spillover2_gvlab GVLAB3=$R/spillover3_gvlab GVLAB4=$R/spillover4_gvlab
echo "$(date -u +%FT%TZ) merge V4 exit $?"
