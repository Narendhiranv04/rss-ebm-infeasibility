#!/bin/zsh
# EXPLORATORY, 3D VAL: 10 mm, then 5 mm, then the shift-0 self-check.  1 worker until hw_tests.log says HW_TESTS_DONE, then 4.
cd "$(dirname "$0")"
PY=../../../approachb_3d_learning/runs/ABF3/env3d/bin/python
for s in 0.01 0.005 0; do $PY grasp_check.py run $s 4 2>&1 | grep -v OMP; done
$PY grasp_check.py report
echo GC_ALL_DONE
