#!/bin/bash
# Remaining V2 mock compute, in order (each step on CPUs 0-11 unless noted; nothing else of this project runs meanwhile):
# candidates (adaptive A, B, fixed x3) -> interrupt/damage -> sweep 8/12/16/18 -> serial -> replays of every stored witness ->
# finger diagnostic -> post-run tests -> integrity -> analysis + figures.
set -u
V2=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$HOME/approachb3d_mock/env/bin/python}; cd "$V2" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
while tmux list-windows -t approachb3d 2>/dev/null | grep -q " pytest"; do sleep 10; done
echo "$(date -u +%FT%TZ) tests done; starting candidates"
bash mock_v2/run_mock_v2.sh adaptive_A adaptive_B fixed
bash mock_v2/run_interrupt_v2.sh
bash mock_v2/run_mock_v2.sh sweep serial
V=mock_v2/validation
for r in reference_v2 adaptive_v1_A adaptive_v1_B fixed_4x1500 fixed_4x5000 fixed_8x5000 adaptive_v1_interrupt serial_adaptive; do
  echo "$(date -u +%FT%TZ) replay $r"; nice -n 10 "$PY" code/replay3d.py ours "mock_v2/runs/$r" mock_v2/roster_v2.jsonl.gz "$V/replay_$r.json" --workers 12 --cpus 0-11 > "$V/replay_$r.log" 2>&1
done
echo "$(date -u +%FT%TZ) finger diagnostic"
for i in $(seq 0 11); do taskset -c "$i" nice -n 10 "$PY" code/finger_diag.py "$i" 12 "$V/finger_diag_$i.json" > "$V/finger_diag_$i.log" 2>&1 & done; wait
echo "$(date -u +%FT%TZ) post-run tests"
RUN3D_TEST_CPU=1 taskset -c 0,1 nice -n 10 "$PY" -m pytest -q -p no:cacheprovider code/test_mock3d_v2.py > "$V/pytest_post_run.txt" 2>&1
echo "$(date -u +%FT%TZ) integrity"
(cd ../dataset_placement_extension_v1 && "$PY" code/audit3d.py integrity audit/source_manifest.json.gz) > "$V/integrity_pinned_source.json" 2> "$V/integrity_pinned_source.err"; echo "pinned-source integrity rc=$?"
(cd ../dataset_placement_extension_v1 && sha256sum -c --quiet "$HOME/approachb3d_mock/v1_tree_sha256_before_v2.txt") > "$V/v1_tree_unchanged.txt" 2>&1; echo "v1 tree unchanged rc=$?" | tee -a "$V/v1_tree_unchanged.txt"
(cd "$HOME/approachb3d_mock/src" && git status --porcelain --untracked-files=no) > "$V/source_git_status.txt"; echo "tracked changes in pinned clone: $(wc -l < "$V/source_git_status.txt")"
echo "$(date -u +%FT%TZ) analysis"
"$PY" code/report3d.py analyze mock_v2 production_v2/production_roster_live.jsonl.gz "$V/analysis_v2.json" > "$V/analysis_v2.log" 2>&1
MUJOCO_GL=egl "$PY" code/figs3d.py plots "$V/analysis_v2.json" mock_v2 mock_v2/figures > mock_v2/figures/plots.log 2>&1
echo "$(date -u +%FT%TZ) ALL DONE"
