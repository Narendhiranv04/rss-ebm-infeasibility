#!/bin/bash
# Robust gvlab chain (V4): run GVLAB3 then GVLAB4 to clean completion (final, stop=null).  A start refused ONLY because worker CPUs are
# momentarily busy (collaborator jobs are unpinned and migrate) is retried every 60 s; a runner safety stop is resumed (<= 3 times per share;
# packed roots are kept); any other refusal (identity / allowlist / disk / memory / temperature) stops the chain for a human.
cd $HOME/approachb3d_v3/2d_q2_handoff_v2
R=$HOME/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2_gvlab/runs
PY=$HOME/approachb3d_v3/2d_q2_spillover/venv313/bin/python
LOG=$HOME/approachb3d_v3/2d_q2_spillover/chain34.log
done_ok() { python3 -c "import json,sys;d=json.load(open(\"$R/$1/status.json\"));sys.exit(0 if d.get(\"final\") and not d.get(\"stop\") else 1)" 2>/dev/null; }
for SH in GVLAB3 GVLAB4; do
  RUN=spillover${SH#GVLAB}_gvlab; OUT=$HOME/approachb3d_v3/2d_q2_spillover/spillover${SH#GVLAB}.log; resumes=0
  while ! done_ok $RUN; do
    [ -f STOP_CHAIN34 ] && { echo "$(date -u +%FT%TZ) STOP_CHAIN34" >> $LOG; exit 0; }
    echo "$(date -u +%FT%TZ) start $SH" >> $LOG
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY spill2d_v2.py run $SH > /tmp/nav_chain34_last.txt 2>&1; cat /tmp/nav_chain34_last.txt >> $OUT
    if grep -q "^REFUSED run:" /tmp/nav_chain34_last.txt; then
      if grep "^REFUSED run:" /tmp/nav_chain34_last.txt | grep -qv "^REFUSED run: worker CPUs busy with other work (> 25%)$"; then
        echo "$(date -u +%FT%TZ) $SH HARD REFUSAL, chain stopped: $(grep "^REFUSED" /tmp/nav_chain34_last.txt)" >> $LOG; exit 1; fi
      echo "$(date -u +%FT%TZ) $SH refused: CPUs momentarily busy; retry in 60 s" >> $LOG; sleep 60; continue
    fi
    done_ok $RUN && break
    resumes=$((resumes+1)); echo "$(date -u +%FT%TZ) $SH ended without clean completion (resume $resumes)" >> $LOG
    [ $resumes -gt 3 ] && { echo "$(date -u +%FT%TZ) $SH too many resumes, chain stopped" >> $LOG; exit 1; }
    sleep 120
  done
  echo "$(date -u +%FT%TZ) $SH COMPLETE" >> $LOG
done
echo "$(date -u +%FT%TZ) chain34 done: GVLAB3 + GVLAB4 complete; ready for merge (RUNBOOK_Q2_V2.md)" >> $LOG
