#!/bin/bash
# Auto-restarting supervisor for one bd_run.py collection (run inside tmux, e.g. `tmux new -d -s bd_mock3dt "bash supervise.sh"`).
# Usage: supervise.sh <domain 2d|3dt> <name> <roots> <workers> <wave_size> <out_dir>
# Clean stop: touch <out_dir>/STOP (the runner finishes its current wave; this loop then exits). Restarts are logged to <out_dir>/restarts.log.
W=${BD_WORKSPACE:-/data/user1/rss_ebm_dataset}
export MUJOCO_GL=egl XARM7_XML=${XARM7_XML:-$W/src/mujoco_menagerie/ufactory_xarm7/xarm7.xml} PYTHONWARNINGS=ignore
DOM=$1 NAME=$2 ROOTS=$3 WORKERS=$4 WAVE=$5 O=$6
mkdir -p $O
cd ${BD_SRC:-$W/src/latent-causal-tamp}
n=0
until ${BD_PYTHON:-$W/env/rss/bin/python} bd_run.py --domain $DOM --name $NAME --roots $ROOTS --workers $WORKERS --wave-size $WAVE --out $O >> $O/run.log 2>&1; do
  rc=$?; n=$((n+1)); echo "$(date +%FT%T) runner exited with code $rc; restart #$n" >> $O/restarts.log
  [ -f $O/STOP ] && break; [ $n -ge 20 ] && echo "$(date +%FT%T) giving up" >> $O/restarts.log && break
  sleep 30
done
echo "$(date +%FT%T) supervisor finished" >> $O/restarts.log
