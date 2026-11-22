#!/bin/sh
# Runs on the MAC: every 10 min, pull the home server's exploratory outputs (results, models, logs) into
# runs/explore/server_backup/ so a server crash loses at most ~10 min.  Pull-only (no --delete); stops after 8 h.
SRC="home@100.65.190.74:Documents/RRC/rss_gen/hz/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/"
DST=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/server_backup/
mkdir -p "$DST"
n=0
while [ $n -lt 48 ]; do
  rsync -a --exclude 'teach_out' -e "ssh -o BatchMode=yes -o ConnectTimeout=20" "$SRC" "$DST" 2>/dev/null \
    && echo "$(date +%H:%M) backup ok" || echo "$(date +%H:%M) backup failed (server unreachable?)"
  ssh -o BatchMode=yes -o ConnectTimeout=20 home@100.65.190.74 'grep -q SERVER_DONE ~/Documents/RRC/rss_gen/hz/server_run.log 2>/dev/null' 2>/dev/null \
    && { rsync -a --exclude 'teach_out' -e "ssh -o BatchMode=yes" "$SRC" "$DST" 2>/dev/null; echo "$(date +%H:%M) server done, final backup"; exit 0; }
  sleep 600; n=$((n+1))
done
