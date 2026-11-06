#!/bin/bash
# SHELF66V3 run (dataset_2500_raw_shelf66v3). Keeps the laptop and server runs of the 2,500-scene 6x6 shelf dataset in ONE ledger (same mechanism as merge_with_server.sh): every
# 5 min each side gets the other's new roots / witnesses (never overwritten), then their ledger lines (complete lines only), so both
# runners count every root toward the quotas.  Log: merge_with_server_3ds66_v3.log.   Stop: tmux kill-session -t =merge66v3
B=/home/ayush/codebase/bd_samples; R=user1@10.4.25.63; T=$B/.merge66v3_tmp
REL=3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3
S="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
mkdir -p $T
merge() {   # $1 = laptop run dir, $2 = server run dir (relative to the server home)
  $S $R "cat $2/ledger.jsonl" > $T/remote.jsonl &&
  python3 $B/merge_ledgers.py $1/ledger.jsonl $T/remote.jsonl $T/to_local.jsonl $T/to_remote.jsonl &&
  rsync -a --ignore-existing -e "$S" $R:$2/roots $R:$2/witnesses $1/ &&
  rsync -a --ignore-existing -e "$S" $1/roots $1/witnesses $R:$2/ &&
  rsync -a -e "$S" $T/to_remote.jsonl $R:$2/.merge_in.jsonl &&
  $S $R "python3 yadu_bd/code/bd_samples/append_lines.py $2/.merge_in.jsonl $2/ledger.jsonl && rm $2/.merge_in.jsonl" &&
  python3 $B/append_lines.py $T/to_local.jsonl $1/ledger.jsonl &&
  echo "$(date +%FT%T) +$(wc -l < $T/to_local.jsonl) lines from server, +$(wc -l < $T/to_remote.jsonl) to server"
}
while true; do
  merge $B/$REL yadu_bd/runs/$REL >> $B/merge_with_server_3ds66_v3.log 2>&1 \
    || echo "$(date +%FT%T) merge FAILED (server down, run not started yet, or SSH key locked?)" >> $B/merge_with_server_3ds66_v3.log
  sleep 300
done
