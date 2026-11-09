#!/bin/bash
# 20-minute status tick (R1 on Blackwell since 12:27Z handover): collect R1 / Q2 / training-server state, write physics_pilot/progress/STATUS.json, commit + push on the SAME branch
# (naren/approachb-3d-learning; never creates a branch).  Read-only on every host.
set -u
REPO=/home/naren/Desktop/rss-3d-learning
P=$REPO/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/physics_pilot
mkdir -p $P/progress
G=$(ssh -o BatchMode=yes -o ConnectTimeout=15 user1@10.4.25.63 'cd ~/approachb_3d_learning/physics/out_r1 && python3 -c "
import json, os, glob
s = json.load(open(\"SUPERVISOR_STATE.json\")); L = json.load(open(\"R1_BUDGET_LEDGER.json\"))
fin = {}
for c in L[\"chunks\"]:
    for k, v in c[\"finals\"].items(): fin[k] = fin.get(k, 0) + v
w = sorted(glob.glob(\"work_*of*.jsonl\"), key=os.path.getmtime)
g = json.loads(open(w[-1]).read().splitlines()[-1]) if w else {}
print(json.dumps({\"host\": \"blackwell\", \"state\": s[\"state\"], \"chunk\": s.get(\"chunk\"), \"cpus\": s.get(\"cpus\"), \"supervisor_alive\": os.path.exists(\"/proc/%d\" % s[\"pid\"]),
  \"chunks_done\": len(L[\"chunks\"]), \"cpu_h\": L[\"chunks\"][-1][\"cpu_h_cumulative\"] if L[\"chunks\"] else 0, \"finals\": fin,
  \"discrepancies\": sum(len(c[\"discrepancies\"]) for c in L[\"chunks\"]), \"handover_partial\": [c[\"chunk\"] for c in L.get(\"handover_partial_records\", [])],
  \"stop_files\": [os.path.basename(f) for f in glob.glob(\"STOP_*\")], \"guard\": g.get(\"guard\"), \"items_files\": len(os.listdir(\"items\"))}))"' 2>/dev/null)
Q2G=$(ssh -o BatchMode=yes -o ConnectTimeout=15 gvlab 'ls ~/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2_gvlab/runs/spillover2_gvlab/records | wc -l; echo -n /; ls ~/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2_gvlab/runs/spillover3_gvlab/records 2>/dev/null | wc -l' 2>/dev/null)
Q=/home/naren/Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover/runs/remainder3_laptop
LQ=$(ls $Q/records 2>/dev/null | wc -l)
LS=$(python3 -c "import json;d=json.load(open('$Q/status.json'));print(json.dumps({'updated':d['updated'],'pid':d['pid'],'inflight':d['inflight'],'done':d.get('queries_done_this_session'),'total':d.get('queries_total')}))" 2>/dev/null)
LA=$(flatpak-spawn --host sh -c "ps -p \$(python3 -c \"import json;print(json.load(open('$Q/status.json'))['pid'])\") >/dev/null && echo alive || echo DEAD" 2>/dev/null)
S=$(ssh -o BatchMode=yes -o ConnectTimeout=15 user1@10.4.25.63 'echo "{\"yadu_procs\": $(ps -eo args | grep -c "[y]adu_bd/venv/bin/python"), \"gpu\": \"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)\", \"ours\": $(ps -eo comm,args | grep -c "^python[0-9.]* .*\(venv_phys\|approachb_3d_learning/venv\|venv/bin/python src/\)"), \"idle_cores\": $(python3 -c "import time;r=lambda:[list(map(int,l.split()[1:])) for l in open(\"/proc/stat\") if l.startswith(\"cpu\") and l[3].isdigit()];a=r();time.sleep(1);b=r();print(sum(1 for x,y in zip(a,b) if (y[3]+y[4]-x[3]-x[4])/max(1,sum(y)-sum(x))>0.5))")}"' 2>/dev/null)
GI=$(ssh -o BatchMode=yes -o ConnectTimeout=15 gvlab 'python3 -c "import time;r=lambda:[list(map(int,l.split()[1:])) for l in open(\"/proc/stat\") if l.startswith(\"cpu\") and l[3].isdigit()];a=r();time.sleep(1);b=r();print(sum(1 for x,y in zip(a,b) if (y[3]+y[4]-x[3]-x[4])/max(1,sum(y)-sum(x))>0.5))"' 2>/dev/null)
python3 - "$G" "$LQ" "$LS" "$LA" "$S" "$P/progress/STATUS.json" "$GI" "$Q2G" <<'EOF'
import json, sys, time
g, lq, ls, la, s, out, gi, q2g = sys.argv[1:9]
d = {"utc": time.strftime("%FT%TZ", time.gmtime()), "r1": json.loads(g) if g else "UNREACHABLE", "gvlab_q2_roots_v2/v3": q2g.replace(chr(10),""), "laptop_q2": {"roots": int(lq), "status": json.loads(ls) if ls else None, "coordinator": la},
     "training_server": json.loads(s) if s else "UNREACHABLE", "caps": {"r1_cpu_h": 60, "r1_wall_h": 42, "r2_cpu_h": 6, "gpu_h": 6},
     "gvlab_idle_cores": int(gi) if gi.strip().isdigit() else None, "plan": "EXECUTION_PLAN_20261108.md (rule: >= 4 idle cores on gvlab / Blackwell)"}
open(out, "w").write(json.dumps(d, indent=1))
r = d["r1"]
print(f"{d['utc']} | R1@blackwell {r.get('state')} {r.get('chunk')} done {r.get('chunks_done')}/52 cpu {r.get('cpu_h', 0):.2f}h disc {r.get('discrepancies')} finals {r.get('finals')} "
      f"stops {r.get('stop_files')} temp {(r.get('guard') or {}).get('pkg_temp_c')} bw_idle {(r.get('guard') or {}).get('idle_cores')} gvlab_idle {d['gvlab_idle_cores']} | Q2 gvlab v2/v3 {d['gvlab_q2_roots_v2/v3']} laptop {lq} ({la}) | server {d['training_server']}" if isinstance(r, dict) else str(d))
EOF
cd $REPO && git add $P/progress/STATUS.json && if ! git diff --cached --quiet; then git commit -q -m "approachb_3d_learning: R1 progress tick $(date -u +%FT%TZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin naren/approachb-3d-learning && echo "pushed $(git rev-parse --short HEAD)"; fi
