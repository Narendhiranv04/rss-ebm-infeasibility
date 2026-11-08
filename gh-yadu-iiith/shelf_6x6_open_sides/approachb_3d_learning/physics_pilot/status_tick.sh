#!/bin/bash
# 20-minute status tick: collect R1 / Q2 / training-server state, write physics_pilot/progress/STATUS.json, commit + push on the SAME branch
# (naren/approachb-3d-learning; never creates a branch).  Read-only on every host.
set -u
REPO=/home/naren/Desktop/rss-3d-learning
P=$REPO/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/physics_pilot
mkdir -p $P/progress
G=$(ssh -o BatchMode=yes -o ConnectTimeout=15 gvlab 'cd ~/approachb3d_v3/bwl_physics_pilot/out_r1 && python3 -c "
import json, os, glob
s = json.load(open(\"SUPERVISOR_STATE.json\")); L = json.load(open(\"R1_BUDGET_LEDGER.json\"))
fin = {}
for c in L[\"chunks\"]:
    for k, v in c[\"finals\"].items(): fin[k] = fin.get(k, 0) + v
g = json.loads(open(\"work_0of2.jsonl\").read().splitlines()[-1]) if os.path.exists(\"work_0of2.jsonl\") else {}
q2 = len(os.listdir(os.path.expanduser(\"~/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2_gvlab/runs/spillover_gvlab/records\")))
print(json.dumps({\"state\": s[\"state\"], \"chunk\": s.get(\"chunk\"), \"supervisor_alive\": os.path.exists(\"/proc/%d\" % s[\"pid\"]), \"chunks_done\": len(L[\"chunks\"]),
  \"chunks_done_list\": [c[\"chunk\"] for c in L[\"chunks\"]], \"cpu_h\": L[\"chunks\"][-1][\"cpu_h_cumulative\"] if L[\"chunks\"] else 0, \"finals\": fin,
  \"discrepancies\": sum(len(c[\"discrepancies\"]) for c in L[\"chunks\"]), \"stop_files\": [os.path.basename(f) for f in glob.glob(\"STOP_*\")],
  \"guard\": g.get(\"guard\"), \"items_files\": len(os.listdir(\"items\")), \"q2_gvlab_roots\": q2}))"' 2>/dev/null)
Q=/home/naren/Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover/runs/remainder_laptop
LQ=$(ls $Q/records 2>/dev/null | wc -l)
LS=$(python3 -c "import json;d=json.load(open('$Q/status.json'));print(json.dumps({'updated':d['updated'],'pid':d['pid'],'inflight':d['inflight']}))" 2>/dev/null)
LA=$(flatpak-spawn --host sh -c "ps -p \$(python3 -c \"import json;print(json.load(open('$Q/status.json'))['pid'])\") >/dev/null && echo alive || echo DEAD" 2>/dev/null)
S=$(ssh -o BatchMode=yes -o ConnectTimeout=15 user1@10.4.25.63 'echo "{\"yadu_procs\": $(ps -eo args | grep -c "[y]adu_bd/venv/bin/python"), \"gpu\": \"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)\", \"ours\": $(ps -eo args | grep -c "[v]env/bin/python src/")}"' 2>/dev/null)
python3 - "$G" "$LQ" "$LS" "$LA" "$S" "$P/progress/STATUS.json" <<'EOF'
import json, sys, time
g, lq, ls, la, s, out = sys.argv[1:7]
d = {"utc": time.strftime("%FT%TZ", time.gmtime()), "gvlab_r1": json.loads(g) if g else "UNREACHABLE", "laptop_q2": {"roots": int(lq), "status": json.loads(ls) if ls else None, "coordinator": la},
     "training_server": json.loads(s) if s else "UNREACHABLE", "caps": {"r1_cpu_h": 60, "r1_wall_h": 42, "r2_cpu_h": 6, "gpu_h": 6}}
open(out, "w").write(json.dumps(d, indent=1))
r = d["gvlab_r1"]
print(f"{d['utc']} | R1 {r.get('state')} {r.get('chunk')} done {r.get('chunks_done')}/52 cpu {r.get('cpu_h', 0):.2f}h disc {r.get('discrepancies')} finals {r.get('finals')} "
      f"stops {r.get('stop_files')} temp {(r.get('guard') or {}).get('pkg_temp_c')} | Q2 gvlab {r.get('q2_gvlab_roots')} laptop {lq} ({la}) | server {d['training_server']}" if isinstance(r, dict) else str(d))
EOF
cd $REPO && git add $P/progress/STATUS.json && if ! git diff --cached --quiet; then git commit -q -m "approachb_3d_learning: R1 progress tick $(date -u +%FT%TZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin naren/approachb-3d-learning && echo "pushed $(git rev-parse --short HEAD)"; fi
