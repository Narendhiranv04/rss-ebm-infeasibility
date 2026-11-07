"""Source / host audit and integrity for the SHELF66V3 placement sidecar (read-only on the source dataset).

  inventory OUT_DIR      -> live_source_inventory.json + source_manifest.json.gz (SHA-256 of every selected root, referenced witness,
                            selected.json, ledger, PROGRESS, EXCLUDED) + host facts
  integrity MANIFEST     -> re-hash every manifest entry; exit 1 on any difference
"""
import gzip, hashlib, json, os, platform, re, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("SHELF_SRC", str(HERE.parents[1] / "dataset_bd_classquota_without_home_position_no_overpass")))
WID = re.compile(r"^[0-9a-f]{24}$")


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as err:  # noqa: BLE001
        return f"ERROR {err}"


def wids(x, out):
    """Every witness id referenced anywhere in a root record (keys 'witness' / 'witnesses')."""
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("witness", "witnesses"):
                for w in (v if isinstance(v, list) else [v]):
                    if isinstance(w, str) and WID.match(w):
                        out.add(w)
            else:
                wids(v, out)
    elif isinstance(x, list):
        for v in x:
            wids(v, out)
    return out


def selected_ids(sel):
    return [(k, r) for k, v in sel["selected"].items() for r in v]


def host():
    import mujoco
    egl = sh(f"MUJOCO_GL=egl {sys.executable} -c \"import mujoco,numpy as n;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\\'.1\\'/></worldbody></mujoco>');"
             "r=mujoco.Renderer(m,64,64);r.update_scene(mujoco.MjData(m));print('EGL_OK',r.render().shape)\"")
    cg = Path("/proc/self/cgroup").read_text().strip()
    rel = cg.split("::")[-1] if "::" in cg else ""
    ctl = Path(f"/sys/fs/cgroup{rel}/cgroup.controllers")
    return {"hostname": platform.node(), "python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__,
            "lscpu_model": sh("lscpu | grep 'Model name' | sed 's/.*: *//'"), "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)), "topology_core_of_cpu": sh("lscpu -p=CPU,CORE,SOCKET | grep -v '^#'"),
            "cgroup": cg, "cgroup_controllers": ctl.read_text().strip() if ctl.exists() else None,
            "hard_cpu_quota": bool(ctl.exists() and "cpu" in ctl.read_text().split()), "loadavg": Path("/proc/loadavg").read_text().strip(),
            "meminfo": sh("grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo"), "disk": sh(f"df -h {SRC} | tail -1"),
            "inodes": sh(f"df -i {SRC} | tail -1"), "mount": sh(f"findmnt -T {SRC} -no SOURCE,FSTYPE,OPTIONS"),
            "sensors": sh("for h in /sys/class/hwmon/hwmon*; do echo $h $(cat $h/name); done"),
            "coretemp_package": sh("for h in /sys/class/hwmon/hwmon*; do [ \"$(cat $h/name)\" = coretemp ] && cat $h/temp1_input; done"),
            "gpus": sh("nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader"), "egl_render": egl,
            "generator_processes": sh("ps -eo user,pid,pcpu,etime,args | grep -E 'run_shelf66|bd_3dt|run_3ds66|shelf_scene|bd_shelf' | grep -v grep"),
            "busy_processes": sh("ps -eo user,pid,pcpu,rss,etime,comm --sort=-pcpu | head -12")}


def conventions():
    sys.path.insert(0, str(SRC / "code" / "shelf66_v3"))
    import bd_shelf, mujoco
    G = bd_shelf.env(); S = G["TS"]
    e = bd_shelf.build([{"id": "t0", "shape": "box", "size": 0.02, "height": 0.12, "color": "red"}], {"t0": (2, 2)})
    gs = S.grasp_poses({**e.objects[0], "pos": e.obj_pos(0)}, e.table_z)
    pm = e.static_ck.m; d = mujoco.MjData(pm)
    d.qpos[e.static_ck.qadr] = e.q_ready; mujoco.mj_kinematics(pm, d)
    l, r = pm.geom("left_finger_pad_1").id, pm.geom("right_finger_pad_1").id
    static_gap = float(np.linalg.norm(d.geom_xpos[l] - d.geom_xpos[r]) - 2 * pm.geom_size[l][1])
    return {"robot_protocol": G["protocol"], "teacher_config_hash": G["config_hash"], "teacher_version": bd_shelf.A.TEACHER_VERSION,
            "geometry": G["geometry"], "source_files": G["source_files"], "NO_OVERPASS": S.NO_OVERPASS, "RETURN_TO_READY": S.RETURN_TO_READY,
            "ARM_CEILING": S.ARM_CEILING, "CEILING_MARGIN": S.CEILING_MARGIN, "GRASP_AT": S.GRASP_AT, "n_grasp_poses": len(gs),
            "azimuths": sorted({g["azimuth_deg"] for g in gs}), "grasp_levels": sorted({g["grasp_level"] for g in gs}), "tilts": sorted({g["tilt_deg"] for g in gs}),
            "ik_branches_max": e.n_branches, "FINGER_CLEARANCE": S.FINGER_CLEARANCE, "RRT_ITERS": bd_shelf.A.RRT_ITERS, "CARRY_LIFT": S.CARRY_LIFT,
            "margins": {"static": S.STATIC_MARGIN, "object": S.OBJECT_MARGIN, "free_move": S.FREE_MOVE_MARGIN},
            "finger_check": {"object_checks_use": "_check_width = grasp open_width (object width + 2 x FINGER_CLEARANCE) during plan_pick_place",
                             "structure_check_finger_qpos": "planning model default (static_ck sets only the 7 arm joints)",
                             "structure_check_pad_gap_m": static_gap, "calibrated_gap_range_m": [float(e._grip_cal[0][0]), float(e._grip_cal[0][-1])],
                             "example_open_width_m": float(gs[0]["open_width"])}}


def inventory(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw, sel = SRC / "raw", json.loads((SRC / "selected.json").read_text())
    ids = selected_ids(sel)
    led = [json.loads(x) for x in (raw / "ledger.jsonl").read_text().splitlines() if x.strip()]
    acc = [x for x in led if x.get("accepted")]
    rootfiles = {p.stem for p in (raw / "roots").glob("*.json")}
    wfiles = {p.stem: p for p in (raw / "witnesses").glob("*/*.npz")}
    man, ref, recs, bad, schema = {}, set(), {}, [], Counter()
    for f in ("selected.json", "README.md", "raw/ledger.jsonl", "raw/PROGRESS.json", "raw/EXCLUDED_FIXED_OBJECT.json"):
        man[f] = fsha(SRC / f)
    for k, r in ids:
        p = raw / "roots" / f"{r}.json"
        if not p.exists():
            bad.append({"root": r, "problem": "MISSING_ROOT"}); continue
        man[f"raw/roots/{r}.json"] = fsha(p)
        rec = recs[r] = json.loads(p.read_text())
        ref |= wids(rec.get("labels", {}), set())
    miss, corrupt = [], []
    for w in sorted(ref):
        p = wfiles.get(w)
        if p is None:
            miss.append(w); continue
        man[f"raw/witnesses/{w[:2]}/{w}.npz"] = fsha(p)
        try:
            z = np.load(p); schema[tuple(sorted(z.files))] += 1
            q = z["q"]
            assert q.ndim == 2 and q.shape[1] == 7 and np.isfinite(q).all()
        except Exception as err:  # noqa: BLE001
            corrupt.append([w, str(err)[:80]])
    tv = Counter(rec["ids"]["teacher_version"][:40] for rec in recs.values())
    cls_ok = Counter((k == "NONE" and rec.get("verified_K") is None) or str(rec.get("verified_K")) == k for (k, r), rec in
                     ((x, recs[x[1]]) for x in ids if x[1] in recs))
    sel_set = {r for _, r in ids}
    exc = sel.get("excluded_fixed_object", {})
    inv = {"source": {"path": str(SRC), "git_commit": sh(f"git -C {SRC} rev-parse HEAD"), "git_branch_origin": "yadu/dataset-3d",
                      "selected_status": sel.get("status"), "selection_rule": sel.get("rule")},
           "selected": {"quota": sel["quota"], "counts": sel["counts"], "n": len(ids), "by_class": Counter(k for k, _ in ids),
                        "duplicate_ids": [r for r, n in Counter(r for _, r in ids).items() if n > 1], "class_matches_verified_K": dict(cls_ok),
                        "teacher_version_mix": dict(tv), "robot_protocol_mix": dict(Counter(r["ids"]["robot_protocol"] for r in recs.values())),
                        "config_hash_mix": dict(Counter(r["ids"]["config_hash"] for r in recs.values())),
                        "generator_mix": dict(Counter(r.get("generator_version_plus", "?") for r in recs.values())),
                        "parent_families": len({r["ids"]["parent_family_id"] for r in recs.values()}), "problems": bad},
           "raw": {"root_files": len(rootfiles), "witness_files": len(wfiles), "ledger_lines": len(led), "ledger_accepted": len(acc),
                   "ledger_rejected": len(led) - len(acc), "ledger_duplicate_root_ids": [r for r, n in Counter(x.get("root_id") for x in acc).items() if n > 1],
                   "ledger_gen_mix": dict(Counter(x.get("gen") for x in led)), "ledger_retaught": sum(1 for x in acc if x.get("retaught_from")),
                   "ledger_observed": dict(Counter(str(x.get("observed")) for x in acc)),
                   "accepted_unselected": len({x["root_id"] for x in acc} - sel_set),
                   "accepted_unselected_by_observed": dict(Counter(str(x.get("observed")) for x in acc if x["root_id"] not in sel_set)),
                   "root_files_not_in_ledger": len(rootfiles - {x.get("root_id") for x in led}),
                   "progress_json": json.loads((raw / "PROGRESS.json").read_text())},
           "witnesses": {"referenced_by_selected": len(ref), "missing": miss, "corrupt": corrupt, "schemas": {"|".join(k): v for k, v in schema.items()},
                         "unreferenced_files": len(set(wfiles) - ref)},
           "excluded_fixed_object": {"roots": exc, "any_selected": sorted(set(exc) & sel_set)},
           "conventions": conventions(), "host": host()}
    inv["manifest_digest"] = hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()
    inv["manifest_entries"] = len(man)
    with gzip.open(out / "source_manifest.json.gz", "wt") as f:
        json.dump({"src": str(SRC), "files": man}, f, sort_keys=True)
    (out / "live_source_inventory.json").write_text(json.dumps(inv, indent=1, sort_keys=True, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print(json.dumps({k: inv[k] for k in ("selected", "witnesses", "excluded_fixed_object")}, default=str)[:3000])


def integrity(manifest):
    m = json.load(gzip.open(manifest, "rt"))
    src = Path(m["src"])
    diff = [f for f, h in m["files"].items() if not (src / f).exists() or fsha(src / f) != h]
    git = sh(f"git -C {src} status --porcelain --untracked-files=all -- .")
    print(json.dumps({"checked": len(m["files"]), "changed_or_missing": diff[:20], "n_changed": len(diff), "git_status_source": git or "clean"}))
    return 1 if diff or git else 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit(inventory(sys.argv[2]) if cmd == "inventory" else integrity(sys.argv[2]))
