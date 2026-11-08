"""S1 physics pilot executor for 3D Approach B X / Y recovery (authorized physics host only; MuJoCo).  NEVER on the GPU training server.

The file name deliberately matches none of the 2D Q2 spillover's PROJECT_PROCS substrings (collect3d.py, replay3d.py, run3d.py, ...).
Imports the PINNED 3D teacher unchanged (bd_shelf -> bd_3dt -> shelf_scene, SHELF66V3, mujoco 3.16.0) and writes only under --out.

  X teacher BD3DS66_X_V1:  settle(S0) at q_ready -> replay the stored relocation witnesses of the prefix with the unchanged run_plan, in order,
                           carrying the achieved state (no reset) and applying bd_3dt.seq's checks per move -> CONNECTOR_TO_QREADY_V1:
                           plan_free_move(q_now, q_ready, seed 0; env RRT-Connect 5000 iterations), execute (go_ready's time
                           parameterization, WITHOUT its straight-line fallback), open the gripper to GRIPPER_OPEN, verify: |q - q_ready| <=
                           0.02 rad, nothing displaced > 4 mm during the connector, every object in its expected cell upright, nothing held,
                           no robot-object contact.  X = 1 only if a fresh-process INDEPENDENT replay of all witnesses + the connector path agrees.
                           A failing planning step is re-confirmed by re-running bd_3dt.query from the replayed achieved state.
  Y teacher BD3DS66_Y_REPLAY_V1: settle(endpoint) at q_ready -> replay the stored target witness with run_plan -> seq()'s target checks;
                           done in two fresh processes.  Negatives: re-run bd_3dt.query (deterministic, seed 0) to re-confirm the archived kind.

    PY=$HOME/approachb3d_mock/env/bin/python
    taskset -c 16 nice -n 19 $PY bwlphys.py policy --out OUT
    taskset -c 16 nice -n 19 $PY bwlphys.py work --manifest S1_PILOT_MANIFEST.json --phase SMOKE --shard 0/2 --out OUT
    (internal) item --manifest M --item ID --mode teacher|replay --out OUT"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("BWL_REPO", str(Path.home() / "approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides")))
SRC = REPO / "dataset_bd_classquota_without_home_position_no_overpass"
QREF_AUDIT = np.array([-1.345029, -0.632166, 0.221281, 2.024222, 0.267756, 2.629261, 2.210774])   # reuse audit: all 3,726 + 4,990 witnesses
TOL_Q, TOL_PUSH = 0.02, 0.004


def env():
    sys.path.insert(0, str(SRC / "code/shelf66_v3"))
    import bd_shelf  # noqa: F401  (points bd_3dt at the shelf scene exactly as production does)
    import bd_3dt as A
    return A, A.env()["TS"]


def root(rid):
    return json.loads((SRC / f"raw/roots/{rid}.json").read_text())


def witness(wid):
    """Load a stored witness and re-derive its content id (bd_core.Store.witness rule); a mismatch is fatal."""
    z = np.load(SRC / "raw/witnesses" / wid[:2] / f"{wid}.npz")
    w = {k: z[k] for k in z.files}
    h = hashlib.sha256()
    for k in sorted(w):
        v = np.ascontiguousarray(np.asarray(w[k]))
        h.update(f"{k}|{v.dtype.str}|{v.shape}".encode()); h.update(v.tobytes())
    if h.hexdigest()[:24] != wid:
        raise RuntimeError(f"WITNESS_HASH_MISMATCH {wid}")
    return w


def segments(e, S, name, w, grasp_id):
    i = e.obj_id(name)
    e._focus(i)
    e.objects[i]["pos"] = e.obj_pos(i)
    g = S.grasp_poses(e.objects[i], e.table_z)[grasp_id]
    out = []
    for k, p in enumerate(str(x) for x in w["phases"]):
        if p in ("open", "close", "release"):
            out.append(dict(phase=p, kind="grip", grip=None if p == "close" else g["open_width"]))
        else:
            out.append(dict(phase=p, kind="joint" if p in ("to_hover", "to_ready") else "cart", q=[r for r in w["q"][w["segment"] == k]]))
    return out


def move(e, S, rec, name, dest, w, grasp_id):
    """One stored move with bd_3dt.seq's checks (l.265-292): end cell, |z_err| <= 1 cm, no other object changed cell."""
    before = {n: e.cell_of(n) for n in e.obj_index if n != name}
    plan = {"name": name, "segments": segments(e, S, name, w, grasp_id)}
    try:
        e.run_plan(plan)
    except S.GraspFailed as err:
        return {"ok": False, "failure": f"GRASP_FAILED:{str(err)[:80]}", "end_cell": list(e.cell_of(name) or [])}
    h = next(o["height"] for o in rec["objects"] if o["id"] == name)
    z = float(e.obj_pos(name)[2] - e.table_z - h / 2)
    moved = sorted(n for n in before if e.cell_of(n) != before[n])
    ok = e.cell_of(name) == tuple(dest) and abs(z) <= 0.01 and not moved
    return {"ok": ok, "failure": None if ok else "EXECUTION_NOT_VERIFIED", "end_cell": list(e.cell_of(name) or []), "z_err": z, "displaced": moved,
            "pushed": plan["pushed"], "held_after": e.held}


def state(e):
    return {"q": e.q_now().tolist(), "obj": {n: e.obj_pos(n).tolist() for n in e.obj_index}, "held": e.held}


def contacts(e):
    import mujoco  # noqa: F401
    rg = set(int(g) for g in e._robot_geoms)
    og = set(int(h) for hs in e._obj_geoms for h in hs)
    return sum(1 for c in e.data.contact[:e.data.ncon] if (c.geom1 in rg and c.geom2 in og) or (c.geom2 in rg and c.geom1 in og))


def connector(e, S, expected, path=None):
    """CONNECTOR_TO_QREADY_V1.  path=None: plan (teacher); else execute the stored path (independent replay)."""
    q0, p0 = e.q_now(), {n: e.obj_pos(n).copy() for n in e.obj_index}
    planned = path is None
    if planned:
        try:
            path = e.plan_free_move(q0, e.q_ready, seed=0)
        except RuntimeError as err:
            return {"ok": False, "planned": False, "failure": "CONNECTOR_NO_PLAN", "reason": str(err)[:160]}, None
    traj = S.time_parameterize(path, S.PLAN_VMAX, S.PLAN_AMAX, e.model.opt.timestep)
    lag = e.execute(traj)
    e.set_gripper(True); e.wait(200)
    dq = float(np.max(np.abs(e.q_now() - e.q_ready)))
    disp = {n: float(np.linalg.norm(e.obj_pos(n) - p0[n])) for n in e.obj_index}
    pushed = sorted(n for n, d in disp.items() if d > TOL_PUSH)
    cells_ok = all(e.cell_of(n) == tuple(c) for n, c in expected.items())
    nc = contacts(e)
    ok = dq <= TOL_Q and not pushed and cells_ok and e.held is None and nc == 0
    return ({"ok": ok, "planned": True, "failure": None if ok else "CONNECTOR_EXECUTION_NOT_VERIFIED", "waypoints": len(path), "dq_ready": dq,
             "max_track_err": float(lag), "pushed_mm": {n: round(disp[n] * 1000, 2) for n in pushed}, "cells_ok": cells_ok, "robot_object_contacts": nc,
             "direct": bool(getattr(e, "last_free", {}).get("direct", False)) if planned else None}, np.asarray(path))


def policy_checks(e, A, S, rec):
    g = A.env()
    return {"q_ready_vs_audit_qref_max_abs": float(np.max(np.abs(e.q_ready - QREF_AUDIT))), "source_files_match_root": g["source_files"] == rec.get("source_files"),
            "robot_protocol_match_root": rec["ids"].get("robot_protocol") == g["protocol"], "config_hash_match_root": rec["ids"].get("config_hash") == g["config_hash"],
            "teacher_version_match": rec["ids"].get("teacher_version") == A.TEACHER_VERSION}


def run_y(it, mode, out):
    A, S = env()
    rec = root(it["root_id"])
    cells = {**{k: tuple(v) for k, v in rec["initial_cells"].items()}, **{k: tuple(v) for k, v in it["edit"].items()}}
    e = A.build(rec["objects"], rec["initial_cells"])
    A.settle(e, cells)
    res = {"policy": policy_checks(e, A, S, rec), "start_q_vs_qready": float(np.max(np.abs(e.q_now() - e.q_ready)))}
    if it["action"] == "REPLAY_STORED_TARGET_WITNESS":
        w = witness(it["witness_id"])
        res["witness_first_q_vs_qready"] = float(np.max(np.abs(w["q"][0] - e.q_ready)))
        r = move(e, S, rec, "t0", tuple(rec["goal"]), w, it["grasp_id"])
        res.update(step=r, verdict="REPLAY_PASS" if r["ok"] else "REPLAY_FAIL", final=state(e))
    else:
        p, info = A.query(e, "t0", tuple(rec["goal"]))
        kind = None if p is not None else A.kind(info["grasp_reasons"])
        res.update(replan={"plan_found": p is not None, "kind": kind, "tried": info.get("tried")},
                   verdict="NEGATIVE_RECONFIRMED" if (p is None and kind == it["negative_kind"]) else
                   ("NEGATIVE_KIND_CHANGED" if p is None else "PLAN_FOUND_CONTRADICTS_NEGATIVE"))
    return res


def stored_seq(rec, key):
    moves = key.split(".")
    for b in rec["labels"]["B"]:
        if [f"{o}>{int(c[0])},{int(c[1])}" for o, c in b.get("sequence") or []][:len(moves)] == moves:
            return b
    raise KeyError(key)


def run_x(it, mode, out):
    A, S = env()
    rec = root(it["root_id"])
    b = stored_seq(rec, it["x_key"])
    m = it["m"]
    moves = [(o, tuple(c)) for o, c in b["sequence"][:m]]
    cells0 = {k: tuple(v) for k, v in rec["initial_cells"].items()}
    e = A.build(rec["objects"], rec["initial_cells"])
    A.settle(e, cells0)
    res = {"policy": policy_checks(e, A, S, rec), "steps": [], "boundary_q_vs_recorded": []}
    expected = dict(cells0)
    for j, (o, d) in enumerate(moves):
        st = b["steps"][j]
        if j < len(b["witnesses"]):
            r = move(e, S, rec, o, d, witness(b["witnesses"][j]), st["grasp_id"])
            r["recorded"] = {"ok": st.get("ok"), "end_cell": st.get("end_cell"), "z_err": st.get("z_err")}
            r["agrees_with_record"] = bool(r["ok"]) == bool(st.get("ok")) and (st.get("end_cell") is None or list(st["end_cell"]) == r["end_cell"])
            if j + 1 < len(b["witnesses"]):
                res["boundary_q_vs_recorded"].append(float(np.max(np.abs(e.q_now() - witness(b["witnesses"][j + 1])["q"][0]))))
        else:                                                                 # the stored attempt failed at PLANNING this move: re-confirm
            p, info = A.query(e, o, d)
            kind = None if p is not None else A.kind(info["grasp_reasons"])
            r = {"ok": False, "failure": "PLANNING", "replan_found": p is not None, "kind": kind, "recorded_kind": b.get("negative_kind"),
                 "agrees_with_record": p is None and kind == b.get("negative_kind")}
        res["steps"].append(r)
        if not r["ok"]:
            res["verdict"] = "X0_RELOCATION_FAILED" if r["agrees_with_record"] else "RELOCATION_MISMATCH_WITH_RECORD"
            res["failed_phase"], res["final"] = f"relocation_{j}", state(e)
            return res
        expected[o] = d
    stored = out / "witness" / f"{it['item']}_connector.npy"
    if mode == "replay" and not stored.exists():
        res["verdict"] = "NO_TEACHER_CONNECTOR"
        return res
    path = np.load(stored) if mode == "replay" else None
    c, path = connector(e, S, expected, path)
    res["connector"] = c
    if mode == "teacher" and path is not None:
        stored.parent.mkdir(parents=True, exist_ok=True)
        np.save(stored, path)
        res["connector_witness_sha256"] = hashlib.sha256(stored.read_bytes()).hexdigest()
    res["final"] = state(e)
    res["verdict"] = "X_CANDIDATE_1" if c["ok"] else ("X0_CONNECTOR_NO_PLAN" if not c["planned"] else "X0_CONNECTOR_FAILED")
    res["failed_phase"] = None if c["ok"] else "connector"
    return res


def cmd_trace(a):
    """Film support: re-run an item in REPLAY semantics (stored witnesses + the stored connector path, no planning) and record the full
    simulator state every --every steps; frames are rendered elsewhere from these EXACT states (mj_forward only, no simulation)."""
    A, S = env()
    TR, step0 = [], S.XArm7TableEnv.step

    def step(self, n=1):
        step0(self, n)
        self._k = getattr(self, "_k", 0) + n
        if self._k % a.every < n:
            TR.append(self.data.qpos.copy())
    S.XArm7TableEnv.step = step
    man = json.loads(Path(a.manifest).read_text())
    it = next(x for x in man["items"] if x["item"] == a.item)
    res = (run_y if it["kind"] == "Y" else run_x)(it, "replay", Path(a.out))
    p = Path(a.out) / "trace" / f"{it['item']}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, qpos=np.array(TR), verdict=res.get("verdict"), root=it["root_id"], every=a.every)
    print(json.dumps({"item": it["item"], "frames": len(TR), "verdict": res.get("verdict"), "trace": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}))


def cmd_item(a):
    man = json.loads(Path(a.manifest).read_text())
    it = next(x for x in man["items"] if x["item"] == a.item)
    out = Path(a.out)
    t0, c0 = time.time(), time.process_time()
    try:
        res = (run_y if it["kind"] == "Y" else run_x)(it, a.mode, out)
    except Exception as err:                                                  # an environment error is MISSING, never a label
        res = {"verdict": "ENV_ERROR", "error": repr(err), "trace": traceback.format_exc()[-1200:]}
    res.update(item=it["item"], mode=a.mode, wall_s=round(time.time() - t0, 2), cpu_s=round(time.process_time() - c0, 2), pid=os.getpid(),
               affinity=sorted(os.sched_getaffinity(0)))
    p = out / "items" / f"{it['item']}.{a.mode}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=str))


def guard(lim):
    t = None
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (h / "name").read_text().strip() == "coretemp":
                for lab in h.glob("temp*_label"):
                    if lab.read_text().startswith("Package"):
                        t = int(Path(str(lab).replace("_label", "_input")).read_text()) / 1000
        except OSError:
            pass
    mem = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l}
    swap = (mem["SwapTotal"] - mem["SwapFree"]) / 1024
    f = {"pkg_temp_c": t, "mem_available_gb": mem["MemAvailable"] / 2**20, "swap_used_mb": swap, "load1": os.getloadavg()[0]}
    bad = [k for k, ok in (("TEMP", t is not None and t < lim["temp"]), ("SENSOR", t is not None), ("MEM", f["mem_available_gb"] >= lim["mem_gb"]),
                           ("SWAP_GROWTH", swap - lim["swap0"] <= lim["swap_growth_mb"])) if not ok]
    return f, bad


def cmd_work(a):
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out); (out / "items").mkdir(parents=True, exist_ok=True)
    i, n = map(int, a.shard.split("/"))
    items = [x for k, x in enumerate(x for x in man["items"] if x["phase"] in a.phase.split(",")) if k % n == i]
    lim = {"temp": 75.0, "mem_gb": 16.0, "swap_growth_mb": 256.0, "swap0": guard({"temp": 999, "mem_gb": 0, "swap_growth_mb": 1e9, "swap0": 0})[0]["swap_used_mb"]}
    log = open(out / f"work_{i}of{n}.jsonl", "a")
    t_start = time.time()
    for it in items:
        f, bad = guard(lim)
        log.write(json.dumps({"utc": time.strftime("%FT%TZ", time.gmtime()), "item": it["item"], "guard": f, "violations": bad}) + "\n"); log.flush()
        if bad:
            (out / f"STOP_GUARD_{i}").write_text(json.dumps({"item": it["item"], "violations": bad, "facts": f}))
            return
        if time.time() - t_start > a.wall_cap_s:
            (out / f"STOP_WALLCAP_{i}").write_text(it["item"]); return
        for mode in (("teacher", "replay") if it.get("independent_replay", True) else ("teacher",)):
            if (out / "items" / f"{it['item']}.{mode}.json").exists():
                continue                                                      # idempotent resume
            try:
                subprocess.run([sys.executable, __file__, "item", "--manifest", a.manifest, "--item", it["item"], "--mode", mode, "--out", str(out)],
                               timeout=a.timeout_s, check=False)
            except subprocess.TimeoutExpired:
                (out / "items" / f"{it['item']}.{mode}.json").write_text(json.dumps({"item": it["item"], "mode": mode, "verdict": "CENSORED_TIMEOUT",
                                                                                     "timeout_s": a.timeout_s}))


def cmd_policy(a):
    A, S = env()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import mujoco
    g = A.env()
    rec = root("3ds66-lap3-w0047-s000")
    e = A.build(rec["objects"], rec["initial_cells"]); A.settle(e, {k: tuple(v) for k, v in rec["initial_cells"].items()})
    res = {"mujoco": mujoco.__version__, "numpy": np.__version__, "python": sys.version.split()[0], "teacher_version": A.TEACHER_VERSION,
           "protocol": g["protocol"], "config_hash": g["config_hash"], "source_files": g["source_files"], "rrt_iters": A.RRT_ITERS,
           "return_to_ready": g["geometry"]["return_to_ready"], "checks_on_sample_root": policy_checks(e, A, S, rec),
           "gripper_open_cmd": float(S.GRIPPER_OPEN), "time_parameterize": S.time_parameterize.__module__}
    (out / "POLICY.json").write_text(json.dumps(res, indent=1, default=str)); print(json.dumps(res, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["policy", "work", "item", "trace"]); ap.add_argument("--manifest"); ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="SMOKE"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--item"); ap.add_argument("--mode", default="teacher")
    ap.add_argument("--timeout-s", type=int, default=900); ap.add_argument("--every", type=int, default=40); ap.add_argument("--wall-cap-s", type=int, default=3 * 3600)
    a = ap.parse_args()
    {"policy": cmd_policy, "work": cmd_work, "item": cmd_item, "trace": cmd_trace}[a.cmd](a)
