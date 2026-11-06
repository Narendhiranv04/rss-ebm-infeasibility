"""Placement-only label sidecar, smoke test (planar 3R arm, 2D class-quota roots). Protocol PLACEMENT_SMOKE_V1.

Event V_place(S, o, d): with object o already grasped in scene S (the arm at a declared side-grasp mode, fingers closed on o), can
carry -> release -> straight retreat to cell d be certified? It is NOT pick access from home (no pregrasp / approach), not the
ordered sequence X, not the target's Y. Everything physical is the dataset teacher's own code (latent-causal-tamp: skill, motion,
planner, geometry, config.toml); nothing in dataset_bd_classquota/ is written.

  post-grasp modes  skill.directions (24 tool axes x 2 elbows = 48): IK, open-finger grasp pose clear, closure certified
                    (skill.screen_pick without its approach line, which belongs to pick access)
  place modes       skill.screen_place (48: IK, held pose, release, straight retreat), nominal attachment as in skill.plan_skill
  pairs             48 x 48 = 2,304 nominal; a pair with a definitely invalid mode is geometrically rejected; valid pairs are tried
                    same-branch first, then by joint distance, at most BUDGET["max_pairs"] carry plans (planner.plan) per query
  witness           close_attach -> carry -> release -> retreat, re-certified by motion.certify_trajectory, place error <= tolerance
  label             1 = a witness verified; 0 = the declared budget ran without one (never an impossibility proof unless every pair
                    was rejected by definite screens: is_certificate); null = not attempted / statically ineligible destination

    python placement_smoke.py serial <out>            # Pass A: one root, one object, one destination, one process
    python placement_smoke.py run <out> [workers]     # Pass B: the predeclared roster, resumable, resource-guarded
"""
import hashlib, json, os, resource, signal, sys, threading, time
from multiprocessing import Pool, active_children
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                           # never write __pycache__ into the original dataset's code folders
HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "dataset_bd_classquota"
LCT = DATA / "code" / "latent-causal-tamp"
sys.path[:0] = [str(DATA / "code" / "dataset-bd"), str(DATA / "code"), str(LCT)]
os.chdir(LCT)                                            # config.toml is read relative to the teacher's folder
import bd_2d as A, geometry as ge, motion as mo, planner as pn, skill as sk  # noqa: E402
from scene import scene_from_dict, scene_hash, substitute_pose  # noqa: E402

SCHEMA = "placement_smoke_v1"
BUDGET = {"max_pairs": 8, "pair_max_nodes": 150, "pair_wall_s": 2.0, "query_wall_s": 12.0, "query_hard_s": 40.0, "global_wall_s": 240.0}
GUARD = {"min_mem_available_mb": 2048, "max_swap_growth_mb": 512, "max_temp_c": 95.0, "min_disk_free_mb": 1024}
CPUS = [c for c in range(32) if c % 16 < 14]             # 28 logical CPUs: cores 0-13 with their SMT siblings 16-29
H = lambda s: hashlib.sha256(f"PLACEMENT_SMOKE_V1|{s}".encode()).hexdigest()  # noqa: E731
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
DEFINITE = {"NO_IK", "GRASP_POSE", "CLOSURE", "PLACE_POSE", "RELEASE", "RETREAT"}


def protocol():
    G = A.env()
    src = {f: sha_file(LCT / f)[:16] for f in ("skill.py", "motion.py", "planner.py", "geometry.py", "config.toml")}
    p = {"name": "PLACEMENT_SMOKE_V1", "budget": BUDGET, "teacher_config_hash": G["config_hash"], "robot_protocol": G["protocol"],
         "teacher_sources": src, "script": sha_file(__file__)[:16]}
    return {**p, "placement_protocol_id": hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16],
            "budget_hash": hashlib.sha256(json.dumps(BUDGET, sort_keys=True).encode()).hexdigest()[:16]}


def load_root(rid):
    return json.loads((DATA / "raw" / "roots" / f"{rid}.json").read_text())


def roster():
    """Predeclared before any teacher call: per class 2, 3, 1, 0 the selected root with the smallest hash among those with >= 2
    movable objects (a static property; e.g. the first K0 by hash holds T alone); two objects (the
    archived minimal repair's first two, topped up by hash); four destinations each: the archived parking cell and one cell per
    distance band (near 1-2, mid 3-4, far >= 5 Chebyshev) outside every historical destination; plus two intermediate queries."""
    sel, out = json.loads((DATA / "selected.json").read_text())["selected"], []
    cheb = lambda a, b: max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))  # noqa: E731
    for cls in ("2", "3", "1", "0"):
        rid = next(x for x in sorted(sel[cls], key=H) if len(load_root(x)["roles"]) >= 2)
        r = load_root(rid)
        cells = {o["id"]: o["cell"] for o in r["scene"]["objects"]}
        mini = next((b["sequence"] for b in r["labels"]["B"] if b["role"] == "minimal" and b["y"]), [])
        objs = [o for o, _ in mini[:2]]
        objs += sorted((o for o in r["roles"] if o not in objs), key=H)[:2 - len(objs)]
        add = lambda o, d, tag, prefix=(): out.append({"root": rid, "cls": cls, "prefix": [list(x) for x in prefix], "object": o,  # noqa: E731
                                                       "dest": int(d), "dest_tag": tag})
        for o in objs:
            hist = {p["cell"] for p in r["parking"].get(o, [])} | set((r.get("alt_parking") or {}).get(o, []))
            hist |= {d for b in r["labels"]["B"] for x, d in b["sequence"] if x == o}
            free = [c for c in r["scene"]["legal_cells"] if c not in cells.values() and c not in hist and c != r["goal_cell"]]
            if r["parking"].get(o):
                add(o, r["parking"][o][0]["cell"], "historical_parking")
            for tag, lo, hi in (("near", 1, 2), ("mid", 3, 4), ("far", 5, 7)):
                band = [c for c in free if lo <= cheb(c, cells[o]) <= hi]
                if band and len([t for t in out if t["root"] == rid and t["object"] == o]) < 4:
                    add(o, min(band, key=lambda c: H(f"{rid}|{o}|{c}")), tag)
        if cls in ("2", "3") and len(mini) >= 2:
            (a, p), (b, q) = mini[0], mini[1]
            if cls == "2":
                add(b, q, "historical_next_move", [(a, p)])
            else:                                           # occupancy change: b to the cell a vacated (statically ineligible at the root)
                add(b, cells[a], "vacated_by_prefix", [(a, p)])
                add(b, cells[a], "vacated_by_prefix")
    for t in out:
        t["trial_id"] = H(json.dumps([t["root"], t["prefix"], t["object"], t["dest"]]))[:16]
    return out


def post_grasp(robot, ck0, obj, p):
    """skill.screen_pick's grasp-state checks (IK, open-finger pose, closure) without the approach line."""
    out = []
    for phi, b in sk.directions(p):
        c = sk.Candidate(phi, b, "OK")
        out.append(c)
        q = c.q_end = sk._ik(robot, obj.xy_m, (robot.grasp_x, 0.0), phi).get(b)
        if q is None:
            sk._fail(c, "NO_IK", why=mo.NO_IK)
            continue
        v = ck0[("start", False)].status_at(q, robot.open_gap)
        if v.status != ge.CLEAR:
            sk._fail(c, "GRASP_POSE", v)
            continue
        local = ge.tool_local(robot, q, obj.xy_m)
        gap = 2 * (obj.radius_m + abs(local[1]))
        if abs(local[1]) > p["contact_tol"] or not 0 <= local[0] <= robot.finger_len or gap < robot.closed_gap:
            sk._fail(c, "CLOSURE", why="CLOSURE_GEOMETRY")
            c.certified = True
            continue
        v = ge.certify(ck0[("start", True)], q, q, robot.open_gap, gap, p["evals"])
        if v.status != ge.CLEAR:
            sk._fail(c, "CLOSURE", v)
            continue
        c.gap, c.local = gap, tuple(local)
    return out


def witness_arrays(robot, oid, pick, path, place):
    t = mo.Trajectory(oid, held_local=pick.local, place_xy=tuple(ge.tool_point(robot, place.q_end, pick.local)))
    t.add(pick.q_end, robot.open_gap, None)
    for qs, gap, phase in (([pick.q_end], pick.gap, "close_attach"), (path[1:], pick.gap, "carry"), ([place.q_end], robot.open_gap, "release"),
                           (place.line[1:], robot.open_gap, "retreat")):
        for q in qs:
            t.add(q, gap, phase)
    return t.arrays()


def save_witness(out, robot, s_hash, oid, arrays, provenance):
    """Content-addressed trajectories/<id>.npz: the replayable record plus derived object positions, phase runs and travel."""
    core = {k: np.asarray(v) for k, v in arrays.items()}
    h = hashlib.sha256(f"{s_hash}|{oid}".encode())
    for k in sorted(core):
        h.update(f"{k}|{core[k].dtype.str}|{core[k].shape}".encode() + core[k].tobytes())
    wid = h.hexdigest()[:24]
    path = out / "trajectories" / f"{wid}.npz"
    if not path.exists():
        ph = [str(x) for x in core["phase"]]
        obj = [mo.target_xy(robot, core, ph[max(i - 1, 0)], q, ge.tool_point(robot, core["q"][1], core["held_local"])) for i, q in enumerate(core["q"])]
        runs = [i for i in range(len(ph)) if i == 0 or ph[i] != ph[i - 1]]
        extra = {"object_xy": np.array(obj), "waypoint_index": np.arange(len(core["q"])), "phase_run_start": np.array(runs),
                 "phase_run_name": np.array([ph[i] for i in runs]), "q_start": core["q"][0], "q_end": core["q"][-1],
                 "travel_rad": np.float64(np.linalg.norm(np.diff(core["q"], axis=0), axis=1).sum()),
                 "scene_hash": np.array(s_hash), "object_id": np.array(oid), "provenance": np.array(json.dumps(provenance, sort_keys=True))}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **core, **extra)
        os.replace(tmp, path)
    return wid


def reuse(r, task, robot, scene, picks, places, p, cfg):
    """An archived B step-1 witness of this exact (root scene, object, destination): its close_attach..retreat suffix, kept only if
    its grasp and place configurations are declared modes that pass this protocol's screens and it re-certifies here."""
    if task["prefix"]:
        return None
    for b in r["labels"]["B"]:
        if b["sequence"] and b["sequence"][0] == [task["object"], task["dest"]] and b["witnesses"]:
            z = np.load(DATA / "raw" / "witnesses" / b["witnesses"][0][:2] / f"{b['witnesses'][0]}.npz")
            ph = [str(x) for x in z["phase"]]
            k, end = ph.index("close_attach"), ph.index("retreat") + ph.count("retreat")
            arr = {"q": z["q"][k:end + 1], "gap": z["gap"][k:end + 1], "phase": z["phase"][k:end], "attached": z["attached"][k:end],
                   "held_local": z["held_local"], "place_xy": z["place_xy"]}
            i = next((n for n, c in enumerate(picks) if c.stage == "OK" and np.allclose(c.q_end, arr["q"][1], atol=1e-9)), None)
            rel = arr["q"][ph[k:].index("release")]
            j = next((n for n, c in enumerate(places) if c.stage == "OK" and np.allclose(c.q_end, rel, atol=1e-9)), None)
            v = mo.certify_trajectory(robot, scene, task["object"], arr, p["delta"], p["eps"], p["evals"])
            err = float(np.linalg.norm(arr["place_xy"] - np.asarray(scene.lattice.center(task["dest"]))))
            ok = i is not None and j is not None and all(x.status == ge.CLEAR for x in v) and err <= p["release_tol"]
            return {"archived_witness": b["witnesses"][0], "role": b["role"], "pick": i, "place": j, "recertified": ok, "arrays": arr}
    return None


def _hard(*_):
    raise TimeoutError("query hard wall cap")


def trial(task, out):
    """One placement query -> parts/<trial_id>.json (+ .npz mode arrays, + trajectories). The JSON is written last: its presence
    marks a finished query, so a resumed run skips it."""
    out, tid = Path(out), task["trial_id"]
    part = out / "parts" / f"{tid}.json"
    if part.exists():
        return json.loads(part.read_text())
    G, P = A.env(), protocol()
    cfg, robot = G["cfg"], G["robot"]
    p = sk.params(cfg)
    r = load_root(task["root"])
    scene = scene_from_dict(r["scene"])
    for o, c in task["prefix"]:
        scene = substitute_pose(scene, o, c)
    obj, s_hash = scene.get(task["object"]), scene_hash(scene)
    rec = {"schema_version": SCHEMA, "trial_id": tid, "source_root_id": task["root"], "source_root_hash": sha_file(DATA / "raw" / "roots" / f"{task['root']}.json"),
           "domain": "planar_2d", "split": None, "verified_class": task["cls"], "prefix_key": task["prefix"],
           "parent_reachability": "HYPOTHETICAL" if task["prefix"] else "ROOT", "hypothetical_scene_hash": s_hash,
           "object_id": obj.id, "source_cell": obj.cell, "destination_cell": task["dest"], "dest_tag": task["dest_tag"], "support_id": "table",
           "placement_protocol_id": P["placement_protocol_id"], "budget_hash": P["budget_hash"], "teacher_hash": G["config_hash"],
           "robot_profile_hash": scene.robot_profile_hash, "seed": cfg["planner2d"]["seed"]}
    occ = {o.cell for o in scene.objects}
    why = ("DEST_NOT_LEGAL" if task["dest"] not in scene.legal_cells else "DEST_OCCUPIED" if task["dest"] in occ else None)
    if why:
        rec.update(attempted=False, label=None, excluded=why, witness_refs=[])
        return _commit(out, part, rec, None)
    t0, c0 = time.time(), time.process_time()
    signal.signal(signal.SIGALRM, _hard)
    signal.alarm(int(BUDGET["query_hard_s"]))
    try:
        rec.update(_query(task, out, rec, scene, obj, s_hash, r, G, p, t0))
    except TimeoutError as e:                             # engineering abort: not a label
        rec.update(attempted=False, label=None, engineering_abort=str(e), witness_refs=[])
        rec.pop("_modes", None)
    finally:
        signal.alarm(0)
    rec.update(wall_s=round(time.time() - t0, 3), cpu_s=round(time.process_time() - c0, 3),
               worker_peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    return _commit(out, part, rec, rec.pop("_modes", None))


def _query(task, out, rec, scene, obj, s_hash, r, G, p, t0):
    cfg, robot = G["cfg"], G["robot"]
    dxy = np.asarray(scene.lattice.center(task["dest"]))
    picks = post_grasp(robot, mo.checkers(robot, scene, obj.id, None, None, p["delta"], p["eps"]), obj, p)
    local = (robot.grasp_x, 0.0)
    ck = mo.checkers(robot, scene, obj.id, dxy, local, p["delta"], p["eps"])
    places = sk.screen_place(robot, ck, dxy, local, 2 * obj.radius_m, p)
    n = len(picks)
    cat = np.zeros((n, n), np.int8)                       # 0 geometric reject (definite) 1 screen uncertain 2 untested 3 failed 4 verified
    for i, a in enumerate(picks):
        for j, b in enumerate(places):
            bad = [c for c in (a, b) if c.stage != "OK"]
            cat[i, j] = 2 if not bad else 0 if any(c.certified for c in bad) else 1
    outcome, attempts, wids = np.full((n, n), -1, np.int8), [], []
    old = reuse(r, task, robot, scene, picks, places, p, cfg)
    if old and old["recertified"]:
        i, j = old["pick"], old["place"]
        wids.append(save_witness(out, robot, s_hash, obj.id, old["arrays"], {"kind": "REUSED_ARCHIVED_SUFFIX", "archived_witness": old["archived_witness"],
                                                                             "archived_role": old["role"], "pick_mode": i, "place_mode": j}))
        cat[i, j], outcome[i, j] = 4, 1
        attempts.append({"pick": i, "place": j, "method": "REUSED_ARCHIVED_WITNESS_RECERTIFIED", "witness": wids[-1], "seconds": 0.0, "nodes": 0})
    pairs = sorted(zip(*np.nonzero(cat == 2)), key=lambda ij: (picks[ij[0]].branch != places[ij[1]].branch,
                                                              float(np.linalg.norm(picks[ij[0]].q_end - places[ij[1]].q_end)), int(ij[0]), int(ij[1])))
    pcfg = {**cfg["planner2d"], "max_nodes": BUDGET["pair_max_nodes"], "max_wall_seconds": BUDGET["pair_wall_s"]}
    cut = False
    for i, j in pairs:
        i, j = int(i), int(j)
        if sum(a["method"] != "REUSED_ARCHIVED_WITNESS_RECERTIFIED" for a in attempts) >= BUDGET["max_pairs"]:
            break
        if time.time() - t0 + BUDGET["pair_wall_s"] > BUDGET["query_wall_s"]:     # a new pair could not finish inside the cap
            cut = True
            break
        a, b = picks[i], places[j]
        guide = pose = None
        if a.branch == b.branch:
            guide = mo.carry_line(robot, obj.xy_m, dxy, a.local, a.phi, b.phi, a.branch, p["line_step_m"], p["max_ik_jump_rad"])[0]
            pose = ((*obj.xy_m, a.phi), (*dxy, b.phi), a.local, a.branch, p["line_step_m"], p["max_ik_jump_rad"])
        pl = pn.plan(ck[("held", False)], a.q_end, b.q_end, a.gap, pcfg, p["evals"], f"{rec['trial_id']}/carry{i}.{j}", True, guide, pose)
        att = {"pick": i, "place": j, "method": pl.reason, "status": pl.status, "nodes": pl.nodes, "seconds": round(pl.seconds, 3),
               "wall_clock_cut": pl.wall_clock_cut}
        ok = False
        if pl.status == ge.CLEAR:
            arr = witness_arrays(robot, obj.id, a, pl.path, b)
            v = mo.certify_trajectory(robot, scene, obj.id, arr, p["delta"], p["eps"], p["evals"])
            err = float(np.linalg.norm(arr["place_xy"] - dxy))
            ok = all(x.status == ge.CLEAR for x in v) and err <= p["release_tol"]
            att.update(replay="CERTIFIED" if ok else "REPLAY_REJECTED", place_error_m=err)
            if ok:
                att["witness"] = save_witness(out, robot, s_hash, obj.id, arr, {"kind": "NEW_VERIFIED", "pick_mode": i, "place_mode": j,
                                                                                "carry_method": pl.reason, "trial_id": rec["trial_id"]})
                wids.append(att["witness"])
        cat[i, j], outcome[i, j] = (4, 1) if ok else (3, 0)
        attempts.append(att)
    counts = {"total": n * n, "geometrically_rejected": int((cat == 0).sum()), "screen_uncertain_not_tested": int((cat == 1).sum()),
              "valid_not_motion_tested": int((cat == 2).sum()), "motion_attempted": int((cat >= 3).sum()),
              "verified_success": int((cat == 4).sum()), "budget_failed": int((cat == 3).sum())}
    counts["not_motion_tested"] = counts["screen_uncertain_not_tested"] + counts["valid_not_motion_tested"]
    assert counts["total"] == counts["geometrically_rejected"] + counts["not_motion_tested"] + counts["motion_attempted"]
    lab = decide(picks, places, counts, attempts, cut)
    modes = {"pick_stage": np.array([c.stage for c in picks], "U12"), "pick_definite": np.array([c.certified for c in picks]),
             "place_stage": np.array([c.stage for c in places], "U12"), "place_definite": np.array([c.certified for c in places]),
             "mode_phi_rad": np.array([c.phi for c in picks]), "mode_branch": np.array([c.branch for c in picks], np.int8),
             "pair_category": cat, "attempted_mask": cat >= 3, "pair_outcome": outcome,
             "attempt_pick_place": np.array([[a["pick"], a["place"]] for a in attempts], np.int16).reshape(-1, 2),
             "attempt_seconds": np.array([a["seconds"] for a in attempts], np.float32), "attempt_nodes": np.array([a["nodes"] for a in attempts], np.int32)}
    return {"attempted": True, **lab, "mode_summary": counts, "pick_modes_valid": sum(c.stage == "OK" for c in picks),
            "place_modes_valid": sum(c.stage == "OK" for c in places), "attempts": attempts, "witness_refs": sorted(set(wids)),
            "reuse_check": None if old is None else {k: old[k] for k in ("archived_witness", "role", "pick", "place", "recertified")},
            "query_wall_cut": cut, "_modes": modes}


def decide(picks, places, counts, attempts, cut):
    """Operational binary label. 0 is a certificate only when every pair fell to definite screens (scope: declared grasp set)."""
    if counts["verified_success"]:
        return {"label": 1, "negative_kind": None, "failure_phase": None, "is_certificate": False}
    def phase(cs):
        bad = [c.stage for c in cs if c.stage != "OK"]
        return max(set(bad), key=bad.count) if bad else None
    if counts["geometrically_rejected"] == counts["total"]:
        stage = "POST_GRASP_STATE" if not any(c.stage == "OK" for c in picks) else phase(places)
        return {"label": 0, "negative_kind": "CERTIFIED_SCREEN", "failure_phase": stage, "is_certificate": True}
    if not attempts:
        return {"label": 0, "negative_kind": "NO_PLAN_FOUND", "failure_phase": "SCREEN_UNCERTAIN", "is_certificate": False}
    left = counts["valid_not_motion_tested"] > 0
    replay = any(a.get("replay") == "REPLAY_REJECTED" for a in attempts)
    return {"label": 0, "negative_kind": "BUDGET_EXHAUSTED" if left or cut else "SCHEDULE_EXHAUSTED_NO_WITNESS",
            "failure_phase": "REPLAY" if replay else "CARRY", "is_certificate": False}


def _commit(out, part, rec, modes):
    part.parent.mkdir(parents=True, exist_ok=True)
    if modes is not None:
        tmp = part.with_suffix(f".{os.getpid()}.npz.tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **modes)
        os.replace(tmp, part.with_suffix(".npz"))
    tmp = part.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
    os.replace(tmp, part)
    return rec


def sample():
    """One resource reading: MemAvailable, swap used, CPU temperature (k10temp), disk free, children's summed RSS / PSS."""
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    temp = None
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        if (hw / "name").read_text().strip() == "k10temp":
            temp = int((hw / "temp1_input").read_text()) / 1000
    rss = pss = 0
    for ch in active_children():
        try:
            rss += int(next(l for l in open(f"/proc/{ch.pid}/status") if l.startswith("VmRSS")).split()[1])
            pss += int(next(l for l in open(f"/proc/{ch.pid}/smaps_rollup") if l.startswith("Pss:")).split()[1])
        except (OSError, StopIteration):
            pass
    st = os.statvfs(HERE)
    return {"t": time.time(), "mem_available_mb": mem["MemAvailable"] // 1024, "swap_used_mb": (mem["SwapTotal"] - mem["SwapFree"]) // 1024,
            "temp_c": temp, "load1": os.getloadavg()[0], "disk_free_mb": st.f_bavail * st.f_frsize // 2 ** 20,
            "children": len(active_children()), "children_rss_mb": rss // 1024, "children_pss_mb": pss // 1024}


def run(out, workers=28, tasks=None):
    """Pass B: the roster in `workers` processes pinned to CPUS, nice 10, one BLAS thread each; a guard samples resources every
    second and terminates the pool on a GUARD breach or the global wall cap. Resumable (finished parts are skipped)."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for t in [*out.glob("parts/*.tmp"), *out.glob("trajectories/*.tmp")]:      # torn writes of an interrupted run
        t.unlink()
    keep = {w for f in out.glob("parts/*.json") for w in json.loads(f.read_text())["witness_refs"]}
    for t in out.glob("trajectories/*.npz"):             # witnesses of queries an interrupted run never finished (they re-run)
        if t.stem not in keep:
            t.unlink()
    tasks = tasks or roster()
    os.sched_setaffinity(0, CPUS[:workers])
    os.nice(10)
    t0, r0 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    log, stop, done = [sample()], [], []
    swap0 = log[0]["swap_used_mb"]
    with Pool(workers, maxtasksperchild=4) as pool:
        it = pool.imap_unordered(_job, [(t, str(out)) for t in tasks])
        while len(done) < len(tasks) and not stop:
            try:
                done.append(it.next(timeout=1.0))
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ != "TimeoutError":
                    stop.append(f"ERROR:{type(e).__name__}:{e}")
            s = sample()
            log.append(s)
            stop += [m for m, bad in (("MEMORY", s["mem_available_mb"] < GUARD["min_mem_available_mb"]),
                                      ("SWAP", s["swap_used_mb"] - swap0 > GUARD["max_swap_growth_mb"]),
                                      ("TEMPERATURE", (s["temp_c"] or 0) > GUARD["max_temp_c"]),
                                      ("DISK", s["disk_free_mb"] < GUARD["min_disk_free_mb"]),
                                      ("GLOBAL_WALL", time.time() - t0 > BUDGET["global_wall_s"])) if bad]
        pool.terminate() if stop else pool.close()
        pool.join()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    res = {"workers": workers, "cpus": CPUS[:workers], "tasks": len(tasks), "finished": len(done), "stop": stop, "wall_s": round(time.time() - t0, 1),
           "children_cpu_s": round(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime, 1), "leftover_children": len(active_children()),
           "peak": {k: (min if k in ("mem_available_mb", "disk_free_mb") else max)(s[k] for s in log if s[k] is not None)
                    for k in log[0] if k != "t"}, "samples": log}
    (out / f"run_{int(t0)}.json").write_text(json.dumps(res, indent=1))
    return res


def _job(arg):
    return trial(*arg)


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    if cmd == "roster":
        print(json.dumps(roster(), indent=1))
    elif cmd == "serial":
        r = trial(roster()[0], out)
        print(json.dumps({k: v for k, v in r.items() if k != "attempts"}, indent=1))
    elif cmd == "run":
        res = run(out, int(sys.argv[3]) if len(sys.argv) > 3 else 28)
        print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))
