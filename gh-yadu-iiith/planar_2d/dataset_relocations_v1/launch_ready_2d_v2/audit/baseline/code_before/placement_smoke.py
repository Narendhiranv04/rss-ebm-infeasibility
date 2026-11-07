"""Placement-only label sidecar, smoke test (planar 3R arm, 2D class-quota roots). Protocol PLACEMENT_SMOKE_V2.

Event V_place(S, o, d): object o is ALREADY HELD at its source pose in a declared post-grasp mode; can carry -> release -> local
retreat to cell d be certified? No pick access (pregrasp, approach, open-finger grasp pose, closure motion), no start or home
connector, no X, no Y. Physics is the dataset teacher's own code (latent-causal-tamp: skill, motion, planner, geometry,
config.toml), imported unmodified; nothing in dataset_bd_classquota/ is written.

  held modes   skill.directions (24 tool axes x 2 elbows = 48): exact IK of the held tool-object transform, the contact model of
               skill.screen_pick (|lateral offset| <= tol, centre within the fingers, closed gap >= closed_gap), and the attached
               configuration clear in the carry context (robot, held disc, other objects, support). skill.screen_pick itself
               (approach line, open-finger pose, closure motion) runs as a DIAGNOSTIC only and never vetoes a label.
  place modes  skill.screen_place with each held mode's own attachment (held_local, gap).
  pairs        48 x 48 = 2,304; definite screen failures are hard-screened; valid pairs are planned in a fixed order (same elbow
               branch first, then joint distance, then index), at most WORK["pairs_per_query"] per query.
  carry plan   planner.plan's direct -> guide -> link-0 certificate, then RRT-Connect with the planner's own spaces, stopped only
               by counted work: extension attempts (failed ones included), collision evaluations, accepted nodes; shortcutting
               runs its full fixed number of rounds. Wall time never decides an outcome.
  witness      carry -> release -> retreat from the held configuration, re-certified phase by phase (verify_witness).
  label        1 = a verified witness; 0 = the whole declared work policy finished without one (is_certificate only when every
               pair fell to definite screens); null = ineligible, not attempted, or interrupted by a safety guard.

    python placement_smoke.py run <out> <workers> <cpus: physical16 | logical28 | serial> [ceiling_s]
"""
import hashlib, json, math, os, resource, signal, sys, time
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

SCHEMA = "placement_smoke_v2"
WORK = {"pairs_per_query": 8, "rrt_attempts": 1000, "rrt_evals": 20000, "rrt_nodes": 200, "shortcut_rounds": 150}
SAFETY = {"query_emergency_s": 1800, "temp_warn_c": 85.0, "temp_pause_c": 90.0, "temp_suspend_c": 92.0, "temp_resume_c": 85.0,
          "temp_cancel_c": 95.0, "temp_cancel_samples": 3,
          "min_mem_available_mb": 2048, "max_swap_growth_mb": 512, "min_disk_free_mb": 1024}
CPUSETS = {"serial": [0], "physical16": list(range(16)), "logical28": [c for c in range(32) if c % 16 < 14]}
EVENT = ("V_place: object already held at its source in a declared post-grasp mode (48 = 24 tool axes x 2 elbows); carry -> "
         "release -> straight local retreat to the destination cell, certified by the dataset teacher's geometry; no pick access, "
         "no closure motion, no home or q_ref connector, no target task")
CAT = {0: "HARD_SCREENED", 1: "SCREEN_INCONCLUSIVE_NOT_PLANNED", 2: "VALID_NOT_PLANNED", 3: "PLANNED_FAILED", 4: "PLANNED_VERIFIED",
       5: "REUSED_REVERIFIED"}
H = lambda s: hashlib.sha256(f"PLACEMENT_SMOKE_V1|{s}".encode()).hexdigest()  # noqa: E731   (the frozen roster's hash)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
sha_file = lambda p: sha(Path(p).read_bytes())  # noqa: E731


class Spent(Exception):
    pass


def protocol():
    G = A.env()
    src = {f: sha_file(LCT / f)[:16] for f in ("skill.py", "motion.py", "planner.py", "geometry.py", "config.toml")}
    sem = {"name": "PLACEMENT_SMOKE_V2", "event": EVENT, "work": WORK, "teacher_config_hash": G["config_hash"], "robot_protocol": G["protocol"],
           "teacher_sources": src, "planner_seed": G["cfg"]["planner2d"]["seed"]}
    return {"placement_protocol_id": sha(json.dumps(sem, sort_keys=True).encode())[:16], "budget_hash": sha(json.dumps(WORK, sort_keys=True).encode())[:16],
            "semantics": sem, "code_sha": sha_file(__file__)[:16]}


def load_root(rid):
    return json.loads((DATA / "raw" / "roots" / f"{rid}.json").read_text())


def roster():
    """The frozen PLACEMENT_SMOKE_V1 roster (unchanged code; compared byte-for-byte with roster.json by the checker)."""
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
            else:
                add(b, cells[a], "vacated_by_prefix", [(a, p)])
                add(b, cells[a], "vacated_by_prefix")
    for t in out:
        t["trial_id"] = H(json.dumps([t["root"], t["prefix"], t["object"], t["dest"]]))[:16]
    return out


def held_modes(robot, scene, obj, p):
    """The 48 declared post-grasp modes of obj resting at its source: (Candidate, carry Checker for its attachment). A mode is
    rejected only by its own held geometry: IK, the contact model, or the attached configuration not clear for carrying."""
    out = []
    for phi, b in sk.directions(p):
        c = sk.Candidate(phi, b, "OK")
        q = c.q_end = sk._ik(robot, obj.xy_m, (robot.grasp_x, 0.0), phi).get(b)
        ck = None
        if q is None:
            sk._fail(c, "NO_IK", why=mo.NO_IK)
        else:
            local = ge.tool_local(robot, q, obj.xy_m)
            gap = 2 * (obj.radius_m + abs(local[1]))
            if abs(local[1]) > p["contact_tol"] or not 0 <= local[0] <= robot.finger_len or gap < robot.closed_gap:
                sk._fail(c, "HELD_CONTACT_MODEL", why="CLOSURE_GEOMETRY")
                c.certified = True
            else:
                c.gap, c.local = gap, tuple(local)
                ck = mo.checkers(robot, scene, obj.id, None, c.local, p["delta"], p["eps"])[("held", False)]
                v = ck.status_at(q, gap)
                if v.status != ge.CLEAR:
                    sk._fail(c, "HELD_STATE", v)
        out.append(c)
    return out


def det_rrt(space, za, zb, ck, w):
    """planner._rrt_connect's extend/connect loop, stopped by counted work only (w: limits in, counters out)."""
    trees = [{"z": [za], "parent": [-1], "wp": [None]}, {"z": [zb], "parent": [-1], "wp": [None]}]
    e0 = ck.n_evals
    count = lambda: len(trees[0]["z"]) + len(trees[1]["z"])  # noqa: E731

    def spent():
        w["nodes"], w["search_evals"] = count(), ck.n_evals - e0
        stop = ("NODES" if w["nodes"] >= WORK["rrt_nodes"] else "ATTEMPTS" if w["attempts"] >= WORK["rrt_attempts"]
                else "EVALS" if w["search_evals"] >= WORK["rrt_evals"] else None)
        w["stop"] = stop
        return stop is not None

    def extend(tree, target):
        w["attempts"] += 1                                # every extension consumes one attempt, failed or not
        nodes = np.array(tree["z"])
        i = space.near(nodes, target)
        new = space.steer(nodes[i], target)
        wp = space.edge(nodes[i], new)
        if wp is None:
            return None
        tree["z"].append(new), tree["parent"].append(i), tree["wp"].append(wp)
        return new
    while not spent():
        new = extend(trees[0], space.sample())
        reached = None if new is None else extend(trees[1], new)
        while reached is not None and not np.array_equal(reached, new) and not spent():
            reached = extend(trees[1], new)
        if reached is not None and np.array_equal(reached, new):
            spent()
            w["stop"] = "SOLVED"
            nodes, edges = pn._join(trees, za)
            e1 = ck.n_evals
            path, _ = pn._shortcut(space, nodes, edges, WORK["shortcut_rounds"], math.inf)    # all rounds, no clock
            w["shortcut_evals"] = ck.n_evals - e1
            return path
        trees.reverse()
    return None


def plan_carry(ck, qa, qb, gap, cfg, p, key, guide=None, pose=None):
    """planner.plan with det_rrt in place of its wall-capped RRT. -> (status, path, method, work)."""
    pc, e0 = cfg["planner2d"], ck.n_evals
    w = {"attempts": 0, "nodes": 0, "search_evals": 0, "shortcut_evals": 0, "stop": None}
    if ge.certify(ck, qa, qb, gap, gap, p["evals"]).status == ge.CLEAR:
        return ge.CLEAR, [qa, qb], "DIRECT", {**w, "pre_evals": ck.n_evals - e0}
    if guide is not None and np.allclose(guide[0], qa) and np.allclose(guide[-1], qb):
        path = [qa] + list(guide[1:-1]) + [qb]
        if mo.certify_chain(ck, path, gap, p["evals"]).status == ge.CLEAR:
            return ge.CLEAR, path, "GUIDE_LINE", {**w, "pre_evals": ck.n_evals - e0}
    blocker = pn.link0_sweep_blocker(ck, qa, qb)
    w["pre_evals"] = ck.n_evals - e0
    if blocker:
        return ge.COLLISION, [], f"LINK0_SWEEP:{blocker}", w
    rng = np.random.default_rng(pn.call_seed(pc["seed"], key))                       # crc32 of the query/mode key
    if pose is None:
        space, za, zb, name = pn.JointSpace(ck, gap, pc["edge_max_evals"], rng, pc["extend_step_rad"]), qa, qb, "RRT_JOINT"
    else:
        (za, zb, local, branch, step, jump), name = pose, "RRT_OBJECT_POSE"
        space = pn.PoseSpace(ck, gap, pc["edge_max_evals"], rng, pc["pose_step_m"], pc["pose_phi_weight_m"], local, branch, step, jump)
        za, zb = np.asarray(za, float), np.asarray(zb, float)
    path = det_rrt(space, za, zb, ck, w)
    if path is None:
        return ge.UNRESOLVED, [], f"WORK_BUDGET_EXHAUSTED:{w['stop']}", w
    if not (np.allclose(path[0], qa, atol=1e-9) and np.allclose(path[-1], qb, atol=1e-9)):
        return ge.UNRESOLVED, [], "ENDPOINT_IK_MISMATCH", w
    return ge.CLEAR, [qa] + path[1:-1] + [qb], name, w


def witness_arrays(robot, oid, pick, path, place):
    t = mo.Trajectory(oid, held_local=pick.local, place_xy=tuple(ge.tool_point(robot, place.q_end, pick.local)))
    t.add(pick.q_end, pick.gap, None)                    # already held: the record starts attached, fingers closed on o
    for qs, gap, phase in ((path[1:], pick.gap, "carry"), ([place.q_end], robot.open_gap, "release"), (place.line[1:], robot.open_gap, "retreat")):
        for q in qs:
            t.add(q, gap, phase)
    return t.arrays()


def verify_witness(robot, scene, oid, arr, dest_xy, p):
    """Phase-aware replay of a placement-only record: carry+ release retreat+, attached exactly while carrying, constant held gap,
    the object at its source at the start, released at the destination; then motion.certify_trajectory segment by segment."""
    ph, gap = [str(x) for x in arr["phase"]], np.asarray(arr["gap"])
    k = ph.index("release") if "release" in ph else -1
    shape = k > 0 and set(ph[:k]) == {"carry"} and ph[k + 1:] and set(ph[k + 1:]) == {"retreat"} and ph.count("release") == 1
    held = bool(shape) and np.array_equal(arr["attached"], np.array([x == "carry" for x in ph])) and np.all(gap[:k + 1] == gap[0]) \
        and gap[0] < robot.open_gap and np.all(gap[k + 1:] == robot.open_gap)
    err = float(np.linalg.norm(np.asarray(arr["place_xy"]) - dest_xy))
    try:
        v = mo.certify_trajectory(robot, scene, oid, arr, p["delta"], p["eps"], p["evals"])
        cert, why = sum(x.status == ge.CLEAR for x in v), next((f"{i}:{x.status}:{x.pair}" for i, x in enumerate(v) if x.status != ge.CLEAR), None)
    except ValueError as e:                               # check_record: limits, gaps, attachment, source / place consistency
        cert, why = 0, f"RECORD:{e}"
    ok = bool(shape and held and why is None and err <= p["release_tol"])
    return ok, {"phase_shape": bool(shape), "attachment_and_gaps": bool(held), "segments_certified": int(cert), "segments": len(ph),
                "first_failure": why, "place_error_m": err, "scope": "PLACEMENT_ONLY_FROM_HELD_STATE"}


def save_witness(out, robot, ident, oid, arrays, provenance):
    """Content-addressed trajectories/<id>.npz; id = sha256(protocol | scene | object | canonical arrays)."""
    core = {k: np.asarray(v) for k, v in arrays.items()}
    h = hashlib.sha256(ident.encode())
    for k in sorted(core):
        h.update(f"{k}|{core[k].dtype.str}|{core[k].shape}".encode() + core[k].tobytes())
    wid = h.hexdigest()[:24]
    path = out / "trajectories" / f"{wid}.npz"
    if not path.exists():
        ph = [str(x) for x in core["phase"]]
        obj = [mo.target_xy(robot, core, ph[max(i - 1, 0)], q, ge.tool_point(robot, core["q"][0], core["held_local"])) for i, q in enumerate(core["q"])]
        runs = [i for i in range(len(ph)) if i == 0 or ph[i] != ph[i - 1]]
        extra = {"object_xy": np.array(obj), "waypoint_index": np.arange(len(core["q"])), "phase_run_start": np.array(runs),
                 "phase_run_name": np.array([ph[i] for i in runs]), "q_start": core["q"][0], "q_end": core["q"][-1],
                 "travel_rad": np.float64(np.linalg.norm(np.diff(core["q"], axis=0), axis=1).sum()), "identity": np.array(ident),
                 "object_id": np.array(oid), "provenance": np.array(json.dumps(provenance, sort_keys=True))}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **core, **extra)
        os.replace(tmp, path)
    return wid


def reuse(r, task, robot, scene, picks, places, p):
    """An archived B witness whose FIRST move is this (root scene, object, destination): its carry..retreat suffix, kept when its
    start and release configurations are declared modes valid here and verify_witness accepts it."""
    if task["prefix"]:
        return None
    for b in r["labels"]["B"]:
        if b["sequence"] and b["sequence"][0] == [task["object"], task["dest"]] and b["witnesses"]:
            z = np.load(DATA / "raw" / "witnesses" / b["witnesses"][0][:2] / f"{b['witnesses'][0]}.npz")
            ph = [str(x) for x in z["phase"]]
            k, end = ph.index("carry"), len(ph) - ph[::-1].index("retreat")
            arr = {"q": z["q"][k:end + 1], "gap": z["gap"][k:end + 1], "phase": z["phase"][k:end], "attached": z["attached"][k:end],
                   "held_local": z["held_local"], "place_xy": z["place_xy"]}
            i = next((n for n, c in enumerate(picks) if c.stage == "OK" and np.allclose(c.q_end, arr["q"][0], atol=1e-9)), None)
            rel = arr["q"][[str(x) for x in arr["phase"]].index("release")]
            j = next((n for n, c in enumerate(places.get(i, [])) if c.stage == "OK" and np.allclose(c.q_end, rel, atol=1e-9)), None)
            ok, ver = verify_witness(robot, scene, task["object"], arr, np.asarray(scene.lattice.center(task["dest"])), p)
            return {"archived_witness": b["witnesses"][0], "role": b["role"], "pick": i, "place": j, "reverified": bool(ok and j is not None),
                    "verification": ver, "arrays": arr}
    return None


def _alarm(*_):
    raise TimeoutError("QUERY_EMERGENCY_WALL")


def trial(task, out):
    """One placement query -> parts/<trial_id>.json (+ .npz mode arrays, + trajectories); the JSON is written last. A finished or
    ineligible record is final; an interrupted one (completed=false) is re-run identically on resume."""
    out, tid = Path(out), task["trial_id"]
    part = out / "parts" / f"{tid}.json"
    if final(part):
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
           "edited_objects": [o for o, _ in task["prefix"]], "object_id": obj.id, "source_cell": obj.cell, "destination_cell": task["dest"],
           "dest_tag": task["dest_tag"], "support_id": "table", "placement_protocol_id": P["placement_protocol_id"], "budget_hash": P["budget_hash"],
           "code_sha": P["code_sha"], "teacher_hash": G["config_hash"], "robot_profile_hash": scene.robot_profile_hash}
    why = "DEST_NOT_LEGAL" if task["dest"] not in scene.legal_cells else "DEST_OCCUPIED" if task["dest"] in {o.cell for o in scene.objects} else None
    if why:
        rec.update(eligible=False, attempted=False, completed=False, label=None, excluded=why, witness_refs=[], safety_interruption=None)
        return commit(part, rec, None)
    t0, c0 = time.time(), time.process_time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SAFETY["query_emergency_s"])
    try:
        rec.update(query(task, out, rec, scene, obj, r, G, p, P))
    except TimeoutError as e:                             # emergency only: no observed label
        rec.update(eligible=True, attempted=True, completed=False, label=None, safety_interruption=str(e), witness_refs=[])
        rec.pop("_modes", None)
    finally:
        signal.alarm(0)
    rec["volatile"] = {"wall_s": round(time.time() - t0, 3), "cpu_s": round(time.process_time() - c0, 3),
                       "worker_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    return commit(part, rec, rec.pop("_modes", None))


def query(task, out, rec, scene, obj, r, G, p, P):
    cfg, robot = G["cfg"], G["robot"]
    dxy = np.asarray(scene.lattice.center(task["dest"]))
    ck0 = mo.checkers(robot, scene, obj.id, None, None, p["delta"], p["eps"])
    pick_diag = sk.screen_pick(robot, ck0, obj.xy_m, obj.radius_m, p)             # pick ACCESS, diagnostic only
    picks = held_modes(robot, scene, obj, p)
    groups, places, cks = {}, {}, {}
    for i, c in enumerate(picks):                        # place screens per distinct attachment (held_local, gap)
        if c.stage == "OK":
            key = (round(c.local[0], 12), round(c.local[1], 12), round(c.gap, 12))
            if key not in groups:
                ck = mo.checkers(robot, scene, obj.id, dxy, c.local, p["delta"], p["eps"])
                groups[key] = (sk.screen_place(robot, ck, dxy, c.local, c.gap, p), ck)
            places[i], cks[i] = groups[key]
    n = len(picks)
    cat = np.zeros((n, n), np.int8)
    for i, a in enumerate(picks):
        for j in range(n):
            bad = [a] if a.stage != "OK" else [x for x in [places[i][j]] if x.stage != "OK"]
            cat[i, j] = 2 if not bad else 0 if any(x.certified for x in bad) else 1
    outcome, attempts, wids = np.full((n, n), -1, np.int8), [], []
    ident = lambda: f"{P['placement_protocol_id']}|{rec['robot_profile_hash']}|{rec['hypothetical_scene_hash']}|{obj.id}"  # noqa: E731
    old = reuse(r, task, robot, scene, picks, places, p)
    if old and old["reverified"]:
        i, j = old["pick"], old["place"]
        w = save_witness(out, robot, ident(), obj.id, old["arrays"], {"evidence_source": "REUSED_REVERIFIED", "archived_witness": old["archived_witness"],
                                                                       "archived_role": old["role"], "held_mode": i, "place_mode": j})
        wids.append(w)
        cat[i, j], outcome[i, j] = 5, 1
        attempts.append({"pick": i, "place": j, "evidence_source": "REUSED_REVERIFIED", "method": "ARCHIVED_SUFFIX", "witness": w,
                         "verification": old["verification"], "work": None})
    pairs = sorted(((int(i), int(j)) for i, j in zip(*np.nonzero(cat == 2))),
                   key=lambda ij: (picks[ij[0]].branch != places[ij[0]][ij[1]].branch, float(np.linalg.norm(picks[ij[0]].q_end - places[ij[0]][ij[1]].q_end)), ij))
    for i, j in pairs[:WORK["pairs_per_query"]]:
        a, b = picks[i], places[i][j]
        guide = pose = None
        if a.branch == b.branch:
            guide = mo.carry_line(robot, obj.xy_m, dxy, a.local, a.phi, b.phi, a.branch, p["line_step_m"], p["max_ik_jump_rad"])[0]
            pose = ((*obj.xy_m, a.phi), (*dxy, b.phi), a.local, a.branch, p["line_step_m"], p["max_ik_jump_rad"])
        st, path, method, work = plan_carry(cks[i][("held", False)], a.q_end, b.q_end, a.gap, cfg, p, f"{rec['trial_id']}/carry{i}.{j}", guide, pose)
        att = {"pick": i, "place": j, "evidence_source": "NEW_PLANNED", "method": method, "status": st, "work": work, "certified_blocked": st == ge.COLLISION}
        ok = False
        if st == ge.CLEAR:
            arr = witness_arrays(robot, obj.id, a, path, b)
            ok, att["verification"] = verify_witness(robot, scene, obj.id, arr, dxy, p)
            if ok:
                att["witness"] = save_witness(out, robot, ident(), obj.id, arr, {"evidence_source": "NEW_PLANNED", "held_mode": i, "place_mode": j,
                                                                                 "carry_method": method, "trial_id": rec["trial_id"]})
                wids.append(att["witness"])
        cat[i, j], outcome[i, j] = (4, 1) if ok else (3, 0)
        attempts.append(att)
    c = {CAT[k]: int((cat == k).sum()) for k in CAT}
    m = {"total": n * n, "hard_screened": c["HARD_SCREENED"], "motion_attempted": c["PLANNED_FAILED"] + c["PLANNED_VERIFIED"],
         "reused_reverified": c["REUSED_REVERIFIED"], "not_motion_attempted": c["SCREEN_INCONCLUSIVE_NOT_PLANNED"] + c["VALID_NOT_PLANNED"],
         "attempted_success": c["PLANNED_VERIFIED"], "attempted_failure": c["PLANNED_FAILED"], "screen_inconclusive": c["SCREEN_INCONCLUSIVE_NOT_PLANNED"],
         "valid_not_planned": c["VALID_NOT_PLANNED"], "n_mode_witnesses": c["PLANNED_VERIFIED"] + c["REUSED_REVERIFIED"]}
    assert m["total"] == m["hard_screened"] + m["motion_attempted"] + m["reused_reverified"] + m["not_motion_attempted"]
    lab = decide(picks, places, m, attempts)
    tot = lambda k: sum((a["work"] or {}).get(k, 0) for a in attempts)  # noqa: E731
    modes = {"held_stage": np.array([x.stage for x in picks], "U20"), "held_definite": np.array([x.certified for x in picks]),
             "pick_access_stage": np.array([x.stage for x in pick_diag], "U20"),
             "place_stage": np.array([[x.stage for x in places[i]] if i in places else [""] * n for i in range(n)], "U20"),
             "mode_phi_rad": np.array([x.phi for x in picks]), "mode_branch": np.array([x.branch for x in picks], np.int8),
             "pair_category": cat, "attempted_mask": (cat == 3) | (cat == 4), "pair_outcome": outcome,
             "attempt_pick_place": np.array([[a["pick"], a["place"]] for a in attempts], np.int16).reshape(-1, 2),
             "attempt_rrt_attempts": np.array([(a["work"] or {}).get("attempts", 0) for a in attempts], np.int32),
             "attempt_evals": np.array([sum((a["work"] or {}).get(k, 0) for k in ("pre_evals", "search_evals", "shortcut_evals")) for a in attempts], np.int64)}
    return {"eligible": True, "attempted": True, "completed": True, **lab, "safety_interruption": None, "mode_summary": m,
            "held_modes_valid": sum(x.stage == "OK" for x in picks), "pick_access_modes_ok": sum(x.stage == "OK" for x in pick_diag),
            "place_screen_groups": len(groups), "attempts": attempts, "witness_refs": list(dict.fromkeys(wids)),
            "work_budget_consumed": {"pairs_planned": sum(a["evidence_source"] == "NEW_PLANNED" for a in attempts), "rrt_attempts": tot("attempts"),
                                     "rrt_nodes": tot("nodes"), "evals": tot("pre_evals") + tot("search_evals") + tot("shortcut_evals")},
            "reuse_check": None if old is None else {k: old[k] for k in ("archived_witness", "role", "pick", "place", "reverified", "verification")},
            "_modes": modes}


def decide(picks, places, m, attempts):
    """Operational binary label of a completed query; a certificate only when every pair fell to definite screens."""
    if m["n_mode_witnesses"]:
        return {"label": 1, "negative_kind": None, "negative_scope": None, "failure_phase": None, "is_certificate": False}
    if m["hard_screened"] == m["total"]:
        valid = [i for i, c in enumerate(picks) if c.stage == "OK"]
        bad = [x.stage for i in valid for x in places[i] if x.stage != "OK"]
        return {"label": 0, "negative_kind": "CERTIFIED_SCREEN", "negative_scope": "DECLARED_48x48_CATALOGUE_DEFINITE_STATIC_SCREENS",
                "failure_phase": "HELD_STATE" if not valid else max(set(bad), key=bad.count), "is_certificate": True}
    scope = f"FINITE_POLICY:{WORK['pairs_per_query']}_PAIRS,PER_PAIR_WORK_BUDGET"
    if not attempts:
        return {"label": 0, "negative_kind": "NO_PLAN_FOUND", "negative_scope": scope + ",SCREEN_INCONCLUSIVE", "failure_phase": "SCREEN", "is_certificate": False}
    replay = any(a.get("verification") and not a.get("witness") for a in attempts)
    return {"label": 0, "negative_kind": "WORK_BUDGET_EXHAUSTED" if m["valid_not_planned"] else "SCHEDULE_EXHAUSTED_NO_WITNESS",
            "negative_scope": scope, "failure_phase": "REPLAY" if replay else "CARRY", "is_certificate": False}


def final(part):
    if not part.exists():
        return False
    r = json.loads(part.read_text())
    return bool(r["completed"] or not r["eligible"])


def commit(part, rec, modes):
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
    """One resource reading: MemAvailable, swap used, CPU temperature (k10temp), load, disk free, children's summed RSS / PSS."""
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    temp = None
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        if (hw / "name").read_text().strip() == "k10temp":
            temp = int((hw / "temp1_input").read_text()) / 1000
    rss = pss = 0
    kids = active_children()
    for ch in kids:
        try:
            rss += int(next(l for l in open(f"/proc/{ch.pid}/status") if l.startswith("VmRSS")).split()[1])
            pss += int(next(l for l in open(f"/proc/{ch.pid}/smaps_rollup") if l.startswith("Pss:")).split()[1])
        except (OSError, StopIteration):
            pass
    st = os.statvfs(HERE)
    return {"t": round(time.time(), 2), "mem_available_mb": mem["MemAvailable"] // 1024, "swap_used_mb": (mem["SwapTotal"] - mem["SwapFree"]) // 1024,
            "temp_c": temp, "load1": os.getloadavg()[0], "disk_free_mb": st.f_bavail * st.f_frsize // 2 ** 20, "children": len(kids),
            "children_rss_mb": rss // 1024, "children_pss_mb": pss // 1024}


def run(out, workers, cpus, ceiling_s, tasks):
    """The frozen roster on `workers` processes pinned to `cpus` (affinity, nice 10, one BLAS thread each; no cgroup quota). The
    dispatcher starts a query only below temp_pause_c; at temp_suspend_c every worker is stopped (SIGSTOP) until temp_resume_c
    (budgets count work, so pausing cannot change an outcome); a safety breach terminates the pool, and every query then in flight
    is recorded completed=false / label=null (re-run identically on resume). Finished records are never re-run."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for t in [*out.glob("parts/*.tmp"), *out.glob("trajectories/*.tmp")]:      # no worker alive yet: torn writes and orphans go
        t.unlink()
    keep = {w for f in out.glob("parts/*.json") for w in json.loads(f.read_text())["witness_refs"]}
    orphans = [t for t in out.glob("trajectories/*.npz") if t.stem not in keep]
    for t in orphans:
        t.unlink()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    queue = [t for t in tasks if not final(out / "parts" / f"{t['trial_id']}.json")]
    todo0 = len(queue)
    t0, r0, log, stop, hot, inflight, done = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN), [sample()], None, 0, {}, []
    swap0 = log[0]["swap_used_mb"]
    pool, stopped, events = Pool(workers, maxtasksperchild=4), [], []

    def wake():
        for pid in stopped:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        stopped.clear()
    while (queue or inflight) and stop is None:
        s = sample()
        log.append(s)
        tc = s["temp_c"] or 0.0
        if tc >= SAFETY["temp_suspend_c"] and not stopped:                     # thermal duty cycle: freeze all workers
            stopped[:] = [ch.pid for ch in active_children()]
            for pid in stopped:
                os.kill(pid, signal.SIGSTOP)
            events.append(["SUSPEND", s["t"], tc])
        elif stopped and tc <= SAFETY["temp_resume_c"]:
            wake()
            events.append(["RESUME", s["t"], tc])
        s["suspended"] = bool(stopped)
        hot = hot + 1 if tc >= SAFETY["temp_cancel_c"] else 0
        stop = next((m for m, bad in (("TEMPERATURE", hot >= SAFETY["temp_cancel_samples"]), ("MEMORY", s["mem_available_mb"] < SAFETY["min_mem_available_mb"]),
                                      ("SWAP", s["swap_used_mb"] - swap0 > SAFETY["max_swap_growth_mb"]), ("DISK", s["disk_free_mb"] < SAFETY["min_disk_free_mb"]),
                                      ("RUN_CEILING", time.time() - t0 > ceiling_s)) if bad), None)
        while stop is None and queue and len(inflight) < workers and tc < SAFETY["temp_pause_c"]:
            t = queue.pop(0)
            inflight[t["trial_id"]] = (t, pool.apply_async(trial, (t, str(out))))
        for tid, (t, ar) in list(inflight.items()):
            if ar.ready():
                if not ar.successful():
                    stop = f"ERROR:{tid}"
                    try:
                        ar.get()
                    except Exception as e:  # noqa: BLE001
                        stop += f":{type(e).__name__}:{str(e)[:200]}"
                done.append(tid)
                del inflight[tid]
        time.sleep(0.5)
    wake()                                               # a stopped worker could not receive SIGTERM
    if stop:
        pool.terminate()
    else:
        pool.close()
    pool.join()
    for tid, (t, _) in inflight.items():                 # interrupted by a safety guard: no observed label
        part = out / "parts" / f"{tid}.json"
        if not final(part):
            commit(part, {"schema_version": SCHEMA, "trial_id": tid, "source_root_id": t["root"], "eligible": True, "attempted": True,
                          "completed": False, "label": None, "safety_interruption": stop, "witness_refs": []}, None)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    temps = [s["temp_c"] for s in log if s["temp_c"] is not None]
    res = {"workers": workers, "cpus": cpus, "hard_cpu_quota": False, "enforcement": "sched_setaffinity + fixed process count + nice 10 + 1 BLAS thread",
           "ceiling_s": ceiling_s, "safety": SAFETY, "tasks": len(tasks), "queued_at_start": todo0,
           "finished_now": len(done), "interrupted": sorted(inflight), "not_started": [t["trial_id"] for t in queue], "stop": stop,
           "orphans_removed_at_start": len(orphans), "wall_s": round(time.time() - t0, 1), "children_cpu_s": round(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime, 1),
           "leftover_children": len(active_children()), "samples_over_warn": sum(x >= SAFETY["temp_warn_c"] for x in temps),
           "samples_paused": sum(x >= SAFETY["temp_pause_c"] for x in temps), "thermal_events": events,
           "suspended_s": round(sum(b[1] - a[1] for a, b in zip(events[::2], events[1::2])), 1), "suspensions": len(events[::2]),
           "peak": {k: (min if k in ("mem_available_mb", "disk_free_mb") else max)(s[k] for s in log if s.get(k) is not None) for k in log[0] if k != "t"},
           "samples": log}
    (out / f"run_{int(t0)}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "roster":
        print(json.dumps(roster(), indent=1))
    elif cmd == "run":
        out, workers, cpus = Path(sys.argv[2]).resolve(), int(sys.argv[3]), CPUSETS[sys.argv[4]]
        tasks = json.loads((out.parent / "roster.json").read_text())
        assert tasks == roster(), "frozen roster differs from the roster code"
        res = run(out, workers, cpus[:workers], float(sys.argv[5]) if len(sys.argv) > 5 else 3600.0, tasks)
        print(json.dumps({k: v for k, v in res.items() if k not in ("samples", "safety")}, indent=1))
