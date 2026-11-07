"""SHELF66V3 placement teacher, physics layer: held state, carry/release/retreat planning, MuJoCo execution, verification.

Everything geometric is the dataset teacher's own code (shelf_scene via bd_shelf, imported UNMODIFIED): the 6x6 open-sided shelf,
the 32 level grasp poses (16 azimuths x mid/top grip), fixed-seed IK branches clear of the structure + arm ceiling, the arm/hand
check against every other object's no-overpass column with the fingers PRE-SHAPED to the grasp's open width (SHELF66V3), the 2-D
hand-footprint carry route (RRT-Connect, seed 0), the IK chains and the straight retreat.  New here:
  held state  restore the settled scene, put the arm at q_g (a branch of the grasp pose), fingers pre-shaped; certify that this
              pre-closure state touches nothing; close in place (run_plan 'close': 400 steps); certify the hold (both finger sides on
              the object, object <= 10 mm moved, neighbours <= 4 mm, robot touching nothing else).  No free move, hover or approach.
  execution   lift -> carry -> lower -> release -> retreat, servo-executed exactly as shelf_scene.execute (settle 40 per segment),
              recorded every 10 steps, every physics contact classified at every step.
  counted     RRT outer iterations / extensions / edge checks, IK calls, object / structure checks, simulation steps; snapshots at
              500 / 1500 iterations make nested RRT budgets exact (same seed => the 500-iteration search is a prefix of the 5000 one).
"""
import hashlib, json, math, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("SHELF_SRC", str(HERE.parents[1] / "dataset_bd_classquota_without_home_position_no_overpass")))
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC / "code" / "shelf66_v3"))
import bd_shelf  # noqa: E402
import mujoco  # noqa: E402

G = bd_shelf.env()
S = G["TS"]
import rrt_connect as _rc  # noqa: E402
from xarm7_cabinet_env import HAND_ROOT  # noqa: E402

TEACHER, SEED = "SHELF66V3_PLACE_V1", 0
HELD_SETTLE, CLOSE_STEPS, RELEASE_STEPS, SEG_SETTLE, FINAL_SETTLE, REC_EVERY = 50, 400, 300, 40, 200, 10
TOL = {"xy_m": 0.020, "z_m": 0.010, "tilt_deg": 10.0, "neighbour_m": 0.004, "hold_shift_m": 0.010, "attach_drift_m": 0.010,
       "lift_frac": 0.3, "settled_mps": 0.02, "envelope_margin_m": 0.0, "envelope_every": 2}
MILESTONES = (500, 1500, 5000)
PH = ("hold", "lift", "carry", "lower", "release", "retreat", "settle")
REC_HOOK = None                                                # (callable, every n steps): rendering only, never alters the physics
W, CUR = {}, {"iters": 5000, "screen_only": False, "last": {}}
KEYS = ("rrt_iters", "rrt_extend", "rrt_edge", "rrt_is_free", "shortcut_edge", "ik_calls", "object_checks", "static_checks", "sim_steps")
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
sha = lambda x: hashlib.sha256(x if isinstance(x, bytes) else json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()  # noqa


def reset_work():
    W.clear(); W.update({k: 0 for k in KEYS})


def _inc(k, f):
    def g(*a, **kw):
        W[k] += 1
        return f(*a, **kw)
    return g


class CountedRRT(_rc.RRTConnect):
    """rrt_connect.RRTConnect.plan with the wall-clock test removed (iteration ceiling only, as bd_shelf), work counted, and counter
    snapshots at the milestone iterations.  CUR['screen_only']: stop after the start/goal validity tests (footprint certificates)."""
    def __init__(self, is_free, edge_free, lo, hi, **kw):
        kw["max_time"], kw["max_iters"] = float("inf"), CUR["iters"]
        super().__init__(_inc("rrt_is_free", is_free), _inc("rrt_edge", edge_free), lo, hi, **kw)

    def _extend(self, T, q):
        W["rrt_extend"] += 1
        return super()._extend(T, q)

    def plan(self, q_start, q_goal):
        start, goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        self.stats = CUR["last"] = dict(iters=0, direct=False, found=False, milestones={})
        if not self.is_free(start):
            raise RuntimeError("RRT-Connect: start configuration is in collision")
        if not self.is_free(goal):
            raise RuntimeError("RRT-Connect: goal configuration is in collision")
        if CUR["screen_only"]:
            return [start, goal]
        if self.try_direct and self.edge_free(start, goal):
            self.stats.update(direct=True, found=True)
            return [start, goal]
        trees, a, it = [_rc._Tree(start, 2 * self.max_iters + 2), _rc._Tree(goal, 2 * self.max_iters + 2)], 0, 0
        for it in range(1, self.max_iters + 1):
            W["rrt_iters"] += 1
            A, B = trees[a], trees[1 - a]
            status, k = self._extend(A, self.rng.uniform(self.lo, self.hi))
            if status != _rc.TRAPPED:
                status_b, j = self._connect(B, A.Q[k])
                if status_b == _rc.REACHED:
                    Sx, s_node, Gx, g_node = (A, k, B, j) if a == 0 else (B, j, A, k)
                    self.stats.update(iters=it, found=True)
                    return Sx.branch(s_node)[::-1] + Gx.branch(g_node)[1:]
            if it in MILESTONES:
                self.stats["milestones"][it] = {**W, "cpu": time.process_time()}
            a = 1 - a
        self.stats.update(iters=it)
        return None


S.RRTConnect = CountedRRT
S.shortcut = (lambda f: lambda path, edge_free, **kw: f(path, _inc("shortcut_edge", edge_free), **kw))(S.shortcut)
for _n, _k in (("ik", "ik_calls"), ("_object_hits", "object_checks"), ("_static_hits", "static_checks")):
    setattr(S.XArm7TableEnv, _n, _inc(_k, getattr(S.XArm7TableEnv, _n)))
_ikc0 = S.XArm7TableEnv.ik_candidates
# IK branches are solved for the pose ROUNDED to the source's own cache key (0.1 mm), so the per-lineage shared cache is a pure
# function of its key (results never depend on which query filled it first; the source solves at the unrounded pose).
S.XArm7TableEnv.ik_candidates = lambda self, p, R, q_prefer, limit=4: _ikc0(self, np.round(np.asarray(p, float), 4), R, q_prefer, limit)
reset_work()


def source_hashes():
    sd = Path(S.__file__).parent
    files = [Path(__file__), Path(S.__file__), Path(_rc.__file__), sd / "rrt_star.py", sd / "xarm7_cabinet_env.py", Path(bd_shelf.__file__),
             SRC / "code/shelf66_v3/bd_3dt.py", SRC / "code/dataset-bd/bd_core.py"]
    return {p.name: sha(p.read_bytes()) for p in files}


def protocol():
    sem = {"name": TEACHER, "robot_protocol": G["protocol"], "source_teacher_config": G["config_hash"], "tol": TOL, "seed": SEED,
           "steps": [HELD_SETTLE, CLOSE_STEPS, RELEASE_STEPS, SEG_SETTLE, FINAL_SETTLE, REC_EVERY], "milestones": MILESTONES,
           "check_width": "grasp open_width in every object check (SHELF66V3); structure check as source (default finger qpos)",
           "mode_order": "policy3d.ranked: k-th azimuth nearest the robot, grip height alternating, then the other heights; IK branches nearest q_ready",
           "ik_candidates": "solved at the pose rounded to the source cache key (1e-4 m), cache shared per object set",
           "sources": {k: v[:16] for k, v in source_hashes().items()}}
    return sha(sem)[:16], sem


class Scene:
    """One hypothetical scene: the source env built at `cells`, settled once; restore() returns to exactly that state."""
    def __init__(self, objects, cells, ikc=None):
        self.e = e = bd_shelf.build(objects, {k: tuple(v) for k, v in cells.items()})
        if ikc is not None:                                    # shared within one object set (same arm ceiling)
            e._ikc = ikc.setdefault(round(max(o["height"] for o in objects), 9), {})
        self.st = np.empty(mujoco.mj_stateSize(e.model, STATE)); mujoco.mj_getState(e.model, e.data, self.st, STATE)
        m = e.model
        bid = m.geom_bodyid
        rob = set(S.XArm7TableEnv._subtree(m, m.body("link_base").id)) - {m.body("link_base").id}
        hand = set(S.XArm7TableEnv._subtree(m, m.body(HAND_ROOT).id))
        coll = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        self.robot = np.array([bool(coll[g]) and bid[g] in rob for g in range(m.ngeom)])
        self.hand = np.array([bid[g] in hand for g in range(m.ngeom)])
        self.left = np.array([bid[g] in hand and "left" in m.body(bid[g]).name for g in range(m.ngeom)])
        self.right = np.array([bid[g] in hand and "right" in m.body(bid[g]).name for g in range(m.ngeom)])
        self.struct = np.array([bool(coll[g]) and bid[g] == 0 for g in range(m.ngeom)])
        self.owner = np.full(m.ngeom, -1)
        for j, gs in enumerate(e._obj_geoms):
            self.owner[gs] = j

    def restore(self):
        e = self.e
        mujoco.mj_setState(e.model, e.data, self.st, STATE); mujoco.mj_forward(e.model, e.data)
        e.held, e._check_width, e._excuse = None, None, set()

    def contacts(self, i):
        """Contact counts now: robot-structure, robot-other object, held-other object, hand-held; left / right finger on held."""
        d = self.e.data
        if d.ncon == 0:
            return 0, 0, 0, 0, False, False
        g = d.contact.geom[:d.ncon]; a, b = g[:, 0], g[:, 1]
        oa, ob = self.owner[a], self.owner[b]
        other_a, other_b = (oa >= 0) & (oa != i), (ob >= 0) & (ob != i)
        rs = (self.robot[a] & self.struct[b]) | (self.robot[b] & self.struct[a])
        ro = (self.robot[a] & other_b) | (self.robot[b] & other_a)
        ho = ((oa == i) & other_b) | ((ob == i) & other_a)
        hh = ((oa == i) & self.hand[b]) | ((ob == i) & self.hand[a])
        L = ((oa == i) & self.left[b]) | ((ob == i) & self.left[a])
        R = ((oa == i) & self.right[b]) | ((ob == i) & self.right[a])
        return int(rs.sum()), int(ro.sum()), int(ho.sum()), int(hh.sum()), bool(L.any()), bool(R.any())


def poses(sc, name):
    e = sc.e
    sc.restore()
    i = e.obj_id(name); e._focus(i); e.objects[i]["pos"] = e.obj_pos(i)
    return S.grasp_poses(e.objects[i], e.table_z)


def screen(sc, name, dest, g):
    """Deterministic static classification of one nominal pose -> (status, held IK branches, reason)."""
    e = sc.e
    sc.restore()                                               # every static test runs on the settled scene
    e._focus(e.obj_id(name))
    e._check_width = g["open_width"]
    try:
        cands = e.ik_candidates(g["position"], g["R"], e.q_ready)
        held = [q for q in cands if not e._object_hits(q)]
        if not cands:
            return "IK_INVALID", [], "no IK branch clear of the structure / arm ceiling"
        if not held:
            return "HELD_INVALID", [], "every IK branch comes within 5 mm of another object (pre-shaped fingers)"
        CUR["screen_only"] = True
        try:
            e.plan_carry_xy(name, tuple(dest), g, seed=SEED, start_xy=g["position"][:2])
        except S.ReachError as err:
            return "GEOMETRY_INVALID", held, str(err)[:140]
        finally:
            CUR["screen_only"] = False
        return "VALID", held, None
    finally:
        e._check_width = None


def hold(sc, name, g, q_g):
    """Instantiate and certify the post-grasp state.  -> (ok, evidence, object positions before)."""
    e, m = sc.e, sc.e.model
    i = e.obj_id(name)
    sc.restore()
    d = e.data
    before = [e.obj_pos(j) for j in range(len(e.objects))]
    d.qpos[e.qadr], d.qvel[:] = q_g, 0.0
    d.qpos[e.grip_qadr] = e._finger_q(g["open_width"])
    d.ctrl[e.act_arm] = q_g; e.set_gripper_width(g["open_width"])
    mujoco.mj_forward(m, d)
    pre = sc.contacts(i)
    e.wait(HELD_SETTLE); e.set_gripper(False); e.wait(CLOSE_STEPS)
    W["sim_steps"] += HELD_SETTLE + CLOSE_STEPS
    rs, ro, ho, hh, L, R = sc.contacts(i)
    shift = float(np.linalg.norm(e.obj_pos(i) - before[i]))
    nb = max([float(np.linalg.norm(e.obj_pos(j) - p)) for j, p in enumerate(before) if j != i] or [0.0])
    ev = {"pre_contacts": list(pre[:4]), "both_fingers": bool(L and R), "object_shift_m": shift, "neighbour_shift_m": nb,
          "pad_gap_m": e.pad_gap(), "width_m": float(g["width"]), "contacts_after": [rs, ro, ho],
          "arm_err_rad": float(np.max(np.abs(d.qpos[e.qadr] - q_g)))}
    ok = sum(pre[:4]) == 0 and L and R and shift <= TOL["hold_shift_m"] and nb <= TOL["neighbour_m"] and rs + ro + ho == 0
    e.held = i if ok else None
    return bool(ok), ev, before


def plan(sc, name, dest, g, branches, iters):
    """shelf_scene._plan_carry + release + _plan_retreat (same calls, same order) for the first IK branch that plans -- the branch
    loop of plan_pick_place from the grasp on.  The route does not depend on the branch: planned once per mode."""
    e = sc.e
    sc.restore()                                               # planned on the settled scene (as the source plans before executing)
    e._focus(e.obj_id(name))
    CUR["iters"], route, rinfo, first = iters, None, {}, None
    e._check_width = g["open_width"]
    R, pg = g["R"], g["position"]
    lift = pg + np.array([0, 0, S.CARRY_LIFT])
    try:
        for b, q_g in enumerate(branches):
            phase = "LIFT"
            try:
                segs = [dict(phase="lift", kind="cart", q=e.chain(q_g, pg, [lift], R))]
                if route is None:
                    phase = "ROUTE"
                    r0 = {**W, "cpu": time.process_time()}
                    try:
                        route = e.plan_carry_xy(name, tuple(dest), g, seed=SEED, start_xy=pg[:2])
                    finally:
                        rinfo = {**{k: CUR["last"].get(k) for k in ("iters", "direct", "found", "milestones")}, "start": r0,
                                 "end": {**W, "cpu": time.process_time()}}
                phase = "CARRY"
                pts = [np.array([x, y, lift[2]]) for x, y in route[1:]]
                segs.append(dict(phase="carry", kind="cart", q=e.chain(segs[-1]["q"][-1], lift, pts, R)))
                phase = "LOWER"
                put = np.array([route[-1][0], route[-1][1], pg[2]])
                segs.append(dict(phase="lower", kind="cart", q=e.chain(segs[-1]["q"][-1], pts[-1] if pts else lift, [put], R)))
                segs.append(dict(phase="release", kind="grip", grip=g["open_width"]))
                phase = "RETREAT"
                segs += e._plan_retreat(g, segs[-2]["q"][-1], put, SEED)
                return {"ok": True, "branch": b, "q_g": q_g, "segs": segs, "carry_xy": np.asarray(route), "route": rinfo}
            except (RuntimeError, ValueError) as err:
                first = first or (phase, str(err)[:140])
                if isinstance(err, S.ReachError):              # plan_pick_place: another IK branch cannot fix the route
                    return {"ok": False, "phase": "ROUTE", "reason": str(err)[:140], "route": rinfo,
                            "budget_limited": "no sideways route" in str(err) and rinfo.get("iters") == iters}
        return {"ok": False, "phase": first[0], "reason": first[1], "budget_limited": False, "route": rinfo}
    finally:
        e._check_width = None


def execute(sc, name, pl, before):
    """Servo execution of the planned segments; recorded every REC_EVERY steps, contacts classified every step."""
    e, m, d = sc.e, sc.e.model, sc.e.data
    i, dt = e.obj_id(name), m.opt.timestep
    o = e.objects[i]
    rec = {k: [] for k in ("q", "gap", "obj", "phase", "step")}
    ev = {"robot_structure_steps": 0, "robot_other_steps": 0, "held_other_steps": 0, "max_attach_drift_m": 0.0,
          "max_neighbour_shift_m": 0.0, "fault": None, "lift_m": None}
    n, off0 = [0], float(np.linalg.norm(e.obj_pos(i) - d.site_xpos[e.site_id]))

    def tick(ph, attached):
        mujoco.mj_step(m, d)
        n[0] += 1
        if REC_HOOK is not None and n[0] % REC_HOOK[1] == 0:
            REC_HOOK[0]()
        rs, ro, ho = sc.contacts(i)[:3]
        ev["robot_structure_steps"] += rs > 0; ev["robot_other_steps"] += ro > 0; ev["held_other_steps"] += ho > 0
        if attached:
            ev["max_attach_drift_m"] = max(ev["max_attach_drift_m"], abs(float(np.linalg.norm(e.obj_pos(i) - d.site_xpos[e.site_id])) - off0))
        if n[0] % REC_EVERY == 0:
            rec["q"].append(d.qpos[e.qadr].copy()); rec["gap"].append(e.pad_gap()); rec["phase"].append(ph); rec["step"].append(n[0])
            rec["obj"].append(d.qpos[e.obj_qadr[i]:e.obj_qadr[i] + 7].copy())
            ev["max_neighbour_shift_m"] = max([ev["max_neighbour_shift_m"]] + [float(np.linalg.norm(e.obj_pos(j) - p))
                                                                                for j, p in enumerate(before) if j != i])
    for seg in pl["segs"]:
        ph, att = PH.index(seg["phase"]), seg["phase"] in ("lift", "carry", "lower")
        if seg["kind"] == "grip":
            e.set_gripper_width(seg["grip"])
            for _ in range(RELEASE_STEPS):
                tick(ph, False)
            e.held = None
            continue
        traj = e.segment_traj(seg)
        qd = np.gradient(traj, dt, axis=0) if len(traj) > 1 else np.zeros_like(traj)
        for q, v in zip(traj, qd):
            d.ctrl[e.act_arm] = q + e.servo_lead * v
            tick(ph, att)
        d.ctrl[e.act_arm] = traj[-1]
        for _ in range(SEG_SETTLE):
            tick(ph, att)
        if seg["phase"] == "lift":
            ev["lift_m"] = float(e.obj_pos(i)[2] - o["height"] / 2 - e.table_z)
            if ev["lift_m"] < TOL["lift_frac"] * S.CARRY_LIFT:
                ev["fault"] = "LIFT_SLIP"
                break
    for _ in range(FINAL_SETTLE):
        tick(PH.index("settle"), False)
    W["sim_steps"] += n[0]
    ev["steps"] = n[0]
    return {k: np.asarray(v) for k, v in rec.items()}, ev


def envelope_hits(sc, name, rec):
    """Recorded arm configurations vs the structure + arm ceiling (default fingers, as the source) and every other object's
    no-overpass column / body (fingers at the recorded opening), at margin TOL['envelope_margin_m'] (0 = penetration)."""
    e, ck, bad = sc.e, sc.e.static_ck, []
    e._focus(e.obj_id(name))
    for k in range(0, len(rec["q"]), TOL["envelope_every"]):
        if PH[rec["phase"][k]] == "settle":
            continue
        ck.margin = TOL["envelope_margin_m"]
        try:
            s = ck.collisions(rec["q"][k])
        finally:
            ck.margin = S.STATIC_MARGIN
        e._check_width = float(rec["gap"][k])
        try:
            ob = e._object_hits(rec["q"][k], margin=TOL["envelope_margin_m"])
        finally:
            e._check_width = None
        if s or ob:
            bad.append([int(rec["step"][k]), PH[rec["phase"][k]], len(s), list(ob)])
    return bad


def verify(sc, name, dest, rec, ev):
    """Independent end-state + trajectory gates; all must hold for a verified witness."""
    e, m, d = sc.e, sc.e.model, sc.e.data
    i = e.obj_id(name)
    o = e.objects[i]
    p = e.obj_pos(i)
    zax = d.xmat[m.body(o["name"]).id].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(zax[2], -1, 1))))
    dof = m.joint(o["name"] + "_free").dofadr[0]
    hh = sc.contacts(i)[3]
    env = envelope_hits(sc, name, rec)
    xy, z = float(np.linalg.norm(p[:2] - e.cell_xy(dest))), float(p[2] - e.table_z - o["height"] / 2)
    g = {"no_exec_fault": ev["fault"] is None, "cell": e.cell_of(name) == tuple(dest), "xy": xy <= TOL["xy_m"], "z": abs(z) <= TOL["z_m"],
         "upright": tilt <= TOL["tilt_deg"], "settled": float(np.linalg.norm(d.qvel[dof:dof + 3])) <= TOL["settled_mps"],
         "attached": ev["max_attach_drift_m"] <= TOL["attach_drift_m"], "neighbours": ev["max_neighbour_shift_m"] <= TOL["neighbour_m"],
         "released": hh == 0, "no_robot_structure_contact": ev["robot_structure_steps"] == 0,
         "no_robot_object_contact": ev["robot_other_steps"] == 0, "no_held_object_contact": ev["held_other_steps"] == 0,
         "envelope": not env}
    return all(g.values()), {"gates": g, "first_failed": next((k for k, v in g.items() if not v), None), "xy_err_m": xy, "z_err_m": z,
                             "tilt_deg": tilt, "end_cell": e.cell_of(name), "envelope_hits": env[:5], "n_envelope_hits": len(env)}


def pack(pl, g, hold_ev, rec):
    """Witness arrays: planned waypoints q (N,7) with segment offsets (source-compatible), executed samples, held-state evidence."""
    cart = [s for s in pl["segs"] if s["kind"] == "cart"]
    q = np.vstack([np.asarray(s["q"], float) for s in cart])
    off = np.cumsum([0] + [len(s["q"]) if s["kind"] == "cart" else 0 for s in pl["segs"]])
    return {"q": q, "segment": np.concatenate([[k] * len(s["q"]) for k, s in enumerate(pl["segs"]) if s["kind"] == "cart"]).astype(np.int16),
            "seg_phase": np.asarray([s["phase"] for s in pl["segs"]]), "seg_off": off.astype(np.int32),
            "seg_grip": np.asarray([s.get("grip") or np.nan for s in pl["segs"]], float), "carry_xy": np.asarray(pl["carry_xy"], float),
            "q_g": np.asarray(pl["q_g"], float), "grasp": np.asarray([g["id"], g["azimuth_deg"], g["_mi"], g["open_width"], g["width"]]),
            "grasp_pos": np.asarray(g["position"], float), "grasp_R": np.asarray(g["R"], float),
            "hold_ev": np.asarray(json.dumps(hold_ev, sort_keys=True)),
            "q_exec": rec["q"].astype(np.float32), "gap_exec": rec["gap"].astype(np.float32), "obj_exec": rec["obj"].astype(np.float32),
            "phase_exec": rec["phase"].astype(np.int8), "step_exec": rec["step"].astype(np.int32), "phases": np.asarray(PH)}


def unpack(wz):
    """Witness arrays -> planned segments for replay."""
    segs = []
    for k, ph in enumerate(wz["seg_phase"]):
        a, b = int(wz["seg_off"][k]), int(wz["seg_off"][k + 1])
        segs.append(dict(phase=str(ph), kind="grip", grip=float(wz["seg_grip"][k])) if ph == "release" else
                    dict(phase=str(ph), kind="cart", q=[r for r in wz["q"][a:b]]))
    return {"ok": True, "segs": segs, "q_g": wz["q_g"], "carry_xy": wz["carry_xy"]}
