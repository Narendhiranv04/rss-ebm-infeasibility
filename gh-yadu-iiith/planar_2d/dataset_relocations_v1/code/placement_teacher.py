"""Placement teacher for the launch-ready sidecar: one planned query under a budget policy, atomic per-query parts, per-root
packing and witness pack / unpack. Event, held-mode semantics, per-pair counted work (det_rrt), witness construction and
phase-aware certification are the repaired smoke's (placement_smoke.py, imported unchanged); only the query-level schedule is
parameterised: the first B valid (held, place) pairs in the smoke's fixed order, FULL (plan all B) or FIRST_SUCCESS coverage.

Label of a completed query: 1 iff a verified carry -> release -> retreat witness exists (planned or a re-verified archived
suffix); 0 iff the declared schedule finished without one (operational, policy-limited; a certificate only when every pair fell
to definite static screens); null if statically ineligible or interrupted by a safety guard (re-run identically on resume).
"""
import functools, io, json, os, resource, signal, time
from pathlib import Path

import numpy as np
import placement_smoke as P
from placement_smoke import ge, mo, sk

TEACHER = "PLACEMENT_LAUNCH_V1"
RESUME_VERSION = 1
STAGES = ("", "OK", "NO_IK", "HELD_CONTACT_MODEL", "HELD_STATE", "PLACE_POSE", "RELEASE", "RETREAT", "GRASP_POSE", "APPROACH", "CLOSURE")
PHASES = ("carry", "release", "retreat")
METHODS = ("ARCHIVED_SUFFIX", "DIRECT", "GUIDE_LINE", "RRT_JOINT", "RRT_OBJECT_POSE", "LINK0_SWEEP", "WORK_BUDGET_EXHAUSTED", "ENDPOINT_IK_MISMATCH")
STOPS, STATUS = ("", "NODES", "ATTEMPTS", "EVALS", "SOLVED"), (ge.CLEAR, ge.COLLISION, ge.UNRESOLVED)
PER_PAIR = {k: P.WORK[k] for k in ("rrt_attempts", "rrt_evals", "rrt_nodes", "shortcut_rounds")}
HELD_POLICY = "HELD_48_MODES_ANY_VERIFIED_V1: 24 tool axes x 2 elbows, attached at the source; success under >= 1 scheduled pair"
CORE = ("q", "gap", "phase", "attached", "held_local", "place_xy")
ATT = ("a_pick", "a_place", "a_rank", "a_method", "a_status", "a_stop", "a_attempts", "a_nodes", "a_pre_evals", "a_search_evals",
       "a_shortcut_evals", "a_verified", "a_stored", "a_witness")


def pair_protocol():
    """How ONE (held, place) pair is planned and certified (independent of the query budget): witness ids use this."""
    G = P.A.env()
    src = {f: P.sha_file(P.LCT / f)[:16] for f in ("skill.py", "motion.py", "planner.py", "geometry.py", "config.toml")}
    sem = {"name": TEACHER, "event": P.EVENT, "per_pair_work": PER_PAIR, "teacher_config_hash": G["config_hash"], "robot_protocol": G["protocol"],
           "teacher_sources": src, "planner_seed": G["cfg"]["planner2d"]["seed"], "seed_rule": "planner.call_seed(seed, '<trial_id>/carry<i>.<j>')",
           "schedule": "valid pairs sorted by (elbow switch, joint distance, (i, j))", "smoke_code_sha256": P.sha_file(P.__file__)}
    return P.sha(json.dumps(sem, sort_keys=True).encode())[:16], sem


def budget_hash(policy):
    return P.sha(json.dumps([pair_protocol()[0], policy], sort_keys=True).encode())[:16]


def code(table, s):
    return table.index(s)                                # an unknown value raises: never silently mis-coded


def scene_for(task):
    s = P.scene_from_dict(P.load_root(task["root"])["scene"])
    for o, c in task["prefix"]:
        s = P.substitute_pose(s, o, c)
    return s


def wid_of(ident, core):
    h = P.hashlib.sha256(ident.encode())
    for k in sorted(core):
        a = np.asarray(core[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}".encode() + a.tobytes())
    return h.hexdigest()[:24]


def put_witness(d, wid, core):
    f = d / f"w_{wid}.npz"
    if not f.exists():
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **{k: np.asarray(core[k]) for k in CORE})
        os.replace(tmp, f)


def _cpu_alarm(*_):
    raise TimeoutError("QUERY_EMERGENCY_CPU")


def run_query(task, out, policy, emergency_cpu_s):
    """task: plan row + root context. Writes parts/<root>/<trial_id>.json (+ .npz, + w_<id>.npz); returns the record."""
    d = Path(out) / "parts" / task["root"]
    part = d / f"{task['trial_id']}.json"
    if final(part):
        return json.loads(part.read_text())
    d.mkdir(parents=True, exist_ok=True)
    scene = scene_for(task)
    obj, s_hash = scene.get(task["object"]), P.scene_hash(scene)
    rec = {k: task[k] for k in ("trial_id", "block", "prefix_kind", "source_cell", "source_bin", "dest_bin", "stratum_k", "stratum_n", "obj_rank", "dest_rank")}
    rec.update(prefix_key=task["prefix"], prefix_provenance="HYPOTHETICAL" if task["prefix"] else "ROOT", object_id=obj.id,
               context_selected_with_protected_metadata=task["prefix_kind"] == "ARCHIVED_FIRST_MOVE", hypothetical_scene_hash=s_hash,
               source_xy=list(map(float, obj.xy_m)),
               destination_cell=task["dest"], destination_xy=list(map(float, scene.lattice.center(task["dest"]))),
               query_seed_key=task["trial_id"], resume_version=RESUME_VERSION)
    why = "DEST_NOT_LEGAL" if task["dest"] not in scene.legal_cells else "DEST_OCCUPIED" if task["dest"] in {o.cell for o in scene.objects} else None
    if why:
        rec.update(eligible=False, attempted=False, completed=False, label=None, excluded=why, witness_refs=[], safety_interruption=None)
        return commit(part, rec, None)
    t0, c0 = time.time(), time.process_time()
    signal.signal(signal.SIGVTALRM, _cpu_alarm)          # CPU-time guard: frozen (SIGSTOP) workers do not advance it
    signal.setitimer(signal.ITIMER_VIRTUAL, emergency_cpu_s)
    try:
        body, modes, vol = query(task, d, rec, scene, obj, policy)
        rec.update(body)
    except TimeoutError as e:
        rec.update(eligible=True, attempted=True, completed=False, label=None, safety_interruption=str(e), witness_refs=[])
        modes, vol = None, {}
    finally:
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    rec["volatile"] = {"wall_s": round(time.time() - t0, 3), "cpu_s": round(time.process_time() - c0, 3), **vol,
                       "worker_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    return commit(part, rec, modes)


def query(task, d, rec, scene, obj, policy):
    G = P.A.env()
    cfg, robot, p, B = G["cfg"], G["robot"], sk.params(G["cfg"]), policy["pairs_per_query"]
    c0 = time.process_time()
    dxy = np.asarray(scene.lattice.center(task["dest"]))
    pick_diag = sk.screen_pick(robot, mo.checkers(robot, scene, obj.id, None, None, p["delta"], p["eps"]), obj.xy_m, obj.radius_m, p)
    picks = P.held_modes(robot, scene, obj, p)
    groups, places, cks = {}, {}, {}
    for i, c in enumerate(picks):
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
    ident = f"{pair_protocol()[0]}|{scene.robot_profile_hash}|{rec['hypothetical_scene_hash']}|{obj.id}"
    rows, cpu, stored, refs, vol = [], [], 0, [], {"screen_cpu_s": round(time.process_time() - c0, 3)}
    old = P.reuse(P.load_root(task["root"]), {"prefix": task["prefix"], "object": obj.id, "dest": task["dest"]}, robot, scene, picks, places, p) \
        if policy["reuse_archived"] else None
    if old and old["reverified"]:
        w = wid_of(ident, old["arrays"])
        put_witness(d, w, old["arrays"])
        refs.append(w)
        cat[old["pick"], old["place"]] = 5
        rows.append([old["pick"], old["place"], -1, 0, 0, 0, 0, 0, 0, 0, 0, True, True, w])
    sched = sorted(((int(i), int(j)) for i, j in zip(*np.nonzero(cat == 2))),
                   key=lambda ij: (picks[ij[0]].branch != places[ij[0]][ij[1]].branch, float(np.linalg.norm(picks[ij[0]].q_end - places[ij[0]][ij[1]].q_end)), ij))
    for rank, (i, j) in enumerate(sched[:B]):
        if policy["coverage"] == "FIRST_SUCCESS" and any(r[11] for r in rows):
            break
        t = time.process_time()
        a, b = picks[i], places[i][j]
        guide = pose = None
        if a.branch == b.branch:
            guide = mo.carry_line(robot, obj.xy_m, dxy, a.local, a.phi, b.phi, a.branch, p["line_step_m"], p["max_ik_jump_rad"])[0]
            pose = ((*obj.xy_m, a.phi), (*dxy, b.phi), a.local, a.branch, p["line_step_m"], p["max_ik_jump_rad"])
        st, path, method, work = P.plan_carry(cks[i][("held", False)], a.q_end, b.q_end, a.gap, cfg, p, f"{rec['trial_id']}/carry{i}.{j}", guide, pose)
        ok, w, keep = False, "", False
        if st == ge.CLEAR:
            arr = P.witness_arrays(robot, obj.id, a, path, b)
            ok = P.verify_witness(robot, scene, obj.id, arr, dxy, p)[0]
            if ok:
                w = wid_of(ident, arr)
                keep = stored < policy["witness_store_max"]
                if keep:
                    put_witness(d, w, arr)
                    refs.append(w)
                    stored += 1
        cat[i, j] = 4 if ok else 3
        rows.append([i, j, rank, code(METHODS, method.split(":")[0]), code(STATUS, st), code(STOPS, work["stop"] or ""), work["attempts"], work["nodes"],
                     work["pre_evals"], work["search_evals"], work["shortcut_evals"], ok, keep, w])
        cpu.append(round(time.process_time() - t, 4))
    vol["attempt_cpu_s"] = cpu
    c = {P.CAT[k]: int((cat == k).sum()) for k in P.CAT}
    m = {"total": n * n, "hard_screened": c["HARD_SCREENED"], "motion_attempted": c["PLANNED_FAILED"] + c["PLANNED_VERIFIED"],
         "reused_reverified": c["REUSED_REVERIFIED"], "not_motion_attempted": c["SCREEN_INCONCLUSIVE_NOT_PLANNED"] + c["VALID_NOT_PLANNED"],
         "attempted_success": c["PLANNED_VERIFIED"], "attempted_failure": c["PLANNED_FAILED"], "screen_inconclusive": c["SCREEN_INCONCLUSIVE_NOT_PLANNED"],
         "valid_not_planned": c["VALID_NOT_PLANNED"], "n_mode_witnesses": c["PLANNED_VERIFIED"] + c["REUSED_REVERIFIED"], "valid_pairs": len(sched)}
    assert m["total"] == m["hard_screened"] + m["motion_attempted"] + m["reused_reverified"] + m["not_motion_attempted"]
    replay = any(r[4] == 0 and not r[11] for r in rows)
    lab = decide(m, picks, places, policy, bool(rows), replay)
    planned = any(r[11] for r in rows if r[2] >= 0)
    modes = {"held_stage": np.array([code(STAGES, x.stage) for x in picks], np.int8), "held_definite": np.array([x.certified for x in picks]),
             "pick_access_stage": np.array([code(STAGES, x.stage) for x in pick_diag], np.int8),
             "place_stage": np.array([[code(STAGES, x.stage) for x in places[i]] if i in places else [0] * n for i in range(n)], np.int8), "pair_category": cat}
    cols = list(zip(*rows)) if rows else [[]] * len(ATT)
    for k, col, dt in zip(ATT, cols, (np.int16,) * 3 + (np.int8,) * 3 + (np.int32,) * 5 + (bool, bool, "U24")):
        modes[k] = np.array(col, dt)
    body = {"eligible": True, "attempted": True, "completed": True, **lab, "safety_interruption": None, "mode_counts": m,
            "label_source": None if not lab["label"] else "BOTH" if planned and old and old["reverified"] else "PLANNED" if planned else "REUSED",
            "held_modes_valid": sum(x.stage == "OK" for x in picks), "pick_access_modes_ok": sum(x.stage == "OK" for x in pick_diag),
            "place_screen_groups": len(groups), "witness_refs": refs, "n_verified_pairs": m["n_mode_witnesses"],
            "work": {"pairs_planned": sum(r[2] >= 0 for r in rows), "rrt_attempts": int(sum(r[6] for r in rows)), "rrt_nodes": int(sum(r[7] for r in rows)),
                     "evals": int(sum(r[8] + r[9] + r[10] for r in rows))},
            "privileged_teacher": None if old is None else {"archived_witness": old["archived_witness"], "archived_role": old["role"],
                                                            "reverified": old["reverified"], "pick": old["pick"], "place": old["place"]}}
    return body, modes, vol


def decide(m, picks, places, policy, any_rows, replay):
    if m["n_mode_witnesses"]:
        return {"label": 1, "negative_kind": None, "negative_scope": None, "failure_phase": None, "is_certificate": False}
    if m["hard_screened"] == m["total"]:
        valid = [i for i, c in enumerate(picks) if c.stage == "OK"]
        bad = [x.stage for i in valid for x in places[i] if x.stage != "OK"]
        return {"label": 0, "negative_kind": "CERTIFIED_SCREEN", "negative_scope": "DECLARED_48x48_CATALOGUE_DEFINITE_STATIC_SCREENS",
                "failure_phase": "HELD_STATE" if not valid else max(sorted(set(bad)), key=bad.count), "is_certificate": True}
    scope = f"FINITE_POLICY:{policy['pairs_per_query']}_PAIRS,{policy['coverage']},PER_PAIR_WORK_BUDGET"
    if not any_rows:
        return {"label": 0, "negative_kind": "NO_PLAN_FOUND", "negative_scope": scope + ",SCREEN_INCONCLUSIVE", "failure_phase": "SCREEN", "is_certificate": False}
    return {"label": 0, "negative_kind": "WORK_BUDGET_EXHAUSTED" if m["valid_not_planned"] else "SCHEDULE_EXHAUSTED_NO_WITNESS", "negative_scope": scope,
            "failure_phase": "REPLAY" if replay else "CARRY", "is_certificate": False}


def final(part):
    if not part.exists():
        return False
    r = json.loads(part.read_text())
    return bool(r["completed"] or not r["eligible"])


def commit(part, rec, modes):
    if modes is not None:
        tmp = part.with_suffix(f".{os.getpid()}.npz.tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **modes)
        os.replace(tmp, part.with_suffix(".npz"))
    tmp = part.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True))
    os.replace(tmp, part)
    return rec


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pack_root(out, header, tasks):
    """All queries of a root final -> records/<root>.json, modes/<root>.npz, witnesses/<root>.npz (written witnesses-first, the
    JSON last, each atomically); then parts/<root>/ is removed. Content is ordered by the plan, never by completion."""
    out, rid = Path(out), header["root_id"]
    d = out / "parts" / rid
    recs = [json.loads((d / f"{t['trial_id']}.json").read_text()) for t in tasks]
    assert all(r["completed"] or not r["eligible"] for r in recs), rid
    wids = list(dict.fromkeys(w for r in recs for w in r["witness_refs"]))
    ws = [{k: v for k, v in np.load(d / f"w_{w}.npz").items()} for w in wids]
    if ws:
        lens = [len(w["q"]) for w in ws]
        pack = {"ids": np.array(wids, "U24"), "q_offsets": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                "q": np.concatenate([w["q"] for w in ws]), "gap": np.concatenate([w["gap"] for w in ws]),
                "phase_code": np.concatenate([[code(PHASES, str(x)) for x in w["phase"]] for w in ws]).astype(np.int8),
                "attached": np.concatenate([w["attached"] for w in ws]), "held_local": np.stack([w["held_local"] for w in ws]),
                "place_xy": np.stack([w["place_xy"] for w in ws])}
        _atomic(out / "witnesses" / f"{rid}.npz", _npz(pack))
    z = [(i, np.load(d / f"{r['trial_id']}.npz")) for i, r in enumerate(recs) if (d / f"{r['trial_id']}.npz").exists()]
    if z:
        arr = {k: np.stack([x[k] for _, x in z]) for k in ("held_stage", "held_definite", "pick_access_stage", "place_stage", "pair_category")}
        arr["query_index"] = np.array([i for i, _ in z], np.int16)
        arr["a_query"] = np.concatenate([np.full(len(x["a_pick"]), i, np.int16) for i, x in z])
        for k in ATT:
            arr[k] = np.concatenate([x[k] for _, x in z])
        _atomic(out / "modes" / f"{rid}.npz", _npz(arr))
    _atomic(out / "records" / f"{rid}.json", json.dumps({**header, "queries": recs}, separators=(",", ":"), sort_keys=True).encode())
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


def _npz(arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


@functools.lru_cache(maxsize=8)
def _pack(path, mtime_ns):
    return dict(np.load(path))                          # one decompression per pack (keyed by mtime), not one per witness


def witness(out, rid, wid):
    """CORE arrays of a stored witness with its original dtypes (from the root pack, or from a part while the root is open)."""
    f = Path(out) / "witnesses" / f"{rid}.npz"
    if not f.exists():
        z = np.load(Path(out) / "parts" / rid / f"w_{wid}.npz")
        return {k: z[k] for k in CORE}
    z = _pack(str(f), f.stat().st_mtime_ns)
    k = list(z["ids"]).index(wid)
    a, b = z["q_offsets"][k], z["q_offsets"][k + 1]
    s = slice(a - k, b - k - 1)                          # n_k waypoints carry n_k - 1 segment entries
    return {"q": z["q"][a:b], "gap": z["gap"][a:b], "phase": np.array([PHASES[c] for c in z["phase_code"][s]], "U16"),
            "attached": z["attached"][s], "held_local": z["held_local"][k], "place_xy": z["place_xy"][k]}
