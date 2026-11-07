"""Independent witness replay (fresh process), source-witness baseline + compatibility converter, and EGL renders.

  ours RUN_DIR ROSTER OUT.json            re-execute EVERY stored positive from its packed arrays in a fresh process; re-verify all gates;
                                          compare end pose / executed samples with the stored witness
  source N OUT.json                       N source B/D step-0 witnesses: (a) unchanged shelf_scene.run_plan from the settled root scene vs the
                                          recorded step; (b) convert -> placement-only (lift..retreat) -> held-state harness -> our gates
  gif RUN_DIR ROSTER QID OUT.gif          render a stored witness (MUJOCO_GL=egl)
"""
import gzip, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement3d as P  # noqa: E402

S = P.S


def roster_index(roster):
    return {q["qid"]: q for q in (json.loads(x) for x in gzip.open(roster, "rt"))}


def load_pack(run, unit):
    z = np.load(Path(run) / "units" / f"{unit}.npz")
    out = {}
    for k in z.files:
        w, a = k.split(".", 1)
        out.setdefault(w, {})[a] = z[k]
    return out


def replay_one(root, q, wz, frames=None, every=40):
    """Rebuild the scene, certify the hold at the stored q_g with the stored grasp, execute the stored waypoints, verify."""
    sc = P.Scene(root["objects"], q["cells"])
    g = next(x for x in P.poses(sc, q["object"]) if x["id"] == int(wz["grasp"][0]))
    same = bool(np.allclose(g["position"], wz["grasp_pos"]) and np.allclose(g["R"], wz["grasp_R"]))
    ok, hev, before = P.hold(sc, q["object"], g, wz["q_g"])
    pl = P.unpack(wz)
    if frames is not None:
        frames.append(render(sc))
        P.REC_HOOK = (lambda: frames.append(render(sc)), every)
    rec, ev = P.execute(sc, q["object"], pl, before)
    P.REC_HOOK = None
    v, vi = P.verify(sc, q["object"], tuple(q["dest"]), rec, ev)
    n = min(len(rec["q"]), len(wz["q_exec"]))
    phases_ok = set(map(str, wz["seg_phase"])) <= {"lift", "carry", "lower", "release", "retreat"}
    return {"qid": q["qid"], "grasp_pose_identical": same, "hold_ok": ok, "verified": v and phases_ok, "first_failed": vi["first_failed"],
            "placement_phases_only": phases_ok, "hold_identical": json.loads(str(wz["hold_ev"])) == json.loads(json.dumps(hev, default=float)),
            "q_exec_max_abs_diff": float(np.max(np.abs(rec["q"][:n].astype(np.float32) - wz["q_exec"][:n]))) if n else None,   # at stored precision
            "obj_end_diff_m": float(np.linalg.norm(rec["obj"][-1][:3].astype(np.float32) - wz["obj_exec"][-1][:3])), "n_samples": [len(rec["q"]), len(wz["q_exec"])]}


_R = {}


def render(sc, w=480, h=360):
    import mujoco
    e = sc.e
    if id(e) not in _R:
        r = mujoco.Renderer(e.model, h, w)
        c = e.scene["camera"]                                      # the source scene's own camera (others look through the room wall)
        cam = mujoco.MjvCamera(); cam.lookat[:] = c["lookat"]; cam.distance, cam.azimuth, cam.elevation = c["distance"] * 0.9, c["azimuth"], c["elevation"]
        _R.clear(); _R[id(e)] = (r, cam)
    r, cam = _R[id(e)]
    r.update_scene(e.data, cam)
    return r.render().copy()


def ours(run, roster, out):
    idx, res = roster_index(roster), []
    for f in sorted((Path(run) / "units").glob("*.json")):
        doc = json.loads(f.read_text())
        pack = load_pack(run, f.stem) if (Path(run) / "units" / f"{f.stem}.npz").exists() else {}
        for r in doc["records"]:
            if r.get("label") != 1:
                continue
            q = idx[r["qid"]]
            root = json.loads((P.SRC / "raw/roots" / f"{q['root_id']}.json").read_text())
            wz = pack.get(r["witness_id"])
            res.append(replay_one(root, q, wz) if wz is not None else {"qid": r["qid"], "verified": False, "first_failed": "MISSING_WITNESS"})
    summ = {"replayed": len(res), "verified": sum(x["verified"] for x in res), "failures": [x for x in res if not x["verified"]],
            "max_q_exec_diff": max([x.get("q_exec_max_abs_diff") or 0 for x in res] or [0]),
            "max_obj_end_diff_m": max([x.get("obj_end_diff_m") or 0 for x in res] or [0]), "results": res}
    Path(out).write_text(json.dumps(summ, indent=1)); print(json.dumps({k: v for k, v in summ.items() if k != "results"}))


def convert(src_w, grasp):
    """Source B/D witness (q, segment, carry_xy, phases: a whole pick-and-place KINEMATIC plan) -> placement-only witness arrays:
    the 'lift' .. 'retreat' segments, q_g = the first lift waypoint (= the end of the approach), release width = grasp open width.
    The pick prefix (open, to_hover, approach, close) is dropped; it is NOT a placement trajectory by itself."""
    ph = [str(x) for x in src_w["phases"]]
    segs, keep = [], ("lift", "carry", "lower", "release", "retreat")
    for k, name in enumerate(ph):
        if name not in keep:
            continue
        if name == "release":
            segs.append(dict(phase="release", kind="grip", grip=grasp["open_width"]))
        else:
            segs.append(dict(phase=name, kind="cart", q=[r for r in src_w["q"][src_w["segment"] == k]]))
    return {"ok": True, "segs": segs, "q_g": segs[0]["q"][0], "carry_xy": src_w["carry_xy"]}


def source(n, out):
    import plan3d as L
    roots, res = L.load_selected(P.SRC), []
    pick = sorted(roots, key=lambda r: L.u("baseline", r))
    want = {"0": 0, "1": 0, "2": 0, "3": 0}
    for rid in pick:
        v = roots[rid]
        if v["K"] not in want or want[v["K"]] >= n // 4:
            continue
        rec = json.loads((P.SRC / "raw/roots" / f"{rid}.json").read_text())
        b = next((x for x in rec["labels"]["B"] if x.get("y") == 1 and x.get("witnesses")), None)
        if b is None:
            continue
        st, wid = b["steps"][0], b["witnesses"][0]
        z = np.load(P.SRC / "raw/witnesses" / wid[:2] / f"{wid}.npz"); w = {k: z[k] for k in z.files}
        want[v["K"]] += 1
        e = P.bd_shelf.build(rec["objects"], rec["initial_cells"])            # (a) the source path, unchanged
        i = e.obj_id(st["object"]); e._focus(i); e.objects[i]["pos"] = e.obj_pos(i)
        g = S.grasp_poses(e.objects[i], e.table_z)[st["grasp_id"]]
        ph = [str(x) for x in w["phases"]]
        segs = [dict(phase=p, kind="grip", grip=(None if p == "close" else g["open_width"])) if p in ("open", "close", "release")
                else dict(phase=p, kind="joint" if p in ("to_hover", "to_ready") else "cart", q=[r for r in w["q"][w["segment"] == k]])
                for k, p in enumerate(ph)]
        others = {o: e.cell_of(o) for o in e.obj_index if o != st["object"]}
        e.run_plan({"name": st["object"], "segments": segs})
        zerr = float(e.obj_pos(st["object"])[2] - e.table_z - e.objects[i]["height"] / 2)
        a_ok = e.cell_of(st["object"]) == tuple(st["dest"]) and abs(zerr) <= 0.01 and all(e.cell_of(o) == c for o, c in others.items())
        sc = P.Scene(rec["objects"], rec["initial_cells"])                      # (b) converted, placement-only, held-state harness
        g2 = next(x for x in P.poses(sc, st["object"]) if x["id"] == st["grasp_id"])
        pl = convert(w, g2)
        ok, hev, before = P.hold(sc, st["object"], g2, pl["q_g"])
        recd, ev = P.execute(sc, st["object"], pl, before)
        vb, vi = P.verify(sc, st["object"], tuple(st["dest"]), recd, ev)
        res.append({"root": rid, "K": v["K"], "witness": wid, "object": st["object"], "dest": st["dest"], "source_recorded": {"end_cell": st["end_cell"], "z_err": st["z_err"], "ok": st["ok"]},
                    "a_unchanged_replay": {"end_cell": list(e.cell_of(st["object"]) or []), "z_err": zerr, "ok": a_ok, "z_err_abs_diff": abs(zerr - st["z_err"])},
                    "b_placement_only": {"hold_ok": ok, "verified": vb, "first_failed": vi["first_failed"], "end_cell": list(vi["end_cell"] or []),
                                         "xy_err_m": vi["xy_err_m"], "z_err_m": vi["z_err_m"], "tilt_deg": vi["tilt_deg"]}})
        print(json.dumps(res[-1])[:400], flush=True)
        if sum(want.values()) >= n:
            break
    summ = {"n": len(res), "a_matches_recorded_ok": sum(r["a_unchanged_replay"]["ok"] == bool(r["source_recorded"]["ok"]) for r in res),
            "a_max_z_err_diff": max(r["a_unchanged_replay"]["z_err_abs_diff"] for r in res), "b_verified": sum(r["b_placement_only"]["verified"] for r in res),
            "b_hold_ok": sum(r["b_placement_only"]["hold_ok"] for r in res), "results": res}
    Path(out).write_text(json.dumps(summ, indent=1)); print(json.dumps({k: v for k, v in summ.items() if k != "results"}))


def gif(run, roster, qid, out):
    from PIL import Image
    q = roster_index(roster)[qid]
    unit = next(f.stem for f in (Path(run) / "units").glob("*.json") if any(r["qid"] == qid for r in json.loads(f.read_text())["records"]))
    doc = json.loads((Path(run) / "units" / f"{unit}.json").read_text())
    r = next(x for x in doc["records"] if x["qid"] == qid)
    wz = load_pack(run, unit)[r["witness_id"]]
    root = json.loads((P.SRC / "raw/roots" / f"{q['root_id']}.json").read_text())
    frames = []
    res = replay_one(root, q, wz, frames=frames)
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=60, loop=0)
    print(json.dumps({**res, "frames": len(ims), "gif": out}))


if __name__ == "__main__":
    c = sys.argv[1]
    {"ours": lambda: ours(*sys.argv[2:5]), "source": lambda: source(int(sys.argv[2]), sys.argv[3]),
     "gif": lambda: gif(*sys.argv[2:6])}[c]()
