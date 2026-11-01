"""Combined dataset: the B/D labels of dataset-bd (bd_3dt, open tabletop) + the dense per-scene arrays of make_dataset.py
(dataset_v1 / v2 / front0), one .npz per scene, no GIFs.

    python make_combined_3dt.py 3dt_20 3dt_20_combined
    python make_combined_3dt.py 3dt_1000_raw 3dt_1000_combined 1000     # only the first 1000 roots

For every root of the bd_3dt output directory (root-id order -> scene_0000.npz, scene_0001.npz, ...):
  * ONE executed run, recorded exactly as make_dataset.py records it (its Recorder at 50 Hz, its dual quaternions, swept masks and
    quality checks): the root's verified minimal B sequence, then the target's move, re-executed in MuJoCo with bd_3dt's own query()
    + run_plan() (the teacher's planner, grasp order and seeds).  A root without a verified repair keeps the initial state only
    (trajectory_kind 'none', n_blocking -1);
  * the dataset-bd labels as arrays (original-scene label, every D query, every B sequence, blocker masks, roles, parking cells,
    factor spec, ids, splits), the planned-waypoint witnesses the labels point to (ragged arrays), and the whole root record as
    JSON (bd_record_json): each file is self-contained, so roots/ and witnesses/ are not copied.
Smaller than both sources: float arrays are stored as float32 (errors ~1e-7) and the .npz members are LZMA-compressed (np.load
reads them as usual).  manifest.csv has one row per scene; README.md lists every array (SCHEMA below)."""
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dataset-bd"))
import bd_core as bc  # noqa: E402
import bd_3dt as A  # noqa: E402

TS = A.env()["TS"]                                  # the teacher's table scene + planners (its random/ goes on sys.path first)
sys.path.insert(1, os.environ.get("MAKE_DATASET_DIR", os.path.expanduser("~/Downloads")))
import make_dataset as M  # noqa: E402              # Recorder, dual quaternions, swept masks of dataset_v1 / v2
import make_dataset_v2 as V  # noqa: E402           # layout signature (uniqueness key of the earlier datasets)

MAX_RELOC = bc.MAX_ACTIONS - 1                      # B sequences: at most 6 relocations before the target's move
MAX_PARK = max(bc.PARKING_N.values())


class Recorder(M.Recorder):
    """make_dataset.py's recorder (arm, gripper, held object, objects at HZ; every contact) without the GIF frames."""

    def frame(self):
        pass


def replay(r):
    """Re-execute the verified minimal sequence + the target's move; -> env, recorder rows and what happened."""
    lab, goal = r["labels"], tuple(r["goal"])
    B = next((b for b in lab["B"] if b["role"] == "minimal" and b["y"]), None)
    plan = [] if B is None else [(o, tuple(c)) for o, c in B["sequence"]] + [("t0", goal)]
    e = A.build(r["objects"], r["initial_cells"])
    A.settle(e, r["initial_cells"])
    init = np.array([e.obj_pos(j) for j in range(len(e.objects))])
    init_cells = np.array([e.cell_of(o["name"]) for o in e.objects])
    rec = Recorder(e)
    rec.log()
    rec.t_log = e.data.time
    moves, notes, azimuths = [], [], []
    for o, dst in plan:
        p, qi = A.query(e, o, dst)
        if p is None:
            notes.append(f"{o} -> {dst}: no plan on replay")
            break
        e.current = dict(name=o, i=e.obj_id(o))     # the handled object: its contacts with the hand are not collisions
        try:
            e.run_plan(p)
        except TS.GraspFailed as err:
            notes.append(f"{o} -> {dst}: GraspFailed: {str(err)[:100]}")
            break
        finally:
            e.current = None
        moves.append((o, dst))
        azimuths.append(qi["azimuth"])
        notes += [f"{o} -> {dst}: pushed {n} {mm} mm during {ph}" for n, ph, mm in p.get("pushed", [])]
    e.wait(100)
    rec.log()
    rec.close()
    return e, rec.rows, rec.collisions, init, init_cells, moves, notes, azimuths, B


def save_npz(path, arrays):
    """np.savez with float64 -> float32 and LZMA instead of deflate (about 2.7x smaller here; np.load reads it as usual)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_LZMA) as z:
        for k, v in arrays.items():
            v = np.asanyarray(v)
            with z.open(k + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, v.astype(np.float32) if v.dtype == np.float64 else v, allow_pickle=False)


def witnesses(src, r):
    """The planned-waypoint witnesses (witnesses/<aa>/<id>.npz) that the root's labels point to, as ragged arrays (rows of witness k:
    witness_q[witness_q_start[k]:witness_q_start[k + 1]], same for phases and carry_xy), and the index each label points to (-1 none)."""
    lab, ids = r["labels"], []

    def ix(w):
        if not w:
            return -1
        if w not in ids:
            ids.append(w)
        return ids.index(w)
    orig = ix(lab["original"].get("witness"))
    D = np.array([ix(d.get("witness")) for d in lab["D"]], np.int16)
    B = np.full((len(lab["B"]), bc.MAX_ACTIONS), -1, np.int16)
    for k, b in enumerate(lab["B"]):
        for j, w in enumerate(b.get("witnesses", [])):
            B[k, j] = ix(w)
    W = [dict(np.load(src / "witnesses" / w[:2] / f"{w}.npz")) for w in ids]
    cat = lambda key, empty: np.concatenate([w[key] for w in W]) if W else empty  # noqa: E731
    start = lambda key: np.cumsum([0] + [len(w[key]) for w in W]).astype(np.int32)  # noqa: E731
    return dict(witness_ids=np.array(ids, str), witness_q=cat("q", np.zeros((0, 7))), witness_q_start=start("q"),
                witness_segment=cat("segment", np.zeros(0, int)).astype(np.int8), witness_phases=cat("phases", np.zeros(0, str)),
                witness_phases_start=start("phases"), witness_carry_xy=cat("carry_xy", np.zeros((0, 2))), witness_carry_xy_start=start("carry_xy"),
                original_witness=np.int16(orig), D_witness=D, B_witness=B)


def pad(rows, width, fill=-1, dtype=np.int16):
    out = np.full((max(len(rows), 1), width), fill, dtype)
    for k, row in enumerate(rows):
        out[k, :len(row)] = row
    return out


def strs(xs):
    return np.array([str(x) for x in xs] or ["none"])


def labels(r, names, splits):
    """dataset-bd labels of one root as arrays (object axis = the scene's object order: t0, o00, o01, ...)."""
    lab, idx, n = r["labels"], {o: k for k, o in enumerate(names)}, len(names)
    orig, blk = lab["original"], lab["blocker"]
    mask = lambda objs: np.array([o in objs for o in names], bool)  # noqa: E731
    park = np.full((n, MAX_PARK, 2), -1, np.int16)
    for o, cells in r["parking"].items():
        for k, c in enumerate(cells[:MAX_PARK]):
            park[idx[o], k] = c
    bseq = np.full((len(lab["B"]), MAX_RELOC, 3), -1, np.int16)
    for k, b in enumerate(lab["B"]):
        for j, (o, c) in enumerate(b["sequence"]):
            bseq[k, j] = [idx[o], *c]
    out = dict(
        root_id=r["ids"]["root_id"], parent_family_id=r["ids"]["parent_family_id"], construction_seed=np.int64(r["ids"]["construction_seed"]),
        config_hash=r["ids"]["config_hash"], generator_version=r["ids"]["generator_version"], teacher_version=r["ids"]["teacher_version"],
        robot_protocol=r["ids"]["robot_protocol"], label_protocol=r["ids"]["label_protocol"],
        verified_K=np.int16(-1 if r["verified_K"] is None else r["verified_K"]),
        y_original=np.int8(orig["y"]), original_negative_kind=str(orig["negative_kind"] or "none"), original_reason=str(orig["reason"]),
        original_hit_objects=strs(orig.get("hit_objects", [])),
        object_role=np.array(["target"] + [r["roles"].get(o, "none") for o in names[1:]]), intended_blocker_mask=mask(r["intended_blockers"]),
        parking_cells=park,
        D_edit_mask=np.array([mask(d["edit"]) for d in lab["D"]] or np.zeros((0, n), bool)).reshape(-1, n),
        D_edit_cells=np.array([[d["edit"].get(o, [-1, -1]) for o in names] for d in lab["D"]] or np.zeros((0, n, 2)), np.int16).reshape(-1, n, 2),
        D_y=np.array([d["y"] for d in lab["D"]], np.int8), D_negative_kind=np.array([str(d["negative_kind"] or "none") for d in lab["D"]]),
        blocker_supervision_valid=np.int8(blk["blocker_supervision_valid"]), blocker_why_invalid=str(blk["why_invalid"] or "none"),
        blocker_scope_mask=mask(blk["scope"]),
        blocker_mask=np.array([(blk["mask"] or {}).get(o, -1) for o in names], np.int8),
        blocker_minimal_sets=np.array([mask(s) for s in blk["minimal_sets"]] or np.zeros((0, n), bool)).reshape(-1, n),
        B_role=np.array([b["role"] for b in lab["B"]]), B_y=np.array([b["y"] for b in lab["B"]], np.int8),
        B_negative_kind=np.array([str(b["negative_kind"] or "none") for b in lab["B"]]),
        B_n_actions=np.array([b["n_actions"] for b in lab["B"]], np.int16),
        B_failed_step=np.array(["none" if b["failed_step"] is None else str(b["failed_step"]) for b in lab["B"]]),
        B_sequence=bseq,
        B_excess_actions=np.array([-1 if b.get("excess_actions") is None else b["excess_actions"] for b in lab["B"]], np.int16),
        bd_record_json=json.dumps(r, sort_keys=True, default=bc._jd))
    out.update({f"spec_{f}": str(r["spec"][f]) for f in bc.FACTORS})
    out.update({f"split_{s}": v.get(r["ids"]["root_id"], "none") for s, v in splits.items()})
    return out


def build(r, path, index, splits, src):
    e, R, coll, init, init_cells, moves, notes, azimuths, B = replay(r)
    names = [o["name"] for o in e.objects]
    goal, start, K = tuple(r["goal"]), tuple(r["target_src"]), r["verified_K"]
    q, track = np.array(R["q"]), np.array(R["obj"])
    final = np.array([e.obj_pos(j) for j in range(len(names))])
    final_cells = np.array([e.cell_of(n) if e.cell_of(n) is not None else (-1, -1) for n in names])
    masks, origin = M.swept_masks(e, track, np.array([TS.footprint_radius(o) for o in e.objects]))
    reached = e.cell_of("t0") == goal
    in_limits = bool(np.all(q >= e.jrange[:, 0] - 1e-6) and np.all(q <= e.jrange[:, 1] + 1e-6))
    n_coll = int(sum(coll.values()))
    carried = {o for o, _ in moves}
    fallen = [n for j, n in enumerate(names) if final[j, 2] < init[j, 2] - 0.01 or final_cells[j][0] < 0]
    pushed = [n for j, n in enumerate(names) if n not in carried and np.linalg.norm(final[j, :2] - init[j, :2]) > 0.005]
    held = np.array(R["held"])
    picks = [int(held[k]) for k in range(1, len(held)) if held[k] != held[k - 1] and held[k] >= 0]
    repicked = len(picks) != len(set(picks))
    clean = bool(n_coll == 0 and reached and not fallen and not pushed and not repicked and in_limits)
    moved_obs = [o for o, _ in moves if o != "t0"]
    kind = "none" if B is None else "direct" if K == 0 else "rearrangement"
    match = -1 if B is None else int(all(tuple(final_cells[names.index(o)]) == tuple(c) for o, c in B["final_cells"].items()))
    park_dist = [float(np.linalg.norm(e.cell_xy(c) - e.cell_xy(tuple(r["initial_cells"][o])))) for o, c in moves if o != "t0"]
    blk = r["labels"]["blocker"]
    lab, wit = labels(r, names, splits), witnesses(src, r)
    minimal = next((k for k, b in enumerate(r["labels"]["B"]) if b["role"] == "minimal" and b["y"]), None)
    save_npz(path, dict(
        # --- make_dataset.py: 1. label + trajectory (the replayed run) ---------------------------------------
        feasible=np.int8(r["labels"]["original"]["y"]), trajectory_kind=np.array(kind),
        time=np.array(R["t"]), joint_traj=q, gripper_gap=np.array(R["gripper"]), held_object=held.astype(np.int8),
        within_joint_limits=np.int8(in_limits), collision_free=np.int8(n_coll == 0),
        collision_counts=strs(f"{k}:{v}" for k, v in sorted(coll.items())), goal_reached=np.int8(reached),
        executed_moves=strs(f"{o}->{c}" for o, c in moves), notes=strs(notes), joint_limits=e.jrange,
        # --- 2. centroids ------------------------------------------------------------------------------------------
        object_names=np.array(names), object_shape=np.array([o["shape"] for o in e.objects]),
        object_size=np.array([o["size"] for o in e.objects]), object_height=np.array([o["height"] for o in e.objects]),
        centroids_initial=init, centroids_final=final, centroid_track=track,
        cells_initial=init_cells.astype(np.int16), cells_final=final_cells.astype(np.int16),
        start_cell=np.array(start, np.int16), goal_cell=np.array(goal, np.int16),
        # --- 3. / 4. blocking and redundant obstacles (of the executed run; obstacle index 0-based among obstacles) -
        n_obstacles=np.int16(len(names) - 1), n_blocking=np.int16(-1 if K is None else K),
        blocking_sets=pad([[names.index(o) - 1 for o in moved_obs]] if B is not None else [], max(len(moved_obs), 1)),
        n_redundant=np.int16(-1 if B is None else len(names) - 1 - len(moved_obs)),
        redundant_mask=np.array([B is not None and n not in moved_obs for n in names[1:]], bool),
        # --- 5. robot pose as dual quaternions -----------------------------------------------------------------
        ee_dual_quat=M.continuous(np.array(R["ee_dq"])),
        link_dual_quat=np.stack([M.continuous(np.array(R["link_dq"])[:, k]) for k in range(len(M.LINKS))], axis=1),
        link_names=np.array(M.LINKS),
        # --- 6. swept area ---------------------------------------------------------------------------------------
        swept_mask=masks, swept_mask_origin_xy=origin, swept_mask_res=np.float32(M.MASK_RES),
        swept_cells=M.swept_cells(e, masks, origin), swept_area_m2=masks.sum(axis=(1, 2)) * M.MASK_RES ** 2,
        object_moved=np.linalg.norm(track[-1] - track[0], axis=1) > 0.005,
        seed=np.int64(r["ids"]["construction_seed"]), hz=np.int16(M.HZ), table_z=np.float32(e.table_z), pitch=np.float32(e.pitch),
        grid_x0=np.float32(e.x0), object_color=np.array([M._rgba(o["color"]) for o in e.objects]),
        # --- quality checks of the replayed run --------------------------------------------------------------------
        objects_fallen=strs(fallen), objects_pushed=strs(pushed), repicked=np.int8(repicked), clean=np.int8(clean),
        parking_object=strs(moved_obs), parking_mode=strs("nearest_clear" for _ in moved_obs),
        parking_distance_m=np.array(park_dist or [0.0]),
        # --- dataset_v2 / front0 bookkeeping -----------------------------------------------------------------------
        dataset=np.array("bd_3dt_combined"), scene_index=np.int32(index),
        scene_class=np.array("unresolved" if K is None else "feasible" if K == 0 else f"infeasible-{K}"),
        layout_signature=np.array(V.signature(start, goal, init_cells[1:], [o["shape"] for o in e.objects[1:]])),
        n_removed=np.int16(len(moved_obs)), extra_removed=np.int16(0),
        n_blocking_kinematic=np.int16(min(map(len, blk["minimal_sets"])) if blk["blocker_supervision_valid"] else -1),
        grasp_family=np.array("front_level"), grasp_tilts_allowed=np.array(TS.TILTS_DEG, float), grasp_n_azimuth=np.int64(TS.N_AZIMUTH),
        grasp_tilt_deg=np.full(len(azimuths), TS.TILTS_DEG[0]) if len(TS.TILTS_DEG) == 1 else np.full(len(azimuths), np.nan),
        grasp_azimuth_deg=np.array(azimuths, float), table_height=np.float32(e.table_z),
        # --- dataset-bd ---------------------------------------------------------------------------------------------
        executed_witness=wit["B_witness"][minimal][:len(moves)] if minimal is not None else np.zeros(0, np.int16),
        replay_matches_label=np.int8(match), **wit, **lab))
    return dict(index=index, file=path.stem, root_id=r["ids"]["root_id"], n_obstacles=len(names) - 1, feasible=r["labels"]["original"]["y"],
                verified_K="NONE" if K is None else K, kind=kind, executed_moves=" ".join(f"{o}->{c[0]},{c[1]}" for o, c in moves) or "-",
                goal_reached=int(reached), clean=int(clean), collisions=" ".join(f"{k}:{v}" for k, v in sorted(coll.items())) or "-",
                pushed=" ".join(pushed) or "-", fallen=" ".join(fallen) or "-", replay_matches_label=match, steps=len(q),
                mask_valid=int(blk["blocker_supervision_valid"]), n_D=len(r["labels"]["D"]), n_B=len(r["labels"]["B"]),
                **{f"spec_{f}": r["spec"][f] for f in bc.FACTORS}, split_iid=lab["split_iid"], notes=" | ".join(notes) or "-")


SCHEMA = """# {name}: combined B/D + dense format

One self-contained `scene_XXXX.npz` per dataset-bd root (no GIFs), made by `make_combined_3dt.py` from `{src}` (which keeps the
original dataset-bd layout: roots/, witnesses/, MANIFEST.json, SPLITS.json). Everything of a root record is inside its npz: the
labels as arrays, the planned-waypoint witnesses as ragged arrays, and the full record as `bd_record_json`.
`manifest.csv`: one row per scene (root id, verified K, the replayed run's checks, spec factors, iid split).

Storage: float arrays are float32 (max error ~1e-7 rad / 3e-8 m against the float64 recording) and the npz members are
LZMA-compressed. `np.load` reads them as usual; a non-Python npz reader must support LZMA zip members.

Object axis everywhere: the scene's order `t0` (the target), `o00`, `o01`, ... = `object_names`.
Cells: (row, col), row 0 nearest the robot, col 0 the -y edge (table_scene).

## Dense run (make_dataset.py fields)
The root's verified minimal B sequence, then the target's move, re-executed in MuJoCo with bd_3dt's planner (same grasp order and
seeds as the labels) and recorded with make_dataset.py's Recorder at `hz` = 50 Hz. A root without a verified repair
(`verified_K` = -1) has the initial state only: `trajectory_kind` = 'none', `n_blocking` = -1, `n_redundant` = -1.

| array | shape | meaning |
|---|---|---|
| feasible | () | 1 = the target's move plans in the original scene (= y_original) |
| trajectory_kind | () | direct (K = 0), rearrangement (K > 0) or none |
| time, joint_traj, gripper_gap, held_object | (T), (T,7), (T), (T) | the run at 50 Hz: s, xArm7 joints (rad), pad gap (m), held object index (-1 none) |
| ee_dual_quat, link_dual_quat, link_names | (T,8), (T,8,8), (8) | grasp-site and link poses as sign-continuous unit dual quaternions [w,x,y,z, dual] |
| centroids_initial / _final, centroid_track | (N,3), (N,3), (T,N,3) | object centres (m) |
| cells_initial / _final, start_cell, goal_cell | (N,2), (N,2), (2), (2) | cells |
| executed_moves, notes | (M) | "o00->(3, 6)" ...; replay notes (pushes seen by run_plan) |
| n_blocking, blocking_sets, n_redundant, redundant_mask | (), (1,K), (), (N-1) | = verified K and the obstacles the run moved (index among obstacles) |
| n_blocking_kinematic | () | smallest blocker-lattice minimal set (target-only replanning after teleporting), -1 if not valid |
| swept_mask, swept_cells, swept_area_m2, object_moved | (N,H,W), (N,8,8), (N), (N) | area each object's footprint swept (5 mm pixels, origin swept_mask_origin_xy) |
| collision_free, collision_counts, goal_reached, within_joint_limits, objects_fallen, objects_pushed, repicked, clean | | make_dataset.py checks of the replayed run (clean = all pass) |
| parking_object, parking_mode, parking_distance_m | (M) | obstacles moved, 'nearest_clear' = dataset-bd's first parking cell, distance (m) |
| grasp_azimuth_deg, grasp_tilt_deg | (M) | side grasp used for each executed move |
| replay_matches_label | () | 1 = every object ended in the cell the B label recorded, 0 = not, -1 = nothing to replay |
| executed_witness | (M) | witness index of each executed move's planned waypoints |

## dataset-bd labels
| array | shape | meaning |
|---|---|---|
| verified_K | () | verified number of relocations (-1 = NONE: no verified repair within the declared budget) |
| y_original, original_negative_kind, original_reason, original_hit_objects | | the target's query in the original scene |
| object_role, intended_blocker_mask | (N), (N) | target / intended_blocker / distractor:<role>; built-in blockers |
| parking_cells | (N,3,2) | dataset-bd parking candidates per object (-1 padded) |
| D_edit_mask, D_edit_cells, D_y, D_negative_kind | (D,N), (D,N,2), (D), (D) | D queries: these obstacles teleported to these cells, then only the target re-planned |
| blocker_scope_mask, blocker_mask, blocker_minimal_sets, blocker_supervision_valid, blocker_why_invalid | (N), (N), (S,N), (), () | blocker supervision (mask -1 = outside scope / invalid) |
| B_role, B_y, B_negative_kind, B_n_actions, B_failed_step, B_excess_actions | (B) | executed B sequences: minimal, redundant+1, noop_irrelevant, wrong_destination, wrong_order, intended_union |
| B_sequence | (B,6,3) | relocations before the target's move: [object index, row, col], -1 padded |
| spec_<factor> | () | requested factors: K, distractors, source, displacement, mechanism, parking, distractor_role, start_pose |
| split_<name> | () | iid / ood_K / ood_obstacle_count / ood_source_region / ood_distractor_density |
| root_id, parent_family_id, construction_seed, config_hash, *_version, robot_protocol, label_protocol | () | ids |
| bd_record_json | () | the whole root record (roots/<root_id>.json of the source) |

## Planned-waypoint witnesses (dataset-bd witnesses/, ragged)
Witness k: `witness_q[witness_q_start[k]:witness_q_start[k+1]]` (planned joint waypoints, rad) with `witness_segment` (index into
its phases), `witness_phases[witness_phases_start[k]:...]` (to_hover, open, approach, close, lift, carry, lower, release, retreat,
to_ready) and `witness_carry_xy[witness_carry_xy_start[k]:...]` (the carried object's route, m). `witness_ids` are the source ids.
Labels point to witnesses by index (-1 = none): `original_witness` (), `D_witness` (D), `B_witness` (B, 7) one per executed action.
"""


def main():
    src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    splits = json.loads((src / "SPLITS.json").read_text())["splits"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None       # the first n roots (wave, slot order): runs overshoot by up to a wave
    recs = [json.loads(p.read_text()) for p in sorted((src / "roots").glob("*.json"))[:n]]
    rows = []
    for k, r in enumerate(recs):
        row = build(r, dst / f"scene_{k:04d}.npz", k, splits, src)
        rows.append(row)
        print({c: row[c] for c in ("file", "root_id", "verified_K", "executed_moves", "clean", "collisions", "replay_matches_label", "steps")}, flush=True)
    with open(dst / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (dst / "README.md").write_text(SCHEMA.format(name=dst.name, src=src.name))
    print(dst)


if __name__ == "__main__":
    main()
