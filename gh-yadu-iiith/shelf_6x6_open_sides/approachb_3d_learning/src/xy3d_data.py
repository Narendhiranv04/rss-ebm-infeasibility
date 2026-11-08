"""3D X/Y data interface (BWL-2 / BWL-3): learner-visible root views, GEO3D observable proxy features, the 3D Y-provenance audit and the
physics-host-only manifests (Y witness replay sample, authentic ordered-prefix X pilot).  Reads the pinned snapshot read-only; runs no physics.

    N="taskset -c 7 nice -n 15 venv/bin/python"
    $N src/xy3d_data.py yaudit        classify every 3D target-only record (original + D) by provenance; no label is changed
    $N src/xy3d_data.py manifest-y    predeclared physics-host Y replay / re-plan sample (NOT executed here)
    P="env CUDA_VISIBLE_DEVICES=0 taskset -c 7 nice -n 10 venv/bin/python"
    $P src/xy3d_data.py manifest-x    predeclared authentic X pilot (frozen cVAE destinations need one tiny GPU inference; NOT executed)"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import xy3d as XY  # noqa: E402
from data3d import COLS, PITCH, ROWS, X0, cell_xy  # noqa: E402

SNAP = ROOT / "snapshots/v4full_82926152fa9cc345"
UP = SNAP / "upstream/gh-yadu-iiith/shelf_6x6_open_sides"
RAW = UP / "dataset_bd_classquota_without_home_position_no_overpass/raw/roots"
V3 = UP / "dataset_placement_extension_v2/production_v3"
OUT = ROOT / "runs/XY3D_BWL"
# learner-visible root fields; everything else in a raw root (roles, intended_blockers, parking, construction, greedy_blockers, mechanism_pools,
# empty_table_T_solve, grasp_azimuths, verified_K, observed, spec, label_consistency, independent_moves, teacher_calls, label outcomes) is privileged
GEOM = ("x0", "pitch", "rows", "cols", "arm_ceiling", "no_overpass", "return_to_ready")
# declared reference state: the SHELF66 'confined' ready pose (shelf_scene.py l.338-339: hand down 5 cm outside the shelf front, 0.20 m above the
# floor, azimuth 180, tilt 0, gripper free).  Joint values are not stored in the roots; q0 = q_ref for every root (spec start_pose 'home').
REF = [X0 - 0.5 * ROWS * PITCH - 0.05, 0.0, 0.20, math.cos(math.pi), math.sin(math.pi), 0.0, 1.0]
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def cell_of(p):
    return int(round((p[0] - X0) / PITCH + 2.5)), int(round(p[1] / PITCH + 2.5))


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def open_dirs(cell, occ):
    """Observable access proxy: compass directions whose two hand cells are free (open sides and front count free; the back wall does not)."""
    n = 0
    for dr, dc in DIRS:
        ok = True
        for k in (1, 2):
            r, c = cell[0] + k * dr, cell[1] + k * dc
            if r >= ROWS or (0 <= r < ROWS and 0 <= c < COLS and (r, c) in occ):
                ok = False
                break
        n += ok
    return n


def line_cells(a, b):
    out = []
    for i in range(1, 28):                                                    # bd_3dt.construct's discretization (29 points, ends excluded)
        t = i / 28
        x = (int(round((1 - t) * a[0] + t * b[0])), int(round((1 - t) * a[1] + t * b[1])))
        if x not in out and x not in (a, b):
            out.append(x)
    return out


class GEO3D:
    """Structural port of bxy_p17 GEO2 (phi_y = 8 clearances + 5 access values of the endpoint at q_ref; phi_x = 8 statistics of the actions in
    their IDEAL predecessor scenes, or in S0 for the order-invariant twin) with 3D OBSERVABLE OCCUPANCY PROXIES: there is no audited 3D access map,
    so these are not teacher-derived grasp-mode counts (declared limitation)."""

    @staticmethod
    def phi_y(v, xy):
        cells, t, goal = [cell_of(p) for p in xy], v["tid"], cell_of(v["goal"])
        others = [c for i, c in enumerate(cells) if i != t]
        occ, T = set(others), cells[t]
        line = line_cells(T, goal)
        p, q = open_dirs(T, occ), open_dirs(goal, occ)
        h = lambda i: v["attrs"][i][1]  # noqa: E731
        clear = [min((cheb(T, c) for c in others), default=5) / 5, min((cheb(goal, c) for c in others), default=5) / 5,
                 sum(cheb(T, c) <= 1 for c in others) / 8, sum(cheb(goal, c) <= 1 for c in others) / 8, sum(c in line for c in others) / 4,
                 sum(c[1] == T[1] and c[0] < T[0] for c in others) / 5, sum(c[1] == goal[1] and c[0] < goal[0] for c in others) / 5,
                 max((h(i) for i, c in enumerate(cells) if i != t and cheb(T, c) <= 1), default=0.0) / h(t)]
        return clear + [p / 8, q / 8, float(p > 0), float(q > 0), float(goal in occ)]

    @staticmethod
    def phi_x(v, seq, ordered):
        cells0 = [cell_of(p) for p in v["xy0"]]
        cur, P, Q = list(cells0), [], []
        for i, dest in seq:
            base = cur if ordered else cells0
            occ = {c for j, c in enumerate(base) if j != i}
            d = cell_of(dest)
            P.append(open_dirs(base[i], occ) / 8)
            Q.append(open_dirs(d, occ) / 8)
            if ordered:
                cur[i] = d
        mean = lambda x: sum(x) / len(x)  # noqa: E731
        return [mean(P), min(P), mean(Q), min(Q), mean([p == 0 for p in P]), mean([q == 0 for q in Q]), len(P) / 6, 0.0]


def visible(raw):
    """The ONLY root fields the X/Y scorer reads."""
    return {"root_id": raw["ids"]["root_id"], "objects": [{k: o[k] for k in ("id", "shape", "size", "height")} for o in raw["objects"]],
            "cells": {k: list(v) for k, v in raw["initial_cells"].items()}, "goal": list(raw["goal"]), "geometry": {k: raw["geometry"][k] for k in GEOM}}


def design(raw):
    """Outcome-free design of the root's historical queries: D endpoint edits (poses only) and B ordered sequences (DESIGN_ONLY, never X)."""
    L = raw["labels"]
    return [dict(r["edit"]) for r in L["D"]], [[(o, list(c)) for o, c in r["sequence"]] for r in L["B"] if r.get("sequence")]


def view3d(raw, extra_edits=(), extra_seqs=(), use_history=True):
    vis = visible(raw)
    ids = sorted(o["id"] for o in vis["objects"])
    slot = {o: i for i, o in enumerate(ids)}
    ob = {o["id"]: o for o in vis["objects"]}
    attrs = [[ob[o]["size"], ob[o]["height"], float(ob[o]["shape"] == "cylinder"), float(o == "t0"), float(o != "t0")] for o in ids]
    tg, g = ob["t0"], vis["geometry"]
    ctx = REF + list(cell_xy(*vis["goal"])) + [tg["size"], tg["height"], float(g["arm_ceiling"]), float(g["no_overpass"]), float(g["return_to_ready"])]
    edits, seqs = design(raw) if use_history else ([], [])
    y_edits = [{slot[o]: cell_xy(*c) for o, c in e.items()} for e in list(edits) + list(extra_edits)]
    x_seqs = [[(slot[o], cell_xy(*c)) for o, c in s] for s in list(seqs) + list(extra_seqs)]
    v, ym, xm = XY.make_view(vis["root_id"], [cell_xy(*vis["cells"][o]) for o in ids], attrs, slot["t0"], cell_xy(*vis["goal"]), (0.0, 0.0), ctx,
                             y_edits, x_seqs, GEO3D)
    return v, ym, xm, ids


def labels3d(raw, v, ym, xm):
    """PLANNING-ONLY Y labels of the historical target-only queries (original -> Y0, D -> its endpoint).  No X label exists (B is composite)."""
    return XY.labels_for(v, ym, xm, obs_y={k: r["y"] for k, r in enumerate(raw["labels"]["D"])}, noop_y=raw["labels"]["original"]["y"])


def qc_roots():
    split = json.loads((V3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
    qc = set(json.loads((V3 / "QC_ELIGIBLE_ROOTS.json").read_text())["roots"])
    return {r: split[r] for r in sorted(qc)}


def load(rid):
    return json.loads((RAW / f"{rid}.json").read_text())


# ---------------------------------------------------------------- BWL-3: Y provenance audit
def classify(rec, kind_ok):
    if not kind_ok:
        return "Y_SEMANTICS_INCOMPATIBLE"
    if rec["y"] == 1:
        return "Y_PLANNING_ONLY"
    return {"CERTIFIED": "Y_POLICY_FAILED:CERTIFIED_GRASP_SCREEN", "NO_PLAN_FOUND": "Y_POLICY_FAILED:PLANNER_BUDGET"}.get(rec.get("negative_kind"),
                                                                                                                       "Y_UNATTEMPTED_OR_UNRESOLVED")


def cmd_yaudit(a):
    """Every original / D record of the 1,582 QC roots.  Source facts (bd_3dt.py sha256 62dc853c..., rss-ebm-infeasibility 6f042be1):
    q_t (l.245-249) = settle (l.145: reset to the READY pose, teleport edits) + query (l.96: plan_pick_place, 'Nothing moves') -> NO MuJoCo execution
    or replay of the target; only seq() (l.265-292) runs plans (run_plan l.280), from the achieved state with the target appended (B records)."""
    roots, rows, c, issues = qc_roots(), [], Counter(), Counter()
    for rid, sp in roots.items():
        raw = load(rid)
        ids, L = raw["ids"], raw["labels"]
        tv_ok = ids.get("teacher_version", "").startswith("BD3DS66_TEACHER_V2") and "pre-shaped" in ids.get("teacher_version", "")
        proto_ok = ids.get("label_protocol") == "BD_BINARY_BUDGET_V1" and raw["spec"].get("start_pose") == "home"
        cells0 = {k: tuple(v) for k, v in raw["initial_cells"].items()}
        k0_exec = any(b.get("role") == "minimal" and b.get("n_actions") == 1 and b.get("y") == 1 for b in L["B"])
        seen = set()
        for fam, rec in [("original", L["original"])] + [("D", r) for r in L["D"]]:
            edit = {o: tuple(cc) for o, cc in rec.get("edit", {}).items()}
            occ = {v for k, v in cells0.items() if k not in edit}
            ok_edit = all(o in cells0 and o != "t0" and 0 <= cc[0] < ROWS and 0 <= cc[1] < COLS and cc not in occ for o, cc in edit.items()) and \
                len(set(edit.values())) == len(edit)
            sig = tuple(sorted(edit.items()))
            dup = sig in seen
            seen.add(sig)
            wit_ok = bool(rec.get("witness")) == (rec["y"] == 1)
            kind_ok = tv_ok and proto_ok and ok_edit and wit_ok and rec["y"] in (0, 1) and not dup
            for name, flag in (("teacher_version", tv_ok), ("protocol", proto_ok), ("edit_illegal", ok_edit), ("witness_mismatch", wit_ok), ("duplicate", not dup)):
                issues[name] += not flag
            cat = classify(rec, kind_ok)
            row = {"root_id": rid, "split": sp, "family": fam, "edit": {o: list(v) for o, v in edit.items()}, "edit_size": len(edit), "y": rec["y"],
                   "negative_kind": rec.get("negative_kind"), "category": cat, "witness_id": rec.get("witness"),
                   "executed_once_in_teacher": bool(fam == "original" and rec["y"] == 1 and k0_exec), "replay_verified": False}
            rows.append(row)
            c[(cat, fam, sp)] += 1
    by = {}
    for (cat, fam, sp), n in c.items():
        by.setdefault(cat, {}).setdefault(fam, {})[sp] = n
    pos_orig = [r for r in rows if r["family"] == "original" and r["y"] == 1]
    res = {"records": len(rows), "roots": len(roots), "by_category": by, "category_totals": dict(Counter(r["category"] for r in rows)),
           "Y_REPLAY_VERIFIED": 0, "semantic_check_failures": dict(issues),
           "original_positives_executed_once_in_teacher": sum(r["executed_once_in_teacher"] for r in pos_orig), "original_positives": len(pos_orig),
           "edit_size_of_positives": dict(Counter(r["edit_size"] for r in rows if r["y"] == 1)),
           "source": {"file": "dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_v3/bd_3dt.py",
                      "sha256": "62dc853ccb5055eaad738a9b2eb810c1de5cb12621c1039bbd68c20b54058815", "repo_commit": "6f042be12de3384f884a6f831e1e8c76d8516c5a",
                      "facts": ["q_t l.245-249: settle(cells) then query(e, 't0', goal); no run_plan -> target outcome is PLANNING ONLY",
                                "settle l.145-150: e.reset() = home cells + READY pose, then teleport -> fixed q_ref, ideal (teleported) endpoint",
                                "query l.96-108: plan_pick_place over the scene's grasp poses, first feasible wins, nothing moves",
                                "kind l.124-128: CERTIFIED iff every grasp fails a deterministic geometric check (finite grasp set of the protocol)",
                                "seq l.265-292: run_plan executes relocations from the achieved state and appends the target (B: composite, no q_ref connector)",
                                "K0 roots: seq([]) executes the target once from the ready pose (teacher run, not an independent replay)"]},
           "rows_file": "runs/XY3D_BWL/Y_PROVENANCE_ROWS.json"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Y_PROVENANCE_ROWS.json").write_text(json.dumps(rows))
    (OUT / "Y_PROVENANCE_AUDIT.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "source"}, indent=1))


def pick(rows, n, tag):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{tag}|{r['root_id']}|{json.dumps(r['edit'], sort_keys=True)}".encode()).hexdigest())[:n]


def cmd_manifest_y(a):
    rows = json.loads((OUT / "Y_PROVENANCE_ROWS.json").read_text())
    strata = {"original_pos_K0_executed_once": lambda r: r["family"] == "original" and r["y"] == 1,
              "D_pos_single_object_edit": lambda r: r["family"] == "D" and r["y"] == 1 and r["edit_size"] == 1,
              "D_pos_multi_object_edit": lambda r: r["family"] == "D" and r["y"] == 1 and r["edit_size"] >= 2,
              "original_neg_certified": lambda r: r["family"] == "original" and r["negative_kind"] == "CERTIFIED",
              "D_neg_certified": lambda r: r["family"] == "D" and r["negative_kind"] == "CERTIFIED",
              "original_neg_planner_budget": lambda r: r["family"] == "original" and r["negative_kind"] == "NO_PLAN_FOUND",
              "D_neg_planner_budget": lambda r: r["family"] == "D" and r["negative_kind"] == "NO_PLAN_FOUND"}
    n = {"original_pos_K0_executed_once": 24, "D_pos_single_object_edit": 24, "D_pos_multi_object_edit": 24, "original_neg_certified": 12,
         "D_neg_certified": 12, "original_neg_planner_budget": 12, "D_neg_planner_budget": 12}
    sel = []
    for s, f in strata.items():
        for r in pick([r for r in rows if f(r) and r["category"] != "Y_SEMANTICS_INCOMPATIBLE"], n[s], "BWL3_YREPLAY"):
            sel.append({"manifest_row": f"BWL3Y-{len(sel):04d}", "stratum": s, **{k: r[k] for k in ("root_id", "split", "family", "edit", "y", "negative_kind", "witness_id")}})
    man = {"manifest": "BWL3_Y_REPLAY_SAMPLE_V1", "status": "PREDECLARED_NOT_EXECUTED", "host": "authorized physics host only (MuJoCo + the SHELF66V3 scene)",
           "selection": "per stratum, first n by sha256('BWL3_YREPLAY|'+root_id+'|'+edit) among semantically compatible records; fixed before any replay",
           "protocol": ["settle(cells0 + edit) exactly as bd_3dt.settle (reset to the READY pose = q_ref, teleport the edit), gripper free",
                        "POSITIVES: (1) independent replay of the stored target witness (witness_id) with MuJoCo run_plan from q_ref; pass iff T ends in the goal "
                        "cell upright (|z_err| <= 0.01 m) and no other object changes cell; (2) fresh re-plan with seeds 0, 1, 2 at RRT max_iters 5000 and execute "
                        "each; record plan success, execution success, end cell, z_err, displaced objects, joint travel of the executed target path",
                        "NEGATIVES: CERTIFIED -> re-run the deterministic grasp screen (expect every grasp blocked, same reasons); NO_PLAN_FOUND -> re-plan with "
                        "seeds 0, 1, 2 at the same budget and once at 4x budget (budget sensitivity, never relabels the archived record)",
                        "statuses: REPLAY_PASS / REPLAY_FAIL / PLAN_IRREPRODUCIBLE / CENSORED (budget) / ENV_ERROR; never convert a censored run into 0"],
           "decision_use": "Y_REPLAY_VERIFIED is granted only to records whose own witness passes (1); the sample estimates the planning-to-execution "
                           "gap of the 2,645 planning positives; promotion of the full pool requires replaying ALL of them (recommended: 2,645 replays)",
           "counts": dict(Counter(r["stratum"] for r in sel)), "rows": sel}
    man["sha256_of_rows"] = hashlib.sha256(json.dumps(sel, sort_keys=True).encode()).hexdigest()
    (OUT / "PHYSICS_Y_REPLAY_MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "rows"}, indent=1))


# ---------------------------------------------------------------- BWL-3: authentic ordered-prefix X pilot (physics host only)
def legal_cells(cells):
    occ = set(map(tuple, cells.values()))
    return [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in occ]


def band_pick(cells, o, k, tag):
    src = tuple(cells[o])
    leg = legal_cells(cells)
    bands = [[x for x in leg if cheb(x, src) == 1], [x for x in leg if 2 <= cheb(x, src) <= 3], [x for x in leg if cheb(x, src) >= 4]]
    band = next((b for j in range(3) for b in [bands[(k + j) % 3]] if b), leg)
    return min(band, key=lambda x: hashlib.sha256(f"{tag}|{x}".encode()).hexdigest())


def cvae_dest(cells, objs, o, tag, net, sc, dev):
    import numpy as np
    import torch
    from data3d import scene_features
    f, m, occ, leg = scene_features({k: objs[k] for k in cells}, {k: tuple(v) for k, v in cells.items()}, o)
    f = (f - np.array(sc["mean"], np.float32)) / np.array(sc["std"], np.float32) * m[..., None]
    g = torch.Generator(device=dev).manual_seed(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))
    T = lambda x, dt=torch.float32: torch.as_tensor(x[None], dtype=dt, device=dev)  # noqa: E731
    d, _ = net.sample(T(f), T(m, torch.bool), T(occ), T(leg, torch.bool), 1, gen=g)
    k = int(d[0, 0])
    return (k // COLS, k % COLS)


def cmd_manifest_x(a):
    import torch
    import guard, train as TR
    _, dev = TR.preflight({}, guard.load_cfg(a.guard_config), False)
    wd = TR.start_watchdog(OUT / "manifest_x", a.guard_config)
    try:
        net = TR.load_model(ROOT / "runs/SAMPLER_V1/cvae_s0/best.pt", dev)[0]
        sc = json.loads((SNAP.parent / (SNAP.name + "_derived") / "scalers.json").read_text())
        roots = {r: s for r, s in qc_roots().items() if s in ("train", "val")}
        byk = {}
        for rid, sp in roots.items():
            byk.setdefault((load(rid)["verified_K"], sp), []).append(rid)
        chosen = []
        for K in (0, 1, 2, 3):
            for sp, n in (("train", 8), ("val", 2)):
                chosen += sorted(byk.get((K, sp), []), key=lambda r: hashlib.sha256(f"BWL3_XPILOT|{r}".encode()).hexdigest())[:n]
        rows, yends = [], {}
        for rid in chosen:
            raw = load(rid)
            vis = visible(raw)
            cells = {k: tuple(v) for k, v in vis["cells"].items()}
            objs = {o["id"]: o for o in vis["objects"]}
            hist = [o for o, _ in (next((b["sequence"] for b in raw["labels"]["B"] if b.get("role") == "minimal"), []) or [])]
            movers = (hist + sorted(o for o in cells if o != "t0" and o not in hist))[:4]
            park = {o: tuple(v[0]) for o, v in raw["parking"].items() if v}

            def dest_sources(o, sc_cells, tag):
                d = {"historical_parking": park.get(o), "stratified_band": band_pick(sc_cells, o, len(tag) % 3, tag),
                     "uniform_random": min(legal_cells(sc_cells), key=lambda x: hashlib.sha256(f"U|{tag}|{x}".encode()).hexdigest()),
                     "sampler_cvae_s0": cvae_dest(sc_cells, objs, o, tag, net, sc, dev)}
                return {k: v for k, v in d.items() if v is not None and v not in set(sc_cells.values())}

            def add(seq, source, kind):
                key = tuple((o, tuple(d)) for o, d in seq)
                dup = next((r for r in rows if r["root_id"] == rid and r["x_key"] == key), None)
                if dup is not None:                                           # one physical query: merge its provenance tags
                    dup["also"] = sorted(set(dup.get("also", [])) | {f"{kind}:{source}"})
                    return
                end = dict(cells)
                for o, d in seq:
                    end[o] = tuple(d)
                if len(set(end.values())) != len(end):
                    return                                                    # illegal endpoint under the static rule (never queried)
                ysig = json.dumps(sorted((k, list(v)) for k, v in end.items()))
                yends.setdefault((rid, ysig), f"Y-{len(yends):05d}")
                rows.append({"root_id": rid, "split": roots[rid], "verified_K_reporting_only": raw["verified_K"], "x_key": key, "query_kind": kind,
                             "ordered_moves": [[o, list(d)] for o, d in seq], "m": len(seq), "destination_source": source, "y_endpoint": yends[(rid, ysig)]})
            for o in movers:                                                  # one-move stopping prefixes (all sources)
                for src_name, d in dest_sources(o, cells, f"{rid}|{o}").items():
                    add([(o, d)], src_name, "ONE_MOVE")
            for i in range(len(movers)):                                      # AB vs BA, same endpoint (parking and cVAE destinations)
                for j in range(i + 1, min(len(movers), 3)):
                    A, B = movers[i], movers[j]
                    for src_name in ("historical_parking", "sampler_cvae_s0"):
                        da, db = dest_sources(A, cells, f"{rid}|{A}").get(src_name), dest_sources(B, cells, f"{rid}|{B}").get(src_name)
                        if da and db and da != db:
                            add([(A, da), (B, db)], src_name, "PAIR_AB")
                            add([(B, db), (A, da)], src_name, "PAIR_BA_SAME_ENDPOINT")
            if len(movers) >= 3:                                              # three-move prefixes, 4 orders, all single omissions
                A, B, C = movers[:3]
                ds = {o: park.get(o) or dest_sources(o, cells, f"{rid}|{o}").get("stratified_band") for o in (A, B, C)}
                if len(set(ds.values())) == 3 and all(ds.values()):
                    for order in ((A, B, C), (A, C, B), (B, A, C), (C, B, A)):
                        add([(o, ds[o]) for o in order], "historical_parking", "TRIPLE_ORDER")
                    for t in range(3):
                        add([(o, ds[o]) for k, o in enumerate((A, B, C)) if k != t], "historical_parking", "OMISSION_OF_TRIPLE")
            if hist:                                                          # historical intended candidate, its reverse and its omissions
                hs = [(o, park[o]) for o in hist if o in park]
                if hs:
                    add(hs, "historical_intended", "HISTORICAL_MINIMAL")
                    if len(hs) >= 2:
                        add(list(reversed(hs)), "historical_intended", "HISTORICAL_REVERSED")
                        for t in range(len(hs)):
                            add([x for k, x in enumerate(hs) if k != t], "historical_intended", "HISTORICAL_OMISSION")
            nuis = [o for o in cells if o != "t0" and o not in hist]
            if nuis and hist:                                                 # nuisance move before the intended repair
                d = dest_sources(nuis[0], cells, f"{rid}|N").get("uniform_random")
                if d:
                    add([(nuis[0], d)] + [(o, park[o]) for o in hist if o in park], "uniform_random+historical", "NUISANCE_PLUS_REPAIR")
        for r in rows:
            r["x_key"] = ".".join(f"{o}>{d[0]},{d[1]}" for o, d in r["x_key"])
        noop = [{"root_id": rid, "split": roots[rid], "query_kind": "NO_OP", "m": 0, "X": "1 by convention (not queried)", "y_endpoint": "Y0 (root scene)"} for rid in chosen]
        man = {"manifest": "BWL3_AUTHENTIC_X_PILOT_V1", "status": "PREDECLARED_NOT_EXECUTED", "host": "authorized physics host only (never this GPU server)",
               "roots": {"train": sum(roots[r] == "train" for r in chosen), "val": sum(roots[r] == "val" for r in chosen), "by_K": dict(Counter(load(r)["verified_K"] for r in chosen))},
               "selection": "QC roots of the inherited train / val split only (no test root), 8 train + 2 val per K0..K3 by sha256('BWL3_XPILOT|'+root); "
                            "all queries and destination sources fixed here BEFORE any outcome (cVAE destinations: frozen SAMPLER_V1 cvae_s0, seeded by the query tag)",
               "X_protocol": ["start S0 / q0 = the READY pose (= q_ref), gripper free; relocate in the listed order with the pinned grasp / motion policy "
                              "(BD3DS66_TEACHER_V2 planner, RRT max_iters 5000, fingers pre-shaped) carrying the ACHIEVED robot state between moves (no reset)",
                              "each move verified: mover ends in its destination cell upright (|z_err| <= 0.01 m), no other object changes cell",
                              "after the last move (stopping prefix): plan AND execute a free-gripper connector to q_ref in the edited scene",
                              "X = 1 iff every phase and the connector succeed; then an INDEPENDENT replay of the full stored X witness must reproduce it",
                              "X = 0 only for an attempted query (record failed phase: grasp / transfer / place / release / retreat / transition / connector, "
                              "negative_kind CERTIFIED vs PLANNER_BUDGET vs EXECUTION, attempts, budget); unattempted / env errors are MISSING"],
               "Y_protocol": ["for every unique y_endpoint: settle the IDEAL endpoint (teleport) at q_ref, plan the original target T -> goal, EXECUTE it in "
                              "MuJoCo, verify T in the goal cell upright and nothing displaced, then replay the witness independently (Y measured even when X = 0)",
                              "Y0 (no-op endpoint) uses the same protocol on the root scene"],
               "travel_J": "joint-space path length (rad, summed over the 7 joints) of the executed X witness (relocation phases + transitions + connector) "
                           "plus the executed Y target path from q_ref (November 5 convention); only for complete (1, 1) witnesses",
               "record_schema": "XYTeacher of BLACKWELL_APPROACH_B_EXECUTION_CONTRACT_v1 section 12 (root_id, X / Y keys, observed, value, status and negative "
                                "scope, connector_verified, qref_verified, witnesses + hashes, replay_status, attempts / budget, failed_phase, travel_J, split)",
               "never_infer_X_from": ["single-object placement labels", "endpoint-only checks", "B composite sequences (target from the achieved state, no connector)",
                                      "a successful order for its permutations", "teleported ideal edits"],
               "counts": {"x_queries": len(rows), "unique_y_endpoints": len(yends) + len(chosen), "by_kind": dict(Counter(r["query_kind"] for r in rows)),
                          "tag_occurrences_incl_merged": dict(Counter([r["query_kind"] for r in rows] + [t.split(":")[0] for r in rows for t in r.get("also", [])])),
                          "same_endpoint_order_groups": sum(1 for v in Counter((r["root_id"], r["y_endpoint"]) for r in rows if r["m"] >= 2).values() if v >= 2),
                          "by_source": dict(Counter(r["destination_source"] for r in rows)), "by_m": dict(Counter(r["m"] for r in rows))},
               "go_no_go_after_pilot": "scale only if the pilot yields decided same-endpoint AB/BA contrasts with discordant X, all four (X, Y) cells, "
                                       "replay-reproducible witnesses and a measured teacher cost; otherwise the X/Y architecture comparison is EVIDENCE_BLOCKED",
               "no_op_rows": noop, "rows": rows}
        man["sha256_of_rows"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        (OUT / "PHYSICS_X_PILOT_MANIFEST.json").write_text(json.dumps(man, indent=1))
        print(json.dumps({k: v for k, v in man.items() if k not in ("rows", "no_op_rows")}, indent=1))
    finally:
        wd.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["yaudit", "manifest-y", "manifest-x"]); ap.add_argument("--guard-config")
    a = ap.parse_args()
    {"yaudit": cmd_yaudit, "manifest-y": cmd_manifest_y, "manifest-x": cmd_manifest_x}[a.cmd](a)
