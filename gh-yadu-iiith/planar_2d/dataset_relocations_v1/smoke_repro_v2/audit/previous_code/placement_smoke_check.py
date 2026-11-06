"""Assemble, re-open and audit a placement smoke output (placement_smoke.py), and render figures from the saved files only.

    python placement_smoke_check.py <out>      # records/, modes/, manifest.json, summary.json, verification.json, visuals/

Audit: every predeclared query has exactly one finished record; attempted -> binary label, not attempted -> null; a 1 has >= 1
witness and a 0 none; pair-category counts reconcile with the mode arrays; every witness file reopens, matches its content id,
has no NaN, starts at closure and ends with the retreat, and re-certifies (motion.certify_trajectory) in the scene it was found
in; hypothetical scenes re-hash to the recorded value; the original dataset folder is unchanged against git.
"""
import hashlib, io, json, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from placement_smoke import A, DATA, ge, mo, sk, H, SCHEMA, load_root, roster, sha_file, scene_from_dict, scene_hash, substitute_pose  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, Circle  # noqa: E402
from PIL import Image  # noqa: E402
import viz  # noqa: E402

MODE_KEYS = ("pick_stage", "pick_definite", "place_stage", "place_definite", "mode_phi_rad", "mode_branch", "pair_category",
             "attempted_mask", "pair_outcome")


def scene_of(rec):
    s = scene_from_dict(load_root(rec["source_root_id"])["scene"])
    for o, c in rec["prefix_key"]:
        s = substitute_pose(s, o, c)
    return s


def assemble(out, tasks):
    order = {t["trial_id"]: k for k, t in enumerate(tasks)}
    recs = sorted((json.loads(p.read_text()) for p in out.glob("parts/*.json")), key=lambda r: order.get(r["trial_id"], 1e9))
    by = defaultdict(list)
    for r in recs:
        by[r["source_root_id"]].append(r)
    for d in ("records", "modes"):
        (out / d).mkdir(exist_ok=True)
    for rid, rs in by.items():
        att = [r for r in rs if r["attempted"]]
        for k, r in enumerate(att):
            r["modes_index"] = k
        (out / "records" / f"{rid}.json").write_text(json.dumps({"schema_version": SCHEMA, "source_root_id": rid,
                                                                 "source_root_hash": rs[0]["source_root_hash"], "domain": "planar_2d",
                                                                 "split": None, "trials": rs}, indent=1, sort_keys=True))
        z = [np.load(out / "parts" / f"{r['trial_id']}.npz") for r in att]
        arr = {k: np.stack([x[k] for x in z]) for k in MODE_KEYS}
        arr["trial_id"] = np.array([r["trial_id"] for r in att])
        arr["attempt_trial"] = np.concatenate([np.full(len(x["attempt_seconds"]), k, np.int16) for k, x in enumerate(z)])
        for k in ("attempt_pick_place", "attempt_seconds", "attempt_nodes"):
            arr[k] = np.concatenate([x[k] for x in z])
        buf = io.BytesIO()
        np.savez_compressed(buf, **arr)
        (out / "modes" / f"{rid}.npz").write_bytes(buf.getvalue())
    return recs


def verify(out, tasks, recs):
    G = A.env()
    robot, p = G["robot"], sk.params(G["cfg"])
    chk, bad = Counter(), []

    def ok(name, cond, detail=""):
        chk[(name, bool(cond))] += 1
        if not cond:
            bad.append(f"{name}: {detail}")
    ids = [r["trial_id"] for r in recs]
    ok("one record per predeclared query", sorted(ids) == sorted(t["trial_id"] for t in tasks), f"{len(ids)} vs {len(tasks)}")
    ok("no torn temp files", not list(out.rglob("*.tmp")))
    used = set()
    for r in recs:
        ok("schema", r["schema_version"] == SCHEMA)
        ok("no UNRESOLVED label", "UNRESOLVED" not in json.dumps([r.get("label"), r.get("negative_kind")]))
        if not r["attempted"]:
            ok("not attempted -> null", r["label"] is None and (r.get("excluded") or r.get("engineering_abort")) and not r["witness_refs"], r["trial_id"])
            if r.get("excluded") == "DEST_OCCUPIED":
                ok("excluded cell really occupied", r["destination_cell"] in {o.cell for o in scene_of(r).objects})
            continue
        ok("attempted -> binary", r["label"] in (0, 1), r["trial_id"])
        ok("1 <-> witness", (r["label"] == 1) == bool(r["witness_refs"]) == (r["mode_summary"]["verified_success"] > 0), r["trial_id"])
        if r["label"] == 0:
            ok("0 has provenance", r["negative_kind"] and r["failure_phase"] and isinstance(r["is_certificate"], bool), r["trial_id"])
        m, z = r["mode_summary"], np.load(out / "modes" / f"{r['source_root_id']}.npz")
        k = list(z["trial_id"]).index(r["trial_id"])
        cat = z["pair_category"][k]
        ok("counts reconcile", m["total"] == 2304 == m["geometrically_rejected"] + m["not_motion_tested"] + m["motion_attempted"]
           and m["motion_attempted"] == m["verified_success"] + m["budget_failed"], r["trial_id"])
        ok("counts match arrays", [m["geometrically_rejected"], m["screen_uncertain_not_tested"], m["valid_not_motion_tested"], m["budget_failed"],
                                   m["verified_success"]] == [int((cat == c).sum()) for c in range(5)], r["trial_id"])
        ok("attempted_mask exact", np.array_equal(z["attempted_mask"][k], cat >= 3) and np.array_equal(z["pair_outcome"][k] >= 0, cat >= 3))
        s = scene_of(r)
        ok("scene hash", scene_hash(s) == r["hypothetical_scene_hash"], r["trial_id"])
        if r["prefix_key"]:
            ok("hypothetical differs from root", r["parent_reachability"] == "HYPOTHETICAL"
               and scene_hash(scene_from_dict(load_root(r["source_root_id"])["scene"])) != r["hypothetical_scene_hash"])
        dxy = np.asarray(s.lattice.center(r["destination_cell"]))
        for w in r["witness_refs"]:
            used.add(w)
            f = np.load(out / "trajectories" / f"{w}.npz")
            core = {x: f[x] for x in ("q", "gap", "phase", "attached", "held_local", "place_xy")}
            h = hashlib.sha256(f"{r['hypothetical_scene_hash']}|{r['object_id']}".encode())
            for x in sorted(core):
                h.update(f"{x}|{core[x].dtype.str}|{core[x].shape}".encode() + core[x].tobytes())
            ok("witness content id", h.hexdigest()[:24] == w, w)
            ok("witness finite", all(np.isfinite(f[x]).all() for x in ("q", "gap", "held_local", "place_xy", "object_xy")), w)
            ph = [str(x) for x in f["phase"]]
            ok("witness phases", ph[0] == "close_attach" and ph[-1] == "retreat" and "carry" in ph and "release" in ph, w)
            v = mo.certify_trajectory(robot, s, r["object_id"], core, p["delta"], p["eps"], p["evals"])
            ok("witness re-certifies in its scene", all(x.status == ge.CLEAR for x in v), w)
            ok("witness ends at destination", np.linalg.norm(f["place_xy"] - dxy) <= p["release_tol"]
               and np.allclose(f["object_xy"][-1], f["place_xy"]), w)
    files = sorted(out.glob("trajectories/*.npz"))
    ok("no orphan or missing witness", {x.stem for x in files} == used, f"{len(files)} files, {len(used)} referenced")
    st = subprocess.run(["git", "status", "--porcelain", "--", str(DATA)], capture_output=True, text=True, cwd=DATA).stdout
    ok("original dataset folder unchanged (git)", st.strip() == "", st[:300])
    for rid in {r["source_root_id"] for r in recs}:
        blob = subprocess.run(["git", "rev-parse", f"HEAD:./raw/roots/{rid}.json"], capture_output=True, text=True, cwd=DATA).stdout.strip()
        now = subprocess.run(["git", "hash-object", f"raw/roots/{rid}.json"], capture_output=True, text=True, cwd=DATA).stdout.strip()
        ok("source root file equals its git blob", blob == now, rid)
    return {"passed": sum(n for (k, v), n in chk.items() if v), "failed": sum(n for (k, v), n in chk.items() if not v),
            "checks": sorted({k for k, _ in chk}), "failures": bad}


COL = {1: "#2ca02c", 0: "#d62728"}


def placements(out, recs):
    G = A.env()
    robot = G["robot"]
    paths = []
    for rid in sorted({r["source_root_id"] for r in recs}):
        rs = [r for r in recs if r["source_root_id"] == rid]
        panels = sorted({(json.dumps(r["prefix_key"]), r["object_id"]) for r in rs})
        fig, axs = plt.subplots(1, len(panels), figsize=(max(5.2 * len(panels), 11), 6.4), squeeze=False)
        for ax, (pk, oid) in zip(axs[0], panels):
            sub = [r for r in rs if json.dumps(r["prefix_key"]) == pk and r["object_id"] == oid]
            s = scene_of(sub[0])
            viz.draw_support(ax, s, cell_ids=True)
            occ = {o.cell for o in s.objects}
            for c in range(64):
                rr, cc = divmod(c, 8)
                if c in occ or c not in s.legal_cells:
                    ax.add_patch(Rectangle((cc * .08, rr * .08), .08, .08, fc="none", ec="#888", hatch="////", lw=0, zorder=1))
            for r in sub:
                rr, cc = divmod(r["destination_cell"], 8)
                fc = COL.get(r["label"], "#bbbbbb")
                ax.add_patch(Rectangle((cc * .08 + .004, rr * .08 + .004), .072, .072, fc=fc, alpha=.55, ec=fc, lw=1.5, zorder=2))
            viz.draw_objects(ax, s)
            o = s.get(oid)
            ax.add_patch(Circle(o.xy_m, o.radius_m + .008, fc="none", ec="#1f3a93", lw=2.2, zorder=6))
            gc = load_root(rid)["goal_cell"]
            ax.add_patch(Rectangle(((gc % 8) * .08, (gc // 8) * .08), .08, .08, fc="none", ec="#ff7f0e", lw=2, ls="--", zorder=2))
            viz.draw_robot(ax, robot, np.asarray(s.robot.q_rad), robot.open_gap, alpha=.35, zorder=4)
            ctx = "root scene" if pk == "[]" else f"hypothetical after {json.loads(pk)}"
            ax.set(xlim=(-.42, .7), ylim=(-.08, .92), aspect="equal", xticks=[], yticks=[])
            ax.set_title(f"{oid} ({ctx})\n" + ", ".join(f"{r['destination_cell']}:{'-' if r['label'] is None else r['label']}" for r in sub), fontsize=8, loc="left")
        handles = [Rectangle((0, 0), 1, 1, fc=COL[1], alpha=.55), Rectangle((0, 0), 1, 1, fc=COL[0], alpha=.55),
                   Rectangle((0, 0), 1, 1, fc=viz.COLORS["support"], ec=viz.COLORS["cell_edge"]), Rectangle((0, 0), 1, 1, fc="none", hatch="////", ec="#888"),
                   Rectangle((0, 0), 1, 1, fc="none", ec="#ff7f0e", ls="--")]
        fig.legend(handles, ["attempted, label 1 (verified witness)", "attempted, label 0 (budget, no witness)", "legal free cell, not attempted",
                             "statically ineligible (occupied)", "target goal cell"], loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.suptitle(f"{rid}: placement queries for the ringed object (arm drawn faintly at its start configuration)", fontsize=10)
        fig.tight_layout(rect=(0, .09, 1, .95))
        path = out / "visuals" / f"{rid}_placements.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def witness_gif(out, rec, wid, n=40):
    """Frames from the saved trajectory file: the full arm bodies (geometry.bodies), fingers, held disc; s = normalised path length
    (the 2D teacher's witnesses are untimed joint-space waypoints)."""
    G = A.env()
    robot = G["robot"]
    f = np.load(out / "trajectories" / f"{wid}.npz")
    core = {x: f[x] for x in ("q", "gap", "phase", "attached", "held_local", "place_xy")}
    s = scene_of(rec)
    rest = s.get(rec["object_id"]).xy_m
    others = [o for o in s.objects if o.id != rec["object_id"]]
    frames, strip = [], []
    smp = mo.samples(core, n)
    for k, (q, gap, phase) in enumerate(smp):
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        viz.draw_support(ax, s, cell_ids=False)
        for o in others:
            ax.add_patch(Circle(o.xy_m, o.radius_m, fc=viz.COLORS["target" if o.kind == "target" else "obstacle"], ec="k", lw=.5, zorder=3))
        held = (rec["object_id"], s.get(rec["object_id"]).radius_m, tuple(core["held_local"])) if mo.CONTEXT[phase][0] == "held" else None
        if held is None:
            xy = mo.target_xy(robot, core, phase, q, rest)
            ax.add_patch(Circle(xy, s.get(rec["object_id"]).radius_m, fc="#1f3a93", ec="k", zorder=3))
        viz.draw_robot(ax, robot, q, gap, held=held)
        if held is not None:                              # the palette draws a held disc in the target's colour: repaint it as the mover
            ax.add_patch(Circle(ge.tool_point(robot, q, held[2]), held[1], fc="#1f3a93", ec="k", zorder=9))
        rr, cc = divmod(rec["destination_cell"], 8)
        ax.add_patch(Rectangle((cc * .08, rr * .08), .08, .08, fc="none", ec="#2ca02c", lw=2, zorder=2))
        ax.set(xlim=(-.42, .7), ylim=(-.08, .92), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"{rec['object_id']} -> cell {rec['destination_cell']}   phase: {phase}\n"
                     f"s = {k / (n - 1):.2f} of the joint path (untimed waypoints)   witness {wid[:10]}", fontsize=8, loc="left")
        fig.tight_layout()
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        frames.append(img)
        if k in np.linspace(0, n - 1, 6).astype(int):
            strip.append(img)
        plt.close(fig)
    gif = out / "visuals" / f"{rec['source_root_id']}_witness.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:] + [frames[-1]] * 8, duration=120, loop=0)
    w, h = strip[0].size
    sheet = Image.new("RGB", (w * len(strip), h), "white")
    for i, im in enumerate(strip):
        sheet.paste(im, (i * w, 0))
    sheet.save(out / "visuals" / f"{rec['source_root_id']}_witness_filmstrip.png")
    return gif


def main(out):
    out = Path(out).resolve()
    tasks = json.loads((out / "roster_predeclared.json").read_text())
    ok_roster = all(t in roster() for t in tasks)
    recs = assemble(out, tasks)
    ver = verify(out, tasks, recs)
    ver["roster_matches_code"] = ok_roster
    figs = placements(out, recs)
    new = [(r, a["witness"]) for r in recs if r["attempted"] for a in r["attempts"] if a.get("witness") and a["method"] != "REUSED_ARCHIVED_WITNESS_RECERTIFIED"]
    pick = next(((r, w) for r, w in new if r["prefix_key"]), new[0] if new else None)
    gif = witness_gif(out, *pick) if pick else None
    att = [r for r in recs if r["attempted"]]
    summ = {"queries_predeclared": len(tasks), "attempted": len(att), "label_1": sum(r["label"] == 1 for r in att),
            "label_0": sum(r["label"] == 0 for r in att), "not_attempted": len(recs) - len(att),
            "not_attempted_reasons": dict(Counter(r.get("excluded") or r.get("engineering_abort") for r in recs if not r["attempted"])),
            "negative_kinds": dict(Counter(r["negative_kind"] for r in att if r["label"] == 0)),
            "failure_phases": dict(Counter(r["failure_phase"] for r in att if r["label"] == 0)),
            "certificates": sum(bool(r.get("is_certificate")) for r in att),
            "modes": {k: sum(r["mode_summary"][k] for r in att) for k in att[0]["mode_summary"]} if att else {},
            "witnesses": {"files": len(list(out.glob("trajectories/*.npz"))),
                          "new_verified": sum(a.get("witness") is not None and a["method"] != "REUSED_ARCHIVED_WITNESS_RECERTIFIED" for r in att for a in r["attempts"]),
                          "reused_archived": sum(a["method"] == "REUSED_ARCHIVED_WITNESS_RECERTIFIED" for r in att for a in r["attempts"]),
                          "reuse_checked": [r["reuse_check"] for r in att if r.get("reuse_check")]},
            "query_wall_cut": sum(r["query_wall_cut"] for r in att), "pair_wall_clock_cut": sum(a.get("wall_clock_cut", False) for r in att for a in r["attempts"]),
            "query_wall_s_max": max(r["wall_s"] for r in att), "query_cpu_s_sum": round(sum(r["cpu_s"] for r in att), 1),
            "worker_peak_rss_mb_max": max(r["worker_peak_rss_mb"] for r in att), "verification": ver,
            "visuals": [str(x.relative_to(out)) for x in figs] + ([str(gif.relative_to(out))] if gif else [])}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    files = sorted(x for x in out.rglob("*") if x.is_file() and x.name != "manifest.json")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=DATA).stdout.strip()
    man = {"schema_version": SCHEMA, "source_commit": commit, "protocol": json.loads((out / "parts" / f"{att[0]['trial_id']}.json").read_text())["placement_protocol_id"] if att else None,
           "roster_sha256": sha_file(out / "roster_predeclared.json"), "split_policy": "none exists in the 2D dataset; recorded as null, not regenerated",
           "mode_arrays": {k: "leading axis = attempted query (trial_id); pick/place axis = skill.directions order (phi = 2 pi k/24, elbow +1/-1)" for k in MODE_KEYS},
           "pair_category_codes": {0: "geometrically rejected (definite screen)", 1: "screen uncertain, not tested", 2: "valid, not motion-tested",
                                   3: "motion attempted, no witness", 4: "verified witness"},
           "files": {str(x.relative_to(out)): {"sha256": sha_file(x), "bytes": x.stat().st_size} for x in files}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    return summ


if __name__ == "__main__":
    s = main(sys.argv[1])
    print(json.dumps({k: v for k, v in s.items() if k != "verification"} | {"verification": {k: v for k, v in s["verification"].items() if k != "checks"}}, indent=1))
