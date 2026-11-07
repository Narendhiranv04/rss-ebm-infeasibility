"""Audit, fingerprint, compare and draw placement smoke runs (placement_smoke.py), from the saved files only.

    python placement_smoke_check.py audit <run>                 # records/, modes/, summary.json, verification.json, fingerprint.json
    python placement_smoke_check.py compare <v2dir> <run>...    # comparisons.json: per-query canonical digests across runs
    python placement_smoke_check.py visuals <run> <outdir>      # placement maps, mode-category matrix, witness GIF + filmstrip
    python placement_smoke_check.py integrity <v2dir>           # original dataset and the failed run against their recorded hashes

Canonical fingerprint of a query = sha256 over its record without `volatile` (wall time, CPU, RSS), plus sha256 of every mode array
(dtype, shape, bytes; not the zip container) and of every referenced witness's arrays. Run order, worker count, PIDs, timing and
temperature never enter it.
"""
import hashlib, io, json, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import placement_smoke as P  # noqa: E402  (sets sys.path and the teacher folder)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
import viz  # noqa: E402

MODE_KEYS = ("held_stage", "held_definite", "pick_access_stage", "place_stage", "mode_phi_rad", "mode_branch", "pair_category",
             "attempted_mask", "pair_outcome", "attempt_pick_place", "attempt_rrt_attempts", "attempt_evals")
CORE = ("q", "gap", "phase", "attached", "held_local", "place_xy")


def arr_hash(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def scene_of(rec):
    s = P.scene_from_dict(P.load_root(rec["source_root_id"])["scene"])
    for o, c in rec["prefix_key"]:
        s = P.substitute_pose(s, o, c)
    return s


def records(run):
    tasks = json.loads((run.parent / "roster.json").read_text())
    got = {json.loads(f.read_text())["trial_id"]: json.loads(f.read_text()) for f in run.glob("parts/*.json")}
    return tasks, [got[t["trial_id"]] for t in tasks if t["trial_id"] in got]


def digest(run, rec):
    canon = {k: v for k, v in rec.items() if k != "volatile"}
    extra = {}
    if (run / "parts" / f"{rec['trial_id']}.npz").exists():
        z = np.load(run / "parts" / f"{rec['trial_id']}.npz")
        extra["modes"] = {k: arr_hash(z[k]) for k in MODE_KEYS}
    extra["witnesses"] = [[w, {k: arr_hash(np.load(run / "trajectories" / f"{w}.npz")[k]) for k in CORE}] for w in rec["witness_refs"]]
    return hashlib.sha256(json.dumps([canon, extra], sort_keys=True).encode()).hexdigest(), canon, extra


def audit(run):
    run = Path(run).resolve()
    tasks, recs = records(run)
    G = P.A.env()
    robot, p = G["robot"], P.sk.params(G["cfg"])
    chk, bad = Counter(), []

    def ok(name, cond, detail=""):
        chk[(name, bool(cond))] += 1
        if not cond:
            bad.append(f"{name}: {detail}")
    ok("frozen roster equals the roster code", tasks == P.roster())
    ok("one record per roster query", len(recs) == len(tasks), f"{len(recs)}/{len(tasks)}")
    ok("no torn temp files", not list(run.rglob("*.tmp")))
    refs = set()
    for r in recs:
        ok("schema", r["schema_version"] == P.SCHEMA, r["trial_id"])
        ok("no UNRESOLVED class", "UNRESOLVED" not in json.dumps([r.get("label"), r.get("negative_kind")]))
        if not r["eligible"]:
            ok("ineligible -> not attempted, label null", not r["attempted"] and r["label"] is None and r["excluded"] and not r["witness_refs"], r["trial_id"])
            ok("ineligible destination really occupied", r["excluded"] != "DEST_OCCUPIED" or r["destination_cell"] in {o.cell for o in scene_of(r).objects})
            continue
        if not r["completed"]:
            ok("interrupted -> label null with reason", r["label"] is None and r["safety_interruption"] and not r["witness_refs"], r["trial_id"])
            continue
        m = r["mode_summary"]
        ok("completed -> binary label", r["label"] in (0, 1) and r["safety_interruption"] is None, r["trial_id"])
        ok("1 <-> witness", (r["label"] == 1) == bool(r["witness_refs"]) == (m["n_mode_witnesses"] > 0), r["trial_id"])
        if r["label"] == 0:
            ok("0 carries kind, scope, phase", r["negative_kind"] and r["negative_scope"] and r["failure_phase"] and isinstance(r["is_certificate"], bool))
        ok("partition reconciles", m["total"] == 2304 == m["hard_screened"] + m["motion_attempted"] + m["reused_reverified"] + m["not_motion_attempted"]
           and m["motion_attempted"] == m["attempted_success"] + m["attempted_failure"]
           and m["not_motion_attempted"] == m["screen_inconclusive"] + m["valid_not_planned"], r["trial_id"])
        z = np.load(run / "parts" / f"{r['trial_id']}.npz")
        cat = z["pair_category"]
        ok("counts equal the mode arrays", [m["hard_screened"], m["screen_inconclusive"], m["valid_not_planned"], m["attempted_failure"],
                                            m["attempted_success"], m["reused_reverified"]] == [int((cat == c).sum()) for c in range(6)], r["trial_id"])
        ok("attempted_mask and outcomes exact", np.array_equal(z["attempted_mask"], (cat == 3) | (cat == 4))
           and np.array_equal(z["pair_outcome"] == 1, (cat == 4) | (cat == 5)) and np.array_equal(z["pair_outcome"] == 0, cat == 3), r["trial_id"])
        ok("held / pick-access counts match arrays", r["held_modes_valid"] == int((z["held_stage"] == "OK").sum())
           and r["pick_access_modes_ok"] == int((z["pick_access_stage"] == "OK").sum()))
        ok("planned pairs within the schedule", m["motion_attempted"] <= P.WORK["pairs_per_query"]   # limits are checked before each
           and all(a["work"]["attempts"] <= P.WORK["rrt_attempts"] + 1 and a["work"]["nodes"] <= P.WORK["rrt_nodes"] + 1  # RRT-Connect step, whose
                   for a in r["attempts"] if a["work"]))                                                                  # second extension may add one
        s = scene_of(r)
        ok("scene hash recomputes", P.scene_hash(s) == r["hypothetical_scene_hash"], r["trial_id"])
        root_hash = P.scene_hash(P.scene_from_dict(P.load_root(r["source_root_id"])["scene"]))
        ok("prefix provenance", (r["parent_reachability"] == "HYPOTHETICAL") == bool(r["prefix_key"]) and (root_hash != r["hypothetical_scene_hash"]) == bool(r["prefix_key"]))
        dxy = np.asarray(s.lattice.center(r["destination_cell"]))
        listed = [a["witness"] for a in r["attempts"] if a.get("witness")]
        ok("witness_refs = attempt witnesses in order", r["witness_refs"] == list(dict.fromkeys(listed)), r["trial_id"])
        for w in r["witness_refs"]:
            refs.add(w)
            f = np.load(run / "trajectories" / f"{w}.npz")
            core = {k: f[k] for k in CORE}
            h = hashlib.sha256(str(f["identity"]).encode())
            for k in sorted(core):
                h.update(f"{k}|{core[k].dtype.str}|{core[k].shape}".encode() + core[k].tobytes())
            ok("witness content id", h.hexdigest()[:24] == w and str(f["identity"]).startswith(r["placement_protocol_id"]), w)
            ok("witness finite", all(np.isfinite(f[k]).all() for k in ("q", "gap", "held_local", "place_xy", "object_xy")), w)
            good, ver = P.verify_witness(robot, s, r["object_id"], core, dxy, p)
            ok("witness re-verifies phase by phase in its scene", good, f"{w} {ver}")
            ok("witness starts at the source and ends at the destination", np.allclose(f["object_xy"][0], s.get(r["object_id"]).xy_m, atol=1e-9)
               and np.allclose(f["object_xy"][-1], f["place_xy"]), w)
            src = json.loads(str(f["provenance"]))["evidence_source"]
            ok("witness provenance matches its attempt", src in {a["evidence_source"] for a in r["attempts"] if a.get("witness") == w}, w)
    files = {x.stem for x in run.glob("trajectories/*.npz")}
    ok("no orphan or missing witness file", files == refs, f"{len(files)} files / {len(refs)} referenced")
    ver = {"passed": sum(n for (k, v), n in chk.items() if v), "failed": sum(n for (k, v), n in chk.items() if not v), "failures": bad,
           "checks": sorted({k for k, _ in chk})}
    comp = [r for r in recs if r["eligible"] and r["completed"]]
    att = [a for r in comp for a in r["attempts"]]
    summ = {"queries": len(tasks), "records": len(recs), "ineligible": sum(not r["eligible"] for r in recs),
            "completed": len(comp), "interrupted": sum(r["eligible"] and not r["completed"] for r in recs),
            "label_1": sum(r["label"] == 1 for r in comp), "label_0": sum(r["label"] == 0 for r in comp),
            "negative_kinds": dict(Counter(r["negative_kind"] for r in comp if r["label"] == 0)),
            "failure_phases": dict(Counter(r["failure_phase"] for r in comp if r["label"] == 0)), "certificates": sum(r["is_certificate"] for r in comp),
            "modes": {k: sum(r["mode_summary"][k] for r in comp) for k in comp[0]["mode_summary"]} if comp else {},
            "held_modes_valid": sum(r["held_modes_valid"] for r in comp), "pick_access_modes_ok": sum(r["pick_access_modes_ok"] for r in comp),
            "pairs_planned": sum(a["evidence_source"] == "NEW_PLANNED" for a in att), "planned_methods": dict(Counter(a["method"].split(":")[0] for a in att)),
            "rrt_stops": dict(Counter(a["work"]["stop"] for a in att if a["work"] and a["work"]["stop"])),
            "witness_files": len(files), "new_witnesses": sum(bool(a.get("witness")) and a["evidence_source"] == "NEW_PLANNED" for a in att),
            "reused_witnesses": sum(a["evidence_source"] == "REUSED_REVERIFIED" for a in att),
            "work": {k: sum(r["work_budget_consumed"][k] for r in comp) for k in ("rrt_attempts", "rrt_nodes", "evals")},
            "query_wall_s_max": max((r["volatile"]["wall_s"] for r in comp), default=None), "query_cpu_s_sum": round(sum(r["volatile"]["cpu_s"] for r in comp), 1),
            "worker_peak_rss_mb_max": max((r["volatile"]["worker_peak_rss_mb"] for r in comp), default=None)}
    fp = {r["trial_id"]: digest(run, r)[0] for r in recs}
    fp_all = hashlib.sha256(json.dumps(sorted(fp.items())).encode()).hexdigest()
    for d in ("records", "modes"):
        (run / d).mkdir(exist_ok=True)
    by = defaultdict(list)
    for r in recs:
        by[r["source_root_id"]].append(r)
    for rid, rs in by.items():                           # per-root sidecar files assembled from the per-query commits
        (run / "records" / f"{rid}.json").write_text(json.dumps({"schema_version": P.SCHEMA, "source_root_id": rid, "split": None, "trials": rs}, indent=1, sort_keys=True))
        z = [(r["trial_id"], np.load(run / "parts" / f"{r['trial_id']}.npz")) for r in rs if (run / "parts" / f"{r['trial_id']}.npz").exists()]
        if z:
            arr = {k: np.stack([x[k] for _, x in z]) for k in MODE_KEYS[:9]}
            arr["trial_id"] = np.array([t for t, _ in z])
            arr["attempt_trial"] = np.concatenate([np.full(len(x["attempt_evals"]), i, np.int16) for i, (_, x) in enumerate(z)])
            for k in MODE_KEYS[9:]:
                arr[k] = np.concatenate([x[k] for _, x in z])
            buf = io.BytesIO()
            np.savez_compressed(buf, **arr)
            (run / "modes" / f"{rid}.npz").write_bytes(buf.getvalue())
    for name, obj in (("summary.json", summ), ("verification.json", ver), ("fingerprint.json", {"run_fingerprint": fp_all, "queries": fp})):
        (run / name).write_text(json.dumps(obj, indent=1, sort_keys=True))
    return summ, ver, fp_all


def compare(v2, runs):
    """Per-query canonical digests across runs; for a mismatch, the differing fields, attempts and first differing waypoint."""
    v2 = Path(v2).resolve()
    R = {n: {r["trial_id"]: r for r in records(v2 / n)[1]} for n in runs}
    tids = [t["trial_id"] for t in json.loads((v2 / "roster.json").read_text())]
    out, mism = {"runs": runs, "queries": {}}, 0
    for t in tids:
        have = {n: R[n][t] for n in runs if t in R[n] and (R[n][t]["completed"] or not R[n][t]["eligible"])}
        ds = {n: digest(v2 / n, r) for n, r in have.items()}
        uniq = sorted({d[0] for d in ds.values()})
        row = {"compared_runs": sorted(have), "digests": {n: d[0][:16] for n, d in ds.items()}, "match": len(uniq) == 1,
               "missing_or_interrupted": [n for n in runs if n not in have]}
        if len(uniq) > 1:
            mism += 1
            ref, refd = next(iter(ds.items()))
            diffs = {}
            for n, d in ds.items():
                if d[0] == refd[0]:
                    continue
                keys = sorted(k for k in set(d[1]) | set(refd[1]) if d[1].get(k) != refd[1].get(k))
                att = next((i for i, (a, b) in enumerate(zip(d[1].get("attempts", []), refd[1].get("attempts", []))) if a != b), None)
                wp = None
                for (w1, _), (w0, _) in zip(d[2]["witnesses"], refd[2]["witnesses"]):
                    if w1 != w0:
                        q1, q0 = np.load(v2 / n / "trajectories" / f"{w1}.npz")["q"], np.load(v2 / ref / "trajectories" / f"{w0}.npz")["q"]
                        wp = next((i for i in range(min(len(q0), len(q1))) if not np.array_equal(q0[i], q1[i])), min(len(q0), len(q1)))
                        break
                diffs[n] = {"vs": ref, "fields": keys, "first_attempt_diff": att,
                            "attempt_work": [(d[1]["attempts"][att] if att is not None else None), (refd[1]["attempts"][att] if att is not None else None)],
                            "first_differing_waypoint": wp, "mode_arrays_differ": [k for k in d[2].get("modes", {}) if d[2]["modes"][k] != refd[2].get("modes", {}).get(k)]}
            row["diff"] = diffs
        out["queries"][t] = row
    out["mismatches"] = mism
    out["queries_compared_in_all_runs"] = sum(not v["missing_or_interrupted"] for v in out["queries"].values())
    out["run_fingerprints"] = {n: json.loads((v2 / n / "fingerprint.json").read_text())["run_fingerprint"] for n in runs}
    (v2 / "comparisons.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


def integrity(v2):
    v2 = Path(v2).resolve()
    res = {}
    for name, base in (("source_before.sha", P.DATA.parent), ("failed_run_before.sha", v2.parent / "smoke_20261106")):
        lines = [l.split(maxsplit=1) for l in (v2 / "audit" / name).read_text().splitlines()]
        changed = [f for h, f in lines if P.sha_file(base / f.strip()) != h]
        res[name] = {"files": len(lines), "changed": changed}
    st = subprocess.run(["git", "status", "--porcelain", "--ignored", "--", str(P.DATA)], capture_output=True, text=True, cwd=P.DATA).stdout
    res["git_status_original_dataset"] = st.strip() or "clean"
    (v2 / "integrity.json").write_text(json.dumps(res, indent=1))
    return res


COL = {1: "#2ca02c", 0: "#d62728"}


def visuals(run, outdir):
    """Scene-backed maps per root (labels per destination), the 48 x 48 pair categories of the reference query, a witness GIF."""
    run, outdir = Path(run).resolve(), Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    _, recs = records(run)
    robot = P.A.env()["robot"]
    made = []
    for rid in sorted({r["source_root_id"] for r in recs}):
        rs = [r for r in recs if r["source_root_id"] == rid]
        panels = sorted({(json.dumps(r["prefix_key"]), r["object_id"]) for r in rs})
        fig, axs = plt.subplots(1, len(panels), figsize=(max(5.2 * len(panels), 11), 6.6), squeeze=False)
        for ax, (pk, oid) in zip(axs[0], panels):
            sub = [r for r in rs if json.dumps(r["prefix_key"]) == pk and r["object_id"] == oid]
            s = scene_of(sub[0])
            viz.draw_support(ax, s, cell_ids=True)
            occ = {o.cell for o in s.objects}
            for c in occ:
                ax.add_patch(Rectangle(((c % 8) * .08, (c // 8) * .08), .08, .08, fc="none", ec="#888", hatch="////", lw=0, zorder=1))
            for r in sub:
                cc, rr = r["destination_cell"] % 8, r["destination_cell"] // 8
                screen_only = r.get("label") == 0 and not r["mode_summary"]["motion_attempted"]
                fc = "#bbbbbb" if r.get("label") is None else COL[r["label"]]
                ax.add_patch(Rectangle((cc * .08 + .004, rr * .08 + .004), .072, .072, fc=fc, alpha=.55, ec=fc, lw=1.5, hatch="xx" if screen_only else None, zorder=2))
            viz.draw_objects(ax, s)
            o = s.get(oid)
            ax.add_patch(Circle(o.xy_m, o.radius_m + .008, fc="none", ec="#1f3a93", lw=2.2, zorder=6))
            gc = P.load_root(rid)["goal_cell"]
            ax.add_patch(Rectangle(((gc % 8) * .08, (gc // 8) * .08), .08, .08, fc="none", ec="#ff7f0e", lw=2, ls="--", zorder=2))
            viz.draw_robot(ax, robot, np.asarray(s.robot.q_rad), robot.open_gap, alpha=.3, zorder=4)
            ax.set(xlim=(-.42, .7), ylim=(-.08, .92), aspect="equal", xticks=[], yticks=[])
            ctx = "root scene" if pk == "[]" else f"hypothetical object edit {json.loads(pk)}"
            ax.set_title(f"{oid} ({ctx})\n" + ", ".join(f"{r['destination_cell']}:{'-' if r.get('label') is None else r['label']}" for r in sub), fontsize=8, loc="left")
        h = [Rectangle((0, 0), 1, 1, fc=COL[1], alpha=.55), Rectangle((0, 0), 1, 1, fc=COL[0], alpha=.55), Rectangle((0, 0), 1, 1, fc=COL[0], alpha=.55, hatch="xx"),
             Rectangle((0, 0), 1, 1, fc="#bbbbbb", alpha=.55), Rectangle((0, 0), 1, 1, fc="none", hatch="////", ec="#888"), Rectangle((0, 0), 1, 1, fc="none", ec="#ff7f0e", ls="--")]
        fig.legend(h, ["label 1: verified placement witness", "label 0: planned pairs, work budget finished", "label 0: no pair motion-tested (screens only)",
                       "no label (ineligible or interrupted)", "occupied cell (statically ineligible)", "target goal cell"], loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.suptitle(f"{rid}: placement-only queries for the ringed object (already held; arm drawn faintly at the scene's start pose)", fontsize=10)
        fig.tight_layout(rect=(0, .1, 1, .95))
        made.append(outdir / f"{rid}_placements.png")
        fig.savefig(made[-1], dpi=110)
        plt.close(fig)
    ref = next(r for r in recs if r["eligible"] and r.get("completed"))   # the Pass A reference query (roster position 0)
    z = np.load(run / "parts" / f"{ref['trial_id']}.npz")
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    cmap = ListedColormap(["#d9d9d9", "#fdd0a2", "#ffffff", "#d62728", "#2ca02c", "#1f77b4"])
    ax.imshow(z["pair_category"], cmap=cmap, vmin=-.5, vmax=5.5, interpolation="nearest")
    ax.set(xlabel="place mode (24 tool axes x 2 elbows)", ylabel="held mode (24 tool axes x 2 elbows)")
    ax.set_title(f"{ref['source_root_id']} {ref['object_id']} -> cell {ref['destination_cell']}: the 2,304 (held, place) pairs\n"
                 + ", ".join(f"{P.CAT[k].lower()} {int((z['pair_category'] == k).sum())}" for k in list(P.CAT)[:3]) + "\n"
                 + ", ".join(f"{P.CAT[k].lower()} {int((z['pair_category'] == k).sum())}" for k in list(P.CAT)[3:]), fontsize=8, loc="left")
    fig.legend([Rectangle((0, 0), 1, 1, fc=c, ec="#999") for c in cmap.colors], [P.CAT[k] for k in P.CAT], loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    made.append(outdir / f"{ref['trial_id']}_mode_pairs.png")
    fig.savefig(made[-1], dpi=110)
    plt.close(fig)
    pos = [(r, a["witness"]) for r in recs if r.get("label") == 1 for a in r["attempts"] if a.get("witness") and a["evidence_source"] == "NEW_PLANNED"]
    rec, wid = next(((r, w) for r, w in pos if r["prefix_key"]), pos[0])
    made += witness_gif(run, rec, wid, outdir)
    return made


def witness_gif(run, rec, wid, outdir, n=40):
    """Frames drawn from the saved witness: full arm bodies (geometry.bodies), fingers, the held disc, release and retreat. The
    2D teacher's witnesses are untimed joint waypoints, so s is the normalised joint-path parameter, not time."""
    robot = P.A.env()["robot"]
    f = np.load(run / "trajectories" / f"{wid}.npz")
    core = {k: f[k] for k in CORE}
    s = scene_of(rec)
    oid, rad = rec["object_id"], s.get(rec["object_id"]).radius_m
    frames, strip = [], []
    for k, (q, gap, phase) in enumerate(P.mo.samples(core, n)):
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        viz.draw_support(ax, s, cell_ids=False)
        for o in s.objects:
            if o.id != oid:
                ax.add_patch(Circle(o.xy_m, o.radius_m, fc=viz.COLORS["target" if o.kind == "target" else "obstacle"], ec="k", lw=.5, zorder=3))
        held = (oid, rad, tuple(core["held_local"])) if phase == "carry" else None
        viz.draw_robot(ax, robot, q, gap, held=held)
        xy = P.ge.tool_point(robot, q, core["held_local"]) if held else core["place_xy"]
        ax.add_patch(Circle(xy, rad, fc="#1f3a93", ec="k", zorder=9))
        cc, rr = rec["destination_cell"] % 8, rec["destination_cell"] // 8
        ax.add_patch(Rectangle((cc * .08, rr * .08), .08, .08, fc="none", ec="#2ca02c", lw=2, zorder=2))
        ax.set(xlim=(-.42, .7), ylim=(-.08, .92), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"{oid} held from cell {rec['source_cell']} -> cell {rec['destination_cell']}   phase: {phase}\n"
                     f"s = {k / (n - 1):.2f} of the joint path (untimed waypoints), witness {wid[:10]}", fontsize=8, loc="left")
        fig.tight_layout()
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        frames.append(img)
        if k in np.linspace(0, n - 1, 6).astype(int):
            strip.append(img)
        plt.close(fig)
    gif = outdir / f"{rec['trial_id']}_witness.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:] + [frames[-1]] * 8, duration=120, loop=0)
    sheet = Image.new("RGB", (strip[0].size[0] * len(strip), strip[0].size[1]), "white")
    for i, im in enumerate(strip):
        sheet.paste(im, (i * strip[0].size[0], 0))
    sheet.save(outdir / f"{rec['trial_id']}_witness_filmstrip.png")
    return [gif, outdir / f"{rec['trial_id']}_witness_filmstrip.png"]


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "audit":
        s, v, fp = audit(args[0])
        print(json.dumps({"summary": s, "verification": {k: v[k] for k in ("passed", "failed", "failures")}, "run_fingerprint": fp}, indent=1))
    elif cmd == "compare":
        c = compare(args[0], args[1:])
        print(json.dumps({k: c[k] for k in ("mismatches", "queries_compared_in_all_runs", "run_fingerprints")}, indent=1))
    elif cmd == "visuals":
        print("\n".join(str(x) for x in visuals(args[0], args[1])))
    elif cmd == "integrity":
        print(json.dumps(integrity(args[0]), indent=1))
