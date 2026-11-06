"""Audit, compare and budget-analyse launch-readiness runs from saved files only (no planner call; witnesses are re-certified
with the teacher's geometry, which is replay, not planning).

    python launch_check.py audit <run>                         # verification.json, summary.json, fingerprint.json in <run>
    python launch_check.py compare <out.json> <run> <run>...   # canonical per-query digests on the queries the runs share
    python launch_check.py nested <out.json> <run64> <run8>    # independent 8-pair run vs the first-8 accounting of the 64-pair run
    python launch_check.py budget <launch_dir> <run64> <out.json>   # 8/16/32/64 x FULL/FIRST_SUCCESS tables, Wilson CIs, flips, strata
    python launch_check.py interrupt <out.json> <run> <snapshot.sha256> <parts_copy>   # SIGKILL + resume preserved commits
    python launch_check.py integrity <launch_dir>              # original dataset, both smoke stores, smoke code vs the baseline hashes
"""
import hashlib, io, json, math, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement_teacher as T  # noqa: E402
P = T.P
MODE_KEYS = ("held_stage", "held_definite", "pick_access_stage", "place_stage", "pair_category")
BUDGETS = (8, 16, 32, 64)
PROTECTED = ("roles", "intended_blockers", "labels", "verified_K", "observed", "construction", "parking", "verified_class")


def ah(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def load(run, roots=None):
    """[(header, query, modes-or-None, attempts dict)] for every packed root (or the given ones), in plan order within a root."""
    out = []
    for f in sorted(Path(run).glob("records/*.json")) if roots is None else [Path(run) / "records" / f"{r}.json" for r in roots]:
        h = json.loads(f.read_text())
        mf = Path(run) / "modes" / f"{h['root_id']}.npz"
        z = dict(np.load(mf)) if mf.exists() else {}
        qi = list(z.get("query_index", []))
        for k, q in enumerate(h["queries"]):
            m = a = None
            if k in qi:
                i, sel = qi.index(k), z["a_query"] == k
                m, a = {x: z[x][i] for x in MODE_KEYS}, {x: z[x][sel] for x in T.ATT}
            out.append(({x: v for x, v in h.items() if x != "queries"}, q, m, a))
    return out


def digest(run, h, q, m, a):
    canon = {k: v for k, v in q.items() if k != "volatile"}
    extra = {"modes": {k: ah(v) for k, v in (m or {}).items()}, "attempts": {k: ah(v) for k, v in (a or {}).items()},
             "witnesses": [[w, {k: ah(v) for k, v in T.witness(run, h["root_id"], w).items()}] for w in q["witness_refs"]],
             "header": {k: h[k] for k in ("root_id", "split", "budget_hash", "pair_protocol_id", "original_scene_hash", "root_file_sha256")}}
    return hashlib.sha256(json.dumps([canon, extra], sort_keys=True).encode()).hexdigest()


def _audit_roots(args):
    """Per-root checks (run in a process pool): -> (check counter, failures, packed witness ids per root)."""
    run, man, split, roots = args
    rows = load(run, roots)
    G = P.A.env()
    robot, p = G["robot"], P.sk.params(G["cfg"])
    chk, bad, packed = Counter(), [], {}
    B, pol = man["policy"]["pairs_per_query"], man["policy"]

    def ok(name, cond, detail=""):
        chk[(name, bool(cond))] += 1
        if not cond:
            bad.append(f"{name}: {detail}")
    for h, q, m, a in rows:
        tid = q["trial_id"]
        ok("header pins budget / protocol / split", h["budget_hash"] == man["budget_hash"] and h["pair_protocol_id"] == man["pair_protocol_id"]
           and h["split"] == split[h["root_id"]] and P.sha_file(P.DATA / "raw" / "roots" / f"{h['root_id']}.json") == h["root_file_sha256"], tid)
        ok("protected fields only under privileged_teacher", not any(k in q for k in PROTECTED) and set(h["privileged_teacher"]) == {"verified_class"}, tid)
        task = {"root": h["root_id"], "prefix": q["prefix_key"]}
        s = T.scene_for(task)
        ok("scene hash recomputes; prefix provenance", P.scene_hash(s) == q["hypothetical_scene_hash"]
           and (q["prefix_provenance"] == "HYPOTHETICAL") == bool(q["prefix_key"]), tid)
        if not q["eligible"]:
            ok("ineligible -> unattempted, unlabeled, really occupied", not q["attempted"] and q["label"] is None and not q["witness_refs"]
               and (q["excluded"] != "DEST_OCCUPIED" or q["destination_cell"] in {o.cell for o in s.objects}), tid)
            continue
        if not q["completed"]:
            ok("interrupted -> label null with reason", q["label"] is None and q["safety_interruption"], tid)
            continue
        mc, cat = q["mode_counts"], m["pair_category"]
        ok("completed -> binary label, 1 <-> verified pair", q["label"] in (0, 1) and (q["label"] == 1) == (mc["n_mode_witnesses"] > 0) == bool(a["a_verified"].any()), tid)
        if q["label"] == 0:
            ok("0 carries kind, scope, phase", q["negative_kind"] and q["negative_scope"] and q["failure_phase"] and isinstance(q["is_certificate"], bool), tid)
        ok("partition reconciles and equals the arrays", mc["total"] == 2304 == mc["hard_screened"] + mc["motion_attempted"] + mc["reused_reverified"] + mc["not_motion_attempted"]
           and [mc["hard_screened"], mc["screen_inconclusive"], mc["valid_not_planned"], mc["attempted_failure"], mc["attempted_success"], mc["reused_reverified"]]
           == [int((cat == c).sum()) for c in range(6)], tid)
        rk = a["a_rank"][a["a_rank"] >= 0]
        first = next((int(r) for r, v in zip(a["a_rank"], a["a_verified"]) if r >= 0 and v), None)
        n_exp = min(B, mc["valid_pairs"]) if pol["coverage"] == "FULL" or first is None else first + 1
        ok("schedule: ranks 0..n-1, n per coverage rule, outcomes match categories", list(rk) == list(range(len(rk))) and len(rk) == n_exp
           and all(cat[i, j] == (5 if r < 0 else 4 if v else 3) for i, j, r, v in zip(a["a_pick"], a["a_place"], a["a_rank"], a["a_verified"])), tid)
        ok("counted work within limits (+1 extension of the RRT-Connect step)", all(x <= T.PER_PAIR["rrt_attempts"] + 1 for x in a["a_attempts"])
           and all(x <= T.PER_PAIR["rrt_nodes"] + 1 for x in a["a_nodes"]), tid)
        st = [w for w, s_ in zip(a["a_witness"], a["a_stored"]) if s_]
        ok("witness_refs = stored verified pairs, within the store cap", q["witness_refs"] == list(dict.fromkeys(st))
           and sum(r >= 0 for r, s_ in zip(a["a_rank"], a["a_stored"]) if s_) <= pol["witness_store_max"], tid)
        ident = f"{h['pair_protocol_id']}|{s.robot_profile_hash}|{q['hypothetical_scene_hash']}|{q['object_id']}"
        dxy = np.asarray(s.lattice.center(q["destination_cell"]))
        for w in q["witness_refs"]:
            core = T.witness(run, h["root_id"], w)
            packed.setdefault(h["root_id"], set()).add(w)
            ok("witness content id recomputes", T.wid_of(ident, core) == w, w)
            good, ver = P.verify_witness(robot, s, q["object_id"], core, dxy, p)
            ok("witness re-certifies phase by phase (carry, release, retreat)", good and [str(x) for x in dict.fromkeys(core["phase"])] == list(T.PHASES), f"{w} {ver}")
            ok("witness starts attached at the source", np.allclose(P.ge.tool_point(robot, core["q"][0], core["held_local"]), s.get(q["object_id"]).xy_m, atol=1e-6), w)
    return chk, bad, packed


def audit(run, procs=16):
    run = Path(run).resolve()
    man, rows = json.loads((run / "run_manifest.json").read_text()), load(run)
    split = {r["root_id"]: r["split"] for r in json.loads((_launch(run) / "split_manifest.json").read_text())["roots"]}
    chk, bad, packed = Counter(), [], {}

    def ok(name, cond, detail=""):
        chk[(name, bool(cond))] += 1
        if not cond:
            bad.append(f"{name}: {detail}")
    ok("one record per planned query", len(rows) == man["queries"], f"{len(rows)}/{man['queries']}")
    ok("no open roots, parts or temp files", not list(run.glob("parts/*/*")) and not list(run.rglob("*.tmp")))
    roots = sorted({h["root_id"] for h, _, _, _ in rows})
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs) as pool:
        for c, b_, pk in pool.map(_audit_roots, [(run, man, split, [r]) for r in roots]):
            chk.update(c)
            bad += b_
            packed.update(pk)
    for f in run.glob("witnesses/*.npz"):
        ok("pack holds exactly the referenced witnesses", set(np.load(f)["ids"]) == packed.get(f.stem, set()), f.stem)
    comp = [(h, q, m, a) for h, q, m, a in rows if q["eligible"] and q["completed"]]
    summ = {"queries": len(rows), "static_ineligible": sum(not q["eligible"] for _, q, _, _ in rows), "completed": len(comp),
            "interrupted": sum(q["eligible"] and not q["completed"] for _, q, _, _ in rows), "label_1": sum(q["label"] == 1 for _, q, _, _ in comp),
            "label_0": sum(q["label"] == 0 for _, q, _, _ in comp), "negative_kinds": dict(Counter(q["negative_kind"] for _, q, _, _ in comp if q["label"] == 0)),
            "label_sources": dict(Counter(q["label_source"] for _, q, _, _ in comp if q["label"] == 1)),
            "modes": {k: sum(q["mode_counts"][k] for _, q, _, _ in comp) for k in comp[0][1]["mode_counts"]} if comp else {},
            "pairs_planned": sum(q["work"]["pairs_planned"] for _, q, _, _ in comp), "stored_witnesses": sum(len(v) for v in packed.values()),
            "methods": dict(Counter(T.METHODS[x] for _, _, _, a in comp for x in a["a_method"])),
            "query_cpu_s_sum": round(sum(q["volatile"]["cpu_s"] for _, q, _, _ in comp), 1),
            "bytes": {d: sum(f.stat().st_size for f in run.glob(f"{d}/*")) for d in ("records", "modes", "witnesses")}}
    fp = {q["trial_id"]: digest(run, h, q, m, a) for h, q, m, a in rows}
    out = {"verification": {"passed": sum(n for (_, v), n in chk.items() if v), "failed": sum(n for (_, v), n in chk.items() if not v), "failures": bad[:50],
                            "checks": sorted({k for k, _ in chk})}, "summary": summ,
           "fingerprint": {"run_fingerprint": hashlib.sha256(json.dumps(sorted(fp.items())).encode()).hexdigest(), "queries": fp}}
    for k, v in out.items():
        (run / f"{k}.json").write_text(json.dumps(v, indent=1, sort_keys=True))
    return out


def _launch(run):
    return next(x for x in Path(run).resolve().parents if (x / "split_manifest.json").exists())


def compare(dst, runs):
    fps = {r: json.loads((Path(r) / "fingerprint.json").read_text())["queries"] for r in runs}
    shared = sorted(set.intersection(*(set(v) for v in fps.values())))
    mism = [t for t in shared if len({fps[r][t] for r in runs}) > 1]
    out = {"runs": [str(r) for r in runs], "shared_queries": len(shared), "mismatches": len(mism), "mismatched": mism,
           "per_run_queries": {str(r): len(v) for r, v in fps.items()}}
    Path(dst).write_text(json.dumps(out, indent=1))
    return out


def virtual(q, a, B, cov):
    """The query's outcome under budget B and coverage cov, read from a FULL run with >= B pairs (nested schedules)."""
    reused = bool(((a["a_rank"] < 0) & a["a_verified"]).any())
    keep = (a["a_rank"] >= 0) & (a["a_rank"] < B)
    if cov == "FIRST_SUCCESS":
        first = next((int(r) for r, v in zip(a["a_rank"], a["a_verified"]) if 0 <= r < B and v), None)
        keep &= (a["a_rank"] < (first + 1 if first is not None else B)) & (not reused)
    ver = int((a["a_verified"] & keep).sum())
    cpu = q["volatile"]["screen_cpu_s"] + sum(c for c, k in zip(q["volatile"]["attempt_cpu_s"], keep[a["a_rank"] >= 0]) if k)
    lab = int(ver > 0 or reused)
    mc = q["mode_counts"]
    kind = None if lab else ("CERTIFIED_SCREEN" if mc["hard_screened"] == mc["total"] else "NO_PLAN_FOUND" if mc["valid_pairs"] == 0
                             else "WORK_BUDGET_EXHAUSTED" if mc["valid_pairs"] > B else "SCHEDULE_EXHAUSTED_NO_WITNESS")
    return {"label": lab, "kind": kind, "planned": int(keep.sum()), "verified": ver + reused, "cpu_s": cpu, "valid": mc["valid_pairs"],
            "evals": int(sum(x for x, k in zip(a["a_pre_evals"] + a["a_search_evals"] + a["a_shortcut_evals"], keep) if k))}


def nested(dst, run64, run8):
    A = {q["trial_id"]: (q, a) for _, q, _, a in load(run64) if q["eligible"] and q["completed"]}
    out, bad = Counter(), []
    for _, q, _, a in load(run8):
        if not (q["eligible"] and q["completed"]):
            continue
        q64, a64 = A[q["trial_id"]]
        sel = a64["a_rank"] < 8
        same = all(np.array_equal(a[k], a64[k][sel]) for k in T.ATT if k != "a_stored") and q["label"] == virtual(q64, a64, 8, "FULL")["label"]
        same &= q["negative_kind"] == virtual(q64, a64, 8, "FULL")["kind"] and q["mode_counts"]["valid_pairs"] == q64["mode_counts"]["valid_pairs"]
        out["equal" if same else "different"] += 1
        if not same:
            bad.append(q["trial_id"])
    res = {"compared": sum(out.values()), **out, "different_ids": bad}
    Path(dst).write_text(json.dumps(res, indent=1))
    return res


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    c, h = (k + z * z / 2) / (n + z * z), z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [round(c - h, 4), round(c + h, 4)]


def budget(L, run64, dst):
    L, rows = Path(L), [(h, q, a) for h, q, _, a in load(run64) if q["eligible"] and q["completed"]]
    grp = {r["root_id"]: r["root_group_id"] for r in json.loads((L / "split_manifest.json").read_text())["roots"]}
    V = {(b, c): [virtual(q, a, b, c) for _, q, a in rows] for b in BUDGETS for c in ("FULL", "FIRST_SUCCESS")}
    ref = [v["label"] for v in V[(64, "FULL")]]
    tab = {}
    for b in BUDGETS:
        f, s = V[(b, "FULL")], V[(b, "FIRST_SUCCESS")]
        y = [v["label"] for v in f]
        miss = sum(r == 1 and x == 0 for r, x in zip(ref, y))
        flips = {f"{i}->{j}": sum(x == i and r == j for r, x in zip(ref, y)) for i in (0, 1) for j in (0, 1)}
        strata = {}
        for name in ("dest_bin", "block", "source_bin", "class"):
            d = defaultdict(lambda: [0, 0, 0])
            for (h, q, _), r, x in zip(rows, ref, y):
                k = h["privileged_teacher"]["verified_class"] if name == "class" else q[name]
                d[k][0] += r == 1
                d[k][1] += r == 1 and x == 0
                d[k][2] += x == 0
            strata[name] = {k: {"pos64": v[0], "missed": v[1], "miss_frac": round(v[1] / v[0], 4) if v[0] else None, "ci95": wilson(v[1], v[0]), "label0": v[2]}
                            for k, v in sorted(d.items())}
        neg = [(h, q) for (h, q, _), x in zip(rows, y) if x == 0]
        cpu = lambda vs: {"mean": round(float(np.mean([v["cpu_s"] for v in vs])), 3), "median": round(float(np.median([v["cpu_s"] for v in vs])), 3),  # noqa: E731
                          "p95": round(float(np.percentile([v["cpu_s"] for v in vs], 95)), 3), "sum_h": round(sum(v["cpu_s"] for v in vs) / 3600, 3)}
        tab[b] = {"queries": len(y), "label_1": sum(y), "label_0": len(y) - sum(y), "missed_vs_64": miss, "pos64": sum(ref),
                  "miss_frac": round(miss / sum(ref), 4) if sum(ref) else None, "miss_ci95": wilson(miss, sum(ref)),
                  "rescued_by_64": {"of_label0": len(y) - sum(y), "rescued": miss, "rate": round(miss / (len(y) - sum(y)), 4) if len(y) > sum(y) else None},
                  "flip_matrix_B_to_64": flips, "strata": strata,
                  "negatives": {"queries": len(neg), "roots": len({h["root_id"] for h, _ in neg}), "lineage_groups": len({grp[h["root_id"]] for h, _ in neg}),
                                "classes": dict(Counter(h["privileged_teacher"]["verified_class"] for h, _ in neg)), "objects": len({(h["root_id"], q["object_id"]) for h, q in neg}),
                                "kinds": dict(Counter(v["kind"] for v in f if not v["label"]))},
                  "pairs_planned_mean": round(float(np.mean([v["planned"] for v in f])), 2),
                  "survivor_pair_coverage": round(sum(v["planned"] for v in f) / max(1, sum(v["valid"] for v in f)), 4),
                  "attempted_pair_success": round(sum(v["verified"] for v in f) / max(1, sum(v["planned"] for v in f)), 4),
                  "verified_pairs_total": sum(v["verified"] for v in f), "evals_total": sum(v["evals"] for v in f),
                  "cpu_FULL": cpu(f), "cpu_FIRST_SUCCESS": cpu(s), "pairs_planned_mean_FIRST_SUCCESS": round(float(np.mean([v["planned"] for v in s])), 2)}
    mono = all(all(not (V[(b1, "FULL")][i]["label"] == 1 and V[(b2, "FULL")][i]["label"] == 0) for i in range(len(rows)))
               for b1 in BUDGETS for b2 in BUDGETS if b2 > b1)
    rank = defaultdict(lambda: [0, 0])
    for _, _, a in rows:
        for r, v in zip(a["a_rank"], a["a_verified"]):
            if r >= 0:
                rank[min(BUDGETS, key=lambda b: (r >= b, b))][0] += 1
                rank[min(BUDGETS, key=lambda b: (r >= b, b))][1] += bool(v)
    fail = Counter((T.METHODS[m], T.STOPS[s_]) for _, _, a in rows for m, s_, v, r in zip(a["a_method"], a["a_stop"], a["a_verified"], a["a_rank"]) if r >= 0 and not v)
    out = {"run": str(run64), "budgets": tab, "monotone_no_1_to_0": mono, "pair_success_by_rank_bucket": {f"<{b}": {"planned": v[0], "verified": v[1],
           "rate": round(v[1] / v[0], 4) if v[0] else None} for b, v in sorted(rank.items())}, "failed_attempts_by_method_stop": {f"{a}:{b}": n for (a, b), n in fail.most_common()},
           "queries_by_label_valid_pairs0": sum(v["valid"] == 0 for v in V[(64, "FULL")]), "virtual_rows": {f"{b}/{c}": [[v["label"], round(v["cpu_s"], 3), v["planned"]] for v in vs]
                                                                                                         for (b, c), vs in V.items()},
           "row_ids": [q["trial_id"] for _, q, _ in rows], "row_meta": [[h["root_id"], q["block"], q["dest_bin"], q["source_bin"], h["privileged_teacher"]["verified_class"]] for h, q, _ in rows]}
    Path(dst).write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


def interrupt(dst, run, snap, parts):
    """After SIGKILL + resume: packed files present at the kill are byte-identical; every query committed (final) at the kill
    reappears unchanged (record and arrays) inside its packed root."""
    run, res = Path(run), Counter()
    for line in Path(snap).read_text().splitlines():
        h, f = line.split(maxsplit=1)
        if not f.startswith("parts/"):
            res["packed_same" if (run / f).exists() and P.sha_file(run / f) == h else "packed_changed"] += 1
    now = {q["trial_id"]: (q, m, a) for _, q, m, a in load(run)}
    for f in Path(parts).glob("*/*.json"):
        r = json.loads(f.read_text())
        if not (r["completed"] or not r["eligible"]):
            res["unfinished_part_rerun"] += 1
            continue
        q, m, a = now[r["trial_id"]]
        same = r == q
        if f.with_suffix(".npz").exists():
            z = np.load(f.with_suffix(".npz"))
            same &= all(np.array_equal(z[k], m[k]) for k in MODE_KEYS) and all(np.array_equal(z[k], a[k]) for k in T.ATT)
        res["committed_part_same" if same else "committed_part_changed"] += 1
    Path(dst).write_text(json.dumps(dict(res), indent=1))
    return dict(res)


def integrity(L):
    L, res = Path(L), {}
    base = L / "audit" / "baseline"
    for name, root in (("original_dataset.sha256", P.DATA), ("smoke_20261106.sha256", L.parent / "smoke_20261106"), ("smoke_repro_v2.sha256", L.parent / "smoke_repro_v2")):
        lines = [x.split(maxsplit=1) for x in (base / name).read_text().splitlines()]
        now = {str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()}
        res[name] = {"files": len(lines), "changed": [f for h, f in lines if P.sha_file(root / f.strip()) != h],
                     "added": sorted(now - {f.strip()[2:] for _, f in lines})}
    res["smoke_code_unchanged"] = {f.split()[1]: P.sha_file(P.HERE / f.split()[1]) == f.split()[0] for f in (base / "code_before.sha256").read_text().splitlines()}
    res["git_status_original_dataset"] = subprocess.run(["git", "status", "--porcelain", "--ignored", "--", str(P.DATA)], capture_output=True, text=True, cwd=P.DATA).stdout.strip() or "clean"
    (L / "audit" / "integrity.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "audit":
        r = audit(args[0])
        print(json.dumps({"verification": {k: r["verification"][k] for k in ("passed", "failed", "failures")}, "summary": r["summary"],
                          "run_fingerprint": r["fingerprint"]["run_fingerprint"]}, indent=1))
    elif cmd == "compare":
        print(json.dumps(compare(args[0], args[1:]), indent=1))
    elif cmd == "nested":
        print(json.dumps(nested(*args), indent=1))
    elif cmd == "budget":
        r = budget(*args)
        print(json.dumps({b: {k: v for k, v in t.items() if k not in ("strata",)} for b, t in r["budgets"].items()}, indent=1))
    elif cmd == "interrupt":
        print(json.dumps(interrupt(*args), indent=1))
    elif cmd == "integrity":
        print(json.dumps(integrity(args[0]), indent=1))
