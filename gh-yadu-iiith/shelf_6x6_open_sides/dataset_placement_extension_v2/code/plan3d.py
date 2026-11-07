"""Lineage splits and the geometry-only placement roster (no teacher calls, no labels, no protected metadata).

Inputs per root: object ids / shapes / sizes / heights, initial cells, the 6x6 grid (and T's goal cell ONLY as an ordinary destination
cell for the holdout's goal-cell slot; it is never a filter).  NOT used anywhere here: roles, parking, intended or greedy blockers, B/D
labels or solutions, K-construction spec, X/Y outcomes.  K is a stratification / reporting field only.  Randomness = sha256 of (rule
version, root id, slot) -> fully deterministic.

  split   SRC OUT/split_manifest.json                         root-lineage groups -> train / val / test (70 / 15 / 15)       [V1]
  extend  SRC PARENT_SPLIT OUT                                 extend a frozen split to a newer source selection (inherit; SPLIT_CONFLICT)
  roster  SPLIT OUT [--production]                             V1 mock roster / production roster design                     [V1 rule]
  pin     SRC MANIFEST.json.gz OUT                             source pin: sha256 of every selected root (must equal the manifest)
  roster2 SRC SPLIT V1_ROSTER EXCLUDE OUT                      V2 mock roster = V1 rows (regression) + enriched independent holdout
  leak    ROSTER SPLIT                                         derived-scene layouts never shared across splits
  freeze2 CFG_DIR                                              freeze the V2 selection rule + run plan (before any V2 mock teacher call)
"""
import gzip, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

RULE = "SHELF66V3_PLACE_ROSTER_V1"
ROWS = COLS = 6
SLOTS = [(0, "near"), (0, "mid"), (0, "far"), (0, "any"), (1, "vacated"), (1, "mid"), (1, "far"), (2, "vacated"), (2, "nearmid"), (2, "far")]
REF_PAIRS = [(1, 6), (2, 9), (0, 4), (3, 8), (2, 5), (1, 7), (0, 9), (3, 4)]          # reference subset: 2 slots per root, cycled
BANDS = {"near": (1, 1), "mid": (2, 3), "far": (4, 5), "nearmid": (1, 3), "any": (1, 5)}


def u(*key):
    return int(hashlib.sha256("|".join(map(str, (RULE,) + key)).encode()).hexdigest()[:15], 16) / 16 ** 15


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def band(d):
    return "near" if d <= 1 else "mid" if d <= 3 else "far"


def load_selected(src):
    sel = json.loads((Path(src) / "selected.json").read_text())
    out = {}
    for k, ids in sel["selected"].items():
        for r in ids:
            rec = json.loads((Path(src) / "raw/roots" / f"{r}.json").read_text())
            out[r] = {"root_id": r, "K": k, "family": rec["ids"]["parent_family_id"], "objects": rec["objects"],
                      "cells": {o: tuple(c) for o, c in rec["initial_cells"].items()}, "goal": tuple(rec["goal"])}
    return out


def split(src, out):
    roots = load_selected(src)
    par = {r: r for r in roots}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    by = defaultdict(list)
    for r, v in roots.items():
        by[("fam", v["family"])].append(r)
        lay = json.dumps([sorted((o["id"], o["shape"], round(o["size"], 6), round(o["height"], 6)) for o in v["objects"]),
                          sorted(v["cells"].items())], default=list)
        by[("layout", hashlib.sha256(lay.encode()).hexdigest())].append(r)
    for rs in by.values():
        for r in rs[1:]:
            par[find(r)] = find(rs[0])
    groups = defaultdict(list)
    for r in roots:
        groups[find(r)].append(r)
    assign = {}
    for g, rs in groups.items():
        key = min(rs); x = u("split", key)
        s = "train" if x < 0.70 else "val" if x < 0.85 else "test"
        for r in rs:
            assign[r] = s
    man = {"rule": RULE + ": union-find over parent_family_id and identical (objects, cells) layout; split by sha256(rule|split|min root id)"
           " < 0.70 train, < 0.85 val, else test", "n_roots": len(roots), "n_groups": len(groups),
           "multi_root_groups": sum(len(v) > 1 for v in groups.values()), "counts": Counter(assign.values()),
           "by_class": {s: Counter(roots[r]["K"] for r in roots if assign[r] == s) for s in ("train", "val", "test")},
           "groups": {min(v): sorted(v) for v in groups.values()}, "split": dict(sorted(assign.items()))}
    man["sha256"] = hashlib.sha256(json.dumps(man["split"], sort_keys=True).encode()).hexdigest()
    Path(out).write_text(json.dumps(man, indent=1, sort_keys=True))
    print(json.dumps({k: man[k] for k in ("n_roots", "n_groups", "multi_root_groups", "counts", "by_class", "sha256")}))


def pick_roots(roots, assign, n_per=6):
    """Balanced K0-K3 train roots, distinct lineage groups, diverse (object count, T row band) -- geometry + class only."""
    out = []
    for k in ("0", "1", "2", "3"):
        cand = sorted((r for r, v in roots.items() if v["K"] == k and assign[r] == "train"), key=lambda r: u("root", r))
        used, strata = set(), Counter()
        for cap in (1, 2, 99):
            for r in cand:
                v = roots[r]
                st = (min(len(v["objects"]), 9) // 2, v["cells"]["t0"][0] // 2)
                if len([x for x in out if roots[x]["K"] == k]) >= n_per or r in out or v["family"] in used or strata[st] >= cap:
                    continue
                out.append(r); used.add(v["family"]); strata[st] += 1
    return out


def scene_hash(objects, cells):
    return hashlib.sha256(json.dumps([sorted((o["id"], o["shape"], round(o["size"], 6), round(o["height"], 6)) for o in objects),
                                      sorted((k, list(v)) for k, v in cells.items())]).encode()).hexdigest()[:20]


def _free(cells):
    occ = {tuple(v) for v in cells.values()}
    return [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in occ]


def _edit(rid, cells, depth, exclude):
    obs = sorted((o for o in cells if o != "t0" and o not in exclude), key=lambda o: u("edit_obj", rid, depth, o))
    if not obs:
        return None, None, None                                   # not representable (too few obstacles)
    a = obs[0]
    fr = sorted((c for c in _free(cells) if cheb(c, cells[a]) >= 2), key=lambda c: u("edit_dst", rid, depth, c))
    new = dict(cells); new[a] = fr[0]
    return a, cells[a], new


def roster_for(v):
    """10 eligible queries over the root scene and its 1-edit / 2-edit descendants, plus static contrasts."""
    rid, objs = v["root_id"], v["objects"]
    shape = {o["id"]: o["shape"] for o in objs}
    a, vac1, s1 = _edit(rid, v["cells"], 1, ())
    b, vac2, s2 = _edit(rid, s1, 2, (a,)) if a else (None, None, None)
    scenes = {0: (v["cells"], None, [])}
    if a:
        scenes[1] = (s1, vac1, [a])
    if b:
        scenes[2] = (s2, vac2, [a, b])
    hashes = {d: scene_hash(objs, c) for d, (c, _, _) in scenes.items()}
    q, used = [], Counter()
    for i, (d, bnd) in enumerate(SLOTS):
        if d not in scenes:
            continue
        cells, vac, edited = scenes[d]
        want = "box" if u("shape", rid, i) < 0.5 else "cylinder"
        obs = sorted((o for o in cells if o != "t0"), key=lambda o: (used[o], shape[o] != want, u("obj", rid, i, o)))
        for o in obs:
            if bnd == "vacated":
                cand = [vac] if o not in edited else []
            else:
                lo, hi = BANDS[bnd]
                cand = sorted((c for c in _free(cells) if lo <= cheb(c, cells[o]) <= hi), key=lambda c: u("dst", rid, i, o, c))
            if cand:
                dst = cand[0]; used[o] += 1
                break
        else:
            continue
        dd = cheb(dst, cells[o])
        q.append({"qid": f"{rid}:q{i}", "root_id": rid, "K": v["K"], "slot": i, "depth": d, "scene_hash": hashes[d], "parent_scene": hashes[d - 1] if d else None,
                  "edits": {e: list(scenes[d][0][e]) for e in edited}, "cells": {k: list(c) for k, c in cells.items()}, "object": o, "shape": shape[o],
                  "src": list(cells[o]), "dest": list(dst), "band": "vacated" if bnd == "vacated" else band(dd), "cheb": dd,
                  "dest_is_t_goal": tuple(dst) == v["goal"], "n_obj": len(cells), "clutter": clutter(cells, o, dst), "static": None})
    s0 = scenes[0][0]
    q4 = [x for x in q if x["slot"] == 4]
    others = sorted((x for x in s0 if x not in ("t0", a)), key=lambda x: u("occ", rid, x))
    o = q4[0]["object"] if q4 else (others or [a])[0]
    if o is None:
        return q
    for i, (dst, why) in enumerate(((vac1, "OCCUPIED"), ((ROWS, s0[o][1]), "OUT_OF_BOUNDS"))):
        if dst is None or (why == "OCCUPIED" and o == a):
            continue
        q.append({"qid": f"{rid}:s{i}", "root_id": rid, "K": v["K"], "slot": f"s{i}", "depth": 0, "scene_hash": hashes[0], "parent_scene": None,
                  "edits": {}, "cells": {k: list(c) for k, c in s0.items()}, "object": o, "shape": shape[o], "src": list(s0[o]), "dest": list(dst),
                  "band": "static", "cheb": cheb(dst, s0[o]), "dest_is_t_goal": False, "n_obj": len(s0), "clutter": None, "static": why,
                  "pairs_with": f"{rid}:q4" if why == "OCCUPIED" else None})
    return q


def clutter(cells, o, dst):
    occ = [tuple(c) for k, c in cells.items() if k != o]
    s = tuple(cells[o])
    lo, hi = (min(s[0], dst[0]), min(s[1], dst[1])), (max(s[0], dst[0]), max(s[1], dst[1]))
    return {"nbr_src": sum(cheb(c, s) == 1 for c in occ), "nbr_dst": sum(cheb(c, dst) == 1 for c in occ),
            "between": sum(lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] for c in occ)}


def roster(src, split_path, out, production=False):
    roots, man = load_selected(src), json.loads(Path(split_path).read_text())
    ids = sorted(roots) if production else pick_roots(roots, man["split"])
    rows = []
    for j, r in enumerate(ids):
        qs = roster_for(roots[r])
        refs = {f"{r}:q{s}" for s in REF_PAIRS[j % len(REF_PAIRS)]} if not production else set()
        for x in qs:
            x["split"], x["reference"] = man["split"][r], x["qid"] in refs
        rows += qs
    with gzip.open(out, "wt") as f:
        for x in rows:
            f.write(json.dumps(x, sort_keys=True) + "\n")
    el = [x for x in rows if not x["static"]]
    summ = {"roots": len(ids), "eligible": len(el), "static": len(rows) - len(el), "reference": sum(x["reference"] for x in rows),
            "by_K": Counter(x["K"] for x in el), "by_depth": Counter(x["depth"] for x in el), "by_band": Counter(x["band"] for x in el),
            "by_depth_band": Counter(f"{x['depth']}:{x['band']}" for x in el), "by_shape": Counter(x["shape"] for x in el),
            "dest_is_t_goal": sum(x["dest_is_t_goal"] for x in el), "splits": Counter(x["split"] for x in rows),
            "sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest()}
    print(json.dumps(summ, default=dict))
    return summ


def freeze(cfg_dir):
    """Freeze the predeclared selection rule + run plan BEFORE any roster query is labeled (hashes of roster, split, code, protocol)."""
    import placement3d as P
    import policy3d as Q
    cfg = Path(cfg_dir); root = cfg.parents[1]
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    rule = {
        "version": "SHELF66V3_PLACE_MOCK_V1_SELECTION_RULE", "teacher": P.protocol()[1], "teacher_hash": P.protocol()[0],
        "event": "V_place(S_hyp, o, d; PI_PLACE_3D): certified post-grasp state -> lift -> carry -> lower -> release -> retreat -> settle; no free move,"
                 " hover, approach, closure test, pick, sequencing, ready return or target-goal test",
        "labels": {"1": "a scheduled unit passes every placement3d.verify gate (MuJoCo-executed)", "0": "NEG_CERTIFIED_GEOMETRY | NEG_UNDER_POLICY"
                   " (kind BOUNDED_SEARCH / EXECUTION / DETERMINISTIC_PLAN)", "unlabeled": "static ineligible, NO_VALID_POST_GRASP_STATE,"
                   " INTERRUPTED (CPU guard / admin / safety stop), scene not settled"},
        "candidate_policies": {**Q.POLICIES, **Q.FIXED},
        "policy_selection": {"reference": "reference_v1 on the predeclared reference subset (roster rows with reference=true)",
                             "paired": "reference label in {0,1}; a reference 0 with any UNRESOLVED unit is censored for negative-side metrics only",
                             "rel_missed_positive_max_overall": 0.05, "rel_missed_positive_max_stratum": 0.10, "stratum_min_ref_positives": 10,
                             "strata": ["K", "depth", "band", "shape"], "ci": "Wilson 95% (reported; not a pass criterion at n<=60)",
                             "choose": "lowest virtual mean CPU-s/query among passing policies; adaptive_v1 preferred if within 1.15x of it",
                             "broad_pilot_policy": "adaptive_v1 (predeclared initial modest policy; its labels equal fixed_16x5000 labels by construction)"},
        "gates": {
            "semantics": ["witness phases subset of {lift,carry,lower,release,retreat,settle}", "every label 1 has a stored witness with all verify gates true",
                          "pick_inaccessible_held_placeable fixture labeled 1 (mode level: every placement-verified pose of the query is pick-inaccessible"
                          " under the source pick policy, 5000-iteration free move; the all-32-pose query-level case was searched in 400+ dev queries and not found)",
                          "no unlabeled status ever stored with label 0", "pytest suite passes (15 tests)"],
            "determinism": ["0 canonical mismatches: pilot run A vs run B (chosen workers)", "serial vs parallel on shared queries",
                            "sweep runs (unit=query, cold cache) vs pilot (unit=root) on shared queries",
                            "virtual adaptive_v1 replayed from reference_v1 == real adaptive_v1 on every reference query (label, unit states, verified rank,"
                            " witness id, counted plan/exec work)"],
            "coverage": {"strata": {"K": ["0", "1", "2", "3"], "depth": [0, 1, 2], "band": ["near", "mid", "far", "vacated"], "shape": ["box", "cylinder"],
                                    "between_band": ["0", "1-2", "3+"]}, "min_labeled_per_stratum": 1, "else": "COVERAGE_BLOCKED for that stratum"},
            "label_quality": {"min_ref_positive_lineages": 10, "min_negative_lineages": 5},
            "safety": {"ceiling_logical_cpus": 18, "monitor_violations": 0, "reserve_outside_mask": 6, "repeats_at_chosen_workers": 2},
            "storage": ["every packed witness loads and re-hashes to its id", "fresh-process replay re-verifies every stored positive (0 failures)"],
            "interrupt": "SIGKILL of the test-owned process group mid-run, resume with the same command: canonical output == uninterrupted run",
            "source_integrity": "audit/source_manifest.json.gz re-hash: 0 changes; git status of the source dataset clean",
            "split_leakage": "every mock query train; lineage groups closed; roster code reads no protected metadata",
            "finger_regression": "object checks at the grasp open width (instrumented test); held state both fingers in contact",
            "baseline_replay": {"a_unchanged_source_replay_agreement_min": 0.95, "b_placement_only_conversion": "reported + analysed"},
            "visualization": ["scene stills root / 1-edit / 2-edit", ">=1 verified witness GIF", ">=1 negative case figure", "budget/mode plots",
                              "throughput + resource traces"],
            "runtime": {"forecast": "production roster x per-stratum measured CPU, root-lineage bootstrap (2000)", "limit_days_with_buffer": 7.0, "buffer": 1.5}},
        "worker_choice": "highest queries/wall-s on the matched sweep subset; a config with fewer workers within 5% wins",
        "hashes": {"roster": h(root / "mock_v1/roster.jsonl.gz"), "split": h(root / "mock_v1/split_manifest.json"),
                   "production_roster_design": h(root / "mock_v1/production_roster_design.jsonl.gz"), "fixtures": h(cfg / "fixtures.json"),
                   "teacher_code": {n: h(root / "code" / n) for n in ("placement3d.py", "policy3d.py", "plan3d.py", "run3d.py")},
                   "support_code_at_freeze": {p.name: h(p) for p in sorted((root / "code").glob("*.py"))}}}
    plan = {"reference_run": {"policy": "reference_v1", "subset": "reference", "unit": "root"},
            "worker_sweep": {"policy": "adaptive_v1", "subset": "all", "unit": "query", "configs": {"8": "0-7", "12": "0-11", "16": "0-15", "18": "0-17"},
                             "serial": {"subset": "serial", "unit": "query", "cpus": "0"}},
            "broad_pilot": {"policy": "adaptive_v1", "subset": "all", "unit": "root", "repeats": 2},
            "interrupt_test": {"policy": "adaptive_v1", "subset": "all", "unit": "root", "kill_after_units": 8},
            "guard_cpu_s": 1200}
    for name, doc in (("selection_rule.json", rule), ("run_plan.json", plan)):
        doc["frozen_utc"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
        (cfg / name).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print({n: h(cfg / n) for n in ("selection_rule.json", "run_plan.json", "fixtures.json")})


RULE2 = "SHELF66V3_PLACE_ROSTER_V2_HOLDOUT"
H_SLOTS = [(0, "far"), (0, "farclutter"), (0, "mid"), (0, "tgoal"), (1, "far"), (1, "vacated"), (2, "far"), (2, "farclutter"), (2, "vacated"),
           (2, "mid")]
HOLD_K = {"0": 3, "1": 5, "2": 6, "3": 6}                     # 20 holdout roots, enriched toward high-K (more objects) scenes


def u2(*key):
    return int(hashlib.sha256("|".join(map(str, (RULE2,) + key)).encode()).hexdigest()[:15], 16) / 16 ** 15


def _uf(items, links):
    par = {r: r for r in items}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for rs in links:
        rs = [r for r in rs if r in par]
        for r in rs[1:]:
            par[find(r)] = find(rs[0])
    groups = defaultdict(list)
    for r in items:
        groups[find(r)].append(r)
    return [sorted(v) for v in groups.values()]


def _lay(v):
    return hashlib.sha256(json.dumps([sorted((o["id"], o["shape"], round(o["size"], 6), round(o["height"], 6)) for o in v["objects"]),
                                      sorted(v["cells"].items())], default=list).encode()).hexdigest()


def extend(src, parent, out):
    """Extend a frozen split to a newer selection.  Old groups stay closed; a group containing old roots inherits their (single) split;
    a group bridging old roots of different splits is a SPLIT_CONFLICT (nothing is written, exit 1); new standalone groups use the V1
    rule sha256(rule|split|min root id).  Old roots keep their split even when they left the selection (listed as retired)."""
    roots, old = load_selected(src), json.loads(Path(parent).read_text())
    oa = old["split"]
    items = sorted(set(roots) | set(oa))
    links = [rs for rs in old["groups"].values()]
    by = defaultdict(list)
    for r, v in roots.items():
        by[("fam", v["family"])].append(r); by[("layout", _lay(v))].append(r)
    links += list(by.values())
    assign, conflicts, inherited, fresh = {}, [], [], []
    for g in _uf(items, links):
        olds = {oa[r] for r in g if r in oa}
        if len(olds) > 1:
            conflicts.append({"group": g, "old_splits": sorted(olds)}); continue
        if olds:
            s_ = olds.pop()
            inherited += [r for r in g if r not in oa]
        else:
            x = u("split", min(g)); s_ = "train" if x < 0.70 else "val" if x < 0.85 else "test"; fresh.append(g)
        for r in g:
            assign[r] = s_
    rep = {"rule": "SHELF66V3_PLACE_SPLIT_EXTENSION_V1: union-find over the parent manifest's groups + parent_family_id + identical layout of"
                   " the new selection; inherit the old split; SPLIT_CONFLICT on a bridge; new groups by the V1 rule",
           "parent_sha256": old["sha256"], "conflicts": conflicts}
    if conflicts:
        Path(out).with_suffix(".SPLIT_CONFLICT.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep)[:2000]); sys.exit(1)
    changed = [r for r in oa if assign[r] != oa[r]]
    assert not changed, changed
    groups = _uf(items, links)
    man = {**rep, "n_roots": len(roots), "n_groups": len(groups), "multi_root_groups": sum(len(g) > 1 for g in groups),
           "counts": Counter(assign[r] for r in roots), "by_class": {s_: Counter(roots[r]["K"] for r in roots if assign[r] == s_) for s_ in ("train", "val", "test")},
           "new_roots": sorted(set(roots) - set(oa)), "new_roots_inheriting_old_group": sorted(inherited), "new_groups": len(fresh),
           "retired_old_roots": sorted(set(oa) - set(roots)), "old_assignments_changed": 0,
           "groups": {min(g): g for g in groups}, "split": dict(sorted(assign.items()))}
    man["sha256"] = hashlib.sha256(json.dumps(man["split"], sort_keys=True).encode()).hexdigest()
    Path(out).write_text(json.dumps(man, indent=1, sort_keys=True))
    print(json.dumps({k: man[k] for k in ("n_roots", "n_groups", "counts", "by_class", "sha256", "parent_sha256", "old_assignments_changed")}
                     | {"new_roots": len(man["new_roots"]), "inherited": len(inherited), "retired": len(man["retired_old_roots"])}, default=dict))


def pin(src, manifest, out):
    """Source pin for the collector: sha256 of every selected root, which must equal the frozen source manifest entry."""
    m = json.loads(gzip.open(manifest, "rt").read())
    ent = m.get("files") or m.get("entries") or m
    roots = load_selected(src)
    cur = {r: hashlib.sha256((Path(src) / "raw/roots" / f"{r}.json").read_bytes()).hexdigest() for r in roots}
    bad = [r for r in cur if ent.get(f"raw/roots/{r}.json") != cur[r]]
    if bad:
        sys.exit(f"SOURCE DRIFT vs manifest: {len(bad)} roots, e.g. {bad[:3]}")
    doc = {"snapshot": os.environ.get("SHELF_SNAPSHOT", "yadu/dataset-3d@8032ad0bf842a490897a8247ee46312036c12546"),
           "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(), "n_roots": len(cur), "roots": dict(sorted(cur.items()))}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True)); print({k: doc[k] for k in ("snapshot", "manifest_sha256", "n_roots")})


def hardness(v):
    """Geometry only: far (Chebyshev 4-5) (non-target object, free cell) pairs with >= 3 objects in their bounding box."""
    c = v["cells"]
    return sum(clutter(c, o, d)["between"] >= 3 for o in c if o != "t0" for d in _free(c) if 4 <= cheb(d, c[o]) <= 5)


def pick_holdout(roots, assign, exclude):
    """Independent holdout: train roots of lineage groups never touched by V1 (roster, fixtures, fixture searches, baseline replays),
    enriched (>= the class median object count, >= 1 far high-clutter pair), then hash order, one root per lineage group / family."""
    grp = {}
    for g in _uf(sorted(roots), [[r for r, v in roots.items() if v["family"] == f] for f in {v["family"] for v in roots.values()}]
                 + [[r for r, v in roots.items() if _lay(v) == h] for h in {_lay(v) for v in roots.values()}]):
        for r in g:
            grp[r] = min(g)
    bad = {grp[r] for r in exclude if r in grp}
    out, stats = [], {}
    for k, n in HOLD_K.items():
        cand = [r for r, v in roots.items() if v["K"] == k and assign[r] == "train" and grp[r] not in bad]
        med = sorted(len(roots[r]["objects"]) for r in cand)[len(cand) // 2]
        el = sorted((r for r in cand if len(roots[r]["objects"]) >= med and hardness(roots[r]) >= 1), key=lambda r: u2("holdout", r))
        used, take = set(), []
        for r in el:
            if len(take) < n and grp[r] not in used:
                take.append(r); used.add(grp[r])
        out += take
        stats[k] = {"candidates": len(cand), "median_objects": med, "eligible_enriched": len(el), "taken": len(take)}
    return out, stats


def roster_holdout(v):
    """10 enriched queries per holdout root: far / far-high-clutter / mid / T's goal cell (as an ordinary destination) on the root scene,
    far / vacated after 1 edit, far / far-high-clutter / vacated / mid after 2 edits.  Same edit rule as V1 (_edit)."""
    rid, objs = v["root_id"], v["objects"]
    shape = {o["id"]: o["shape"] for o in objs}
    a, vac1, s1 = _edit(rid, v["cells"], 1, ())
    b, vac2, s2 = _edit(rid, s1, 2, (a,)) if a else (None, None, None)
    scenes = {0: (v["cells"], None, [])}
    if a:
        scenes[1] = (s1, vac1, [a])
    if b:
        scenes[2] = (s2, vac2, [a, b])
    hashes = {d: scene_hash(objs, c) for d, (c, _, _) in scenes.items()}
    q, used = [], Counter()
    for i, (d, bnd) in enumerate(H_SLOTS):
        if d not in scenes:
            continue
        cells, vac, edited = scenes[d]
        want = "box" if u2("shape", rid, i) < 0.5 else "cylinder"
        obs = sorted((o for o in cells if o != "t0"), key=lambda o: (used[o], shape[o] != want, u2("obj", rid, i, o)))
        pick = None
        if bnd == "farclutter":
            pairs = [(clutter(cells, o, c)["between"], o, c) for o in obs for c in _free(cells) if 4 <= cheb(c, cells[o]) <= 5]
            if pairs:
                best = max(x[0] for x in pairs)
                pick = sorted(((o, c) for bt, o, c in pairs if bt == best), key=lambda x: (used[x[0]], u2("fc", rid, i, x[0], x[1])))[0]
        elif bnd == "tgoal":
            gc = tuple(v["goal"])
            if gc in _free(cells):
                pick = next(((o, gc) for o in obs if cheb(gc, cells[o]) >= 1), None)
        else:
            for o in obs:
                if bnd == "vacated":
                    cand = [vac] if o not in edited else []
                else:
                    lo, hi = BANDS[bnd]
                    cand = sorted((c for c in _free(cells) if lo <= cheb(c, cells[o]) <= hi), key=lambda c: u2("dst", rid, i, o, c))
                if cand:
                    pick = (o, cand[0]); break
        if pick is None:
            continue
        o, dst = pick
        used[o] += 1
        dd = cheb(dst, cells[o])
        q.append({"qid": f"{rid}:h{i}", "root_id": rid, "K": v["K"], "slot": i, "slot_kind": bnd, "depth": d, "scene_hash": hashes[d],
                  "parent_scene": hashes[d - 1] if d else None, "edits": {e: list(scenes[d][0][e]) for e in edited},
                  "cells": {k: list(c) for k, c in cells.items()}, "object": o, "shape": shape[o], "src": list(cells[o]), "dest": list(dst),
                  "band": "vacated" if bnd == "vacated" else band(dd), "cheb": dd, "dest_is_t_goal": tuple(dst) == tuple(v["goal"]),
                  "n_obj": len(cells), "clutter": clutter(cells, o, dst), "static": None})
    return q


def touched_roots(v1_dir):
    """Every root id V1 touched (roster, fixtures, fixture-search logs, baseline replays) -> excluded lineages for the holdout."""
    import re
    pat, out = re.compile(r"3ds66-[a-z0-9]+-w\d{4}-s\d{3}"), set()
    for p in list(Path(v1_dir).rglob("*.json")) + list(Path(v1_dir).rglob("*.log")) + list(Path(v1_dir).rglob("*.jsonl.gz")) + list(Path(v1_dir).rglob("*.txt")):
        if "runs" in p.parts and p.suffix != ".txt" and "dev" not in p.parts:
            continue                                           # run units only contain roster roots (already in the roster)
        if p.name == "production_roster_design.jsonl.gz" or p.name == "split_manifest.json":
            continue                                           # every selected root appears there: not a 'touch'
        txt = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text(errors="ignore")
        out |= set(pat.findall(txt))
    return out


def roster2(src, split_path, v1_roster, exclude_out, out):
    roots, man = load_selected(src), json.loads(Path(split_path).read_text())
    v1 = [json.loads(x) for x in gzip.open(v1_roster, "rt")]
    excl = touched_roots(Path(v1_roster).parent) | {r["root_id"] for r in v1}
    Path(exclude_out).write_text(json.dumps(sorted(excl), indent=0))
    hold, stats = pick_holdout(roots, man["split"], excl)
    rows = [{**r, "cohort": "v1_regression", "reference": True} for r in v1]
    for r in hold:
        for x in roster_holdout(roots[r]):
            x["split"], x["reference"], x["cohort"] = man["split"][r], True, "holdout"
            rows.append(x)
    el = [x for x in rows if not x["static"]]
    n_roots = len({x["root_id"] for x in rows})
    assert n_roots <= 48 and len(el) <= 500 and all(x["split"] == "train" for x in rows), (n_roots, len(el))
    with gzip.open(out, "wt") as f:
        for x in rows:
            f.write(json.dumps(x, sort_keys=True) + "\n")
    summ = {"roots": n_roots, "holdout_roots": len(hold), "eligible": len(el), "static": len(rows) - len(el), "holdout_stats": stats,
            "excluded_v1_touched_roots": len(excl), "by_cohort": Counter(x["cohort"] for x in el), "by_K": Counter(x["K"] for x in el),
            "by_depth": Counter(x["depth"] for x in el), "by_band": Counter(x["band"] for x in el), "by_shape": Counter(x["shape"] for x in el),
            "between3plus": sum((x["clutter"] or {}).get("between", 0) >= 3 for x in el), "dest_is_t_goal": sum(x["dest_is_t_goal"] for x in el),
            "far_holdout": sum(x["band"] == "far" and x["cohort"] == "holdout" for x in el), "sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest()}
    print(json.dumps(summ, default=dict))


def leak(roster_path, split_path):
    """Every scene layout of the roster (root scenes and their 1- / 2-edit descendants) and every selected root's initial layout must
    belong to ONE split."""
    man = json.loads(Path(split_path).read_text())
    seen = defaultdict(set)
    for x in (json.loads(l) for l in gzip.open(roster_path, "rt")):
        seen[x["scene_hash"]].add(man["split"][x["root_id"]])
    bad = {h: sorted(s_) for h, s_ in seen.items() if len(s_) > 1}
    print(json.dumps({"scene_layouts": len(seen), "cross_split_layouts": len(bad), "examples": list(bad.items())[:5]}))
    return not bad


def freeze2(cfg_dir):
    """Freeze the V2 selection rule + run plan BEFORE any V2 mock teacher call (hashes of roster, split, source pin, code, protocol)."""
    import placement3d as P
    import policy3d as Q
    cfg = Path(cfg_dir); root = cfg.parents[1]
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    cand = {k: (Q.POLICIES | Q.FIXED)[k] for k in ("adaptive_v1", "fixed_4x1500", "fixed_4x5000", "fixed_8x5000")}
    rule = {
        "version": "SHELF66V3_PLACE_MOCK_V2_SELECTION_RULE", "teacher": P.protocol()[1], "teacher_hash": P.protocol()[0],
        "registered_as": "a NEW rule registered before any V2 teacher call; the V1 rule (mock_v1/configs/selection_rule.json) and its literal"
                         " choice fixed_4x1500 stay on record unchanged.  Reason for the new rule: V1 evaluated policies on a 40-query subset with"
                         " < 10 far reference positives, while the broad V1 pilot showed fixed_4x1500 missing 6/45 far positives.",
        "event": "V_place(S_hyp, o, d; PI_PLACE_3D) conditional on a certified post-grasp state of o: lift -> carry -> lower -> release -> retreat"
                 " -> settle; no free move, hover, approach, closure test, pick, sequencing, ready return or target-goal test",
        "labels": {"1": "a scheduled unit's eligible branch passes every placement3d.verify gate (MuJoCo-executed)",
                   "0": "NEG_CERTIFIED_GEOMETRY (scope: 32 poses x <= 4 source IK candidates x the 2-D footprint carry model) | NEG_UNDER_POLICY"
                        " (BOUNDED_SEARCH / EXECUTION / DETERMINISTIC_PLAN, with its work scope)",
                   "unlabeled": "static ineligible, NO_VALID_POST_GRASP_STATE, INTERRUPTED (CPU guard / admin / safety stop), scene not settled"},
        "reference": {"policy": "reference_v2", "stages": Q.POLICIES["reference_v2"], "early_stop": False,
                      "meaning": "finite reference recall: every valid unit x every eligible branch at 5000 RRT iterations; NOT physical truth"},
        "candidate_policies": cand,
        "comparison": "every candidate is RUN for real on every mock query (12 workers, CPUs 0-11); branch-aware virtual replay from the"
                      " reference is a cross-check only (prefix-equivalence must hold exactly on every query, else virtual results are not used)",
        "policy_selection": {"paired": "per query, reference label in {0,1}; candidate misses = reference 1 & candidate 0",
                             "missed_positive_max_overall": 0.05, "missed_positive_max_stratum": 0.10, "stratum_min_ref_positives": 10,
                             "difficult_strata": {"band": ["far"], "depth": [2], "between": ["3+"], "cohort": ["holdout"], "shape": ["box", "cylinder"],
                                                  "K": ["0", "1", "2", "3"], "dest_is_t_goal": [True]},
                             "ci": "Wilson 95% (reported for every stratum; never used as a pass threshold; 0 observed misses is not zero risk)",
                             "choose": "lowest measured mean CPU-s/query (real 12-worker runs) among passing policies; adaptive_v1 preferred if within"
                                       " 1.15x of it", "if_none_pass": "MOCK_V2_FAIL for policy selection (report; do not relax)"},
        "gates": {
            "semantics": ["witness phases subset of {lift,carry,lower,release,retreat,settle}", "every label 1 has a stored witness with all verify gates",
                          "no unlabeled status stored with label 0", "branch fixtures: v1 rejects / v2 rescues (natural if found, else injected + labelled synthetic)",
                          "pytest suite passes"],
            "determinism": ["adaptive_v1 run A vs run B: 0 canonical mismatches", "serial subset vs run A: 0 mismatches",
                            "SIGKILL + resume vs run A: 0 mismatches", "virtual (from reference_v2) vs real for every candidate: 0 mismatches or the"
                            " virtual path is reported unusable"],
            "accounting": "0 unaccounted modes and 0 unaccounted branches on every query",
            "storage": ["every unit JSON + NPZ re-validates (sha256 + witness re-hash)", "fresh-process replay re-verifies 100% of stored positives (all runs)"],
            "source_integrity": "pinned source (yadu/dataset-3d@8032ad0b) re-hash: 0 changes; v1 artifacts unchanged",
            "split_leakage": "0 val/test mock roots; holdout lineages disjoint from every V1-touched lineage; 0 cross-split derived layouts",
            "safety": {"ceiling_logical_cpus": 18, "mask": "0-11", "monitor_violations": 0, "sensor_loss_test": "fail closed"},
            "coverage": {"strata": {"K": ["0", "1", "2", "3"], "depth": [0, 1, 2], "band": ["near", "mid", "far", "vacated"], "shape": ["box", "cylinder"]},
                         "K4_NONE": "COVERAGE_BLOCKED (not in the pinned source / not validated)"},
            "runtime": {"forecast": "live snapshot roster x stratum CPU (K x depth x band, backed off) with root-lineage bootstrap (2000)",
                        "limit_days_with_buffer": 7.0, "buffer": 1.5}},
        "hashes": {"roster": h(root / "mock_v2/roster_v2.jsonl.gz"), "split": h(root / "mock_v2/split_manifest_pinned.json"),
                   "source_pin": h(cfg / "source_pin.json"), "excluded_v1_roots": h(root / "mock_v2/excluded_v1_touched_roots.json"),
                   "teacher_code": {n: h(root / "code" / n) for n in ("placement3d.py", "policy3d.py", "plan3d.py", "run3d.py", "guard3d.py")},
                   "support_code_at_freeze": {p.name: h(p) for p in sorted((root / "code").glob("*.py"))}}}
    plan = {"reference_run": {"policy": "reference_v2", "subset": "all", "unit": "root", "workers": 12, "cpus": "0-11"},
            "candidates_real": {k: {"subset": "all", "unit": "root", "workers": 12, "cpus": "0-11"} for k in cand},
            "repeat": {"policy": "adaptive_v1", "name": "adaptive_v1_B"},
            "serial": {"policy": "adaptive_v1", "subset": "depth-0 slot 1 / holdout slot 2 per root", "unit": "query", "workers": 1, "cpus": "0"},
            "interrupt": {"policy": "adaptive_v1", "kill_after_units": 10, "then": "corrupt one NPZ + add an orphan, resume"},
            "sweep": {"policy": "adaptive_v1", "subset": "depth-0 queries", "unit": "query", "configs": {"8": "0-7", "12": "0-11", "16": "0-15", "18": "0-17"},
                      "condition": "only if gvlab has no other heavy load"},
            "guard_cpu_s": 1200, "max_roots": 48, "max_physical_queries": 500}
    for name, doc in (("selection_rule_v2.json", rule), ("run_plan_v2.json", plan)):
        doc["frozen_utc"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
        (cfg / name).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print({n: h(cfg / n) for n in ("selection_rule_v2.json", "run_plan_v2.json", "source_pin.json")})


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    c = sys.argv[1]
    if c == "split":
        split(sys.argv[2], sys.argv[3])
    elif c == "extend":
        extend(sys.argv[2], sys.argv[3], sys.argv[4])
    elif c == "pin":
        pin(sys.argv[2], sys.argv[3], sys.argv[4])
    elif c == "roster2":
        roster2(*sys.argv[2:7])
    elif c == "leak":
        sys.exit(0 if leak(sys.argv[2], sys.argv[3]) else 1)
    elif c == "freeze2":
        freeze2(sys.argv[2])
    elif c == "freeze":
        freeze(sys.argv[2])
    else:
        roster(sys.argv[2], sys.argv[3], sys.argv[4], production="--production" in sys.argv)
