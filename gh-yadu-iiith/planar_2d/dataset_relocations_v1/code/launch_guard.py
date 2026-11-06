"""Launch guard for the full 2D placement collection: metadata-only preflight (no teacher call, counted), the confirmation token
of the current frozen files, and the production entry that refuses unless --execute-full, a matching --confirm token and a
passing preflight are all present. Nothing here starts work except full(..., execute=True) after every check has passed."""
import fcntl, gzip, json, os
from collections import Counter
from pathlib import Path

import psutil
import launch_runner as R
from launch_runner import LP, P, T, cfg_of, plan_rows, read_temp, sensors, ac_online, topology, tasks_for


def preflight(L, full=False):
    """Metadata only: no teacher call (counted). Dataset, plan, split, config, policy, topology, sensor, power, memory, disk."""
    L, C = Path(L), cfg_of(L)
    calls = Counter()
    orig = (P.plan_carry, P.held_modes)
    P.plan_carry = lambda *a, **k: calls.update(["plan_carry"]) or orig[0](*a, **k)
    P.held_modes = lambda *a, **k: calls.update(["held_modes"]) or orig[1](*a, **k)
    try:
        return _checks(L, C, full, calls)
    finally:
        P.plan_carry, P.held_modes = orig


def _checks(L, C, full, calls):
    h, chk = json.loads((L / "plan_hashes.json").read_text()), {}
    plan = plan_rows(L)
    chk["selected_unchanged"] = P.sha_file(P.DATA / "selected.json") == h["source_selected_sha256"]
    chk["roots_5000_unique"] = len({p["root_id"] for p in plan}) == len(plan) == 5000 == len(LP.selected())
    chk["root_files_unchanged"] = all(P.sha_file(P.DATA / "raw" / "roots" / f"{p['root_id']}.json") == p["root_file_sha256"] for p in plan)
    chk["query_plan_hash"] = P.sha(gzip.decompress((L / "query_plan.jsonl.gz").read_bytes())) == h["query_plan_sha256"]
    chk["split_manifest_hash"] = P.sha_file(L / "split_manifest.json") == h["split_manifest_sha256"]
    b = LP.build()
    chk["plan_recomputes_from_dataset"] = P.sha("".join(json.dumps(p, separators=(",", ":"), sort_keys=True) + "\n" for p in b["plan"]).encode()) == h["query_plan_sha256"]
    chk["teacher_sources_pinned"] = T.pair_protocol()[0] == C["pair_protocol_id"]
    chk["config_hash"] = P.sha_file(L / "configs" / "launch_config.json") == (json.loads((L / "configs" / "frozen_hashes.json").read_text())["launch_config_sha256"])
    cpus, res, topo = topology(C["execution"]["workers"], C["execution"]["reserved_logical_cpus"])
    chk["cpusets_match_config"] = cpus == C["execution"]["worker_cpus"] and res == C["execution"]["reserved_cpus"]
    lab = sensors()
    chk["sensor_k10temp_tctl"] = lab is not None and read_temp(lab) is not None
    chk["ac_power"] = ac_online() is True
    vm, du = psutil.virtual_memory(), psutil.disk_usage(str(L))
    chk["memory_headroom"] = vm.available // 2 ** 20 >= C["execution"]["workers"] * 250 + C["safety"]["min_mem_available_mb"]
    pol_f = L / "configs" / "production_policy.json"
    pol = json.loads(pol_f.read_text()) if pol_f.exists() else None
    need = (pol or {}).get("forecast", {}).get("disk_mb", 0) + C["safety"]["min_disk_free_mb"]
    chk["disk_headroom"] = du.free // 2 ** 20 >= need
    if full:
        chk["production_policy_frozen"] = pol is not None and P.sha_file(pol_f) == json.loads((L / "configs" / "frozen_hashes.json").read_text()).get("production_policy_sha256")
        tgt = Path(C["full_run_dir"])
        chk["no_conflicting_full_run"] = not (tgt / ".lock").exists() or _unlocked(tgt / ".lock")
    q = Counter()
    sel = (lambda r: LP.in_quota(r, LP.QUOTAS[pol["quota"]])) if pol else (lambda r: False)
    for p in plan:
        for r in p["queries"]:
            if sel(r):
                q[(p["split"], "static" if r[12] else "teacher")] += 1
    rep = {"checks": chk, "passed": all(chk.values()), "teacher_calls": dict(calls), "planner_calls": calls["plan_carry"], "worker_cpus": cpus,
           "reserved_cpus": res, "topology": topo, "hard_cpu_quota": False, "mem_available_mb": vm.available // 2 ** 20, "disk_free_mb": du.free // 2 ** 20,
           "disk_needed_mb": need, "planned_queries": {f"{a}/{b}": n for (a, b), n in sorted(q.items())}, "production_policy": pol,
           "temp_now_c": read_temp(lab) if lab else None, "load1": os.getloadavg()[0]}
    return rep


def _unlocked(f):
    with open(f, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def token(L):
    L = Path(L)
    fz = json.loads((L / "configs" / "frozen_hashes.json").read_text())
    h = json.loads((L / "plan_hashes.json").read_text())
    parts = [h["query_plan_sha256"], h["split_manifest_sha256"], fz["launch_config_sha256"], fz.get("production_policy_sha256") or "NO_POLICY"]
    return "LAUNCH-2D-" + P.sha("|".join(parts).encode())[:20]


def full(L, argv, execute=True):
    """Production entry: refuses unless --execute-full, --confirm <token of the CURRENT frozen files> and a passing preflight.
    execute=False (the `guard` command) stops after the checks: nothing is created and no teacher runs."""
    if "--execute-full" not in argv or "--confirm" not in argv or argv.index("--confirm") + 1 >= len(argv):
        raise SystemExit("REFUSED: full collection needs --execute-full and --confirm <token> (see RUNBOOK.md)")
    if argv[argv.index("--confirm") + 1] != token(L):
        raise SystemExit("REFUSED: confirmation token does not match the current frozen plan / split / config / policy hashes")
    rep = preflight(L, full=True)
    if not rep["passed"]:
        raise SystemExit("REFUSED: preflight failed: " + json.dumps({k: v for k, v in rep["checks"].items() if not v}))
    C, pol = cfg_of(L), rep["production_policy"]
    if not execute:
        return {"guard": "PASSED", "token": token(L), "full_run_dir": C["full_run_dir"], "policy": pol["policy"], "quota": pol["quota"]}
    roots = tasks_for(L, lambda r: LP.in_quota(r, LP.QUOTAS[pol["quota"]]))
    return R.run(L, C["full_run_dir"], roots, pol["policy"], C["execution"]["workers"], C["execution"]["worker_cpus"], C["execution"]["reserved_cpus"], "FULL")
