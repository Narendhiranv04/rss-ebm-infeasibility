#!/usr/bin/env python3
"""Fail-closed, versioned preparation of a 2D placement launch-readiness v2 candidate.

Creates code_v2 and launch_ready_2d_v2 alongside the existing v1 artifacts.
NEVER calls the physical teacher, starts workers, or launches the full collection.
Requires the repository tree from GitHub branch naren/2d-dataset-relocation,
commit 16d1b3b4 (or an explicitly reviewed identical-source checkout).

This is a reviewed patch *candidate*. It is NOT a claim that integration tests or
25-worker thermal tests have been run. See INSTRUCTIONS.md before using.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

BASE_COMMIT = "16d1b3b4d565ae89b833141e136c71157823e1dc"

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    n = source.count(old)
    if n != 1:
        raise RuntimeError(f"FAIL_CLOSED {label}: expected exactly one anchor, found {n}; no files committed")
    return source.replace(old, new, 1)


def patch_plan(s: str) -> str:
    s = replace_once(s, '"version": "PLACEMENT_PLAN_V1"', '"version": "PLACEMENT_PLAN_V2"', "plan version")
    # REVIEW FIX R1: the bare anchor also occurs in QUOTAS["Q1"]; anchor on the PLAN line's own indentation (unique).
    s = replace_once(s, '\n        "root_objects": 4, "root_dests": 12, "prefixes": 2, "prefix_objects": 2, "prefix_dests": 6,',
                         '\n        "root_objects": 4, "root_dests": 12, "prefixes": 3, "prefix_objects": 2, "prefix_dests": 6,', "plan prefix count")
    # All quotas keep the optional two-edit block. No change to the original source roots.
    s = replace_once(s, '"Q1": {"root_objects": 4, "root_dests": 12, "prefixes": 2,',
                         '"Q1": {"root_objects": 4, "root_dests": 12, "prefixes": 3,', "Q1 quota")
    s = replace_once(s, '"Q2": {"root_objects": 4, "root_dests": 8, "prefixes": 2,',
                         '"Q2": {"root_objects": 4, "root_dests": 8, "prefixes": 3,', "Q2 quota")
    s = replace_once(s, '"Q3": {"root_objects": 3, "root_dests": 6, "prefixes": 2,',
                         '"Q3": {"root_objects": 3, "root_dests": 6, "prefixes": 3,', "Q3 quota")
    s = replace_once(s, '"Q4": {"root_objects": 2, "root_dests": 4, "prefixes": 2,',
                         '"Q4": {"root_objects": 2, "root_dests": 4, "prefixes": 3,', "Q4 quota")
    s = s.replace("destinations: legal, empty, not T's goal;", "destinations: legal and empty, independent of T's goal;")
    s = s.replace('one random design edit; one archived executed first move, flagged as chosen',
                  'two geometry-designed edits; plus a third TWO-EDIT hypothetical context, all chosen')
    s = replace_once(s, 'goal, legal = r["goal_cell"], set(r["scene"]["legal_cells"])',
                         'legal = set(r["scene"]["legal_cells"])', "target-goal independence")
    s = replace_once(s, 'c for c in legal if c not in cells.values() and c != goal and c not in extra',
                         'c for c in legal if c not in cells.values() and c not in extra', "remove task-specific destination veto")
    start = '''        arch = sorted({tuple(b["sequence"][0]) for b in r["labels"]["B"] if b["witnesses"] and b["sequence"] and b["sequence"][0][0] in movers
                       and b["sequence"][0][1] in pool(cells0)} - {edits[0][2:]}, key=lambda m: H(rid, "edit2", *m))
        if arch:
            edits.append(("prefix2", "ARCHIVED_FIRST_MOVE", *arch[0]))
        elif len(order) > 1:
            edits.append(("prefix2", "DESIGN_RANDOM_EDIT", order[1], min(pool(cells0), key=lambda c: H(rid, "edit2dest", c))))
    for block, kind, a, c in edits:
        cells, prefix = {**cells0, a: c}, [[a, c]]
        near = sorted((o for o in movers if o != a), key=lambda o: (cheb(cells0[o], cells0[a]), H(rid, block, o)))[:PLAN["prefix_objects"]]
        for oi, o in enumerate(near):
            vac = oi == 0 and cells0[a] != goal
            ds = ([(cells0[a], "vacated", 1, 1)] if vac else []) + dests((rid, block, o), cells[o], pool(cells, [cells0[a]] if vac else []),
                                                                        PLAN["prefix_dests"] - vac)
            for di, (d, b, k, n) in enumerate(ds):
                add(block, prefix, kind, o, cells, d, b, k, n, pilot and oi == 0 and di < PLAN["pilot"]["prefix_dests"], ranks=(oi, di))
            if vac:                                   # the same object and cell before the edit: occupied, a legality record
                add("contrast_" + block, [], None, o, cells0, cells0[a], "vacated", 1, 1, pilot, True)
'''
    repl = '''        # Never use archived solution labels or witness roles to select sampler contexts.
        if len(order) > 1:
            edits.append(("prefix2", "DESIGN_RANDOM_EDIT", order[1], min(pool(cells0), key=lambda c: H(rid, "edit2dest", c))))
    # Third block: two legal, distinct hypothetical relocations; no physical execution claimed.
    # Do not query an object already moved in the prefix (move-once convention).
    if len(order) >= 3 and len(pool(cells0)) >= 2:
        a = order[0]
        c1 = min(pool(cells0), key=lambda c: H(rid, "edit3first", c))
        c_after_a = {**cells0, a: c1}
        b = next(o for o in order if o != a)
        # Avoid reversing into the first mover's source. Both edits affect distinct objects.
        candidates2 = [c for c in pool(c_after_a) if c != cells0[a]]
        if candidates2:
            c2 = min(candidates2, key=lambda c: H(rid, "edit3second", c))
            edits.append(("prefix3", "DESIGN_TWO_EDIT", a, c1, b, c2))
    for edit in edits:
        block, kind = edit[:2]
        moves = [(edit[2], edit[3])] + ([(edit[4], edit[5])] if len(edit) == 6 else [])
        cells = dict(cells0)
        for a, c in moves:
            assert a in cells and c in pool(cells), (rid, block, a, c)
            cells[a] = c
        prefix = [[a, c] for a, c in moves]
        moved = {a for a, _ in moves}
        vacated = cells0[moves[-1][0]]
        near = sorted((o for o in movers if o not in moved),
                      key=lambda o: (min(cheb(cells0[o], cells0[a]) for a in moved), H(rid, block, o)))[:PLAN["prefix_objects"]]
        for oi, o in enumerate(near):
            vac = oi == 0 and vacated not in cells.values()
            ds = ([(vacated, "vacated", 1, 1)] if vac else []) + dests(
                (rid, block, o), cells[o], pool(cells, [vacated] if vac else []), PLAN["prefix_dests"] - int(vac))
            for di, (d, b, k, n) in enumerate(ds):
                add(block, prefix, kind, o, cells, d, b, k, n,
                    pilot and oi == 0 and di < PLAN["pilot"]["prefix_dests"], ranks=(oi, di))
            if vac:
                add("contrast_" + block, [], None, o, cells0, vacated, "vacated", 1, 1, pilot, True)
'''
    s = replace_once(s, start, repl, "replace oracle-dependent edit selection + add two-edit contexts")
    # No changed grouping or source identity. All 5,000 roots retain their original group assignment.
    s = s.replace('"prefixes": 2,', '"prefixes": 3,')
    return s


def patch_teacher(s: str) -> str:
    # The adapter itself is scientific code and must be included in its identity.
    s = replace_once(s, 'def pair_protocol():', '@functools.lru_cache(maxsize=1)\ndef pair_protocol():', "cache pinned protocol")
    s = replace_once(s, '"smoke_code_sha256": P.sha_file(P.__file__)}',
                         '"smoke_code_sha256": P.sha_file(P.__file__), "adapter_sha256": P.sha_file(__file__)}', "pin adapter source hash")
    # Preserve compatibility with the original FIXED/FULL and FIRST_SUCCESS pilot paths.
    s = replace_once(s, 'for rank, (i, j) in enumerate(sched[:B]):\n        if policy["coverage"] == "FIRST_SUCCESS" and any(r[11] for r in rows):\n            break',
'''max_pairs = int(policy.get("rescue_pairs", B)) if policy["coverage"] == "ADAPTIVE_FULL" else B
    if max_pairs < B:
        raise ValueError("rescue_pairs must be >= pairs_per_query")
    for rank, (i, j) in enumerate(sched[:max_pairs]):
        if policy["coverage"] == "FIRST_SUCCESS" and any(r[11] for r in rows):
            break
        # Fully test the first B pairs, then extend to max_pairs only if all first-B modes failed.
        if policy["coverage"] == "ADAPTIVE_FULL" and rank == B and any(r[11] for r in rows):
            break''', "adaptive FULL 8-to-32 search")
    s = replace_once(
        s,
        "scope = f\"FINITE_POLICY:{policy['pairs_per_query']}_PAIRS,{policy['coverage']},PER_PAIR_WORK_BUDGET\"",
        """scope = (f"FINITE_POLICY:{policy['pairs_per_query']}_PAIRS,"
                 f"RESCUE_TO_{policy.get('rescue_pairs', policy['pairs_per_query'])},"
                 f"{policy['coverage']},PER_PAIR_WORK_BUDGET")""",
        "explicit negative scope",
    )
    # REVIEW FIX R2: up to 32 witnesses are stored per query; two pairs yielding identical arrays share one content id, so
    # store and reference each content id once (otherwise witness_refs repeats an id and the audit fails).
    s = replace_once(s, '                keep = stored < policy["witness_store_max"]',
                         '                keep = w not in refs and stored < policy["witness_store_max"]', "content-deduplicated witness storage")
    return s


def patch_check(s: str) -> str:
    s = replace_once(s,
'''        n_exp = min(B, mc["valid_pairs"]) if pol["coverage"] == "FULL" or first is None else first + 1''',
'''        if pol["coverage"] == "ADAPTIVE_FULL":
            first_base = any(0 <= int(r) < B and bool(v) for r, v in zip(a["a_rank"], a["a_verified"]))
            n_exp = min(B if first_base else int(pol["rescue_pairs"]), mc["valid_pairs"])
        else:
            n_exp = min(B, mc["valid_pairs"]) if pol["coverage"] == "FULL" or first is None else first + 1''',
    "audit adaptive schedule")
    return s


def patch_runner(s: str) -> str:
    s = replace_once(s,
'''        pol = {"pairs_per_query": B, "coverage": cov, "witness_store_max": B, "reuse_archived": True}''',
'''        if cov == "ADAPTIVE_FULL":
            rescue = int(C.get("adaptive_rescue_pairs", 32))
            pol = {"pairs_per_query": B, "rescue_pairs": rescue, "coverage": cov,
                   "witness_store_max": rescue, "reuse_archived": False}
        else:
            # V2 pilot is oracle-independent even for the archived-witness baseline.
            pol = {"pairs_per_query": B, "coverage": cov, "witness_store_max": B, "reuse_archived": False}''',
    "runner v2 pilot policy")
    return s


def patch_guard(s: str) -> str:
    s = replace_once(s, '    chk["disk_headroom"] = du.free // 2 ** 20 >= need',
'''    # V2 uses a generous preflight floor because all verified witnesses are stored.
    minimum = int(C.get("v2_min_free_disk_mb", 10240))
    chk["disk_headroom"] = du.free // 2 ** 20 >= max(need, minimum)
    if full:
        # Host power configuration is a reproducible condition, not a runtime command.
        boost = Path("/sys/devices/system/cpu/cpufreq/boost")
        chk["cpu_boost_disabled"] = boost.exists() and boost.read_text().strip() == "0"
        chk["oracle_free_production_policy"] = pol is not None and pol["policy"].get("reuse_archived") is False
        chk["all_discovered_witnesses_saved"] = pol is not None and int(pol["policy"].get("witness_store_max", 0)) >= int(pol["policy"].get("rescue_pairs", pol["policy"].get("pairs_per_query", 0)))
''', "guard disk/boost/witness/archival checks")
    return s


def patch_tests(s: str) -> str:
    return replace_once(s, 'L = Path(__file__).resolve().parents[1] / "launch_ready_2d_v1"',
                        'L = Path(__file__).resolve().parents[1] / "launch_ready_2d_v2"', "tests point at v2 package")

PATCHES = {
    "launch_plan.py": patch_plan,
    "placement_teacher.py": patch_teacher,
    "launch_check.py": patch_check,
    "launch_runner.py": patch_runner,
    "launch_guard.py": patch_guard,
    "test_launch_ready.py": patch_tests,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path, help="Path to the local rss-ebm-infeasibility checkout/worktree")
    ap.add_argument("--apply", action="store_true", help="Prepare new v2 directories; default validates anchors without writing")
    ap.add_argument("--allow-other-commit", action="store_true", help="Only after human code review; still fail-closed on source anchors")
    args = ap.parse_args()
    repo = args.repo.resolve()
    rel = Path("gh-yadu-iiith/planar_2d/dataset_relocations_v1")
    root = repo / rel
    src, oldL = root / "code", root / "launch_ready_2d_v1"
    dst, newL = root / "code_v2", root / "launch_ready_2d_v2"
    if not src.is_dir() or not oldL.is_dir():
        raise SystemExit(f"FAIL_CLOSED: missing v1 source: {src} or {oldL}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if not args.allow_other_commit and head != BASE_COMMIT:
        raise SystemExit(f"FAIL_CLOSED: expected {BASE_COMMIT}, got {head}. Use an exact checkout; never overwrite existing work.")
    if dst.exists() or newL.exists():
        raise SystemExit("FAIL_CLOSED: code_v2 or launch_ready_2d_v2 already exists. Refuse to overwrite.")
    edited = {}
    for name, fn in PATCHES.items():
        original = (src / name).read_text()
        revised = fn(original)
        ast.parse(revised, filename=name)
        edited[name] = revised
    # Confirm expected policy anchors before creating any new files.
    assert '"prefix3", "DESIGN_TWO_EDIT"' in edited["launch_plan.py"]
    assert '"ADAPTIVE_FULL"' in edited["placement_teacher.py"]
    assert '"cpu_boost_disabled"' in edited["launch_guard.py"]
    print("SOURCE PATCH PREVIEW OK; 6 modules parse correctly")
    for n, v in edited.items():
        print(f"  {n}: {len((src/n).read_text().splitlines())} -> {len(v.splitlines())} lines")
    if not args.apply:
        print("DRY RUN. No writes. Re-run with --apply after code review.")
        return 0
    # Only copy existing assets; NEVER touch code/ or launch_ready_2d_v1/.
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    try:
        shutil.copytree(src, dst, ignore=ignore)
        for name, s in edited.items():
            (dst / name).write_text(s)
        newL.mkdir()
        (newL / "configs").mkdir()
        (newL / "audit").mkdir()
        # Baseline hashes are part of provenance, and are copied byte-for-byte.
        shutil.copytree(oldL / "audit" / "baseline", newL / "audit" / "baseline")
        cfg = json.loads((oldL / "configs" / "launch_config.json").read_text())
        cfg["config_version"] = "LAUNCH_READY_2D_V2"
        cfg["full_run_dir"] = str(root / "full_2d_v2")
        cfg["adaptive_rescue_pairs"] = 32
        cfg["v2_min_free_disk_mb"] = 10240
        cfg["storage"]["production_all_verified_witnesses"] = True
        # Pair protocol is dynamically hashed after patching; launch freeze fills identity later.
        cfg["pair_protocol_id"] = "PENDING_FREEZE"
        (newL / "configs" / "launch_config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
        rule = json.loads((oldL / "configs" / "pilot_policy_selection.json").read_text())
        rule["version"] = "PILOT_POLICY_SELECTION_V2"
        rule["adaptive_policy"] = {"base_pairs": 8, "rescue_pairs": 32, "coverage": "ADAPTIVE_FULL", "reuse_archived": False,
                                   "witness_store_max": 32, "decision": "freeze only after new roster pilot and 2 clean 25-worker thermal runs"}
        rule["safety_gate_25_workers"] = "two completed 25-worker ADAPTIVE_FULL runs with boost OFF, zero safety cancels, each independently audited"
        rule["workload_rule"]["min_free_disk_mb_v2"] = 10240
        (newL / "configs" / "pilot_policy_selection.json").write_text(json.dumps(rule, indent=1, sort_keys=True))
        # The v2 shell wrapper is inert without a frozen production policy and token.
        shell = (oldL / "launch_full_2d.sh").read_text()
        shell = shell.replace('CODE="$HERE/../code"', 'CODE="$HERE/../code_v2"')
        shell = shell.replace("full_2d_v1.launch.log", "full_2d_v2.launch.log")
        shell = shell.replace("full_2d_v1", "full_2d_v2")
        (newL / "launch_full_2d.sh").write_text(shell)
        (newL / "launch_full_2d.sh").chmod(0o755)
        # A self-contained v2 regression test module, never a production launcher.
        with_test = Path(__file__).with_name("test_v2_contract.py")
        if not with_test.is_file():
            raise RuntimeError("Missing test_v2_contract.py beside prepare_v2.py")
        shutil.copy2(with_test, dst / "test_v2_contract.py")
        (newL / "RUNBOOK_V2.md").write_text(
            "# V2 readiness only\n\n"
            "Follow the separately supplied `INSTRUCTIONS.md` and `mock_v2_gate.py`.\n"
            "The wrapper in this folder is LOCKED until a fresh v2 readiness decision freezes a policy.\n"
            "Never run production as part of this mock.\n")
        # V2 intentionally starts WITHOUT production_policy, frozen_hashes or query_plan;
        # use the runbook to create them AFTER review, pilot and preflight.
        (newL / "README_PREPARED_NOT_LAUNCHED.txt").write_text(
            "Candidate v2 code is prepared. NO launch/freeze/pilot was executed by this script.\n"
            "Follow INSTRUCTIONS.md. Never run the full collection before independent v2 gates pass.\n")
        print("PREPARED V2; no physical planning, no production launch, and no frozen policy")
        print(f"    Code: {dst}\n    Readiness candidate: {newL}")
    except BaseException:
        # Safe rollback: only remove *new* directories created by this script.
        if dst.exists(): shutil.rmtree(dst)
        if newL.exists(): shutil.rmtree(newL)
        raise
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
