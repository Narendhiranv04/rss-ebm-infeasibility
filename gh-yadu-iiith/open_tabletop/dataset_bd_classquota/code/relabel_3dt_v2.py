"""Relabel the class-quota 3D dataset for teacher V2 (bd_3dt.TEACHER_VERSION): within an executed B sequence the arm returns to the
ready pose ONCE, after the last relocation, then moves T (V1 returned to the ready pose after every relocation).

What V2 changes: only B sequences with >= 2 relocations (their executed steps, outcomes and witnesses) and what the teacher derives
from them: verified K (the repair search executes its clearing sets as sequences), the B plans built on the minimal repair, the
search trace (blocker.clearing_sets_reachable / sequence_tries / relocation_queries), independent_moves, teacher_calls.
What V2 keeps: the original-scene label, every D edit, the blocker lattice (all T-only queries from the ready pose in a teleported
scene), every B sequence with <= 1 relocation (identical protocol: relocation, ready pose, T), construction metadata. These are
kept byte-identical from V1; the V2 re-run recomputes them only as a reproduction check (`relabel.reproduction`).

    python relabel_3dt_v2.py <raw> <out> [workers]     # resumable: <out>/RELABEL.jsonl, <out>/roots/, <out>/witnesses/
"""
import json
import os
import shutil
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dataset-bd"))
sys.path.insert(0, str(HERE))
WALL_S = 5400


def relocations(b):
    return len(b["sequence"])


def needs_v2(r):
    """A root whose labels involve no executed sequence with >= 2 relocations is unchanged by V2."""
    return r["labels"]["original"]["y"] == 0 or any(relocations(b) >= 2 for b in r["labels"]["B"])


def case(r):
    """bd_3dt.teach's input, from a stored record. wrong_dest is not stored: its first cell is the stored wrong_destination move,
    else (route construction) bd_3dt_plus.line(src, goal) minus occupied cells, as construct_route builds it."""
    import bd_3dt_plus as P
    wd = next((b["sequence"][0][1] for b in r["labels"]["B"] if b["role"] == "wrong_destination"), None)
    occ = {tuple(v) for v in r["initial_cells"].values()}
    line = [list(x) for x in P.line(tuple(r["target_src"]), tuple(r["goal"])) if x not in occ][:2]
    if r["construction"]["method"] != "bd_3dt":
        assert wd is None or wd == line[0], (r["ids"]["root_id"], wd, line)
        wd = wd or (line[0] if line else None)
    return {"objs": r["objects"], "cells": r["initial_cells"], "goal": r["goal"], "roles": r["roles"], "parking": r["parking"],
            "intended": r["intended_blockers"], "wrong_dest": [wd] if wd else []}


def q_sig(x):
    return [x["y"], x.get("negative_kind"), x.get("grasp_id")]


def b_sig(b):
    return [b["role"], b["sequence"], b["y"], b["failed_step"], [[s["object"], s.get("grasp_id"), s["ok"]] for s in b["steps"]]]


def blk_sig(x):
    return {k: x[k] for k in ("scope", "full_lattice", "minimal_sets", "mask", "blocker_supervision_valid")}


def _alarm(*_):
    raise TimeoutError("relabel wall cap")


def work(job):
    """One root: copy it unchanged, or re-run the V2 teacher and merge (V1 bytes for everything V2 does not change)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import bd_core as bc
    import bd_3dt as A
    import bd_3dt_plus as P
    src, out = Path(job[0]), Path(job[1])
    r = json.loads(src.read_text())
    rid, t0, c0 = r["ids"]["root_id"], time.time(), time.process_time()
    v1 = {"teacher_version": r["ids"]["teacher_version"], "config_hash": r["ids"]["config_hash"], "verified_K": r["verified_K"],
          "observed": r["observed"], "independent_moves": r.get("independent_moves"), "teacher_calls": r["teacher_calls"],
          "teacher_wall_s": r["teacher_wall_s"]}
    G = A.env()
    r["ids"].update(teacher_version=A.TEACHER_VERSION, config_hash=G["config_hash"])
    line = {"root_id": rid, "K_v1": v1["verified_K"]}
    if not needs_v2(r):
        for b in r["labels"]["B"]:
            for s in b["steps"]:
                s.setdefault("ends_at_ready", True)
        r["relabel"] = {"v1": v1, "action": "KEPT_NO_MULTI_RELOCATION_SEQUENCE", "relabelled": []}
        line.update(action="kept", K_v2=r["verified_K"])
    else:
        A.POOL_CAP, A.MAX_SEQ_TRIES = r["budget"]["pool_cap"], r["budget"]["seq_tries"]
        store = bc.Store(out)
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(WALL_S)
        try:
            c = case(r)
            t = A.teach(c, store, rid)
        finally:
            signal.alarm(0)
        L1, L2 = r["labels"], t["labels"]
        old = {(b["role"], json.dumps(b["sequence"])): b for b in L1["B"]}
        B, changed = [], []
        for b in L2["B"]:
            k = (b["role"], json.dumps(b["sequence"]))
            if relocations(b) <= 1 and k in old:            # protocol-identical: keep the V1 entry and its witnesses
                b1 = old[k]
                for s in b1["steps"]:
                    s.setdefault("ends_at_ready", True)
                B.append(b1)
            else:
                B.append(b)
            p = old.get(k)
            changed.append({"role": b["role"], "relocations": relocations(b), "y_v1": p["y"] if p else None, "y_v2": b["y"],
                            "kept_v1": B[-1] is not b, "grasps_v1": [s.get("grasp_id") for s in p["steps"]] if p else None,
                            "grasps_v2": [s.get("grasp_id") for s in b["steps"]]})
        same_minimal = [b["sequence"] for b in L1["B"] if b["role"] == "minimal"] == [b["sequence"] for b in L2["B"] if b["role"] == "minimal"]
        ms = next((b["sequence"] for b in B if b["role"] == "minimal" and b["y"]), [])
        indep = v1["independent_moves"] if same_minimal else (not ms or P.independent(c["objs"], c["cells"], dict(ms), [o for o, _ in ms], c["goal"]))
        repro = {"original": q_sig(L1["original"]) == q_sig(L2["original"]),
                 "D": [[d["edit"], q_sig(d)] for d in L1["D"]] == [[d["edit"], q_sig(d)] for d in L2["D"]],
                 "blocker": blk_sig(L1["blocker"]) == blk_sig(L2["blocker"]),
                 "B_single": all(b_sig(old[k]) == b_sig(b) for b in L2["B"] for k in [(b["role"], json.dumps(b["sequence"]))]
                                 if relocations(b) <= 1 and k in old)}
        blk = {**L1["blocker"], **{k: L2["blocker"][k] for k in ("clearing_sets_reachable", "sequence_tries", "relocation_queries")}}
        r["labels"] = {"original": L1["original"], "D": L1["D"], "B": B, "B_skipped": L2["B_skipped"], "blocker": blk,
                       "edits_evaluated": L1["edits_evaluated"]}
        r.update(verified_K=t["verified_K"], observed="NONE" if t["verified_K"] is None else t["verified_K"], independent_moves=bool(indep),
                 teacher_calls=t["teacher_calls"], teacher_wall_s=t["teacher_wall_s"])
        r["relabel"] = {"v1": v1, "action": "RELABELLED_V2", "relabelled": ["labels.B", "labels.B_skipped", "labels.blocker.clearing_sets_reachable",
                        "labels.blocker.sequence_tries", "labels.blocker.relocation_queries", "verified_K", "observed", "independent_moves",
                        "teacher_calls", "teacher_wall_s"], "B": changed, "reproduction": repro}
        line.update(action="relabelled", K_v2=r["verified_K"], B=changed, repro=repro, same_minimal=same_minimal)
    obs = str(r["observed"]) if r["independent_moves"] else "DEPENDENT"
    line.update(observed_v1=str(v1["observed"]) if v1["independent_moves"] is not False else "DEPENDENT", observed_v2=obs,
                cpu_s=round(time.process_time() - c0, 1), wall_s=round(time.time() - t0, 1))
    (out / "roots").mkdir(parents=True, exist_ok=True)
    tmp = out / "roots" / f".{rid}.tmp"
    tmp.write_text(json.dumps(r, indent=1, sort_keys=True))
    tmp.replace(out / "roots" / f"{rid}.json")
    return line


def safe(job):
    try:
        return work(job)
    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
        return {"root_id": Path(job[0]).stem, "action": "ERROR", "error": f"{type(e).__name__}:{str(e)[:300]}"}


def main(raw, out, workers=4):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    led = [json.loads(x) for x in (raw / "ledger.jsonl").read_text().splitlines() if x.strip()]
    ids = [e["root_id"] for e in led if e.get("accepted")]
    log = out / "RELABEL.jsonl"
    done = {json.loads(x)["root_id"] for x in log.read_text().splitlines()} if log.exists() else set()
    done = {d for d in done if (out / "roots" / f"{d}.json").exists()}
    jobs = [(str(raw / "roots" / f"{i}.json"), str(out)) for i in ids if i not in done]
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)             # big roots first (K4 lattices), for an even finish
    print(f"{len(ids)} accepted roots, {len(done)} done, {len(jobs)} to go, {workers} workers", flush=True)
    with Pool(workers, maxtasksperchild=10) as pool, log.open("a") as f:
        for n, line in enumerate(pool.imap_unordered(safe, jobs), 1):
            f.write(json.dumps(line, sort_keys=True) + "\n")
            f.flush()
            print(n, line["root_id"], line["action"], line.get("K_v1"), "->", line.get("K_v2"), line.get("cpu_s"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        print(json.dumps(work((sys.argv[2], sys.argv[3])), indent=1))
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
