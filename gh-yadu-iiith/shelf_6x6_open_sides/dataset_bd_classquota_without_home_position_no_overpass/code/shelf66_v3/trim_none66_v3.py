"""NONE trim for the 6 x 6 shelf dataset (2026-11-08).  The dense NONE scenes (run_shelf66_none.py) hold 13-14 obstacles while K0-K3
scenes hold 0-7, so the obstacle count alone would give NONE away.  The certificate needs far fewer: every feasible direction of T with
>= KMIN (5) obstacles that each block it ON THEIR OWN.  An obstacle's blocking is tested with only T and itself on the shelf, so removing
other obstacles never changes it -- as long as the arm ceiling (tallest object + margin) stays the same, which is why the tallest obstacle
is always kept.

Per dense NONE root (ledger order):
  1. the smallest set of obstacles that keeps >= KMIN blockers on every direction of its stored certificate (exact search; ties broken
     by a seeded rng), with the scene's tallest obstacle forced in;
  2. + extra obstacles drawn with P_EXTRA (set to none: every obstacle not needed for the certificate is removed);
  3. obstacles renumbered o00, o01, ... (no gaps in the ids); kept cover = intended blockers, extras = distractors;
  4. the trimmed scene is labelled from scratch like every root: V3 teacher, DEPENDENT / INCONSISTENT checks, then the NONE gate
     none_cert66_v4 (certificate recomputed on the trimmed scene + strong direct + one move first) -- counted only if CONFIRMED.
Root id: the dense root's id with the name part + "t" (3ds66-lapN-w0000-s001 -> 3ds66-lapNt-w0000-s001); "trim" in the record keeps
the source root, the id map and the removed obstacles.

    python shelf66_v3/trim_none66.py --src <dense NONE run> --out <trim run> --cap 18 [--part k --parts n] [--only id,id]
Resumable (roots already in <out>/ledger.jsonl are skipped); <out>/STOP stops it cleanly; <out>/MAX_RUNNING overrides --cap live."""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import multiprocessing as mp
import random
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataset-bd"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

TRIM_VERSION = "NONE_TRIM66_V3"
GEN_NONE = "DENSE_NONE_V2+CERT66_V4+TRIM_V3"
KMIN = 5
P_EXTRA = (1.0, 0.0, 0.0)            # P(0 / 1 / 2 extra obstacles beyond the minimal cover); user 2026-11-08: "remove unnecessary objects"
MAX_INTENDED = 5                     # the teacher also tries "1 irrelevant move + ALL intended" (+ T = 7 actions = bd_core.MAX_ACTIONS): V2 with 6 gave "8 actions > 7"
B_CELLS = 6                          # one-move check per obstacle, as in run_shelf66_none (the certificate is the proof)


def ledger(p):
    out = []
    if p.exists():
        for x in p.read_text().splitlines():
            try:
                out.append(json.loads(x))
            except ValueError:
                pass
    return out


def new_id(rid):
    p = rid.split("-")
    p[1] += "t"
    return "-".join(p)


def choose(r):
    """(kept obstacle ids in old numbering, minimal cover size, number of extras)."""
    cert = r["none_audit"]["certificate"]
    blk = {d: set(v) for d, v in cert["blockers"].items()}
    obs = [o for o in r["objects"] if o["id"] != "t0"]
    tallest = max(obs, key=lambda o: o["height"])["id"]
    rng = random.Random(r["ids"]["construction_seed"])
    rest = sorted({o["id"] for o in obs} - {tallest})
    for k in range(0, len(rest) + 1):
        covers = [set(s) | {tallest} for s in itertools.combinations(rest, k)
                  if all(len(b & (set(s) | {tallest})) >= KMIN for b in blk.values())]
        if covers:
            break
    cover = rng.choice(covers)
    n_extra = rng.choices([0, 1, 2], weights=P_EXTRA)[0]
    others = sorted({o["id"] for o in obs} - cover)
    extra = set(rng.sample(others, min(n_extra, len(others))))
    return cover, extra, len(cover)


def trimmed_scene(r):
    import run_shelf66_full as F
    cover, extra, n_cover = choose(r)
    keep = sorted(cover | extra)
    m = {"t0": "t0", **{o: f"o{i:02d}" for i, o in enumerate(keep)}}
    c = F.scene_of(r)
    c["objs"] = [{**o, "id": m[o["id"]]} for o in r["objects"] if o["id"] in m]
    c["cells"] = {m[k]: v for k, v in r["initial_cells"].items() if k in m}
    top = sorted(cover)[:MAX_INTENDED]               # V3: see MAX_INTENDED
    c["roles"] = {m[o]: ("intended_blocker" if o in top else "distractor:near_path") for o in keep}
    c["parking"] = {m[k]: v for k, v in r["parking"].items() if k in m}
    c["alt_parking"] = {m[k]: v for k, v in r["alt_parking"].items() if k in m}
    c["intended"] = [m[o] for o in top]
    c["greedy"] = [m[o] for o in r["greedy_blockers"] if o in m]
    occ = {tuple(v) for v in c["cells"].values()}
    c["wrong_dest"] = [list(x) for x in __import__("bd_3dt_plus").line(tuple(r["target_src"]), tuple(r["goal"])) if x not in occ][:2]
    con = dict(r["construction"])
    con.pop("certificate", None)                    # recomputed by the NONE gate on the trimmed scene
    con.pop("cert_tries", None)
    con["distractors"] = {"placed": len(extra), "requested": "trim"}
    c["construction"] = con
    trim = {"version": TRIM_VERSION, "from_root": r["ids"]["root_id"], "from_n_obstacles": len(r["objects"]) - 1,
            "n_obstacles": len(keep), "min_cover": n_cover, "extras": len(extra), "id_map": {k: v for k, v in m.items() if k != "t0"},
            "removed": sorted(o["id"] for o in r["objects"] if o["id"] not in m), "tallest_kept": True, "intended_cap": MAX_INTENDED}
    return c, trim


def work(job):
    import bd_core as bc
    import run_3dt_plus as R
    import run_shelf66_full as F
    import run_shelf66_none as N
    e0, srcdir, out = job
    _, A, P = F._modules()
    F.none_gate = N.none_gate
    N.B_CELLS = B_CELLS
    store = bc.Store(out)
    rid0 = e0["root_id"]
    rid = new_id(rid0)
    t0 = time.time()
    signal.signal(signal.SIGALRM, R._alarm)
    signal.alarm(F.TEACH_WALL_S)
    entry = {"gen": P.GENERATOR_VERSION, "gen_none": GEN_NONE, "wave": e0["wave"], "slot": e0["slot"], "spec": e0["spec"], "root_id": rid,
             "construction_seed": e0["construction_seed"], "trimmed_from": rid0}
    try:
        r = json.loads((Path(srcdir) / "roots" / f"{rid0}.json").read_text())
        G = A.env()
        A.POOL_CAP = P.POOL_CAP
        ids = {**r["ids"], "root_id": rid, "config_hash": G["config_hash"], "teacher_version": A.TEACHER_VERSION, "robot_protocol": G["protocol"]}
        c, trim = trimmed_scene(r)
        entry["n_obstacles"] = trim["n_obstacles"]
        entry.update(F.label(e0["spec"], ids, c, store, {"generator_none": GEN_NONE, "trim": trim}))
    except (A.Reject, R.WallCap) as e:
        entry.update(accepted=False, reason=f"{type(e).__name__}:{str(e)[:400]}")
    except Exception as e:  # noqa: BLE001
        entry.update(accepted=False, reason=f"ERROR:{type(e).__name__}:{str(e)[:200]}", trace=traceback.format_exc()[-1500:])
    finally:
        signal.alarm(0)
    entry["wall_s"] = time.time() - t0
    print(f"trim {rid0} -> {rid}: {entry.get('n_obstacles')} obstacles, {entry.get('observed', entry.get('reason'))} ({entry['wall_s']:.0f} s)",
          flush=True)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--part", default="0", help="comma-separated residues: roots i with i %% parts in this set")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    import bd_core as bc
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = bc.Store(out)
    only = set(filter(None, a.only.split(",")))
    done = {e.get("trimmed_from") for e in ledger(out / "ledger.jsonl") if e.get("accepted")}    # V1 ERROR lines are redone
    todo = [e for i, e in enumerate(x for x in ledger(src / "ledger.jsonl") if x.get("accepted") and x.get("observed") == "NONE")
            if i % a.parts in {int(x) for x in a.part.split(",")} and e["root_id"] not in done and (not only or e["root_id"] in only)]
    print(f"{len(todo)} NONE roots to trim (part {a.part}/{a.parts})", flush=True)
    t_start = time.time()

    def cap():
        try:
            return int((out / "MAX_RUNNING").read_text())
        except (OSError, ValueError):
            return a.cap

    with mp.get_context("spawn").Pool(a.cap, maxtasksperchild=1) as pool:
        running = {}
        while True:
            for r in [r for r in running if r.ready()]:
                running.pop(r)
                store.append(r.get())
                led = [e for e in ledger(out / "ledger.jsonl") if e.get("gen_none") == GEN_NONE]
                prog = {"done": len(led), "running": len(running), "left": len(todo),
                        "observed": dict(collections.Counter(str(e.get("observed", e.get("reason", "")[:30])) for e in led)),
                        "n_obstacles": dict(sorted(collections.Counter(e.get("n_obstacles") for e in led if e.get("accepted")).items(),
                                                   key=lambda kv: str(kv[0]))),
                        "elapsed_h": round((time.time() - t_start) / 3600, 2), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
                (out / f"PROGRESS_{a.part.replace(',', '_')[:20]}.json").write_text(json.dumps(prog, indent=1))
            stop = (out / "STOP").exists()
            while not stop and todo and len(running) < min(cap(), a.cap):
                e0 = todo.pop(0)
                running[pool.apply_async(work, ((e0, str(src), str(out)),))] = e0["root_id"]
            if not running and (stop or not todo):
                print(f"done ({'STOP' if stop else 'all'})", flush=True)
                break
            time.sleep(5)


if __name__ == "__main__":
    main()
