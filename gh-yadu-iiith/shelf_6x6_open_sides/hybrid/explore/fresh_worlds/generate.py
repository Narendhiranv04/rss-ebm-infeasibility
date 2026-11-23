#!/usr/bin/env python3
"""FRESH 3D test roots: new runs of the team's UNCHANGED SHELF66V3 generator under names never used before, so no new root can share a
construction seed (= the scene) or a generation batch ("world" = 3ds66-<name>-w<wave>, hz_eval._cluster) with TRAIN / VAL / TEST.
No method is run and nothing is scored here (pre-registration first).  Every stage resumes from its own ledger; clean stop:
touch <out>/gen/<stage dir>/STOP (running roots finish).

Stages = the team's scripts, run as-is from ABF3/fin3d/src/.../code (the published generator code + its NONE tools; the shared files are
byte-identical to dataset_bd_classquota_without_home_position_no_overpass/code):
  k     shelf66_v3/run_shelf66_full.py --name fw1   K0-K3 class-quota run (QUOTA.json: NONE 0, K4 0)            -> <out>/gen/k
  none  shelf66_v3/run_shelf66_none.py --name fwN   dense NONE + blocking certificate (QUOTA.json: NONE only)   -> <out>/gen/none_dense
  trim  shelf66_v3/trim_none66_v3.py   (ids fwNt)   trimmed NONE, re-taught + re-certified, as published       -> <out>/gen/none_trim
  check only the disjointness check and the quotas (no generation)
    python generate.py [--stage all|k|none|trim|check] [--workers 24] [--n 250] [--quota '{"0": 1}'] [--out DIR]
Run it with the generator's python (mujoco 3.16.0, numpy 2.2.6).  `k` and `none` are independent and may run at the same time (split the
workers); `trim` needs `none` to be finished.  Quotas = the 3D TEST class mix scaled to --n, plus ~8 % for roots the build QC drops."""
import argparse, json, math, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SH = HERE.parents[2]                                                   # shelf_6x6_open_sides/
RUNS = SH / "approachb_3d_learning/runs"
ABF = RUNS / "ABF3"
CODE = ABF / "fin3d/src/dataset_bd_classquota_without_home_position_no_overpass/code"
PUB = RUNS / "K3ABL/runs/public"
OUT = RUNS / "explore/fresh_worlds"
TEST_MIX = {"0": 67, "1": 53, "2": 86, "3": 102, "NONE": 62}         # = K counts of private_test_3d.jsonl (370 roots)
NAMES = {"k": "fw1", "none": "fwN", "trim": "fwNt"}                   # trim_none66_v3: dense name + "t"
USED = {"lap", "lap3", "lap4", "lap5", "srv", "srv3", "srv5", "lapN", "srvN", "lapNt", "srvNt",   # published + final3d_v4 runs
        "hx1", "lapx", "hw1"}                                                                     # our earlier runs (home server, pilot, gen_hw)
REMOTE = ("/home/projects/long-horizon/naren_workspace/k3abl/abf/", "/home/projects/long-horizon/abf_v2/root/")   # = hz_eval.REMOTE
SLOTS, MAX_WAVE, CERT_TRIES = 24, 150, 4      # run_3dt_plus wave size; waves checked (N=250 needs ~20; fwN wave 188 slot 5 retry 2 clashes
                                              # by chance, 32-bit seeds); run_shelf66_none.CERT_TRIES (seed + 7919 i).  build.py re-checks the real seeds.
sys.path.insert(0, str(CODE / "dataset-bd"))
import bd_core as bc  # noqa: E402  (ids / sha: the generator's own seed formula)


def local(p):
    for pre in REMOTE:
        if p.startswith(pre):
            return str(ABF / p[len(pre):])
    return p


def targets(n, quota=None):
    """{class: n roots wanted}: TEST mix scaled to n by largest remainder (or --quota as given)."""
    if quota:
        return {k: int(v) for k, v in quota.items()}
    tot = sum(TEST_MIX.values())
    raw = {k: n * v / tot for k, v in TEST_MIX.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:n - sum(out.values())]:
        out[k] += 1
    return out


def gen_quota(t):
    # spare for QC drops (fixed object, Y(empty) != 1); NONE +30 %: the trim stage also loses dense roots (yield not measured)
    return {k: v + round((0.30 if k == "NONE" else 0.08) * v) for k, v in t.items()}


def layout_key(rec):
    """= hybrid/gen/qc_new_roots.layout_key (objects + cells + goal)."""
    import hashlib
    objs = sorted((o["shape"], round(o["size"], 5), round(o["height"], 5), tuple(rec["initial_cells"][o["id"]]), o["id"].startswith("t"))
                  for o in rec["objects"])
    return hashlib.sha256(json.dumps([objs, list(rec["goal"])]).encode()).hexdigest()[:16]


def split_index():
    """root ids, construction seeds and layouts of every TRAIN / VAL / TEST root (3D) + seeds of the generator ledgers on this host."""
    ids, seeds, lays, miss = set(), set(), set(), []
    for s in ("train", "val", "test"):
        for e in map(json.loads, open(PUB / f"private_{s}_3d.jsonl")):
            ids.add(e["root_id"])
            p = Path(local(e["path"]))
            if not p.exists():
                miss.append(e["root_id"]); continue
            rec = json.loads(p.read_text())
            seeds.add(int(rec["ids"]["construction_seed"]))
            lays.add(layout_key(rec))
    for led in (SH / "dataset_bd_classquota_without_home_position_no_overpass/raw/ledger.jsonl",
                ABF / "mirror/rss3d_final_src/gh-yadu-iiith/shelf_6x6_open_sides/dataset_bd_classquota_without_home_position_no_overpass/raw_none/ledger.jsonl"):
        if led.exists():
            seeds |= {int(json.loads(l)["construction_seed"]) for l in open(led) if l.strip()}
    assert not miss, f"{len(miss)} split root files missing on this host (e.g. {miss[:2]}): copy ABF3/mirror first"
    return ids, seeds, lays


def check():
    """Disjointness BEFORE generation: our names are new, and none of the seeds they can produce is a split / ledger seed."""
    ids, seeds, _ = split_index()
    assert len(ids) >= 2400 and seeds, f"split index too thin to check against ({len(ids)} roots, {len(seeds)} seeds)"
    spaces = {r.split("-")[1] for r in ids if r.startswith("3ds66-")}
    ours = set(NAMES.values())
    assert not ours & (USED | spaces), ours & (USED | spaces)
    mine = [int(bc.sha([bc.GENERATOR_VERSION, "3ds66", NAMES[st], w, s])[:8], 16) for st in ("k", "none") for w in range(MAX_WAVE) for s in range(SLOTS)]
    mine += [x + 7919 * i for x in mine[MAX_WAVE * SLOTS:] for i in range(1, CERT_TRIES)]      # dense NONE re-tries
    clash = set(mine) & seeds
    assert not clash, f"seed clash with existing roots: {sorted(clash)[:5]}"
    print(json.dumps({"split_roots": len(ids), "known_seeds": len(seeds), "our_seeds_checked": len(mine), "our_internal_dups": len(mine) - len(set(mine)),
                      "names": NAMES, "disjoint": True}), flush=True)


def run(cmd, cwd):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "PYTHONWARNINGS": "ignore", "MUJOCO_GL": "disable", "PYTHONDONTWRITEBYTECODE": "1"}
    print("RUN", " ".join(map(str, cmd)), flush=True)
    subprocess.run(["nice", "-n", "10", *map(str, cmd)], cwd=cwd, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "k", "none", "trim", "check"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--quota", default=None, help='target counts per class, e.g. {"0": 1, "1": 1} (smoke)')
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t = targets(a.n, json.loads(a.quota) if a.quota else None)
    q = gen_quota(t)
    gen = Path(a.out) / "gen"
    print(json.dumps({"targets": t, "generation_quota": q}), flush=True)
    check()
    py = sys.executable
    for st in (("k", "none", "trim") if a.stage == "all" else () if a.stage == "check" else (a.stage,)):
        if st == "k":
            d = gen / "k"
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "QUOTA.json").exists():                  # never overwritten: a live edit by the user wins
                (d / "QUOTA.json").write_text(json.dumps({"NONE": 0, "0": q.get("0", 0), "1": q.get("1", 0), "2": q.get("2", 0), "3": q.get("3", 0), "4": 0}))
            (d / "K4_MAX_RUNNING").exists() or (d / "K4_MAX_RUNNING").write_text("0")
            run([py, "shelf66_v3/run_shelf66_full.py", "--out", d, "--name", NAMES["k"], "--workers", a.workers], CODE)
        elif st == "none":
            d = gen / "none_dense"
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "QUOTA.json").exists():
                (d / "QUOTA.json").write_text(json.dumps({"NONE": q.get("NONE", 0), "0": 0, "1": 0, "2": 0, "3": 0, "4": 0}))
            if q.get("NONE", 0):
                run([py, "shelf66_v3/run_shelf66_none.py", "--out", d, "--name", NAMES["none"], "--workers", a.workers], CODE)
        elif st == "trim":
            assert (gen / "none_dense/ledger.jsonl").exists(), "run --stage none first"
            run([py, "shelf66_v3/trim_none66_v3.py", "--src", gen / "none_dense", "--out", gen / "none_trim", "--cap", a.workers], CODE)


if __name__ == "__main__":
    main()
