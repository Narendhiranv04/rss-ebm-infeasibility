"""Board-height check (EXPLORATORY, VAL only): does the UNCHANGED 3D teacher give the same X / Y labels when the shelf board is lower,
and with the rig's fixed crossbar + overhead camera as an obstacle?

    python height_check.py select                          -> plans.jsonl  (75 decided successes + 75 failures, VAL, K1-K3)
    python height_check.py run TAG ZB BAR WORKERS ...      -> runs.jsonl   (BAR = none | 0.42 | 0.85); several "ZB:BAR" configs allowed:
    python height_check.py run p1 0.55:none,0.50:none 6
    python height_check.py report                          -> report.md

Teacher = evaluate._verify_job behind construct._boot, exactly as hybrid/src/hz_eval.py _run (V3.4 code, fin3d copy), with these
monkeypatches inside each worker only (no file is edited):
  * shelf_scene.SHELF66["zb"] = ZB  (board, shelf body, objects and the ready pose all move with it; robot, grasps, rules unchanged)
  * BAR != none: a crossbar (box, x = BAR +- 2 cm, y +- 0.6 m, z 0.96..1.00) and a camera + T-mount block (x = BAR +- 2 cm, y +- 6.5 cm,
    z 0.90..0.96) are added to the scene as solid static geoms: planner and physics both see them.
  * BAR == none: the same blocks for BOTH bar positions are added as GHOST geoms (contype = conaffinity = 0: nothing collides with them,
    the planner never sees them); during the executed motion the minimum distance robot -> ghost is recorded = would this unchanged plan
    hit the bar / camera.
Labels read as hz_eval._xy (3D: only resolution EVALUATED counts)."""
from __future__ import annotations

import collections
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SH = HERE.parents[1]                                             # shelf_6x6_open_sides/
ABF = SH / "approachb_3d_learning/runs/ABF3"
K3 = SH / "approachb_3d_learning/runs/K3ABL"
REMOTE = ("/home/projects/long-horizon/naren_workspace/k3abl/abf/", "/home/projects/long-horizon/abf_v2/root/")
os.environ.update(ABF_ROOT=str(ABF), ABF_SNAPSHOT=str(ABF / "snapshot/abf_v32_fb0eef4f5ff433ee"), ABF_P5DIR=str(HERE / "p5"),
                  K3=str(K3), K3_BUNDLE=str(K3 / "stage/bundle"), XY_DEV="cpu", V3OUT=str(HERE / "teach_out"),
                  FIN3D_CODE=str(ABF / "fin3d/code"), FIN3D_SRC=str(ABF / "fin3d/src/dataset_bd_classquota_without_home_position_no_overpass"),
                  FIN3D_STORE=str(HERE / "store/fin3d_store"), MUJOCO_GL="disable", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ABF / "code_v34"))
BARS = {"0.42": 0.42, "0.85": 0.85}


def local(p):
    for pre in REMOTE:
        if p.startswith(pre):
            return str(ABF / p[len(pre):])
    return p


def xy(x):                                                       # = hz_eval._xy("3d", x)
    out = []
    for k in ("x", "y"):
        r = x[k] or {}
        v = r.get(k) if r.get("resolution", "EVALUATED") == "EVALUATED" else None
        out.append(v if v in (0, 1) else None)
    return tuple(out)


def select(seed=0):
    priv = {json.loads(l)["root_id"]: json.loads(l) for l in open(K3 / "runs/public/private_val_3d.jsonl")}
    pool = collections.defaultdict(list)
    for line in open(K3 / "runs/verifier_cache/verify_3d.jsonl"):
        v = json.loads(line)
        e = priv.get(v["root"])
        if not e or e["K"] not in ("1", "2", "3"):
            continue
        for x in v["results"]:
            t = xy(x)
            if None not in t and x["seq"]:
                pool[(e["K"], t == (1, 1))].append((e, x["seq"], t))
    rng, out = random.Random(seed), []
    for K in ("1", "2", "3"):
        for succ in (True, False):
            cand = pool[(K, succ)]
            rng.shuffle(cand)
            seen, pick = set(), []
            for e, seq, t in cand:                               # distinct roots first, then fill
                if e["root_id"] not in seen and len(pick) < 25:
                    seen.add(e["root_id"]); pick.append((e, seq, t))
            for c in cand:
                if len(pick) >= 25:
                    break
                if c not in pick:
                    pick.append(c)
            out += [dict(root_id=e["root_id"], K=K, geom=e["geom"], path=local(e["path"]), ids=e["ids"], seq=seq, cached=list(t))
                    for e, seq, t in pick]
    with open(HERE / "plans.jsonl", "w") as f:
        for i, p in enumerate(out):
            f.write(json.dumps({"plan": i, **p}) + "\n")
    print(len(out), collections.Counter((p["K"], tuple(p["cached"])) for p in out))


# ------------------------------------------------------------------------------------------- worker side
_W = {}


def _init(geom):
    import construct as CO
    CO._boot("3D", geom)
    S = sys.modules["shelf_scene"]
    import bd_3dt_plus as P
    _W.update(S=S, P=P, cfg=None, conf0=S.SCENES["confined"], step0=S.XArm7TableEnv.step, acc={})

    def step(self, n=1):                                         # measure robot -> bar / ghost distance during execution
        hw = getattr(self, "_hw", None)
        if hw is None:
            import mujoco
            import numpy as np
            m = self.model
            robot = set(self._subtree(m, m.body("link_base").id))
            tg = {m.geom(g).name: g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(("ghost_", "hwbar_"))}
            hw = self._hw = dict(mj=mujoco, np=np, rg=np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in robot]), tg=tg, k=0)
        for _ in range(n):
            _W["step0"](self, 1)
            hw["k"] += 1
            if not hw["tg"] or hw["k"] % 10:
                continue
            m, d, np = self.model, self.data, hw["np"]
            rg = hw["rg"]
            hi = rg[d.geom_xpos[rg, 2] + m.geom_rbound[rg] > 0.88]   # only parts that come near the bar's height
            for name, g in hw["tg"].items():
                key = name.split("_")[1]                         # x042 / x085 / bar position tag
                for r in hi:
                    dist = hw["mj"].mj_geomDistance(m, d, int(r), int(g), 0.05, None)
                    if dist < _W["acc"].get(key, (1.0,))[0]:
                        _W["acc"][key] = (float(dist), m.body(m.geom_bodyid[r]).name)
    S.XArm7TableEnv.step = step


def _config(zb, bar):
    if _W["cfg"] == (zb, bar):
        return
    S = _W["S"]
    S.SHELF66["zb"] = zb

    def blocks(x, tag, collide):
        return [S._slab(f"{tag}_bar", x - 0.02, x + 0.02, -0.6, 0.6, 0.96, 1.00, "0.2 0.2 0.2 1", collide=collide),
                S._slab(f"{tag}_cam", x - 0.02, x + 0.02, -0.065, 0.065, 0.90, 0.96, "0.1 0.1 0.1 1", collide=collide)]

    def confined(*a, **kw):
        sc = _W["conf0"](*a, **kw)
        extra = blocks(BARS[bar], "hwbar_x" + bar.replace(".", "")[1:], True) if bar != "none" else \
            blocks(0.42, "ghost_x042", False) + blocks(0.85, "ghost_x085", False)
        sc["statics"] = list(sc["statics"]) + extra
        return sc
    S.SCENES["confined"] = confined
    _W["P"]._REACH.clear()                                       # the only module-level geometry cache (lazy reach map)
    _W["cfg"] = (zb, bar)


def _job(job):
    import evaluate as EV
    _config(job["zb"], job["bar"])
    _W["acc"] = {}
    t0 = time.time()
    r = EV._verify_job({"dom": "3D", "path": job["path"], "seqs": [[[job["ids"][o], c] for o, c in job["seq"]]]})
    x = r["results"][0]
    return dict(tag=job["tag"], plan=job["plan"], zb=job["zb"], bar=job["bar"], got=list(xy(x)), cached=job["cached"],
                x_status=(x["x"] or {}).get("status") or (x["x"] or {}).get("error", "")[:120],
                y_status=(x["y"] or {}).get("status") or (x["y"] or {}).get("error", "")[:120],
                bar_min_dist_m={k: [round(v[0], 4), v[1]] for k, v in _W["acc"].items()}, cpu_s=r["cpu_s"], wall_s=round(time.time() - t0, 1))


def run(tag, configs, workers):
    import multiprocessing as mp
    plans = [json.loads(l) for l in open(HERE / "plans.jsonl")]
    out = HERE / "runs.jsonl"
    done = set()
    if out.exists():
        done = {(r["tag"], r["plan"], r["zb"], r["bar"]) for r in map(json.loads, open(out))}
    jobs = [dict(tag=tag, zb=float(z), bar=b, **p) for c in configs.split(",") for z, b in [c.split(":")] for p in plans
            if (tag, p["plan"], float(z), b) not in done]
    assert {p["geom"] for p in plans} == {"original"}
    (HERE / "teach_out").mkdir(exist_ok=True)
    print(f"{time.strftime('%H:%M:%S')} {tag}: {len(jobs)} jobs, {workers} workers", flush=True)
    with open(out, "a") as f, mp.get_context("spawn").Pool(workers, initializer=_init, initargs=("original",), maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 25 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i}/{len(jobs)}", flush=True)


def report():
    R = [json.loads(l) for l in open(HERE / "runs.jsonl")]
    by = collections.defaultdict(list)
    for r in R:
        by[(r["zb"], r["bar"])].append(r)
    lines = ["| board z | bar x | n | agree (x,y) % | X % | Y % | S->F | F->S | undecided | bar/camera hit (min dist < 0) x=0.42 / 0.85 | closest mm x=0.42 / 0.85 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (zb, bar), rs in sorted(by.items(), key=lambda t: (-t[0][0], t[0][1])):
        n = len(rs)
        pct = lambda k: round(100 * k / n, 1)  # noqa: E731
        ag = sum(tuple(r["got"]) == tuple(r["cached"]) for r in rs)
        ax = sum(r["got"][0] == r["cached"][0] for r in rs)
        ay = sum(r["got"][1] == r["cached"][1] for r in rs)
        sf = sum(tuple(r["cached"]) == (1, 1) and tuple(r["got"]) != (1, 1) for r in rs)
        fs = sum(tuple(r["cached"]) != (1, 1) and tuple(r["got"]) == (1, 1) for r in rs)
        und = sum(None in r["got"] for r in rs)
        keys = ["x042", "x085"] if bar == "none" else ["x" + bar.replace(".", "")[1:]]
        hits = [sum(r["bar_min_dist_m"].get(k, [1])[0] < 0 for r in rs) for k in keys]
        close = [min((r["bar_min_dist_m"][k][0] for r in rs if k in r["bar_min_dist_m"]), default=None) for k in keys]
        lines.append(f"| {zb:.2f} | {bar} | {n} | {pct(ag)} | {pct(ax)} | {pct(ay)} | {sf} | {fs} | {und} | "
                     f"{' / '.join(map(str, hits))} | {' / '.join('-' if c is None else f'{c * 1000:.0f}' for c in close)} |")
    (HERE / "report.md").write_text("# height_check report (generated)\n\nEXPLORATORY, VAL split only.  closest = smallest robot -> bar/camera "
                                    "distance seen in executed motion (only parts above z 0.88 measured; '-' = never within 5 cm).\n\n"
                                    + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "select":
        select()
    elif c == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif c == "report":
        report()
