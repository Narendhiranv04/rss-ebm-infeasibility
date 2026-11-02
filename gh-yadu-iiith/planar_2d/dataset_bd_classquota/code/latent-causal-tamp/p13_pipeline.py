"""P13 orchestration: ledger (runs/p13/guards, contract caps, one guarded process per job), gates, DEV probes / pilot, plan
building, learner records, matched training, the frozen prediction snapshot and the final manifest. P10-P12 artifacts are read-only."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import config as cf

OUT = Path("runs/p13")
CAPS = {"audit": 4000.0, "dev": 10000.0, "training": 35000.0, "teacher": 65000.0, "p12_continuation": 9000.0,
        "execution": 22000.0, "analysis": 5000.0}
JOB = {"cpu_s": 4000.0, "wall_s": 12000.0}


def rd(p):
    return json.loads(Path(p).read_text())


def sha(p) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _cpu(g) -> float:
    """Guard CPU, except training jobs: the first launcher ran guards as threads of ONE parent, so teacher.guard's RUSAGE_CHILDREN delta
    absorbed other children's CPU; a training child is single-threaded (torch.set_num_threads(1)), so its CPU is bounded by its wall."""
    return min(g["cpu_s"], g["wall_s"]) if g["category"] == "training" else g["cpu_s"]


def spent(cat=None) -> float:
    gs = [rd(p) for p in (OUT / "guards").glob("*.json")]
    return sum(_cpu(g) for g in gs if cat is None or g["category"] == cat)


def job(name, call, cpu, module="p13_factors", category="audit", wall=None) -> dict:
    """Guarded child process (process-tree CPU of this child only), charged to one P13 category; refuses past the cap."""
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    cpu = min(cpu, JOB["cpu_s"], CAPS[category] - spent(category))
    if cpu < 5:
        raise SystemExit(f"{category} budget exhausted")
    g = teacher.guard([sys.executable, "-c", f"import torch; torch.set_num_threads(1); import {module}; {module}.{call}"], cpu,
                      wall or JOB["wall_s"], OUT / f"heartbeat_{name}.txt", 15.0, 0.25, Path("."))
    n = 1
    while (OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | \
          {"cpu_cap_s": cpu, "call": f"{module}.{call}", "category": category, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cf.write_json(OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (OUT / f"heartbeat_{name}.txt").unlink(missing_ok=True)
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {category: round(spent(category), 1)}))
    return rec


def gate(stage, **kw) -> None:
    cf.write_json(OUT / f"P13_{stage}_GATE.json", {"stage": f"P13-{stage}", **kw, "spend": {c: round(spent(c), 1) for c in CAPS}})


# ------------------------------------------------------------------ P13-2: construction (dev seeds only until the freeze)
DEV_SEED0 = 2026150000


def probe(n=30, fams=None) -> dict:
    """Static constructibility per P13 family on DEV seeds: acceptance, rejection reasons, CPU per attempt (no teacher call)."""
    import collections
    import config as cf
    import geometry as ge
    import p10_scenes as ps
    import p13_scenes as s13
    cfg = cf.load_config("config.toml")
    cx, out = ps.Ctx(cfg, ge.make_robot(cfg)), {}
    for i, fam in enumerate(fams or s13.FAMS):
        why, c0 = collections.Counter(), time.process_time()
        for a in range(n):
            rec, w = s13.try_root(cx, {"family": fam, "seed": DEV_SEED0 + 97 * i, "index": -1, "wave": 0}, a, set())
            why["ACCEPTED" if rec else w] += 1
        out[fam] = {"attempts": n, "accept_rate": why["ACCEPTED"] / n, "cpu_per_attempt": round((time.process_time() - c0) / n, 2), "reasons": dict(why)}
        print(fam, out[fam], flush=True)
    (OUT / "p13_2").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "p13_2" / "PROBE.json", out)
    return out


def _known_all() -> set:
    """Geometry / fingerprint dedup set: P10a / P10b (p10b_dataset._known) + every P11 / P12 / P13 plan written so far."""
    import g4c
    import p10b_dataset as pd
    from scene import scene_from_dict
    known = _KNOWN0()
    for f in sorted(Path("runs/p11").glob("**/plan.json")) + sorted(Path("runs/p12").glob("**/plan.json")) + sorted(OUT.glob("**/plan.json")):
        for r in rd(f)["roots"].values():
            known.add(g4c.stable_geometry(scene_from_dict(r["scene"]), r["goal_cell"]))
            known.add(tuple(map(str, r["fingerprint"])))
    return known


def _register():
    import p10b_dataset as pd
    import p13_scenes  # noqa: F401  (family dispatch)
    global _KNOWN0
    if pd._known is not _known_all:
        _KNOWN0, pd._known = pd._known, _known_all
    if TEACHER_ON:
        pd.POLICY = {**pd.POLICY, **TEACHER}                    # recorded in every plan built afterwards (execution_policy)
    return pd


DEV_FAMS = ("PC_SERIAL", "PC_SG", "PR_SIDES", "PN_SIMP", "PN_GIMP", "ORD_CHAIN", "OMIT_OPT", "TRI_SERIAL", "TRI_OR", "NAT")


def dev_pilot(half) -> None:
    """DEV teacher pilot: 10 roots (one per family), two halves run as separate guarded processes; dev split, never a roster seed."""
    pd = _register()
    d = OUT / "dev" / f"half{half}"
    specs = [{"family": f, "seed": DEV_SEED0 + 1000 + i, "index": i, "wave": 0, "batch": 0, "split": "dev"} for i, f in enumerate(DEV_FAMS)]
    specs = specs[5 * half: 5 * half + 5]
    if not (d / "plan.json").exists():
        pd.build_plan(d, specs, max_attempts=400, root_cpu=240.0)
    if not (d / "static_admission.json").exists():
        pd.admit(d)
    pd.collect(d, chunk_cpu=4400.0)


# ------------------------------------------------------------------ P13-3: learner records, matched training
MODELS = OUT / "models"


def _sealed(d):
    """Teacher outcomes of a P13 batch directory (blind outcomes only after runs/p13/holdout/UNSEALED)."""
    import p6_run as p6r
    import operational_teacher as ot
    d = Path(d)
    plan = rd(d / "plan.json")
    if any(r["split"] == "blind" for r in plan["roots"].values()) and not (OUT / "holdout" / "UNSEALED").exists():
        raise SystemExit("P13 blind labels are sealed")
    static, pol = rd(d / "static_admission.json")["certificates"], ot.load_policy("teacher_opneg.toml")
    res = {}
    for k, r in plan["roots"].items():
        f = d / ("sealed" if r["split"] == "blind" else "progress") / f"{k}.json"
        if not f.exists():
            continue
        done = rd(f)["done"]
        for q in plan["queries"]:
            if q["root"] == k and q["query"] in done:
                j = p6r.judge(static, pol, q, done[q["query"]])
                res[q["query"]] = {**j, "query": q, "root": k, "split": r["split"], "family": r["family"],
                                   "static_certificate": static.get(q["query"])}
    return res


def records(dirs, split, out_name) -> None:
    """Learner records for P13 plan directories (every pool-subset edit is a scene slot), appended to the P10b trainval records
    for the training set; unlabelled for blind splits."""
    import os
    import torch
    os.environ["P10C_OUT"] = str(MODELS)
    import learn_data as ld
    import p10b_dataset as pd
    import p10c_train_eval as pc
    cfg, dino, recs = cf.load_config("config.toml"), ld.dino(), []
    if out_name == "trainval":
        recs = torch.load("runs/p10c/dataset_trainval.pt", weights_only=False)["roots"]
    for d in dirs:
        plan = rd(Path(d) / "plan.json")
        lab = {} if split == "blind" else pc._labels({q: x for q, x in _sealed(d).items()})
        dl = None if split == "blind" else pd.d_labels(plan, {q: x for q, x in _sealed(d).items()})
        drop = set(sum((v for k_, v in rd(OUT / "DEDUP.json").items() if k_ != "rule"), []))
        for k, r in plan["roots"].items():
            if k in drop:
                continue
            ed = [it for it in r["items"] if it["kind"] == "SCENE_TARGET_ONLY" and it.get("d_subset")]
            sq = [it for it in r["items"] if it["kind"] == "ORDERED_TEACHER"]
            rec, _ = pc._record(cfg, dino, k, r, lab, r["split"], "P13", ed, sq)
            ids = rec["audit"]["ids"]
            rec["audit"] |= {"family": r["family"], "variant": r["variant"]}
            rec["d"] = {"pool": [ids.index(o) for o in r["parking"]], "roles": [{v: kk for kk, v in r["ids"].items()}[o] for o in r["parking"]]}
            if dl is not None:
                x = dl[k]
                rec["d"] |= {"y0": x["y0"], "mask": x["mask"], "type": x["type"], "provenance": x["provenance"], "sets": x["sets"]}
            recs.append(rec)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"version": "P13_RECORDS_V1", "roots": recs}, MODELS / f"dataset_{out_name}.pt")


def _arms():
    """P10c arms plus P13 controls: parameter-matched off arms (scene head widened to restore the removed parameter count) and
    F1-off (lambda_static = 0). Widths are fixed here and verified by parameter count in the gate."""
    import abc7_models as a7
    import abc8
    import abc_models as am
    import p10c_train_eval as pc

    def wide(m, h):
        m.scene = am.mlp(2 * am.W, 1, h=h).double()
        return m
    pc.ARMS.update({"B-F3pm": (lambda: wide(a7.ModelBC7("B", "full", drop=("pair_scene",)).double(), PM["B-F3pm"]), "unique", True),
                    "B-F5spm": (lambda: wide(abc8.build("B-structured"), PM["B-F5spm"]), "unique", True),
                    "B-F5qpm": (lambda: wide(abc8.build("B-quadratic"), PM["B-F5qpm"]), "unique", True),
                    "B-F1off": (lambda: abc8.build("B-full"), "unique", True)})
    return pc


PM = {"B-F3pm": 320, "B-F5spm": 641, "B-F5qpm": 448}       # scene-head widths: (2W + 2) h + 1 restores the removed parameters


def _skipped(name) -> bool:
    f = MODELS / "SKIP.json"
    return f.exists() and name in rd(f)["skip"]


def train(arm, seed) -> None:
    import functools
    import os
    if _skipped(f"P13-{arm}-s{seed}"):
        return
    os.environ["P10C_OUT"] = str(MODELS)
    pc = _arms()
    import unique_factor as uf
    if arm == "B-F1off":
        uf.total = functools.partial(uf.total, lam_static=0.0)
    pc.train_job(f"P13-{arm}-s{seed}", arm, seed, "full")


def train_d(pair, seed, lam) -> None:
    import os
    os.environ["P10C_OUT"] = str(MODELS)
    import p10c_train_eval as pc
    pc.train_d(f"P13-{'D' if pair else 'D-pairless'}-l{lam}-s{seed}", pair, seed, lam)


# ------------------------------------------------------------------ P13-3: frozen predictions (validation-only calibration, 0 physical calls)
HOLD = OUT / "holdout"


def _prepared(name):
    import os
    import torch
    os.environ["P10C_OUT"] = str(MODELS)
    import abc7_train as t7
    ds = torch.load(MODELS / f"dataset_{name}.pt", weights_only=False)
    rs = t7.prepare(ds)
    for r, rec in zip(rs, ds["roots"]):
        r["ids"], r["audit"], r["d"] = rec["audit"]["ids"], rec["audit"], rec.get("d")
    return rs


def _plans():
    import p10c_train_eval as pc
    P = {k: r for b in pc.batches() for k, r in rd(b / "plan.json")["roots"].items()}
    for f in sorted(OUT.glob("data/*/plan.json")) + sorted(HOLD.glob("*/plan.json")):
        P |= rd(f)["roots"]
    return P


def _freeze_ctx():
    import os
    os.environ["P10C_OUT"], os.environ["P10D_OUT"] = str(MODELS), str(HOLD)
    if (HOLD / "PREDICTIONS_FROZEN.json").exists() or any(HOLD.glob("*/static_admission.json")):
        raise SystemExit("test labels exist or predictions already frozen")
    import p10_protocol as pp
    import p9_l3 as l3
    from scene import scene_from_dict
    cfg, P = cf.load_config("config.toml"), _plans()
    sets = {"validation": [r for r in _prepared("trainval") if r["split"] == "validation"], "test": _prepared("test")}
    veto = {(sp, r["key"]): frozenset(c["query"] for c in r["cands"] if c["seq"] and pp.static_veto(
        scene_from_dict(P[r["key"]]["scene"]), [(r["ids"][i], l3.cell(d)) for i, d in c["seq"]], P[r["key"]]["goal_cell"], cfg))
        for sp, rs in sets.items() for r in rs}
    return cfg, P, sets, veto


def freeze_arm(name) -> None:
    """One checkpoint: temperature + threshold on validation; per test root the root-prior marginals, keyed P_S joint (B-family), F1
    static P(valid) on analytic static checks, TOP1 rows and the choice (p10_protocol.top1 rule: admission, analytic veto, then
    (moves, J_pred, C_d, energy, id)). Physical calls are counted and must be zero."""
    import os
    os.environ["P10C_OUT"], os.environ["P10D_OUT"] = str(MODELS), str(HOLD)
    import contextlib
    import copy
    import math
    import torch
    import p10_protocol as pp
    import p10d_execute as px
    import p7_eval
    import p9_l3 as l3
    import p9_l3c as l3c
    import unique_factor as uf
    import unique_factor_keyed as ufk
    cfg, P, sets, veto = _freeze_ctx()
    pc = _arms()
    calls = pp.instrument()
    x = rd(MODELS / "runs" / f"{name}.json")
    m, keyed = pc.load(x["name"])[0], pc.ARMS[x["arm"]][2]
    ctx = ufk.installed if keyed else contextlib.nullcontext
    v0 = (lambda rs: pc._v0(copy.deepcopy(rs))) if x["arm"] == "B-V0" else (lambda rs: rs)
    mode = "factorized" if x["arm"] in ("A", "SF") else "unique"
    with ctx():
        vev = [l3.root_eval9(m, r, mode)["root"] for r in v0(sets["validation"])]
    tau = p7_eval.fit_temperature(vev)
    thr = px.choose_threshold(vev, tau)
    per = {}
    for r in v0(sets["test"]):
        l3c.check_identity(r)
        with ctx(), torch.no_grad():
            if m.kind in ("B", "C"):
                inst = uf.instances(m, r)
                row = {"marginals": {q: float(v) for q, v in uf.prior_marginals(inst).items()},
                       "joint_S": {"queries": inst["S"]["queries"], "logp": [float(v) for v in inst["S"]["logp"]]}}
                rows = l3c.deployed(m, r, inst, tau, thr)[1]
            else:
                import p10_eval as pe
                row = {"marginals": pe.marginals(m, r, False)}
                rows = l3.root_eval9(m, r, "factorized")["root"]["cands"]
            st = []
            for c in r["static_checks"]:
                xy = r["R"].xy0.clone()
                for i, dd in c["seq"]:
                    xy[i] = torch.as_tensor(dd, dtype=torch.float64)
                st.append({"p_valid": float(np.exp(-float(m.static(r["R"], xy)[0]))),
                           "valid": int(all(v != 0 for v in c["su"] + c["sp"] if v is not None))})
        if not all(math.isfinite(y[k]) for y in rows for k in ("u", "J_pred", "C_d", "energy")):
            raise FloatingPointError(f"{r['key']}: nonfinite score")
        keep = [y for y in rows if y["query"] not in veto[("test", r["key"])] and 1 / (1 + math.exp(-y["u"] / tau)) >= thr]
        best = min(keep, key=lambda y: (y["m"], y["J_pred"], y["C_d"], y["energy"], y["query"])) if keep else None
        per[r["key"]] = row | {"static": st, "top1": best["query"] if best else "ABSTAIN", "admitted": len(keep),
                               "rows": [{k: y[k] for k in ("query", "u", "m", "J_pred", "C_d", "energy")} for y in rows]}
    if sum(calls.values()):
        raise RuntimeError(f"physical call during learned prediction: {dict(calls)}")
    (HOLD / "parts").mkdir(parents=True, exist_ok=True)
    cf.write_json(HOLD / "parts" / f"{name}.json", {"arm": x["arm"], "seed": x["seed"], "tau": tau, "threshold": thr, "roots": per,
                                                    "checkpoint_sha256": x["checkpoint_sha256"], "physical_calls": dict(calls)})


def freeze_merge() -> None:
    """D (lambda_B and threshold by validation), geometry heuristic and geometry TOP1, merge of the per-checkpoint parts, hash."""
    import os
    os.environ["P10C_OUT"], os.environ["P10D_OUT"] = str(MODELS), str(HOLD)
    import torch
    import d_blockers as db
    import p10_protocol as pp
    import p10d_execute as px
    cfg, P, sets, veto = _freeze_ctx()
    calls = pp.instrument()
    snap = {"arms": {}, "d": {}, "baselines": {}}
    names = sorted(f.stem for f in (MODELS / "runs").glob("P13-*.json") if rd(f)["arm"] not in ("D", "D-pairless"))
    for n in names:
        part = rd(HOLD / "parts" / f"{n}.json")
        if sum(part["physical_calls"].values()):
            raise RuntimeError(f"{n}: physical calls")
        snap["arms"][n] = {k: v for k, v in part.items() if k != "physical_calls"}
    dsel = {}
    for f in sorted((MODELS / "runs").glob("P13-D*.json")):
        x = rd(f)
        m = db.ModelD(pair=x["pair"]).double()
        m.load_state_dict(torch.load(MODELS / "ckpt" / f"{x['name']}.pt", weights_only=True))
        m.eval()
        snap["d"][x["name"]] = {k: v for k, v in x.items() if k in ("arm", "seed", "lambda_B", "val_nll", "checkpoint_sha256")}
        snap["d"][x["name"]]["roots"] = {sp: {r["key"]: db.predict(m, r["R"], (r.get("d") or {}).get("pool", [])) for r in rs} for sp, rs in sets.items()}
        dsel.setdefault((x["arm"], x["lambda_B"]), []).append(x["val_nll"])
    for arm in ("D", "D-pairless"):
        lam = min((float(np.mean(v)), k[1]) for k, v in dsel.items() if k[0] == arm)[1]
        snap["baselines"][f"{arm}_lambda_B"] = lam
        snap["baselines"][f"{arm}_threshold"] = px.d_threshold(snap, sets["validation"], arm, lam)
    for r in sets["test"]:
        snap["baselines"].setdefault("heuristic", {})[r["key"]] = db.heuristic(r["R"], (r.get("d") or {}).get("pool", []))
        snap["baselines"].setdefault("geometry_top1", {})[r["key"]] = px.geometry_choice(P[r["key"]], r, cfg)
    for d in snap["d"].values():
        d["roots"] = d["roots"]["test"]
    if sum(calls.values()):
        raise RuntimeError(f"physical call during D / baselines: {dict(calls)}")
    snap |= {"version": "P13_PREDICTIONS_V1", "prereg": sha(OUT / "PREREG.json"), "physical_calls": dict(calls),
             "datasets": {n: sha(MODELS / f"dataset_{n}.pt") for n in ("trainval", "test")}, "veto_fired": sum(len(v) for v in veto.values()),
             "checkpoints": {n: snap["arms"][n]["checkpoint_sha256"] for n in names}}
    snap["hash"] = cf.stable_hash(snap)
    cf.write_json(HOLD / "PREDICTIONS_FROZEN.json", snap)


TEACHER = {"version": "P13_S100_V2", "change_vs_P10B_S100_V1": "a censored seed no longer ends the query: the remaining declared seeds run "
           "(still 100 CPU-s each, 330 CPU-s per query, 900 CPU-s per root); adjudication unchanged (OPNEG-v1)", "stop_after_censored_seed": False}


TEACHER_ON = True                                             # dev pilot 1 ran under P10B_S100_V1 (before this switch existed)


def _collect(d, chunk_cpu):
    pd = _register()
    pd.collect(d, chunk_cpu=chunk_cpu)


def dev_pilot2() -> None:
    """DEV pilot 2 (P13_S100_V2 policy) for the censoring-prone families: PC_SERIAL x2, TRI_CTRL x2, TRI_OR, PN_GIMP; new dev seeds."""
    pd = _register()
    d = OUT / "dev" / "pilot2"
    fams = ("PC_SERIAL", "PC_SERIAL", "TRI_CTRL", "TRI_CTRL", "TRI_OR", "PN_GIMP")
    specs = [{"family": f, "seed": DEV_SEED0 + 2000 + i, "index": i, "wave": 0, "batch": 0, "split": "dev"} for i, f in enumerate(fams)]
    if not (d / "plan.json").exists():
        pd.build_plan(d, specs, max_attempts=400, root_cpu=240.0)
    if not (d / "static_admission.json").exists():
        pd.admit(d)
    _collect(d, 5600.0)


# ------------------------------------------------------------------ P13-3: roster (fixed before any train / test label)
SEED0 = {"train": 2026160000, "validation": 2026170000, "blind": 2026180000, "natural": 2026190000}
COUNTS = {"PC_SERIAL": {"train": 16, "validation": 4, "blind": 16}, "PC_SG": {"train": 16, "validation": 4, "blind": 16},
          "PR_SIDES": {"train": 16, "validation": 4, "blind": 14}, "PN_SIMP": {"train": 16, "validation": 4, "blind": 12},
          "PN_GIMP": {"train": 16, "validation": 4, "blind": 12}, "ORD_CHAIN": {"train": 16, "validation": 4, "blind": 14},
          "OMIT_OPT": {"train": 16, "validation": 4, "blind": 14}, "TRI_SERIAL": {"train": 12, "validation": 3, "blind": 12},
          "TRI_CTRL": {"train": 12, "validation": 3, "blind": 12}, "NAT": {"train": 24, "validation": 6, "natural": 48}}
# TRI_OR is NOT run (0 / 2 dev lattices complete at the P13 teacher budget): Delta_ABC = -1 stays NOT_EVALUATED.


def roster() -> list:
    """Interleaved families per split, batches of <= 16 roots; 'natural' roots are the NAT stream held out as a separate blind test."""
    out = []
    for split in ("train", "validation", "blind", "natural"):
        fams = [f for f, c in COUNTS.items() for _ in range(c.get(split, 0))]
        order = sorted(range(len(fams)), key=lambda i: (sum(1 for j in range(i) if fams[j] == fams[i]), fams[i]))
        for n, i in enumerate(order):
            out.append({"family": fams[i], "seed": SEED0[split] + n, "index": n, "wave": 0, "batch": 1 + n // 16,
                        "split": "blind" if split == "natural" else split, "stream": split})
    return out


def batch_dirs(stream):
    base = HOLD if stream in ("blind", "natural") else OUT / "data"
    return sorted(base.glob(f"{stream}_b*"))


def build(stream, b) -> None:
    pd = _register()
    d = (HOLD if stream in ("blind", "natural") else OUT / "data") / f"{stream}_b{b}"
    if not (d / "plan.json").exists():
        pd.build_plan(d, [s for s in roster() if s["stream"] == stream and s["batch"] == b], max_attempts=600, root_cpu=300.0)


def label(stream, b) -> None:
    """Static admission + teacher collection of one batch. Blind batches only after the frozen predictions are on origin."""
    pd = _register()
    d = (HOLD if stream in ("blind", "natural") else OUT / "data") / f"{stream}_b{b}"
    if stream in ("blind", "natural") and not (HOLD / "FREEZE_PUSHED").exists():
        raise SystemExit("blind labels before the pushed freeze are forbidden")
    if not (d / "static_admission.json").exists():
        pd.admit(d)
    _collect(d, 12000.0)


def _on_origin(path) -> bool:
    import subprocess
    subprocess.run(["git", "fetch", "-q", "origin"], check=True)
    r = subprocess.run(["git", "show", f"origin/main:{path}"], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(path).read_bytes()


def mark_freeze_pushed() -> None:
    """Gate for blind labelling: PREREG + frozen predictions byte-identical on origin/main."""
    for p in (OUT / "PREREG.json", HOLD / "PREDICTIONS_FROZEN.json"):
        if not _on_origin(str(p)):
            raise SystemExit(f"{p} is not on origin/main")
    (HOLD / "FREEZE_PUSHED").write_text(sha(HOLD / "PREDICTIONS_FROZEN.json") + "\n")


def label_manifest() -> None:
    """Hash of every sealed blind outcome file (committed and pushed BEFORE unsealing)."""
    files = sorted(str(f) for f in HOLD.glob("*/sealed/*.json"))
    cf.write_json(HOLD / "LABELS_MANIFEST.json", {"files": {f: sha(f) for f in files}, "n": len(files)})


def unseal() -> None:
    if not _on_origin(str(HOLD / "LABELS_MANIFEST.json")):
        raise SystemExit("label manifest not on origin/main")
    man = rd(HOLD / "LABELS_MANIFEST.json")
    if any(sha(f) != h for f, h in man["files"].items()):
        raise SystemExit("sealed outcome changed after the manifest")
    (HOLD / "UNSEALED").write_text(sha(HOLD / "LABELS_MANIFEST.json") + "\n")


ARMS = ("K-B", "K-C", "A", "SF", "B-F2", "B-F3", "B-F3pm", "B-F4", "B-F5s", "B-F5spm", "B-F5q", "B-F5qpm", "B-F1off", "B-V0", "B-U")


def write_prereg() -> None:
    """P13 preregistration: written and pushed BEFORE any train / validation / test teacher label of the roster."""
    import p13_scenes as s13
    if (OUT / "PREREG.json").exists():
        raise SystemExit("PREREG already frozen")
    pre = {
        "version": "P13_PREREG_V1", "start_commit": "021cd00461bfb346053c991303385a9e467769f5",
        "generator": {"module": "p13_scenes.py", "version": s13.VERSION, "sha256": sha("p13_scenes.py"), "families": COUNTS,
                      "not_run": {"TRI_OR": "0 / 2 DEV lattices complete at the teacher budget (Delta_ABC = -1 NOT_EVALUATED)"},
                      "conditioning": "mechanism-conditioned case-control: sound-certificate / pose-screen INTENT predicates per family (recorded "
                                      "per root, never a learner input); NAT = unconditioned natural stream (no prescreen)",
                      "dev_revisions": ["V1: near-start barrier rule after the first probe (DEV)", "V1.1: PC_SERIAL prediction-only after DEV pilot 2"]},
        "roster": {"rule": "p13_pipeline.roster(): seeds 2026160000+ (train), 2026170000+ (validation), 2026180000+ (blind challenge), "
                           "2026190000+ (natural blind); families interleaved; batches of <= 16", "n": len(roster())},
        "dedup": "stable geometry + fingerprint against P10a / P10b / P11 / P12 / every earlier P13 plan; parent lineage = root (no derived roots)",
        "teacher": {"policy": TEACHER, "adjudication": "unchanged OPNEG-v1; VERIFIED / CERTIFIED (scoped) / BUDGET (operational) / UNKNOWN / NOT_RUN"},
        "catalogue": "family tokens (p13_scenes.TOK); every scene edit = a D parking subset of the pool (pairs: A, B, A+B; triples: all 7)",
        "training": {"data": "P10c trainval records + P13 train / validation", "arms": ARMS, "seeds": [0, 1, 2], "updates": 400, "batch": 12,
                     "selection": "lowest validation root NLL every 20 updates (P10b + P13 validation)", "lambda_static": "1 (0 for B-F1off)",
                     "parameter_matched": {k: f"scene head hidden width {v}" for k, v in PM.items()},
                     "D": "D / D-pairless, lambda_B in {0.5, 1, 2} x 3 seeds, 600 updates; lambda_B by mean validation NLL"},
        "calibration": "per checkpoint: temperature (p7_eval.fit_temperature) + admission threshold (P8 rule) on validation only; no test refit",
        "primary": {
            "family": "Holm over exactly these 4 tests, alpha 0.05, two-sided exact sign test (ties removed), unit = independent root, "
                      "paired root bootstrap 95 % CI (2,000 resamples, seed 0)",
            "H_F3": "B vs B-F3pm: per-root joint four-bit log-loss -log P_S(z0, zS[A], zS[B], zS[A+B] = observed), mean over 3 seeds; cohort = "
                    "proof-grounded pair four-tuples of the BLIND CHALLENGE set (all kappa classes)",
            "H_F5": "B vs B-F5spm: same endpoint and cohort",
            "H_F2": "B vs B-F2: per-root mean accuracy P(good) > P(bad) over proof-grounded same-endpoint reversed-order pairs (seed-mean marginals)",
            "H_F4": "B vs B-F4: per-root mean omission log-loss over proof-grounded omissions of VERIFIED >= 2-move repairs (seed-mean marginals)"},
        "minimal_meaningful_effect": {"joint_nll": 0.05, "order_accuracy": 0.10, "omission_logloss": 0.05, "kappa_mae_kappa_plus1": 0.10,
                                      "static_auprc": 0.05, "travel_regret_fraction": 0.05},
        "verdict_rule": "SUPPORTED_CONDITIONAL only if Holm p < 0.05, mean effect >= the minimal effect, 95 % CI excludes 0, AND no significant harm "
                        "(raw p < 0.05 in the harmful direction) on (i) kappa = 0 false-interaction rate (F3 / F5), (ii) the natural-mix version of "
                        "the same endpoint; else NO_DEMONSTRATED_BENEFIT / HARMFUL (significant reversal) / UNDERPOWERED (< 12 roots)",
        "safety_and_secondary": [
            "kappa MAE (seed-mean estimand) on kappa = +1 roots and false-interaction rate (|khat| >= 0.5) on kappa = 0 roots; constant-zero control",
            "same comparisons with the plain off arms (B-F3, B-F5s, B-F5q) - capacity attribution",
            "F5q: B vs B-F5q(pm) on |Delta_hat_ABC - Delta_ABC| and joint 8-bit log-loss over complete proof-grounded triple lattices (>= 12 required)",
            "F1: static-validity AUROC / AUPRC on held-out analytic static checks, B vs B-F1off; deployment path reported (veto / tie counts)",
            "J: travel regret among admitted, VERIFIED, equal-move alternatives with witnessed travel (B vs no-J vs shuffled-J); success non-inferiority",
            "vision B vs B-V0; keying B vs B-U; A / C / SF on every endpoint; natural-mix replication of every primary",
            "strict TOP1 executed success (frozen choices, no fallback) and G+M separately",
            "D / D-pairless / geometry: initial feasibility (natural set has feasible originals) and blocker union F1 / exact on exact lattices; "
            "A / B / C / SF attribution on the same lattices with NO_PREDICTED_CLEARING_SET scored as a miss"],
        "secondary_multiplicity": "Holm within each secondary family; never promoted to confirmed",
        "stopping": "fixed roster and budget; no interim significance look; unavailable / censored roots are reported, never replaced",
        "execution": {"protocol": "strict TOP1_COMMIT, one episode per distinct (root, repair), 180 CPU-s / 600 s, no fallback, SAFE_RETREAT; "
                                  "G+M planner-assisted <= 5 candidates reported separately",
                      "order": "jobs in one frozen random root permutation (seed 13) across all executable blind roots, interleaving families, so "
                               "coverage under the cap is outcome-blind and not batch-prefix biased", "cap_cpu_s": CAPS["execution"]},
        "p12_continuation": "frozen P12 jobs only, frozen order, cap 9,000 CPU-s; supplementary, never P13 confirmation",
        "caps": CAPS}
    pre["hash"] = cf.stable_hash(pre)
    cf.write_json(OUT / "PREREG.json", pre)


def train_all(parallel=14) -> None:
    """All matched jobs not skipped (SKIP.json) and not done, each launched as its OWN OS process running job() (one guard per parent
    process, so RUSAGE_CHILDREN never mixes jobs), at most `parallel` at a time."""
    import subprocess
    todo = [(f"p13_3_train_{a}_s{s}", f"train('{a}', {s})", 2500) for s in (0, 1, 2) for a in ARMS
            if not (MODELS / "runs" / f"P13-{a}-s{s}.json").exists() and not _skipped(f"P13-{a}-s{s}")]
    todo += [(f"p13_3_train_D{int(pr)}_l{lam}_s{s}", f"train_d({pr}, {s}, {lam})", 300) for pr in (True, False) for lam in (0.5, 1.0, 2.0)
             for s in (0, 1, 2) if not (MODELS / "runs" / f"P13-{'D' if pr else 'D-pairless'}-l{lam}-s{s}.json").exists()]
    run = []
    for name, call, cpu in todo:
        while sum(p.poll() is None for p in run) >= parallel:
            time.sleep(5)
        run.append(subprocess.Popen([sys.executable, "-c", f"import p13_pipeline as p; p.job({name!r}, {call!r}, {cpu}, module='p13_pipeline', "
                                     f"category='training', wall=40000)"]))
    for p in run:
        p.wait()


def regression() -> None:
    """Historical suite + P12 / P13 tests + six replays (fixture planner calls itemized by countcalls), outputs under runs/p13/regression."""
    import subprocess
    d = OUT / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "2500", "--wall", "5000"], capture_output=True, text=True)


def manifest() -> None:
    """P13-7 final manifest: sources, prereg / addendum, frozen predictions, labels, execution plan, checkpoints, figures, regression, ledger."""
    import hashlib
    import subprocess
    reg = rd(OUT / "regression" / "regression_result.json")
    files = sorted(set([str(OUT / "PREREG.json"), str(OUT / "PREREG_ADDENDUM_1.json"), str(OUT / "DEDUP.json"), str(HOLD / "PREDICTIONS_FROZEN.json"),
                        str(HOLD / "LABELS_MANIFEST.json"), str(HOLD / "UNSEALED"), str(HOLD / "FREEZE_PUSHED"), str(OUT / "p13_5" / "EXEC_JOBS.json"),
                        str(OUT / "p13_5" / "EXEC_RESULTS.json"), str(OUT / "p13_5" / "P12_FINAL80.json"), str(OUT / "p13_4" / "TARGETED.json"),
                        str(OUT / "p13_4" / "F1_J.json"), str(OUT / "p13_6" / "D13.json"), str(OUT / "regression" / "regression_result.json")]
                       + [str(p) for p in sorted(OUT.glob("p13_[0-2]/*.json"))] + [str(p) for p in sorted(OUT.glob("P13_*_GATE.json"))]))
    out = {"stage": "P13", "start": "021cd00461bfb346053c991303385a9e467769f5",
           "commits": {"p13_0": "719493e", "p13_1": "efc6854", "p13_2_prereg": "a551db6", "plans_dedup": "ac6b4e3", "freeze": "c9756294beed243b16210fde29519de5eb7db395",
                       "labels_hash": "4da8ca297072bda01a21a131d047c01f8472d8f9"},
           "status": "PARTIAL (engineering PASS)", "gates": {g.stem: rd(g).get("engineering") for g in sorted(OUT.glob("P13_*_GATE.json"))},
           "regression": {"tests": reg.get("tests"), "replays": reg.get("replays")},
           "spend_cpu_s": {c: round(spent(c), 1) for c in CAPS}, "caps": CAPS,
           "sources": {p: sha(p) for p in ("p13_scenes.py", "p13_pipeline.py", "p13_factors.py", "p13_execution.py", "tests/test_p13.py", "p10b_dataset.py")},
           "files": {p: sha(p) for p in files if Path(p).exists()},
           "checkpoints": {str(p): sha(p) for p in sorted((MODELS / "ckpt").glob("*.pt"))},
           "datasets": {str(p): sha(p) for p in sorted(MODELS.glob("dataset_*.pt"))},
           "figures": {str(p): sha(p) for p in sorted(Path("visualizations/p13").glob("*")) if p.suffix != ".sha256"},
           "head_at_manifest": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    out["hash"] = "sha256:" + hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    cf.write_json(OUT / "P13_FINAL_MANIFEST.json", out)
