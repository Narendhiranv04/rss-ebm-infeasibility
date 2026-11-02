"""P11: 2D scientific closure. Ledger (runs/p11/guards, contract caps), read-only source-to-result audit (P11-0), collision-free H9
two-palette evaluation (P11-1), F1 / J / ablation component identification (P11-2), short-trained physical execution (P11-3), D versus
A/B/C comparison (P11-5). Every P10 artifact is read-only; P11 outputs are new files under runs/p11/. No teacher / IK / planner call
in any learned path (instrumented); physical episodes only for one committed choice."""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import config as cf

OUT = Path("runs/p11")
CAPS = {"audit_eval": 4000.0, "f1_training": 6000.0, "short_episodes": 5000.0, "newcon_dev": 1200.0, "newholdout_teacher": 8000.0,
        "newholdout_exec": 5000.0, "regression": 1700.0}
JOB = {"cpu_s": 3000.0, "wall_s": 9000.0}


def _rd(p):
    return json.loads(Path(p).read_text())


def spent(cat=None) -> tuple:
    gs = [_rd(p) for p in (OUT / "guards").glob("*.json")]
    gs = [g for g in gs if cat is None or g["category"] == cat]
    return sum(g["cpu_s"] for g in gs), sum(g["wall_s"] for g in gs)


def job(name, call, cpu, module="p11_closure", category="audit_eval", wall=None) -> dict:
    """Process-tree guarded job, ONE process per job, charged to one P11 category; refuses to start past the category cap."""
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    left = CAPS[category] - spent(category)[0]
    cpu = min(cpu, JOB["cpu_s"], left)
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
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {category: round(spent(category)[0], 1)}))
    return rec


def sha(p) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def boot(d, n=2000, seed=0):
    if len(d) < 2:
        return [None, None]
    rng, a = np.random.default_rng(seed), np.asarray(d, float)
    ms = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def sign_p(d):
    from scipy.stats import binomtest
    n, k = sum(x != 0 for x in d), sum(x > 0 for x in d)
    return float(binomtest(k, n).pvalue) if n else 1.0


def paired(per, a, b):
    ks = sorted(set(per.get(a, {})) & set(per.get(b, {})))
    d = [per[a][k] - per[b][k] for k in ks]
    return {"roots": len(d), "mean": float(np.mean(d)) if d else None, "ci95": boot(d), "sign_p": sign_p(d),
            "a_better": sum(x > 0 for x in d), "b_better": sum(x < 0 for x in d)}


def holm(ps: dict) -> dict:
    out, run = {}, 0.0
    for i, k in enumerate(sorted(ps, key=ps.get)):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = {"p": ps[k], "holm_p": run, "reject_0.05": run < 0.05}
    return out


# ------------------------------------------------------------------ P11-0: read-only source-to-result audit
def audit() -> None:
    import subprocess
    import torch
    import p10b_dataset as pd
    import p10c_train_eval as pc
    import p10d_execute as px
    sh = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    A = {"head": sh("rev-parse", "HEAD"), "origin": sh("rev-parse", "origin/main"), "tracked_changes": sh("status", "--short", "--untracked-files=no").splitlines()}
    man = _rd("runs/p10b/SPLIT_MANIFEST.json")
    roots = man["roots"]
    A["split_counts"] = dict(collections.Counter(v["split"] for v in roots.values()))
    A["split_manifest_hash_ok"] = cf.stable_hash({k: v for k, v in man.items() if k != "hash"}) == man["hash"]
    A["sealed_files_unchanged"] = all(sha(f) == h for f, h in man["sealed_file_sha256"].items())
    P = px.plans()
    A["blind_constructor_versions"] = {f"{a}|{b}": n for (a, b), n in collections.Counter((P[k]["family"], P[k]["construction"].split(" + ")[-1][:40]) for k in P if P[k]["split"] == "blind" and P[k]["family"] == "H3").items()}
    A["h3_v1_roots_by_split"] = dict(collections.Counter(P[k]["split"] for k in P if P[k]["family"] == "H3" and "V1.1" not in P[k]["construction"]))
    tv = torch.load("runs/p10c/dataset_trainval.pt", weights_only=False)["roots"]
    A["trainval_sources"] = {f"{a}|{b}": n for (a, b), n in collections.Counter((r["audit"]["split"], r["audit"]["source"].split(":")[0]) for r in tv).items()}
    bl = torch.load("runs/p10c/dataset_blind.pt", weights_only=False)["roots"]
    A["blind_records"] = {"n": len(bl), "labels_present": sum(c["privileged"]["y"] is not None for r in bl for c in r["candidates"]),
                          "disjoint_from_trainval": not ({r["audit"]["root"] for r in bl} & {r["audit"]["root"] for r in tv})}
    ck = {}
    for f in sorted(Path("runs/p10c/runs").glob("*.json")):
        x = _rd(f)
        ck[x["name"]] = sha(f"runs/p10c/ckpt/{x['name']}.pt").removeprefix("sha256:") == x["checkpoint_sha256"].removeprefix("sha256:")
    A["checkpoints"] = {"n": len(ck), "all_hash_ok": all(ck.values())}
    snap = _rd("runs/p10d/predictions.json")
    A["snapshot_hash_ok"] = cf.stable_hash({k: v for k, v in snap.items() if k != "hash"}) == snap["hash"]
    A["snapshot_commit_before_unseal"] = int(sh("log", "-1", "--format=%ct", "162338ea47478bb96acf80c96360e4cd85972f89")) < int(sh("log", "-1", "--format=%ct", "--", "runs/p10b/UNSEALED"))
    # H9 key collision check
    h9 = torch.load("runs/p10c/dataset_h9.pt", weights_only=False)["roots"]
    keys = collections.Counter(r["audit"]["root"] for r in h9)
    a0 = next(a for a in snap["arms"].values() if a["regime"] == "full")
    A["h9"] = {"records": len(h9), "distinct_keys": len(keys), "max_records_per_key": max(keys.values()),
               "snapshot_h9_entries_per_run": len(a0["roots"].get("h9", {})), "overwrite_confirmed": len(a0["roots"].get("h9", {})) < len(h9)}
    # episodes: 975 choices -> 255 deterministic episodes; every choice mapped; committed sequence equals the choice
    jobs = _rd("runs/p10d/episode_jobs.json")
    led = {}
    for f in Path("runs/p10d/episodes").glob("*/ledger.json"):
        led |= {k: (v, f.parent / k) for k, v in _rd(f)["episodes"].items()}
    seqs = {c["query"]: c["sequence"] for r in P.values() for c in r["items"] if c["kind"] == "ORDERED_TEACHER"}
    bad, abst, n_choice = [], 0, 0
    for key, arms in jobs["arms"].items():
        root, ch = key.split("::", 1)
        n_choice += len(arms)
        if ch in ("ABSTAIN", "NOT_RUN"):
            abst += len(arms)
            continue
        j = next(j for j in jobs["jobs"] if j["root"] == root and j["choice"] == ch)
        e, d = led.get(f"{j['arm']}__{root}", (None, None))
        if e is None:
            bad.append((key, "missing episode"))
        elif not (d / "result.json").exists():
            if e.get("outcome") != "UNRESOLVED_CENSORED":
                bad.append((key, f"no result.json but outcome {e.get('outcome')}"))
        elif ch != "GM":
            res = _rd(d / "result.json")
            if [list(map(list, x)) for x in [res["committed_sequence"] or []]][0] not in ([], [list(a) for a in seqs[ch]]) and res["committed_sequence"] is not None:
                bad.append((key, "committed sequence differs"))
    A["episodes"] = {"distinct": len(jobs["jobs"]), "choices": n_choice, "abstain_choices": abst, "mapping_errors": bad[:10], "n_errors": len(bad),
                     "jobs_hash_ok": cf.stable_hash(jobs["jobs"]) == jobs["hash"], "outcomes": dict(collections.Counter(v[0].get("outcome") for v in led.values()))}
    # recompute P1 contrasts + Holm independently from per-root success
    p1 = _rd("runs/p10d/P1_TOP1.json")
    succ = p1["per_root"]
    re = {f"{a}-{b}": paired(succ, a, b) for a, b in (("K-B", "SF"), ("K-B", "A"), ("K-C", "K-B"), ("K-B", "G+M"), ("K-B", "geometry"))}
    A["p1_recomputed"] = {k: {x: v[x] for x in ("mean", "ci95", "sign_p", "roots")} for k, v in re.items()}
    A["p1_matches_report"] = all(abs(re[k]["mean"] - p1["contrasts"][k]["mean"]) < 1e-12 and abs(re[k]["sign_p"] - p1["contrasts"][k]["sign_p"]) < 1e-12 for k in re)
    be = _rd("runs/p10d/BLIND_EVAL.json")
    ps = {"P1": re["K-B-SF"]["sign_p"], "P2": paired(be["per_root"]["H1"], "K-B", "A")["sign_p"],
          "P3": paired(be["per_root"]["order"], "K-C", "K-B")["sign_p"], "P4": paired(be["per_root"]["omission"], "K-B", "SF")["sign_p"]}
    A["holm_recomputed"] = holm(ps)
    A["holm_matches_report"] = all(abs(A["holm_recomputed"][k]["holm_p"] - v["holm_p"]) < 1e-12 for k, v in zip(("P1", "P2", "P3", "P4"), p1["holm"].values()))
    # recompute the blind evaluation into a NEW file and compare with the committed one
    import p10e_report as rr
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    rr.blind_eval("blind", str(OUT / "audit" / "BLIND_EVAL_recomputed.json"))
    nb = _rd(OUT / "audit" / "BLIND_EVAL_recomputed.json")
    A["blind_eval_recompute_identical"] = cf.stable_hash({k: v for k, v in nb.items() if k != "split"}) == cf.stable_hash({k: v for k, v in be.items() if k != "split"})
    A["status_counts_blind"] = be["status_counts"]
    A["regression_p10e"] = {k: _rd("runs/p10b/regression/regression_result.json")[k] for k in ("tests", "replays", "verdict")}
    A["ledger"] = {c: round(pd.spent(c)[0], 1) for c in pd.CAPS} | {"true_training_cpu_s": round(sum(_rd(f)["train_cpu_s"] for f in Path("runs/p10c/runs").glob("*.json")) +
                                                                                              sum(_rd(f)["train_cpu_s"] for f in Path("runs/p10c_prelim/runs").glob("*.json")), 1)}
    A["inference_cost_unit"] = "process CPU seconds per root (time.process_time), single thread"
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "audit" / "P11_0_AUDIT.json", A)


# ------------------------------------------------------------------ P11-1: H9 with collision-free (root, palette) keys (evaluation only)
def h9_variants():
    """[(compound key, base root, palette, prepared root)] for every H9 record; asserts 2 distinct palettes per base root."""
    import p10c_train_eval as pc
    rs = pc.prepared("h9")
    out = [(f"{r['key']}#{r['audit']['palette']}", r["key"], r["audit"]["palette"], r) for r in rs]
    keys = [k for k, *_ in out]
    if len(set(keys)) != len(keys):
        raise ValueError("H9 compound keys collide")
    per = collections.Counter(b for _, b, _, _ in out)
    if set(per.values()) != {2}:
        raise ValueError(f"expected exactly two palettes per root, got {collections.Counter(per.values())}")
    return out


def h9_integrity() -> dict:
    """Non-visual inputs bit-identical to the original record; visual features differ between original, P1 and P2."""
    import torch
    orig = {r["audit"]["root"]: r for n in ("trainval", "blind") for r in torch.load(f"runs/p10c/dataset_{n}.pt", weights_only=False)["roots"]}
    h9 = torch.load("runs/p10c/dataset_h9.pt", weights_only=False)["roots"]
    bad_geom, same_vis, qid = 0, 0, 0
    vis = collections.defaultdict(dict)
    for r in h9:
        o = orig[r["audit"]["root"]]
        bad_geom += any(not np.array_equal(np.asarray(r["inputs"][k]), np.asarray(o["inputs"][k])) for k in r["inputs"] if k != "visual")
        qid += [c["privileged"]["query"] for c in r["candidates"]] != [c["privileged"]["query"] for c in o["candidates"]]
        same_vis += np.allclose(r["inputs"]["visual"], o["inputs"]["visual"])
        vis[r["audit"]["root"]][r["audit"]["palette"]] = np.asarray(r["inputs"]["visual"])
    p12 = sum(np.allclose(v["P1"], v["P2"]) for v in vis.values())
    return {"records": len(h9), "roots": len(vis), "nonvisual_input_mismatch": int(bad_geom), "query_id_mismatch": int(qid),
            "visual_identical_to_original": int(same_vis), "P1_identical_to_P2": int(p12),
            "mask_rule": "p10c_train_eval._recolor colours pixels by the SAME instance mask of the recorded observation (instance > 0 -> object RGB, "
                         "== 0 -> support RGB); masks, ids and geometry are not modified; no label or certificate is read"}


def h9_eval() -> None:
    import contextlib
    import p10_eval as pe
    import p10_protocol as pp
    import p10c_train_eval as pc
    import unique_factor_keyed as ufk
    snap = _rd("runs/p10d/predictions.json")
    integ = h9_integrity()
    if integ["nonvisual_input_mismatch"] or integ["query_id_mismatch"] or integ["P1_identical_to_P2"] or integ["visual_identical_to_original"]:
        raise SystemExit(f"H9 integrity FAILED: {integ}")
    V = h9_variants()
    split = {b: ("blind" if b in snap["arms"][next(iter(snap["arms"]))]["roots"]["blind"] else "validation") for _, b, _, _ in V}
    calls, out = pp.instrument(), {}
    for n, a in snap["arms"].items():
        if a["regime"] != "full":
            continue
        m, x = pc.load(n)
        keyed = pc.ARMS[a["arm"]][2]
        rows = {}
        for ck, base, pal, r in V:
            if a["arm"] == "B-V0":
                import copy
                r = pc._v0([copy.deepcopy(r)])[0]
            with ufk.installed() if keyed else contextlib.nullcontext():
                pm = pe.marginals(m, r, a["arm"] not in ("A", "SF"))
            sel = pp.top1(m, r, a["tau"], a["threshold"], frozenset(), keyed)["choice"]
            o = a["roots"][split[base]][base]
            vetoed_o = o["top1"]
            dp = [abs(pm[q] - p) for q, p in o["marginals"].items() if q in pm]
            cross = [(pm[q] >= 0.5) != (p >= 0.5) for q, p in o["marginals"].items() if q in pm]
            rows[ck] = {"base": base, "palette": pal, "split": split[base], "mean_abs_dp": float(np.mean(dp)), "max_abs_dp": float(np.max(dp)),
                        "crossing_frac": float(np.mean(cross)), "top1": sel, "top1_original": vetoed_o, "top1_changed": sel != vetoed_o, "marginals": pm}
        out[n] = {"arm": a["arm"], "seed": a["seed"], "rows": rows}
    if sum(calls.values()):
        raise RuntimeError(f"physical calls in H9 evaluation: {dict(calls)}")
    (OUT / "h9").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "h9" / "h9_predictions.json", {"integrity": integ, "runs": out, "snapshot": snap["hash"], "physical_calls": dict(calls)})


def h9_report() -> None:
    """Per-root appearance robustness (seed mean per root; palettes separately and averaged); before / after the overwrite fix."""
    import p10d_execute as px
    import p10e_report as rr
    d, snap, P = _rd(OUT / "h9" / "h9_predictions.json"), _rd("runs/p10d/predictions.json"), px.plans()
    tok = {k: {c["token"]: c["query"] for c in P[k]["items"]} for k in P}
    res = {"integrity": d["integrity"], "static_veto": "no H9 base root has a vetoed candidate, so TOP1 recomputation without the veto set equals "
                                                         "the deployed rule", "by_arm": {}}
    for arm in ("K-B", "K-C", "A", "SF", "B-V0", "B-F2", "B-F3", "B-F4", "B-F5s", "B-F5q", "B-U"):
        runs = [r for r in d["runs"].values() if r["arm"] == arm]
        if not runs:
            continue
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in runs:
            orig = next(a for a in snap["arms"].values() if a["arm"] == arm and a["seed"] == r["seed"] and a["regime"] == "full")
            for ck, x in r["rows"].items():
                b, pal = x["base"], x["palette"]
                for m in ("mean_abs_dp", "crossing_frac", "top1_changed"):
                    acc[(m, pal)][b].append(float(x[m]))
                    acc[(m, "both")][b].append(float(x[m]))
                fam = P[b]["family"]
                om = orig["roots"][x["split"]][b]["marginals"]
                if fam == "H1":
                    t = tok[b]
                    k_ = lambda mm: mm[t["edit:A+B"]] - mm[t["edit:A"]] - mm[t["edit:B"]] + mm[t[""]]  # noqa: E731
                    acc[("kappa_abs_change", "both")][b].append(abs(k_(x["marginals"]) - k_(om)))
                for a_, b_ in rr.ORDER_PAIRS.get(fam, []):
                    qa, qb = tok[b].get(a_), tok[b].get(b_)
                    if qa in om and qb in om:
                        acc[("order_flip", "both")][b].append(float((x["marginals"][qa] > x["marginals"][qb]) != (om[qa] > om[qb])))
        out = {}
        for (m, pal), per in acc.items():
            v = {b: float(np.mean(s)) for b, s in per.items()}
            bl = [x for b, x in v.items() if P[b]["split"] == "blind"]
            out[f"{m}|{pal}"] = {"roots": len(v), "mean": float(np.mean(list(v.values()))), "ci95": boot(list(v.values())),
                                 "blind_mean": float(np.mean(bl)) if bl else None, "blind_roots": len(bl)}
        res["by_arm"][arm] = out
    old = _rd("runs/p10d/BLIND_EVAL.json")["H9_mean_abs_change"]
    res["before_after"] = {a: {"P10e_reported (blind, P2 only, query-pooled)": old.get(a), "corrected blind both palettes (per-root)": res["by_arm"][a]["mean_abs_dp|both"]["blind_mean"],
                               "corrected blind P1": res["by_arm"][a]["mean_abs_dp|P1"]["blind_mean"], "corrected blind P2": res["by_arm"][a]["mean_abs_dp|P2"]["blind_mean"]}
                           for a in res["by_arm"] if a in old}
    cf.write_json(OUT / "h9" / "H9_RESULTS.json", res)


def h9_figure() -> None:
    import p10_eval as pe  # noqa: F401
    import p6_viz as pv
    import geometry as ge
    from scene import scene_from_dict
    import data
    import learn_data as ld
    import p10c_train_eval as pc
    import p10d_execute as px
    from viz import plt
    dst = Path("visualizations/p11")
    dst.mkdir(parents=True, exist_ok=True)
    res, d, P = _rd(OUT / "h9" / "H9_RESULTS.json"), _rd(OUT / "h9" / "h9_predictions.json"), px.plans()
    cfg = cf.load_config("config.toml")
    b = next(k for k in P if P[k]["split"] == "blind" and P[k]["family"] == "H1")
    s = scene_from_dict(P[b]["scene"])
    q = data.make_query(cfg, b, s, "t0", P[b]["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
    a = data.appearance_anchor(q, ld.IMAGE_PX)
    run = next(r for r in d["runs"].values() if r["arm"] == "K-B" and r["seed"] == 0)
    f, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(a["rgb"])
    ax[0].set_title(f"{b}\noriginal observation (same geometry in all panels)", fontsize=9)
    for i, (name, pal) in enumerate(pc.PALETTES.items(), 1):
        rgb = a["rgb"].copy()
        rgb[a["instance"] > 0], rgb[a["instance"] == 0] = pal
        x = run["rows"][f"{b}#{name}"]
        ax[i].imshow(rgb)
        ax[i].set_title(f"palette {name}\nK-B s0: mean |dp| {x['mean_abs_dp']:.3f}; TOP1 changed: {x['top1_changed']}", fontsize=9)
    for x in ax[:3]:
        x.axis("off")
    arms = [k for k in ("K-B", "K-C", "A", "SF", "B-V0") if k in res["by_arm"]]
    for j, pal in enumerate(("P1", "P2")):
        v = [res["by_arm"][k][f"mean_abs_dp|{pal}"] for k in arms]
        ax[3].errorbar([i + (j - 0.5) * 0.25 for i in range(len(arms))], [x["mean"] for x in v],
                       yerr=[[x["mean"] - (x["ci95"][0] or x["mean"]) for x in v], [(x["ci95"][1] or x["mean"]) - x["mean"] for x in v]],
                       fmt="o", capsize=3, label=f"palette {pal}", color=["#2a6fdb", "#e07b00"][j])
    ax[3].set(xticks=range(len(arms)), xticklabels=arms, ylabel="per-root mean |p_variant - p_original|", title="H9 (100 roots, both palettes; root CI)")
    ax[3].legend(fontsize=8)
    pv.png(f, dst / "P11_H9_two_palettes.png", "P11-1: same geometry, two palettes (instance masks unchanged); per-root prediction shift per arm and palette")


# ------------------------------------------------------------------ P11-2: F1 static-supervision control, J selector audit, ablation verification
COMP = OUT / "components"


def train_f1off(seed, updates=400) -> None:
    """Keyed B with the F1 static BCE weight set to 0 (same architecture / parameter count / data / sampler / schedule / seed as
    p10c F-K-B, whose only difference is lam_static = 1). Checkpoint selection on P10b validation only."""
    import functools
    import torch
    import abc8
    import p10c_train_eval as pc
    import p9_l3 as l3
    import unique_factor as uf
    import unique_factor_keyed as ufk
    tr, va = pc.data("full")
    uf.total = functools.partial(uf.total, lam_static=0.0)
    l3.build = lambda a: abc8.build("B-full")
    c0 = time.process_time()
    with ufk.installed():
        m, rec = l3.fit("B-full", seed, updates, tr, va, "unique")
    name = f"F1OFF-K-B-s{seed}"
    (COMP / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), COMP / "ckpt" / f"{name}.pt")
    rec["history"] = [{k: h.get(k) for k in ("update", "total", "nll", "static", "val")} for h in rec["history"]]
    rec |= {"name": name, "arm": "K-B", "seed": seed, "lam_static": 0.0, "train_cpu_s": round(time.process_time() - c0, 1),
            "checkpoint_sha256": sha(COMP / "ckpt" / f"{name}.pt"), "train_roots": len(tr), "val_roots": len(va)}
    (COMP / "runs").mkdir(parents=True, exist_ok=True)
    cf.write_json(COMP / "runs" / f"{name}.json", rec)


def _blind_labels():
    """query -> (teacher status, binary label, witnessed J) on the (unsealed) blind roots; read-only."""
    import p10b_dataset as pd
    import p10c_train_eval as pc
    out = {}
    for b in pc.batches():
        for q, r in pd.records(b, ("blind",)).items():
            out[q] = (r["teacher_status"], r["binary_training_label"], r.get("J_first"))
    return out


def _episode_outcomes():
    jobs, led = _rd("runs/p10d/episode_jobs.json"), {}
    for f in Path("runs/p10d/episodes").glob("*/ledger.json"):
        led |= _rd(f)["episodes"]
    return {f"{j['root']}::{j['choice']}": (led.get(f"{j['arm']}__{j['root']}") or {}).get("outcome") for j in jobs["jobs"]}


def _calibrate(m, keyed, val):
    import contextlib
    import p7_eval as ev
    import p9_l3 as l3
    import p10d_execute as px
    import unique_factor_keyed as ufk
    with ufk.installed() if keyed else contextlib.nullcontext():
        vev = [l3.root_eval9(m, r, "unique" if keyed else "factorized")["root"] for r in val]
    tau = ev.fit_temperature(vev)
    return tau, px.choose_threshold(vev, tau)


def _static_metrics(m, rs):
    import torch
    import p10_eval as pe
    rows = []
    with torch.no_grad():
        for r in rs:
            for c in r["cands"] + r.get("static_checks", []):
                xy = m.actions(r["R"], c["seq"])[1] if c["seq"] else r["R"].xy0
                _, lu, lp = m.static(r["R"], xy)
                for lg, ys in ((lu, c["su"]), (lp, c["sp"])):
                    for i, y in enumerate(ys):
                        if y is not None:
                            rows.append({"p": float(torch.sigmoid(lg[i])), "y": int(y), "root": r["key"]})
    return pe.metrics(rows) | {"positives": sum(x["y"] for x in rows), "negatives": sum(1 - x["y"] for x in rows)}


def components_eval() -> None:
    import contextlib
    import copy
    import p10_eval as pe
    import p10_protocol as pp
    import p10c_train_eval as pc
    import p10e_report as rr
    import p9_l3 as l3
    import p9_l3c as l3c
    import unique_factor as uf
    import unique_factor_keyed as ufk
    import abc8
    import torch
    lab, epo, snap = _blind_labels(), _episode_outcomes(), _rd("runs/p10d/predictions.json")
    val = [r for r in pc.prepared("trainval") if r["split"] == "validation"]
    blind = pc.prepared("blind")
    calls, res = pp.instrument(), {"F1": {}, "J": {}}
    # ---- F1: enabled (p10c F-K-B) vs disabled (P11 F1OFF), calibrated on validation only
    mini = {"arms": {}}
    for s in (0, 1, 2):
        for tag, name in (("enabled", f"F-K-B-s{s}"), ("disabled", f"F1OFF-K-B-s{s}")):
            if tag == "enabled":
                m = pc.load(name)[0]
                tau, thr = snap["arms"][name]["tau"], snap["arms"][name]["threshold"]
            else:
                m = abc8.build("B-full")
                m.load_state_dict(torch.load(COMP / "ckpt" / f"{name}.pt", weights_only=True))
                m.eval()
                tau, thr = _calibrate(m, True, val)
            res["F1"][name] = {"tau": tau, "threshold": thr, "static_blind": _static_metrics(m, blind)}
            per = {}
            for r in blind:
                with ufk.installed():
                    pm = pe.marginals(m, r, True)
                ch = pp.top1(m, r, tau, thr, frozenset(), True)["choice"]
                per[r["key"]] = {"marginals": pm, "top1": ch}
                res["F1"][name].setdefault("top1", {})[r["key"]] = {"choice": ch, "label": (lab.get(ch) or ("ABSTAIN",))[0], "executed": epo.get(f"{r['key']}::{ch}")}
            mini["arms"][name] = {"arm": "K-B" if tag == "enabled" else "B-F1off", "seed": s, "regime": "full", "roots": {"blind": per}}
    orig, orig_d = rr._rd, rr.d_eval
    rr._rd = lambda p: mini if str(p) == "runs/p10d/predictions.json" else orig(p)
    rr.d_eval = lambda *a, **k: {"note": "D not evaluated in the F1 control"}
    try:
        ev = rr.blind_eval("blind", str(COMP / "F1_BLIND_EVAL.json"))
    finally:
        rr._rd, rr.d_eval = orig, orig_d
    res["F1"]["contrasts_disabled_minus_enabled"] = {e: paired(ev["per_root"][e] if e != "nll_full" else ev["per_root"]["nll_full"], "B-F1off", "K-B")
                                                     for e in ("nll_full", "H1", "order", "omission")}
    # ---- J: selector variants on frozen K-B / K-C, identical admitted pools
    for arm in ("K-B", "K-C"):
        for s in (0, 1, 2):
            name = f"F-{arm}-s{s}"
            m, a = pc.load(name)[0], snap["arms"][name]
            rng, st = np.random.default_rng(1000 + s), collections.Counter()
            jerr = []
            for r in blind:
                with ufk.installed():
                    _, rows = l3c.deployed(m, r, uf.instances(m, r), a["tau"], a["threshold"])
                adm = [x for x in rows if 1 / (1 + math.exp(-x["u"] / a["tau"])) >= a["threshold"]]
                for x in rows:
                    j = (lab.get(x["query"]) or (None, None, None))[2]
                    if j is not None and x["m"] > 0 and (lab.get(x["query"]) or ("",))[0] == "VERIFIED_FEASIBLE":
                        jerr.append((x["J_pred"], j))
                if not adm:
                    st["abstain"] += 1
                    continue
                mm = min(x["m"] for x in adm)
                st["tie_reached"] += sum(x["m"] == mm for x in adm) >= 2
                shuf = dict(zip([x["query"] for x in adm], rng.permutation([x["J_pred"] for x in adm])))
                keys = {"default": lambda x: (x["m"], x["J_pred"], x["C_d"], x["energy"], x["query"]),
                        "no_J": lambda x: (x["m"], x["C_d"], x["energy"], x["query"]),
                        "shuffled_J": lambda x: (x["m"], shuf[x["query"]], x["C_d"], x["energy"], x["query"]),
                        "oracle_J_offline": lambda x: (x["m"], (lab.get(x["query"]) or (0, 0, None))[2] if (lab.get(x["query"]) or (0, 0, None))[2] is not None else 9e9, x["C_d"], x["energy"], x["query"])}
                ch = {k: min(adm, key=f)["query"] for k, f in keys.items()}
                for k, q in ch.items():
                    st[f"changed_{k}"] += q != ch["default"]
                    st[f"label_success_{k}"] += (lab.get(q) or ("",))[0] == "VERIFIED_FEASIBLE"
                    ex = epo.get(f"{r['key']}::{q}")
                    st[f"executed_known_{k}"] += ex is not None
                    st[f"executed_success_{k}"] += ex == "SUCCESS"
            pj, oj = np.array([x for x, _ in jerr]), np.array([y for _, y in jerr])
            from scipy.stats import spearmanr
            res["J"][name] = dict(st) | {"roots": len(blind), "J_regression": {"n_witnessed": len(jerr), "mae": float(np.mean(np.abs(pj - oj))) if jerr else None,
                                                                              "mean_witnessed": float(np.mean(oj)) if jerr else None,
                                                                              "spearman": float(spearmanr(pj, oj)[0]) if len(jerr) > 2 else None}}
    # ---- ablation verification (definitions, parameter counts) and exploratory Holm across the ablation family
    ver = {}
    for arm in ("B-F2", "B-F3", "B-F4", "B-F5s", "B-F5q", "B-V0", "B-U"):
        mm = pc.build(arm)
        ver[arm] = {"params": sum(p.numel() for p in mm.parameters()), "has_pairo": hasattr(mm, "pairo"), "order_mode": getattr(mm, "order_mode", None),
                    "has_pair_scene": hasattr(mm, "pair_scene"), "has_pair_omit": hasattr(mm, "pair_omit"), "has_cpl": hasattr(mm, "cpl"),
                    "has_ho": hasattr(mm, "ho_q"), "keyed": pc.ARMS[arm][2], "visual_input_zeroed": arm == "B-V0"}
    ver["K-B"] = {"params": sum(p.numel() for p in pc.build("K-B").parameters())}
    be = _rd("runs/p10d/BLIND_EVAL.json")
    abl = {f"{a}|{e}": paired(be["per_root"][e], a, "K-B") for a in ("B-F2", "B-F3", "B-F4", "B-F5s", "B-F5q", "B-V0", "B-U", "K-C")
           for e in ("nll_full", "H1", "order", "omission")}
    hh = holm({k: v["sign_p"] for k, v in abl.items()})
    res["ablations"] = {"definitions": ver, "contrasts_vs_K-B": {k: v | {"holm_p_family32": hh[k]["holm_p"]} for k, v in abl.items()},
                        "note": "exploratory family (8 arms x 4 endpoints = 32 tests), Holm within the family; nll_full: positive = ablation worse"}
    phys = {k: v for k, v in calls.items() if not k.startswith("operational_teacher.")}   # label READING (pure logic) is not a physical call
    res["physical_calls"] = phys
    res["label_reading_calls"] = {k: v for k, v in calls.items() if k.startswith("operational_teacher.")}
    if sum(phys.values()):
        raise RuntimeError(f"physical calls: {phys}")
    cf.write_json(COMP / "COMPONENTS.json", res)


# ------------------------------------------------------------------ P11-3: short-trained (<= 2 moves) strict TOP1 PHYSICAL execution (POST-HOC)
LEN = OUT / "length"


def length_manifest() -> None:
    """Frozen supplementary manifest: (a) the UNCHANGED short-regime TOP1 choices of the P10d snapshot (calibrated on the full
    validation set at freeze time), (b) a sensitivity set recalibrated on the <= 2-move validation subset only (deterministic rule, no
    outcome use). Episodes keyed by (root, repair) with the P10d seed namespace, so P10d episodes of the same key are reused."""
    import p10_protocol as pp
    import p10c_train_eval as pc
    import p10d_execute as px
    snap, P = _rd("runs/p10d/predictions.json"), px.plans()
    seqs = {c["query"]: c["sequence"] for r in P.values() for c in r["items"] if c["kind"] == "ORDERED_TEACHER"}
    old = {f"{j['root']}::{j['choice']}": j for j in _rd("runs/p10d/episode_jobs.json")["jobs"]}
    val_s = pc.short([r for r in pc.prepared("trainval") if r["split"] == "validation"])
    blind = pc.prepared("blind")
    choices, cal = {"frozen": {}, "shortval": {}}, {}
    for n, a in snap["arms"].items():
        if a["regime"] != "short":
            continue
        choices["frozen"][n] = {k: x["top1"] for k, x in a["roots"]["blind"].items()}
        m, keyed = pc.load(n)[0], pc.ARMS[a["arm"]][2]
        tau, thr = _calibrate(m, keyed, val_s)
        cal[n] = {"frozen_tau": a["tau"], "frozen_threshold": a["threshold"], "shortval_tau": tau, "shortval_threshold": thr, "shortval_roots": len(val_s)}
        choices["shortval"][n] = {r["key"]: pp.top1(m, r, tau, thr, frozenset(), keyed)["choice"] for r in blind}
    jobs, arms = {}, collections.defaultdict(list)
    for variant, d in choices.items():
        for n, per in d.items():
            for k, ch in per.items():
                key = f"{k}::{ch}"
                arms[key].append(f"{variant}|{n}")
                if ch in ("ABSTAIN", "NOT_RUN"):
                    continue
                j = {"arm": "c" + cf.stable_hash(key)[7:19], "root": k, "ranked": [seqs[ch]], "choice": ch}
                if key in old and (old[key]["arm"], old[key]["ranked"]) != (j["arm"], j["ranked"]):
                    raise ValueError(f"episode identity differs from P10d for {key}")
                jobs[key] = j | {"reuse_p10d": key in old}
    LEN.mkdir(parents=True, exist_ok=True)
    man = {"jobs": list(jobs.values()), "arms": arms, "calibration": cal, "snapshot": snap["hash"], "protocol": "TOP1_COMMIT, one episode per (root, repair), "
           "180 CPU-s / 600 s episode cap, no fallback, SAFE_RETREAT", "post_hoc": True,
           "checkpoints": {n: _rd(f"runs/p10c/runs/{n}.json")["checkpoint_sha256"] for n in choices["frozen"]}}
    man["hash"] = cf.stable_hash(man)
    if (LEN / "LENGTH_MANIFEST.json").exists():
        raise SystemExit("length manifest already frozen")
    cf.write_json(LEN / "LENGTH_MANIFEST.json", man)


def length_episodes(name) -> None:
    import p7_oneshot as po
    import p10c_train_eval as pc
    man = _rd(LEN / "LENGTH_MANIFEST.json")
    b = Path("runs/p10b") / name
    po.CAPS.update({"cpu_s": 4900.0, "wall_s": 60000.0, "episode_cpu_s": 180.0, "episode_wall_s": 600.0, "max_episodes": 10 ** 6, "max_rank": 5})
    js = [j for j in man["jobs"] if not j["reuse_p10d"] and j["root"] in _rd(b / "plan.json")["roots"]]
    po.run_episodes(LEN / "episodes" / name, b / "plan.json", js, lambda m: open(LEN / f"episodes_{name}.log", "a").write(m + "\n"))
    (LEN / f"DONE_{name}").write_text("done\n")
    del pc


def length_eval() -> None:
    import p10d_execute as px
    man, P = _rd(LEN / "LENGTH_MANIFEST.json"), px.plans()
    led = {}
    for base in (Path("runs/p10d/episodes"), LEN / "episodes"):
        for f in base.glob("*/ledger.json"):
            led |= {k: (v, f.parent / k) for k, v in _rd(f)["episodes"].items()}
    ep = {f"{j['root']}::{j['choice']}": j for j in man["jobs"]}
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    fails = collections.Counter()
    for key, arms in man["arms"].items():
        root, ch = key.split("::", 1)
        e = led.get(f"{ep[key]['arm']}__{root}", (None,))[0] if key in ep else None
        ok = float(e is not None and e.get("outcome") == "SUCCESS")
        for a in arms:
            variant, n = a.split("|")
            arm = n.split("-s")[0].removeprefix("S-")
            per[f"short_{variant}|{arm}"][root].append(ok)
            fails[(f"short_{variant}|{arm}", "ABSTAIN" if ch == "ABSTAIN" else (e or {}).get("outcome", "MISSING"))] += 1
    full = _rd("runs/p10d/P1_TOP1.json")["per_root"]
    for arm in ("K-B", "K-C", "A", "SF"):
        per[f"full|{arm}"] = {k: [v] for k, v in full[arm].items()}
    succ = {g: {k: float(np.mean(v)) for k, v in d.items()} for g, d in per.items()}
    strata = {"H10_3": lambda k: P[k]["family"] == "H10_3", "H10_4": lambda k: P[k]["family"] == "H10_4",
              "H10_4_SOUND": lambda k: P[k]["family"] == "H10_4" and k in _sound_roots(), "1-2 move families": lambda k: not P[k]["family"].startswith("H10")}
    res = {"success": {g: {s: {"mean": float(np.mean([v for k, v in d.items() if f(k)])) if any(f(k) for k in d) else None, "roots": sum(f(k) for k in d)}
                           for s, f in strata.items()} | {"all": {"mean": float(np.mean(list(d.values()))), "roots": len(d)}} for g, d in succ.items()},
           "outcomes": {f"{g}|{o}": n for (g, o), n in fails.items()},
           "short_vs_full_H10": {arm: paired({"s": {k: v for k, v in succ[f"short_frozen|{arm}"].items() if P[k]["family"].startswith("H10")},
                                              "f": {k: v for k, v in succ[f"full|{arm}"].items() if P[k]["family"].startswith("H10")}}, "s", "f")
                                 for arm in ("K-B", "K-C", "A", "SF")},
           "short_B_vs_others_H10": {f"K-B-{b}": paired({"a": {k: v for k, v in succ["short_frozen|K-B"].items() if P[k]["family"].startswith("H10")},
                                                         "b": {k: v for k, v in succ[f"short_frozen|{b}"].items() if P[k]["family"].startswith("H10")}}, "a", "b")
                                     for b in ("A", "SF", "K-C")},
           "episodes": {"total_distinct": len(man["jobs"]), "reused_p10d": sum(j["reuse_p10d"] for j in man["jobs"]),
                        "new": sum(not j["reuse_p10d"] for j in man["jobs"])}, "post_hoc": True}
    cf.write_json(LEN / "LENGTH_RESULTS.json", res)


def _sound_roots():
    import p10e_report as rr
    import p10b_dataset as pd
    import p10c_train_eval as pc
    out = set()
    for b in pc.batches():
        plan, rec = _rd(b / "plan.json"), pd.records(b, ("train", "validation", "blind"))
        out |= {k for k in plan["roots"] if plan["roots"][k]["family"] == "H10_4" and rr.root_yield(plan, rec, k)["minimality_relative_to_catalogue"] == "SOUND"}
    return out


def length_figure() -> None:
    """Recorded short-trained TOP1 episodes on H10 roots: one success GIF (real arrays) and static failure panels (no motion invented)."""
    import g4d_run as gr
    import geometry as ge
    import p6_viz as pv
    import p10d_execute as px
    from scene import scene_from_dict
    from viz import COLORS, plt
    man, P, dst = _rd(LEN / "LENGTH_MANIFEST.json"), px.plans(), Path("visualizations/p11")
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    led = {}
    for base in (Path("runs/p10d/episodes"), LEN / "episodes"):
        for f in base.glob("*/ledger.json"):
            led |= {k: (v, f.parent / k) for k, v in _rd(f)["episodes"].items()}
    ok, bad = None, []
    for j in man["jobs"]:
        arms = man["arms"].get(f"{j['root']}::{j['choice']}", [])
        if not P[j["root"]]["family"].startswith("H10") or not any("frozen|S-K-B" in a for a in arms):
            continue
        e, d = led.get(f"{j['arm']}__{j['root']}", (None, None))
        if e and e.get("outcome") == "SUCCESS" and ok is None and len(j["ranked"][0]) >= 3 and (d / "relocation.npz").exists():
            ok = (j, d)
        elif e and e.get("outcome") != "SUCCESS" and len(bad) < 3:
            bad.append((j, e))
    if ok:
        j, d = ok
        s, seq, frames = scene_from_dict(P[j["root"]]["scene"]), [tuple(a) for a in j["ranked"][0]], []
        for qq, g, pos, mv in pv.trace_frames(robot, s, seq, gr._arrays(d / "relocation.npz"), gr._arrays(d / "target_witness.npz"), n_move=10, n_target=12):
            f, a = plt.subplots(figsize=(5, 5))
            pv._panel(a, cfg, robot, s, P[j["root"]]["goal_cell"], f"{j['root'][-3:]} {P[j['root']]['family']}: SHORT-trained (<=2) keyed B TOP1\n{len(seq)} moves, RECORDED episode", q=qq, gap=g, pos=pos, held=mv)
            pv._tag(a, "TOP1_COMMIT SUCCESS - recorded trajectory", COLORS["valid"])
            frames.append(pv._rgb(f))
        pv.gif(dst / "P11_short_trained_long_success.gif", frames, "P11-3: recorded short-trained keyed-B TOP1 success on a blind H10 root")
    if bad:
        f, ax = plt.subplots(1, len(bad), figsize=(5 * len(bad), 5), squeeze=False)
        for a, (j, e) in zip(ax[0], bad):
            s = scene_from_dict(P[j["root"]]["scene"])
            pv._panel(a, cfg, robot, s, P[j["root"]]["goal_cell"], f"{j['root'][-3:]} {P[j['root']]['family']}: short keyed-B TOP1 ({len(j['ranked'][0])} moves)")
            pv._arrows(a, s, [tuple(x) for x in j["ranked"][0]], pv.COL["CERTIFIED_INFEASIBLE"])
            pv._tag(a, f"{e['outcome']} - NO VERIFIED TRACE", pv.COL["UNKNOWN"], size=6)
        pv.png(f, dst / "P11_short_trained_long_failures.png", "P11-3: short-trained keyed-B TOP1 failures on blind H10 roots (committed repair, recorded outcome)")
    res = _rd(LEN / "LENGTH_RESULTS.json")
    f, ax = plt.subplots(figsize=(9, 4.5))
    arms = ["K-B", "K-C", "A", "SF"]
    for i, st in enumerate(("H10_3", "H10_4")):
        for j_, rg in enumerate(("full", "short_frozen")):
            ax.bar([x + i * 0.42 + j_ * 0.2 for x in range(4)], [res["success"][f"{rg}|{a}"][st]["mean"] for a in arms], 0.18,
                   color=[["#2a6fdb", "#9bbcf0"], ["#e07b00", "#f3c48a"]][i][j_], label=f"{st}: trained {'1-4 moves' if rg == 'full' else '<=2 moves'}")
    ax.set(xticks=[x + 0.3 for x in range(4)], xticklabels=arms, ylabel="executed TOP1 success (9 roots each)", title="P11-3 (POST-HOC): physically executed 3- / 4-move repairs")
    ax.legend(fontsize=7)
    pv.png(f, dst / "P11_length_executed.png", "P11-3: executed TOP1 success on blind H10 roots, short-trained vs full-trained")


# ------------------------------------------------------------------ P11-5: D versus A / B / C / SF (initial feasibility; blocker identity)
def _feas_metrics(pairs, seeds_pairs=None):
    """pairs: [(p, y)] one per root (seed mean). Threshold 0.5 on p(feasible)."""
    import p10_eval as pe
    p, y = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
    pred = p >= 0.5
    tp, tn, fp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    rec1 = tp / max(1, tp + fn) if (y == 1).any() else None
    rec0 = tn / max(1, tn + fp) if (y == 0).any() else None
    m = pe.metrics([{"p": float(a), "y": int(b), "root": i} for i, (a, b) in enumerate(pairs)])
    out = {"N": len(pairs), "feasible_GT": int(y.sum()), "infeasible_GT": int((1 - y).sum()), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
           "accuracy": float((pred == (y == 1)).mean()), "balanced_accuracy": (rec1 + rec0) / 2 if rec1 is not None and rec0 is not None else None,
           "feasible_recall": rec1, "infeasible_recall": rec0, "auroc": m["auroc"], "nll": m["nll"], "brier": m["brier"], "ece_10bins": m["ece"]}
    if seeds_pairs:
        accs = [float(np.mean([(a >= 0.5) == (b == 1) for a, b in sp])) for sp in seeds_pairs]
        nlls = [float(np.mean([-math.log(min(max(a if b else 1 - a, 1e-9), 1)) for a, b in sp])) for sp in seeds_pairs]
        out["per_seed_accuracy"], out["per_seed_nll"] = accs, nlls
    return out


def _set_metrics(gt, pred):
    tp = sum(a and b for a, b in zip(gt, pred))
    fp = sum((not a) and b for a, b in zip(gt, pred))
    prec = tp / (tp + fp) if (tp + fp) else 1.0           # no predicted blocker = no false attribution
    rec = tp / sum(gt) if sum(gt) else 1.0
    f1 = 1.0 if not any(gt) and not any(pred) else 2 * tp / max(1, sum(gt) + sum(pred))
    u = sum(a or b for a, b in zip(gt, pred))
    neg = sum(1 - a for a in gt)
    return {"precision": prec, "recall": rec, "f1": f1, "exact": float(list(gt) == list(pred)), "jaccard": tp / u if u else 1.0,
            "card_err": abs(sum(pred) - sum(gt)), "fp_rate_irrelevant": fp / neg if neg else None, "tp": tp, "fp": fp, "fn": sum(a and not b for a, b in zip(gt, pred))}


def _agg(per_root):
    keys = ("precision", "recall", "f1", "exact", "jaccard", "card_err", "fp_rate_irrelevant")
    out = {"roots": len(per_root)}
    for k in keys:
        v = [x[k] for x in per_root.values() if x.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
        out[f"{k}_ci95"] = boot(v) if k in ("f1", "exact", "precision", "recall") else None
    tp, fp, fn = (sum(x[k] for x in per_root.values()) for k in ("tp", "fp", "fn"))
    out["micro_f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    return out


def _derived_mask(st_p, n):
    """Blocker mask from predicted subset feasibilities: minimal predicted-feasible subsets (no predicted-feasible proper subset); union."""
    feas = [s for s, p in st_p.items() if p >= 0.5]
    if frozenset() in feas:
        return [0] * n
    mins = [s for s in feas if not any(t < s for t in feas)]
    return [int(any(i in s for s in mins)) for i in range(n)]


def d_compare(cohort="p10b", out=None) -> dict:
    import p10b_dataset as pd
    import p10c_train_eval as pc
    import p10d_execute as px
    snap = _rd("runs/p10d/predictions.json") if cohort == "p10b" else _rd(OUT / "new_holdout" / "predictions_newholdout.json")
    P, rec, dl = {}, {}, {}
    if cohort == "p10b":
        for b in pc.batches():
            plan = _rd(b / "plan.json")
            r = pd.records(b, ("train", "validation", "blind"))
            P |= plan["roots"]
            rec |= r
            dl |= pd.d_labels(plan, r)
        roots = [k for k in P if P[k]["split"] == "blind"]
    else:
        import p11_holdout as ph
        for b in ("batch1", "batch2"):
            plan = _rd(ph.OUTD / b / "plan.json")
            r = ph.sealed_records(b)
            P |= plan["roots"]
            rec |= r
            dl |= pd.d_labels(plan, r)
        roots = list(P)
    split = "blind"
    res = {"cohort": cohort, "roots": len(roots)}
    # ---- 1) initial feasibility on the identical decided cohort
    y0 = {k: dl[k]["y0"] for k in roots if dl[k]["y0"] is not None}
    arms = {}
    for n, a in snap["arms"].items():
        if a["regime"] == "full" and a["arm"] in ("A", "SF", "K-B", "K-C"):
            for k in y0:
                p = a["roots"][split].get(k, {}).get("marginals", {}).get(f"seq:{k}:empty")
                if p is not None:
                    arms.setdefault(a["arm"], {}).setdefault(k, {})[a["seed"]] = p
    for arm in ("D", "D-pairless"):
        lam = snap["baselines"][f"{arm}_lambda_B"]
        for n, x in snap["d"].items():
            if x["arm"] == arm and x["lambda_B"] == lam:
                for k in y0:
                    if k in x["roots"][split]:
                        arms.setdefault(arm, {}).setdefault(k, {})[x["seed"]] = x["roots"][split][k]["p_feasible"]
    common = sorted(set.intersection(*[set(v) for v in arms.values()]))
    prov = collections.Counter(rec[f"seq:{k}:empty"]["label_source"] for k in common)
    res["feasibility"] = {"cohort_N": len(common), "excluded_undecided_y0": len(roots) - len(y0), "excluded_missing_prediction": len(y0) - len(common),
                          "provenance": dict(prov), "arms": {}}
    for arm, d in arms.items():
        pairs = [(float(np.mean(list(d[k].values()))), y0[k]) for k in common]
        seeds = sorted({s for k in common for s in d[k]})
        res["feasibility"]["arms"][arm] = _feas_metrics(pairs, [[(d[k][s], y0[k]) for k in common if s in d[k]] for s in seeds])
    # ---- 2) blocker identity on complete masks (union of inclusion-minimal sufficient parking subsets)
    gt = {k: dl[k] for k in roots if dl[k]["mask"] is not None and dl[k]["n"] > 0}
    preds = {}
    for arm in ("D", "D-pairless"):
        lam, thr = snap["baselines"][f"{arm}_lambda_B"], snap["baselines"][f"{arm}_threshold"]
        for n, x in snap["d"].items():
            if x["arm"] == arm and x["lambda_B"] == lam:
                for k in gt:
                    p = x["roots"][split][k]
                    preds.setdefault(arm, {}).setdefault(k, []).append([int(p["p_feasible"] < 0.5 and v >= thr) for v in p["marg"]])
    preds["geometry_heuristic"] = {k: [snap["baselines"]["heuristic"][split][k]] for k in gt}

    def score(arm, ks):
        per = {}
        for k in ks:
            ms = [_set_metrics(gt[k]["mask"], pm) for pm in preds[arm][k]]
            per[k] = {m: (float(np.mean([x[m] for x in ms])) if ms[0][m] is not None else None) for m in ms[0]}
        return per
    strata = {"all": list(gt), "proof_grounded": [k for k in gt if gt[k]["provenance"] == "proof_grounded"],
              "operational": [k for k in gt if gt[k]["provenance"] != "proof_grounded"]}
    strata |= {f"blockers={c}": [k for k in gt if sum(gt[k]["mask"]) == c] for c in range(5)}
    res["blockers"] = {"complete_masks": len(gt), "types": dict(collections.Counter(dl[k]["type"] for k in roots)),
                       "alternative_minimal_sets": sum(len(gt[k]["sets"]) > 1 for k in gt),
                       "union_differs_from_necessary": sum(gt[k]["mask"] != gt[k]["necessary"] for k in gt), "by_stratum": {}}
    per_all = {arm: score(arm, list(gt)) for arm in preds}
    for s, ks in strata.items():
        if ks:
            res["blockers"]["by_stratum"][s] = {arm: _agg({k: per_all[arm][k] for k in ks}) | {"provenance_counts": dict(collections.Counter(gt[k]["provenance"] for k in ks))}
                                                for arm in preds}
    f1 = {arm: {k: v["f1"] for k, v in per_all[arm].items()} for arm in preds}
    ex = {arm: {k: v["exact"] for k, v in per_all[arm].items()} for arm in preds}
    res["blockers"]["paired"] = {"D-D-pairless_F1": paired(f1, "D", "D-pairless"), "D-heuristic_F1": paired(f1, "D", "geometry_heuristic"),
                                 "D-D-pairless_exact": paired(ex, "D", "D-pairless"),
                                 "D-D-pairless_F1_proof_only": paired({a: {k: v for k, v in f1[a].items() if gt[k]["provenance"] == "proof_grounded"} for a in f1}, "D", "D-pairless"),
                                 "D-D-pairless_F1_3_4_operational": paired({a: {k: v for k, v in f1[a].items() if sum(gt[k]["mask"]) >= 3} for a in f1}, "D", "D-pairless")}
    # ---- 3) strictly matched A / B / C / SF counterfactual attribution (pool = 2 edit roles; parking subsets == zS edit queries)
    sup, why = [], collections.Counter()
    for k in gt:
        r = P[k]
        items = {it["token"]: it for it in r["items"]}
        e = [items.get(t) for t in ("edit:A", "edit:B", "edit:A+B")]
        ok = dl[k]["n"] == 2 and all(e) and all(x.get("d_subset") for x in e) and \
            all(x["edit"] == {o: r["parking"][o] for o in r["parking"] if {v: kk for kk, v in r["ids"].items()}[o] in x["d_subset"]} for x in e)
        (sup.append(k) if ok else why.update(["pool != 2 or parking subsets differ from the zS edits"]))
    for n, a in snap["arms"].items():
        if a["regime"] != "full" or a["arm"] not in ("A", "SF", "K-B", "K-C"):
            continue
        for k in sup:
            m, roles = a["roots"][split][k]["marginals"], dl[k]["roles"]
            items = {it["token"]: it for it in P[k]["items"]}
            st = {frozenset(): m[f"seq:{k}:empty"]}
            for t in ("edit:A", "edit:B", "edit:A+B"):
                st[frozenset(roles.index(x) for x in items[t]["d_subset"])] = m[items[t]["query"]]
            preds.setdefault(a["arm"], {}).setdefault(k, []).append(_derived_mask(st, 2))
    res["matched"] = {"matched_N": len(sup), "unsupported_N": len(gt) - len(sup), "unsupported_reasons": dict(why),
                      "proof_only_matched_N": sum(gt[k]["provenance"] == "proof_grounded" for k in sup),
                      "arms": {arm: _agg(score(arm, sup)) for arm in ("D", "D-pairless", "geometry_heuristic", "A", "K-B", "K-C", "SF") if arm in preds and sup}}
    mf1 = {arm: {k: v["f1"] for k, v in score(arm, sup).items()} for arm in preds if sup}
    res["matched"]["paired_vs_D"] = {f"D-{a}": paired(mf1, "D", a) for a in ("K-B", "K-C", "A", "SF", "D-pairless", "geometry_heuristic") if a in mf1}
    res["D_capacity"] = {"exact_enumeration_cap": 6, "benchmark_pool_max": max((dl[k]["n"] for k in roots), default=0)}
    out = out or (OUT / "d_compare" / f"D_VS_ABC_{cohort}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cf.write_json(Path(out), res)
    return res


def d_figures() -> None:
    import geometry as ge
    import p10b_dataset as pd
    import p10c_train_eval as pc
    import p10d_execute as px
    import p6_viz as pv
    from matplotlib.patches import Circle
    from scene import scene_from_dict
    from viz import plt
    dst = Path("visualizations/p11")
    r = _rd(OUT / "d_compare" / "D_VS_ABC_p10b.json")
    f, ax = plt.subplots(1, 3, figsize=(21, 5.5))
    fa = r["feasibility"]["arms"]
    arms = ["D", "D-pairless", "SF", "K-B", "K-C", "A"]
    ax[0].barh(arms, [fa[a]["nll"] for a in arms], color=["#2a6fdb", "#e07b00", "#8e5bd0", "#1f4e9e", "#c56a00", "#2e9e5b"])
    for i, a in enumerate(arms):
        ax[0].text(fa[a]["nll"], i, f"  bal.acc {fa[a]['balanced_accuracy']:.3f}  TP/TN/FP/FN {fa[a]['TP']}/{fa[a]['TN']}/{fa[a]['FP']}/{fa[a]['FN']}", va="center", fontsize=7)
    ax[0].set(xlabel="NLL (lower better)", title=f"Initial feasibility, identical {r['feasibility']['cohort_N']} roots (4 feasible / 60 infeasible)")
    st = r["blockers"]["by_stratum"]
    for j, s in enumerate(("proof_grounded", "operational", "all")):
        for i, a in enumerate(("D", "D-pairless", "geometry_heuristic")):
            x = st[s][a]
            ax[1].bar(j + (i - 1) * 0.27, x["f1"], 0.25, yerr=[[x["f1"] - x["f1_ci95"][0]], [x["f1_ci95"][1] - x["f1"]]], capsize=3,
                      color=["#2a6fdb", "#e07b00", "#7f7f7f"][i], label=a if j == 0 else None)
    ax[1].set(xticks=range(3), xticklabels=[f"{s}\n(N={st[s]['D']['roots']})" for s in ("proof_grounded", "operational", "all")], ylim=(0, 1.05),
              ylabel="per-object F1 (macro by root)", title="Blocker identity, all complete masks (union-minimal-set GT)")
    ax[1].legend(fontsize=8)
    m = r["matched"]["arms"]
    ma = ["D", "geometry_heuristic", "D-pairless", "K-C", "K-B", "SF", "A"]
    for i, a in enumerate(ma):
        ax[2].errorbar(m[a]["f1"], i, xerr=[[m[a]["f1"] - m[a]["f1_ci95"][0]], [m[a]["f1_ci95"][1] - m[a]["f1"]]], fmt="o", capsize=3, color="#333")
    ax[2].set(yticks=range(len(ma)), yticklabels=ma, xlabel="per-object F1", title=f"Strictly matched cohort N={r['matched']['matched_N']} (A/B/C/SF via exact zS edits)")
    pv.png(f, dst / "P11_D_vs_ABC.png", "P11-5: initial feasibility (identical cohort) and blocker identity (all complete masks; strictly matched cohort)")
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    P, snap, dl = px.plans(), _rd("runs/p10d/predictions.json"), {}
    for b in pc.batches():
        plan = _rd(b / "plan.json")
        dl |= pd.d_labels(plan, pd.records(b, ("train", "validation", "blind")))
    lam, thr = snap["baselines"]["D_lambda_B"], snap["baselines"]["D_threshold"]
    d0 = next(x for x in snap["d"].values() if x["arm"] == "D" and x["lambda_B"] == lam and x["seed"] == 0)
    picks = []
    for want in (("proof_grounded", 2), ("operational", 3), ("proof_grounded", 1)):
        k = next((k for k in sorted(dl) if P[k]["split"] == "blind" and dl[k]["mask"] is not None and dl[k]["provenance"] == want[0] and sum(dl[k]["mask"]) == want[1]), None)
        if k:
            picks.append(k)
    f, ax = plt.subplots(1, len(picks), figsize=(6 * len(picks), 6), squeeze=False)
    for a, k in zip(ax[0], picks):
        s, x = scene_from_dict(P[k]["scene"]), dl[k]
        p = d0["roots"]["blind"][k]
        pred = [int(p["p_feasible"] < 0.5 and v >= thr) for v in p["marg"]]
        pv._panel(a, cfg, robot, s, P[k]["goal_cell"], f"{k[-3:]} {P[k]['family']} ({x['provenance']}): GT union {x['roles']} -> {x['mask']}\nD s0 pred {pred}  (p_feasible {p['p_feasible']:.3f})")
        for i, o in enumerate(x["pool"]):
            xy = s.get(o).xy_m
            if x["mask"][i]:
                a.add_patch(Circle(xy, 0.036, fc="none", ec="#2e9e5b", lw=2.5, zorder=20))
            if pred[i]:
                a.plot(*xy, marker="x", color="#2a6fdb", ms=12, mew=3, zorder=21)
            tag = "TP" if x["mask"][i] and pred[i] else "FN" if x["mask"][i] else "FP" if pred[i] else "TN"
            a.text(xy[0], xy[1] - 0.045, f"{x['roles'][i]}:{tag}", ha="center", fontsize=8, weight="bold", zorder=22)
        pv._tag(a, "GT = union of minimal parking subsets (HYPOTHETICAL edits; not executed)", "#34495e", size=6)
    pv.png(f, dst / "P11_D_attribution_scenes.png", "P11-5: blind roots with GT union-minimal blockers (green rings) and D predictions (blue x)")


# ------------------------------------------------------------------ P11-6: end-to-end resource accounting and outcome classes
def resources() -> None:
    import geometry as ge
    import p10d_execute as px
    jobs, P = _rd("runs/p10d/episode_jobs.json"), px.plans()
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    led = {}
    for f in Path("runs/p10d/episodes").glob("*/ledger.json"):
        led |= {k: (v, f.parent / k) for k, v in _rd(f)["episodes"].items()}
    jmap = {f"{j['root']}::{j['choice']}": j for j in jobs["jobs"]}
    gm_cpu = {}
    for k in sorted({j["root"] for j in jobs["jobs"]}):
        c0 = time.process_time()
        px.gm_rank(cfg, robot, P[k])
        gm_cpu[k] = time.process_time() - c0
    inf = _rd("runs/p10d/INFERENCE_COST.json")
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    cls = collections.defaultdict(collections.Counter)
    for key, arms in jobs["arms"].items():
        root, ch = key.split("::", 1)
        e, d = (led.get(f"{jmap[key]['arm']}__{root}") if key in jmap else None) or (None, None)
        res = _rd(d / "result.json") if d is not None and (d / "result.json").exists() else {}
        for a in arms:
            arm = a if a in ("geometry", "uniform", "G+M") else a.split("-s")[0].removeprefix("F-")
            ep_cpu = (e or {}).get("cpu_s", 0.0)
            pre = gm_cpu[root] if arm == "G+M" else 0.0
            nn = (inf.get(arm) or {}).get("mean_s", 0.0) if arm in ("K-B", "K-C", "A", "SF") else 0.0
            per[arm]["total_cpu_per_root_incl_abstain"].append(ep_cpu + pre + nn)
            per[arm]["episode_cpu_executed"] += [ep_cpu] if e else []
            per[arm]["relocation_plans"].append(res.get("relocation_plans", 0) or 0)
            per[arm]["target_calls"].append(res.get("target_calls", 0) or 0)
            per[arm]["candidates_physically_tried"].append(res.get("tried", 0) or 0)
            per[arm]["precommit_ranking_cpu"].append(pre)
            per[arm]["wall_s"] += [(e or {}).get("guard", {}).get("wall_s", 0.0)] if e else []
            cls[arm]["ABSTAIN" if ch in ("ABSTAIN", "NOT_RUN") else (e or {}).get("outcome", "MISSING")] += 1
    out = {a: {k: {"mean": float(np.mean(v)) if v else None, "n": len(v)} for k, v in d.items()} for a, d in per.items()}
    out["_outcome_classes"] = {a: dict(c) for a, c in cls.items()}
    out["_notes"] = {"inference": "learned arms add per-root network + exact inference CPU (INFERENCE_COST.json); D is not an execution arm",
                     "G+M": "pose-screen ranking (pose screens + sector claim, measured now) + planning of up to 5 ranked candidates in the episode",
                     "TOP1": "one committed candidate; abstention = 0 episode CPU and counts as failure"}
    cf.write_json(OUT / "RESOURCES.json", out)


def outcome_figure() -> None:
    import p6_viz as pv
    from viz import plt
    r = _rd(OUT / "RESOURCES.json")["_outcome_classes"]
    order = ["K-B", "K-C", "A", "SF", "geometry", "uniform", "G+M"]
    classes = ["SUCCESS", "NO_REPAIR_WITNESS_WITHIN_BUDGET", "TARGET_CERTIFIED_INFEASIBLE", "TARGET_UNRESOLVED", "UNRESOLVED_CENSORED", "ABSTAIN"]
    colors = ["#2e9e5b", "#e07b00", "#c0392b", "#8e5bd0", "#7f7f7f", "#d0d0d0"]
    f, ax = plt.subplots(figsize=(11, 5))
    for i, a in enumerate(order):
        tot, left = sum(r[a].values()), 0.0
        for c, col in zip(classes, colors):
            v = r[a].get(c, 0) / tot
            ax.barh(i, v, left=left, color=col, edgecolor="white", hatch="//" if a == "G+M" else None, label=c if i == 0 else None)
            left += v
        ax.text(1.01, i, f"n={tot} choices (65 roots)", va="center", fontsize=8)
    ax.set(yticks=range(len(order)), yticklabels=[a + (" (planner-assisted)" if a == "G+M" else "") for a in order], xlim=(0, 1.25),
           xlabel="fraction of arm x seed choices", title="P10d blind TOP1 outcome classes (strict TOP1; G+M hatched = planner-assisted, up to 5 candidates)")
    ax.legend(fontsize=7, loc="lower right")
    pv.png(f, Path("visualizations/p11") / "P11_TOP1_outcome_classes.png", "Outcome classes per arm; G+M differentiated")


def regression() -> None:
    """Historical suite + six replays + P10/P11 tests after the P11 code changes (fixture planner calls itemized by countcalls)."""
    import subprocess
    d = OUT / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "1600", "--wall", "3400"], capture_output=True, text=True)
