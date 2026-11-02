"""P8 read-only audit: P7 evidence recomputed from saved records, manifest integrity, and the factor-scope proof of
ABC7_V1 coupling modes (which marginals / conditionals each factor can move under the chain normalization).
No training, no teacher call. Writes runs/p8/audit/{factor_mapping.json, replay_audit.json, data_coverage.csv,
baseline_status.json}."""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import torch

import abc7_models as a7
import abc7_train as t7
import config as cf
import g4d_run as gr
import operational_teacher as ot
import p6_run as p6r

C7 = Path("runs/p7/collection")
MANIFESTS = ("runs/g4c_d2_poc_s", "runs/g4c_pilot", "runs/g4d_evidence", "runs/g4c_lf_p4/pilot", "runs/p5/d3lite", "runs/p5/g4e",
             "runs/p6/collect", "runs/p7/collection")
REPORTED = {"VERIFIED_FEASIBLE": 269, "CERTIFIED_INFEASIBLE": 368, "BUDGET_INFEASIBLE": 125, "UNKNOWN": 8, "NOT_RUN": 201}


def manifests() -> dict:
    out = {}
    for d in MANIFESTS:
        m = json.loads(Path(d, "manifest.json").read_text())
        bad = [f for f, h in m["files"].items() if not Path(d, f).exists() or gr._sha(Path(d, f)) != h]
        out[d] = {"files": len(m["files"]), "mismatched_or_missing": bad[:10], "n_bad": len(bad)}
    return out


def p7_recount() -> dict:
    plan = json.loads((C7 / "plan.json").read_text())
    rec = json.loads((C7 / "decisions.json").read_text()) | json.loads((C7 / "sealed_test.json").read_text())
    st = collections.Counter(r["teacher_status"] for r in rec.values())
    roots = {k: v for k, v in plan["roots"].items() if v["queries"]}
    kap = p6r.kappas({"roots": roots}, rec)
    eff = ot.effects(rec, roots)
    led = json.loads(Path("runs/p7/execution/ledger.json").read_text())
    rows = []
    for k, v in roots.items():
        mine = [r for r in rec.values() if r["query"]["lineage"] == k]
        c = collections.Counter(r["teacher_status"] for r in mine)
        kk = next((x for x in kap if x["root"] == k), None)
        rows.append({"root": k, "family": v["family"], "split": v["split"], "wave": v["wave"], "queries": len(mine), **{s: c[s] for s in REPORTED},
                     "kappa": kk and kk["kappa"], "kappa_operational": kk and kk["operational"]})
    return {"status": dict(st), "matches_report": dict(st) == {**REPORTED, **({"FATAL_CONTRADICTION": st["FATAL_CONTRADICTION"]} if st["FATAL_CONTRADICTION"] else {})},
            "kappa": {f"{a}|{'operational' if b else 'certified'}": n for (a, b), n in collections.Counter((x["kappa"], x["operational"]) for x in kap).items()},
            "kappa_by_split": {f"{a}|{b}": n for (a, b), n in collections.Counter((roots[x["root"]]["split"], x["kappa"]) for x in kap).items()},
            "order_contrasts": sum(s["decided_contrast"] for s in eff["swap"]), "episodes": len(led["episodes"]),
            "episode_outcomes": {a: dict(collections.Counter(e["outcome"] for e in led["episodes"].values() if e["arm"] == a)) for a in "ABCU"},
            "rows": rows}


def scope_tests() -> dict:
    """Controlled value tests: perturb one named factor's parameters, measure what moves."""
    import sys
    sys.path.insert(0, "tests")
    import test_p7_models as fx                        # synthetic root fixture (no data dependence)
    r = t7.prepare({"roots": [fx.fixture(0)]})[0]
    out = {}
    for mode in ("full", "quadratic", "structured", "none"):
        torch.manual_seed(0)
        m = a7.build("B" + ("" if mode == "full" else f"-{mode}" if mode != "none" else "-no-coupling"))
        mods = sorted({n.split(".")[0] for n, _ in m.named_parameters()})
        cand = next(c for c in r["cands"] if len(c["seq"]) == 4)
        base = t7.forward(m, r)
        res = {"modules": mods}
        for name in ("pair_scene", "pair_omit", "cpl", "ho_s", "ho_q", "scene", "order_s"):
            if not hasattr(m, name):
                res[name] = "ABSENT"
                continue
            with torch.no_grad():
                saved = [p.clone() for p in getattr(m, name).parameters()]
                for p in getattr(m, name).parameters():
                    p.add_(0.5 * torch.randn_like(p))
            new = t7.forward(m, r)
            o0, o1 = base[r["cands"].index(cand)], new[r["cands"].index(cand)]
            ps0 = torch.exp(o0["logP_S"])
            ps1 = torch.exp(o1["logP_S"])
            res[name] = {"scene_block_P_S_change": float((ps1 - ps0).abs().max().detach()),
                         "repair_conditional_change": float((torch.exp(o1["logP_Q"]) - torch.exp(o0["logP_Q"])).abs().max().detach()),
                         "zR_marginal_change": float(abs(o1["p"] - o0["p"]).detach())}
            with torch.no_grad():
                for p, s in zip(getattr(m, name).parameters(), saved):
                    p.copy_(s)
        out[mode] = res
    return out


def run(args=None) -> int:
    out = Path("runs/p8/audit")
    out.mkdir(parents=True, exist_ok=True)
    man = manifests()
    rc = p7_recount()
    with open(out / "data_coverage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rc["rows"][0]))
        w.writeheader()
        w.writerows(rc["rows"])
    scopes = scope_tests()
    cf.write_json(out / "factor_mapping.json", {"model_version": a7.VERSION, "scope_tests": scopes, "chain_normalization": (
        "P(z|r) = P_S(z_s|c0) P_Q(z_q|z_s,r): E_Q is normalized per z_s, so NO repair-side or scene<->repair parameter (pair_omit, "
        "ho_q, cross-block cpl pairs, order_s) can change P_S; only scene logits, pair_scene, scene-block cpl pairs and ho_s can.")})
    cf.write_json(out / "replay_audit.json", {"manifests": man, "all_intact": all(v["n_bad"] == 0 for v in man.values())})
    cf.write_json(out / "baseline_status.json", {k: v for k, v in rc.items() if k != "rows"})
    print(json.dumps({"manifests_intact": all(v["n_bad"] == 0 for v in man.values()), "status": rc["status"], "matches": rc["matches_report"],
                      "kappa": rc["kappa"], "order": rc["order_contrasts"], "episodes": rc["episodes"]}, indent=1, default=str))
    print(json.dumps(scopes, indent=1))
    return 0


if __name__ == "__main__":
    run()
