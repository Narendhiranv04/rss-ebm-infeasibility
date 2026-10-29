"""Which requested K levels fail? Per requested K: attempts, construction rejections / admin caps / errors by reason, accepted roots by
VERIFIED K (0-4 or NONE), exact-match rate, and for NONE roots where the repair broke down.

    python bd_kfail.py <out_dir> [<out_dir> ...]          # prints a table, writes <out_dir>/KFAIL.json"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def none_cause(r):
    """Where a root with no verified repair broke down (tabletop records carry the relocation search; others fall back to the mask)."""
    bl = r["labels"]["blocker"]
    if r["labels"]["original"]["y"]:
        return "T_FEASIBLE_BUT_EXECUTION_FAILED"
    if not bl.get("minimal_sets"):
        return "NO_T_CLEARING_SET_IN_SCOPE"
    reach = bl.get("clearing_sets_reachable")
    if reach is not None and not any(reach.values()):
        return "CLEARING_SET_NOT_REACHABLE_BY_PLANNABLE_RELOCATIONS"
    if reach is not None:
        return "REACHABLE_BUT_EXECUTION_NOT_VERIFIED"
    return "CLEARING_SET_EXISTS_NO_VERIFIED_SEQUENCE"


def analyse(out):
    out = Path(out)
    led = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines() if x.strip()]
    rows = {}
    for k in range(5):
        L = [e for e in led if e["spec"]["K"] == k]
        acc = [e for e in L if e.get("accepted")]
        rej = collections.Counter(e["reason"].split(":")[0] + ":" + e["reason"].split(":")[1][:48] for e in L if not e.get("accepted"))
        vk = collections.Counter("NONE" if e.get("verified_K") is None else e["verified_K"] for e in acc)
        nc = collections.Counter()
        for e in acc:
            if e.get("verified_K") is None:
                p = out / "roots" / f"{e['root_id']}.json"
                nc[none_cause(json.loads(p.read_text())) if p.exists() else "ROOT_FILE_MISSING"] += 1
        rows[k] = {"attempts": len(L), "accepted": len(acc), "rejected": dict(rej.most_common()), "verified_K": {str(a): b for a, b in sorted(vk.items(), key=str)},
                   "exact": vk.get(k, 0), "exact_rate": round(vk.get(k, 0) / max(1, len(acc)), 3), "none_rate": round(vk.get("NONE", 0) / max(1, len(acc)), 3),
                   "none_causes": dict(nc.most_common()), "wall_s_mean": round(sum(e["wall_s"] for e in L) / max(1, len(L)), 1)}
    obs = collections.Counter("NONE" if e.get("verified_K") is None else e["verified_K"] for e in led if e.get("accepted"))
    res = {"out": str(out), "ledger": len(led), "accepted": sum(e.get("accepted", False) for e in led), "observed_K": {str(a): b for a, b in sorted(obs.items(), key=str)},
           "by_requested_K": rows}
    (out / "KFAIL.json").write_text(json.dumps(res, indent=1))
    return res


def show(res):
    print(f"\n{res['out']}  ledger={res['ledger']} accepted={res['accepted']} observed K={res['observed_K']}")
    print(f"{'req K':>5} {'tries':>6} {'acc':>5} {'exact':>6} {'NONE':>6}  verified K of accepted        rejections / NONE causes")
    for k, r in res["by_requested_K"].items():
        print(f"{k:>5} {r['attempts']:>6} {r['accepted']:>5} {r['exact_rate']:>6.0%} {r['none_rate']:>6.0%}  {str(r['verified_K']):<28}  {r['rejected']} {r['none_causes']}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        show(analyse(d))
