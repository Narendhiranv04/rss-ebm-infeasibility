"""EXPLORATORY give-up analysis (no new compute; item #7, user 2026-11-22): why ours gives up too rarely in 2D.
Reads the stored pools (frozen rule 0) and labels; for 2D and, for contrast, 3D, seeds 0-2 pooled, TEST and VAL:
  * on ABSTAIN scenes (nothing works): how often ours still tries a repair, and how confident it was (best plan score p11)
  * on task scenes: the p11 of plans that really succeed vs fail, so the overlap with the abstain scenes is visible
  * what each eta would give (abstain caught vs task successes lost), all from stored scores (decision = REPAIR iff p11 >= eta)
    python giveup2d.py   -> runs/explore/giveup.json
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402  (frozen; read-only)

OUT = H.HZ.ABF.parent / "explore"
ETAS = (0.3, 0.5, 0.7, 0.9)
BINS = (0, 0.3, 0.5, 0.7, 0.9, 0.95, 1.01)


def hist(v):
    return {f"{a}-{b}": sum(a <= x < b for x in v) for a, b in zip(BINS, BINS[1:])}


def one(dom, split):
    priv, L = H.priv_of(dom, split), H.labels(dom)
    ab_rep, task_ok, task_bad, ab_all = [], [], [], 0
    for s in (0, 1, 2):
        for r in H.ours(dom, split, s):
            m, d, p = priv[r["root_id"]]["masks"], r["decisions"][0], r["p11"][0]
            if m["eval_abstain"]:
                ab_all += 1
                if d:                                     # non-empty move list = REPAIR on a scene where nothing works
                    ab_rep.append(p)
            elif m["eval_task"] and d:
                o = H.V.outcome(d, L, r["root_id"])
                (task_ok if o == "SUCCESS" else task_bad).append(p)
    assert ab_all and (task_ok or task_bad), f"{dom} {split}: empty groups"
    return {"abstain_scenes_runs": ab_all, "abstain_scenes_repaired": len(ab_rep),
            "p11_hist": {"abstain_scene_repairs": hist(ab_rep), "task_repair_success": hist(task_ok), "task_repair_fail": hist(task_bad)},
            "if_eta": {e: {"abstain_repairs_kept": sum(x >= e for x in ab_rep), "task_successes_kept": sum(x >= e for x in task_ok),
                           "task_failures_kept": sum(x >= e for x in task_bad)} for e in ETAS},
            "note": "eta counts only re-threshold the frozen pick; a repair dropped below eta becomes ABSTAIN"}


if __name__ == "__main__":
    res = {"label": "EXPLORATORY, stored pools, frozen rule (noop gate 0.5, eta 0.3), seeds 0-2 pooled (counts are runs)"}
    for dom in ("2d", "3d"):
        for split in ("val", "test"):
            res[f"{dom}_{split}"] = one(dom, split)
    (OUT / "giveup.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if k != "label":
            print(k, v["abstain_scenes_runs"], v["abstain_scenes_repaired"], json.dumps(v["if_eta"]))
