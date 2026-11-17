"""The 2D evaluation set exactly as Approach B sees it: selection V3 rows, the scene file of each root and its evaluation masks
(approach_b_final/contracts.py compile_domain, 2D branch, recomputed from the finalized X/Y label manifest)."""
from __future__ import annotations

import collections
import functools
import json
from pathlib import Path

from . import paths as P


@functools.lru_cache(maxsize=1)
def selection() -> dict:
    return {r["root_id"]: r for r in json.loads(P.SEL2.read_text())["rows"]}


def path_of(rid: str) -> Path:
    r = selection()[rid]
    if r["source"] == "ORIGINAL_SELECTED":
        return P.DS2 / "raw/roots" / f"{rid}.json"
    hs = r["host_store"]
    return P.FDG / hs[hs.index("final_dataset_generation/") + len("final_dataset_generation/"):] / "roots" / f"{rid}.json"


def record(rid: str) -> dict:
    return json.loads(path_of(rid).read_text())


def geom_of(rec: dict) -> str:
    return "fastgeom1" if rec.get("geometry_impl") == "fastgeom1" else "original"


@functools.lru_cache(maxsize=1)
def masks() -> dict:
    """{root_id: {eval_task, eval_repair, eval_noop, eval_abstain, K, split, K_status}} as contracts.compile_domain computes them."""
    by = collections.defaultdict(list)
    for l in json.loads(P.XY2.read_text())["rows"]:
        by[l["root_id"]].append(l)
    out = {}
    for rid, r in selection().items():
        K = r["K"]
        X, Y = {}, {}
        for l in by.get(rid, []):
            seq = [[o, int(d)] for o, d in l["relocations"]]
            ek = json.dumps(sorted(map(tuple, seq)))
            if l["y"] in (0, 1):
                Y.setdefault(ek, l["y"])
            if seq and l["x"] in (0, 1):
                X.setdefault(json.dumps(seq), {"seq": seq, "x": l["x"], "kind": l["row_kind"]})
        sel = [q for q in X.values() if q["kind"] == "SELECTED_ASSIGNMENT_ORDER"]
        yof = lambda q: Y.get(json.dumps(sorted(map(tuple, q["seq"]))))  # noqa: E731
        has = any(q["x"] == 1 and yof(q) == 1 for q in sel)
        om = any(q["kind"] == "OMISSION_YNEG" and q["x"] == 1 and yof(q) == 1 for q in X.values())
        ks = ("NOT_APPLICABLE_K0" if K == "0" else "NOT_APPLICABLE_NONE" if K == "NONE" else
              "UNCERTIFIED_NO_FINALIZED_LABELS" if not sel else "FLAGGED_SHORTER_REPAIR_PROVEN" if om else
              "FLAGGED_NO_FINALIZED_X1Y1_ORDER" if not has else "LEGACY_MINIMAL_AND_FINALIZED_REPAIR_VERIFIED")
        out[rid] = {"K": K, "split": r["split"], "K_status": ks, "eval_task": K in ("1", "2", "3"),
                    "eval_repair": ks == "LEGACY_MINIMAL_AND_FINALIZED_REPAIR_VERIFIED",
                    "eval_noop": K == "0" and Y.get("[]") == 1, "eval_abstain": K == "NONE"}
    return out


def roots(split: str = "test") -> list[str]:
    """Root ids of a split in a fixed order (class, then id)."""
    order = {"0": 0, "1": 1, "2": 2, "3": 3, "NONE": 4}
    return sorted((rid for rid, r in selection().items() if r["split"] == split), key=lambda rid: (order[selection()[rid]["K"]], rid))
