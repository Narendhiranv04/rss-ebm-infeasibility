"""P5.4: one common learner-view dataset for A, B and C (no teacher call).

Two namespaces per root. INPUTS (learner-visible): data.learner_tensors_v2 of the initial
observation (allowlisted fields only), frozen DINOv2-small patch tokens mask-pooled per object
from the initial appearance anchor, canonical candidate sequences as (object slot, destination)
with ideal prefix positions, and predeclared fixed-robot edits as pose changes (same anchor).
PRIVILEGED (training targets only): teacher label, source, weight, first-witness cost, V2 z-slot
labels and analytic static-factor labels. Root keys / lineage / split are audit metadata and
never enter a tensor. Holdout roots are loaded only by `load(..., sealed=True)`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import data
import proposals as pr
import zlayout as zl
from scene import scene_from_dict

DINO = ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056")
IMAGE_PX, GRID = 224, 16
INPUT_KEYS = ("object_xy", "object_radius", "is_active_target", "is_obstacle", "robot_q", "goal_xy", "robot_fk_joints", "visual")


def dino():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    path = snapshot_download(DINO[0], revision=DINO[1], local_files_only=True, allow_patterns=["*.json", "*.safetensors"])
    model = AutoModel.from_pretrained(path, local_files_only=True).eval()
    pre = json.loads((Path(path) / "preprocessor_config.json").read_text())
    return model, np.array(pre["image_mean"]), np.array(pre["image_std"])


@torch.no_grad()
def visual_features(model, mean, std, anchor) -> np.ndarray:
    """Frozen patch tokens (16 x 16 x 384) of the initial RGB, area-weighted per instance mask."""
    rgb = anchor["rgb"].astype(np.float64) / 255.0
    if rgb.shape[:2] != (IMAGE_PX, IMAGE_PX):
        raise ValueError(f"rendered observation {rgb.shape} is not {IMAGE_PX} square")
    x = torch.tensor(((rgb - mean) / std).transpose(2, 0, 1)[None], dtype=torch.float32)
    tok = model(pixel_values=x).last_hidden_state[0, 1:].reshape(GRID, GRID, -1).numpy()
    p = IMAGE_PX // GRID
    out = []
    for k in range(1, len(anchor["ids"]) + 1):
        w = (anchor["instance"] == k).reshape(GRID, p, GRID, p).mean((1, 3))
        out.append((tok * w[..., None]).sum((0, 1)) / max(w.sum(), 1e-9))
    return np.array(out, np.float32)


def _static(scene, cells, ids, cfg):
    """Analytic static labels (support per object endpoint, pair compatibility); gray band -> None."""
    st = pr.static_endpoints(scene, cells, "t0", cfg)
    bad = {(f["factor"], tuple(sorted(f["objects"]))): f["status"] for f in st["failed"]}
    lab = lambda key: None if bad.get(key) == "UNRESOLVED" else int(key not in bad)  # noqa: E731
    unary = [lab(("support", (i,))) for i in ids]
    pair = {f"{a}|{b}": lab(("target_at_rest" if "t0" in (a, b) else "pair", (a, b))) for i, a in enumerate(ids) for b in ids[i + 1:]}
    return {"unary": unary, "pair": pair}


def root_record(cfg, model, mean, std, key, scene, goal, cands, edits, labels, split, source) -> dict:
    q = data.make_query(cfg, key, scene, "t0", goal, {}, "SAFE_RETREAT", data.PROTOCOL_V2)
    anchor = data.appearance_anchor(q, IMAGE_PX)
    t = data.learner_tensors_v2(q, cfg, anchor)
    ids = [o["id"] for o in q["learner"]["scene"]["objects"]]
    if ids != list(anchor["ids"]):
        raise ValueError("object storage order differs from the appearance anchor")
    inputs = {k: np.asarray(t[k], np.float32) for k in INPUT_KEYS if k != "visual"} | {"visual": visual_features(model, mean, std, anchor)}
    lat = scene.lattice
    rows = []
    for c in cands:
        seq = [(ids.index(o), list(lat.center(d))) for o, d in c["sequence"]]
        lab = labels.get(c["query"], {})
        zq = c["z_layout"]["query"]
        rows.append({"actions": seq, "z_slots": list(zq), "z_queries": list(zq.values()),
                     "privileged": {"query": c["query"], "y": lab.get("y"), "source": lab.get("source"), "weight": lab.get("weight", 1.0),
                                    "cost": lab.get("cost"), "z": [labels.get(x, {}).get("y") for x in zq.values()],
                                    "z_source": [labels.get(x, {}).get("source") for x in zq.values()],
                                    "static": _static(scene, {o: d for o, d in c["sequence"]}, ids, cfg)}})
    ed = [{"kind": e["kind"], "poses": {ids.index(o): list(lat.center(d)) for o, d in e["edit"].items()}, "eligible": e["eligible"],
           "privileged": {"query": e["query"], "y": labels.get(e["query"], {}).get("y"),
                         "source": labels.get(e["query"], {}).get("source")}} for e in edits]
    return {"inputs": inputs, "candidates": rows, "edits": ed, "audit": {"root": key, "split": split, "source": source,
                                                                         "ids": ids, "anchor_hash": anchor["hash"]}}


def _labels(decisions: dict, masks: dict) -> dict:
    return {q: {"y": r["binary_training_label"], "source": r["label_source"], "weight": r["training_weight"],
                "cost": (masks.get(q) or {}).get("cost_J")} for q, r in decisions.items()}


def build(cfg, out: Path, p4=Path("runs/g4c_lf_p4/pilot"), d3=Path("runs/p5/d3lite"), sealed=False) -> dict:
    model, mean, std = dino()
    rd = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    roots = []
    pp, lab = rd(p4 / "pilot_plan.json"), _labels(rd(p4 / "decisions.json"), rd(p4 / "loss_masks.json"))
    for k, r in pp["roots"].items():
        cat = [{"kind": q["kind"], "edit": q["edit"], "eligible": q["eligible"]} for q in r["queries"]["scene_queries"]]
        seqs = [q["query"]["sequence"] for q in rd(p4 / "decisions.json").values() if q["query"]["root"] == k and q["query"]["kind"] != "SCENE_TARGET_ONLY"]
        cands = [{"sequence": s, "query": f"seq:{k}:{zl.sig(s)}", "z_layout": zl.build(k, s, cat)} for s in seqs]
        edits = [{**c, "query": f"scene:{k}:{zl.sig(sorted(c['edit'].items()))}"} for c in cat]
        roots.append(root_record(cfg, model, mean, std, k, scene_from_dict(r["scene"]), r["goal_cell"], cands, edits, lab, "train", "P4"))
    if not (d3 / "decisions.json").exists():                      # development smoke runs on P4 only
        plan, dec = {"roots": {}}, {}
    else:
        plan = rd(d3 / "plan.json")
        dec = rd(d3 / "decisions.json") | (rd(d3 / "sealed_holdout.json") if sealed else {})
    lab = _labels({q: r for q, r in dec.items() if r["teacher_status"] != "NOT_RUN"}, rd(d3 / "loss_masks.json") if dec else {})
    for k, r in plan["roots"].items():
        if r["split"] == "holdout" and not sealed:
            continue
        if r["queries"] is None:                                   # reference scene: its own one-candidate root
            cands, edits = [{"sequence": [], "query": f"seq:{k}:empty", "z_layout": zl.build(k, [], [])}], []
        else:
            cands = [{"sequence": p["sequence"], "query": p["z_layout"]["query"].get("zR", f"seq:{k}:empty"), "z_layout": p["z_layout"]}
                     for p in r["queries"]["pool"]]
            edits = [{**c, "query": f"scene:{k}:{zl.sig(sorted(c['edit'].items()))}"} for c in r["queries"]["scene_queries"]]
        roots.append(root_record(cfg, model, mean, std, k, scene_from_dict(r["scene"]), r["goal_cell"], cands, edits, lab, r["split"], "D3"))
    ds = {"version": "P5_LEARN_VIEW_V1", "z_layout": zl.VERSION, "dino": list(DINO), "roots": roots, "input_keys": list(INPUT_KEYS)}
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ds, out / ("dataset_sealed.pt" if sealed else "dataset.pt"))
    return ds


def leakage_audit(ds) -> list:
    """Every learner tensor key is allowlisted; no privileged / audit string reaches inputs."""
    bad = []
    for r in ds["roots"]:
        if set(r["inputs"]) != set(INPUT_KEYS):
            bad.append(f"{r['audit']['root']}: input keys {sorted(r['inputs'])}")
        for c in r["candidates"]:
            if set(c) - {"actions", "z_slots", "z_queries", "privileged"}:
                bad.append("candidate carries extra fields")
    return bad


def summary(ds) -> dict:
    n = lambda f: sum(1 for r in ds["roots"] for c in r["candidates"] if f(c["privileged"]))  # noqa: E731
    return {"roots": len(ds["roots"]), "candidates": n(lambda p: True), "decided": n(lambda p: p["y"] is not None),
            "positive": n(lambda p: p["y"] == 1), "negative": n(lambda p: p["y"] == 0), "budget_negative": n(lambda p: p["source"] == "PLANNER_BUDGET"),
            "cost_labels": n(lambda p: p["cost"] is not None), "complete_z": n(lambda p: all(z is not None for z in p["z"]))}
