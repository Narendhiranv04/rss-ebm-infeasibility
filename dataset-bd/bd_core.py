"""Shared B/D dataset core (approach_D_server_dataset_bootstrap_prompt.md): factor specification, deterministic deficit-based wave sampler,
immutable ids, binary label protocol, content-addressed witness store, atomic per-root records, append-only ledger, manifests and
root-grouped split manifests. Domain adapters (bd_2d.py, bd_3d.py) provide construct() and teach(); nothing here plans motion.

One root scene serves both methods: B gets ordered correction sequences, D gets original-target-T queries on the original and on
hypothetical (edited) scenes; both are views of the same record. Labels are binary (1 = verified within the declared per-query teacher
budget, 0 otherwise) with the reason kept as metadata; a 0 is never a proof of physical impossibility."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

GENERATOR_VERSION = "BD_GEN_V1"
LABEL_PROTOCOL = "BD_BINARY_BUDGET_V1"
MAX_ACTIONS, MAX_NECESSARY, MAX_REDUNDANT, MAX_MOVABLE = 7, 4, 2, 6
FACTORS = {"K": [0, 1, 2, 3, 4], "distractors": ["0", "1-2", "3-4"], "source": ["near", "mid", "far"],
           "displacement": ["short", "medium", "long"], "mechanism": ["grasp", "place", "transfer", "mixed"],
           "parking": ["one", "limited", "multiple"], "distractor_role": ["near_T", "near_goal", "near_path", "far"],
           "start_pose": ["home", "nonhome"]}
PAIRS = [("K", "mechanism"), ("K", "distractors"), ("K", "source"), ("K", "displacement"), ("mechanism", "distractor_role")]
TRIPLES = [("K", "mechanism", "distractor_role")]
N_DISTRACT = {"0": (0, 0), "1-2": (1, 2), "3-4": (3, 4)}
PARKING_N = {"one": 1, "limited": 2, "multiple": 3}


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def valid_spec(s) -> bool:
    """K + minimum distractor count within the tested D mask cap; K=0 has no blocker mechanism to vary (mechanism kept as a label only)."""
    return s["K"] + N_DISTRACT[s["distractors"]][0] <= MAX_MOVABLE and (s["distractors"] != "0" or s["distractor_role"] == "far")


SPECS = [dict(zip(FACTORS, v)) for v in itertools.product(*FACTORS.values())]
SPECS = [s for s in SPECS if valid_spec(s)]


def observed_key(rec):
    """Stratum used for coverage: VERIFIED K (never the requested K); 'NONE' = no verified repair within the declared domain."""
    k = rec.get("verified_K")
    return "NONE" if k is None else int(k)


def plan_wave(ledger, n, wave, salt, specs=None):
    """Deterministic greedy deficit sampler. Counts come from ACCEPTED roots with their OBSERVED K and requested other factors; the
    requested K of a new spec is the observed-K level with the largest deficit. Score = marginal + pairwise + selected triple deficits
    (uniform targets over valid levels); ties broken by a salted spec hash. Each pick is added provisionally before the next."""
    acc = [r for r in ledger if r.get("accepted")]
    cnt = {}

    def add(s, w=1.0):
        for f in FACTORS:
            cnt[(f, s[f])] = cnt.get((f, s[f]), 0) + w
        for a, b in PAIRS:
            cnt[(a, s[a], b, s[b])] = cnt.get((a, s[a], b, s[b]), 0) + w
        for t in TRIPLES:
            k = tuple(x for f in t for x in (f, s[f]))
            cnt[k] = cnt.get(k, 0) + w
    for r in acc:
        k = observed_key(r)
        if k != "NONE":
            add({**r["spec"], "K": k})
    total = max(1, len(acc))
    picks = []
    for i in range(n):
        tot = total + i

        def score(s):
            v = sum(tot / len(FACTORS[f]) - cnt.get((f, s[f]), 0) for f in FACTORS)
            v += sum(tot / (len(FACTORS[a]) * len(FACTORS[b])) - cnt.get((a, s[a], b, s[b]), 0) for a, b in PAIRS)
            for t in TRIPLES:
                d = 1
                for f in t:
                    d *= len(FACTORS[f])
                v += tot / d - cnt.get(tuple(x for f in t for x in (f, s[f])), 0)
            return v
        best = max(specs or SPECS, key=lambda s: (round(score(s), 9), sha([salt, wave, i, s])))
        picks.append(best)
        add(best)
    return picks


def check_sequence(steps, K=None):
    """Action-budget contract of every new task record: <= MAX_ACTIONS pick-and-place actions in total = the relocations in `steps`
    followed by exactly one final move of the original target T; never a relocation of T itself. Relocations beyond a verified K are
    'redundant' and may number at most MAX_REDUNDANT. Raises ValueError."""
    if any(o == "t0" for o, _ in steps):
        raise ValueError("T is moved last, once, by the task itself")
    if len(steps) + 1 > MAX_ACTIONS:
        raise ValueError(f"{len(steps) + 1} actions > {MAX_ACTIONS}")
    if K is not None and len(steps) - K > MAX_REDUNDANT:
        raise ValueError(f"{len(steps) - K} redundant relocations > {MAX_REDUNDANT}")
    return len(steps) + 1


# ------------------------------------------------------------------ ids, labels, storage
def ids(domain, name, wave, slot, spec, config_hash, teacher_version, robot_protocol):
    seed = int(sha([GENERATOR_VERSION, domain, name, wave, slot])[:8], 16)
    family = "fam-" + sha([domain, {k: spec[k] for k in FACTORS}])[:10]       # construction template (all requested factor levels)
    return {"root_id": f"{domain}-{name}-w{wave:04d}-s{slot:03d}", "parent_family_id": family, "construction_seed": seed, "variant_id": "orig",
            "config_hash": config_hash, "generator_version": GENERATOR_VERSION, "teacher_version": teacher_version, "robot_protocol": robot_protocol,
            "label_protocol": LABEL_PROTOCOL}


def binary(status_ok: bool, reason, kind=None, **meta) -> dict:
    """One teacher query outcome. y=1 iff verified within the declared budget; y=0 otherwise with negative_kind metadata
    (CERTIFIED / TIMEOUT / NO_PLAN_FOUND / OTHER_FAILURE). Never a third label."""
    if not status_ok and kind not in ("CERTIFIED", "TIMEOUT", "NO_PLAN_FOUND", "OTHER_FAILURE"):
        raise ValueError(f"negative_kind {kind!r}")
    return {"y": int(bool(status_ok)), "reason": reason, "negative_kind": None if status_ok else kind, **meta}


class Store:
    """<root>/roots/<root_id>.json (atomic), <root>/witnesses/<aa>/<sha>.npz (content addressed, never duplicated),
    <root>/ledger.jsonl (append-only: accepted roots and construction rejections with reasons)."""

    def __init__(self, root):
        self.root = Path(root)
        for d in ("roots", "witnesses", "viz"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def witness(self, arrays: dict) -> str:
        h = hashlib.sha256()
        for k in sorted(arrays):
            v = np.ascontiguousarray(np.asarray(arrays[k]))
            h.update(f"{k}|{v.dtype.str}|{v.shape}".encode())
            h.update(v.tobytes())
        wid = h.hexdigest()[:24]
        p = self.root / "witnesses" / wid[:2] / f"{wid}.npz"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
            os.replace(tmp, p)
        return wid

    def load_witness(self, wid):
        z = np.load(self.root / "witnesses" / wid[:2] / f"{wid}.npz")
        return {k: z[k] for k in z.files}

    def save_root(self, rec):
        p = self.root / "roots" / f"{rec['ids']['root_id']}.json"
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=_jd))
        os.replace(tmp, p)

    def append(self, entry):
        with open(self.root / "ledger.jsonl", "a") as f:
            f.write(json.dumps(entry, sort_keys=True, default=_jd) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def ledger(self):
        p = self.root / "ledger.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _jd(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def manifest(store: Store, domain: str) -> dict:
    """Master manifest (root-grouped, factor rich, no split baked in) + split manifests derived from it without touching any data."""
    rows = []
    for p in sorted((store.root / "roots").glob("*.json")):
        r = json.loads(p.read_text())
        rows.append({**r["ids"], "requested": r["spec"], "observed": r["observed"], "verified_K": r["verified_K"], "n_movable": r["n_movable"],
                     "y_original": r["labels"]["original"]["y"], "mask_valid": r["labels"]["blocker"]["blocker_supervision_valid"],
                     "n_D_records": len(r["labels"]["D"]), "n_B_records": len(r["labels"]["B"]), "file": str(p.relative_to(store.root))})
    grp = lambda r: r["parent_family_id"]  # noqa: E731   (all descendants live inside one root file; families stay together)
    splits = {"iid": {}, "ood_K": {}, "ood_obstacle_count": {}, "ood_source_region": {}, "ood_distractor_density": {}}
    for r in rows:
        u = int(sha(["split", grp(r)])[:8], 16) / 2 ** 32
        splits["iid"][r["root_id"]] = "test" if u < 0.15 else "val" if u < 0.25 else "train"
        k = r["verified_K"]
        splits["ood_K"][r["root_id"]] = "unassigned" if k is None else "train" if k <= 2 else "ood_test"
        splits["ood_obstacle_count"][r["root_id"]] = "train" if r["n_movable"] <= 4 else "ood_test"
        splits["ood_source_region"][r["root_id"]] = "ood_test" if r["requested"]["source"] == "far" else "train"
        splits["ood_distractor_density"][r["root_id"]] = "ood_test" if r["requested"]["distractors"] == "3-4" else "train"
    fam = {}
    for r in rows:                       # integrity: a family never straddles iid train / test
        fam.setdefault(grp(r), set()).add(splits["iid"][r["root_id"]])
    assert all(len(v) == 1 for v in fam.values())
    out = {"domain": domain, "generator_version": GENERATOR_VERSION, "label_protocol": LABEL_PROTOCOL, "n_roots": len(rows), "rows": rows,
           "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (store.root / "MANIFEST.json").write_text(json.dumps(out, indent=1, default=_jd))
    (store.root / "SPLITS.json").write_text(json.dumps({"grouping": "parent_family_id", "splits": splits}, indent=1))
    return out
