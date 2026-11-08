"""Separate HARD-ORDER CHALLENGE COHORT proposal for the 75 DEPENDENT 3D roots (outside the frozen 1,582-root cohort; no inherited split).

    python3 -I challenge_cohort.py RAW_DIR V3_DIR OUT.json

Rule (predeclared here, applied mechanically; the frozen cohort split and its test set are NOT changed):
  * lineage link = same parent_family_id as a cohort root, OR an identical layout (objects + initial cells + goal) to a cohort root;
  * linked roots inherit the strictest linked split (test > val > train), so no challenge root can leak lineage into development:
      train -> CHALLENGE_DEV_TRAIN, val -> CHALLENGE_DEV_VAL, test -> CHALLENGE_HOLDOUT;
  * unlinked roots: sha256('BWL_CHALLENGE|'+root) order, the first 40 % -> CHALLENGE_HOLDOUT, the rest -> CHALLENGE_DEV_TRAIN (half of those,
    by the same order, -> CHALLENGE_DEV_VAL).
Disclosure: the stored composite B outcomes of these roots were inspected in aggregate (reuse audit; this script counts order discordance), so
CHALLENGE_HOLDOUT is 'aggregate-inspected', not a pristine blind test; its X / Y labels do not exist yet and must be collected under BD3DS66_X_V1."""
import hashlib, json, sys, collections
from pathlib import Path

RAW, V3, OUT = map(Path, sys.argv[1:4])
split = json.loads((V3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
qc = set(json.loads((V3 / "QC_ELIGIBLE_ROOTS.json").read_text())["roots"])
lay = lambda d: json.dumps([sorted((o["id"], o["shape"], round(o["size"], 6), round(o["height"], 6)) for o in d["objects"]),
                            sorted((k, tuple(v)) for k, v in d["initial_cells"].items()), tuple(d["goal"])])  # noqa: E731
fam, layout, dep = collections.defaultdict(set), collections.defaultdict(set), []
for p in sorted((RAW / "roots").glob("*.json")):
    d = json.loads(p.read_text())
    rid = d["ids"]["root_id"]
    if rid in qc:
        fam[d["ids"].get("parent_family_id")].add(split[rid])
        layout[lay(d)].add(split[rid])
    elif d.get("ledger_observed") == "DEPENDENT":
        dep.append(d)
rank = {"test": 2, "val": 1, "train": 0}
tag = {"train": "CHALLENGE_DEV_TRAIN", "val": "CHALLENGE_DEV_VAL", "test": "CHALLENGE_HOLDOUT"}
unl = sorted((d["ids"]["root_id"] for d in dep if not (fam.get(d["ids"].get("parent_family_id")) or layout.get(lay(d)))),
             key=lambda r: hashlib.sha256(f"BWL_CHALLENGE|{r}".encode()).hexdigest())
n_hold = round(0.4 * len(unl))
rows = []
for d in dep:
    rid = d["ids"]["root_id"]
    linked = (fam.get(d["ids"].get("parent_family_id")) or set()) | (layout.get(lay(d)) or set())
    if linked:
        s = max(linked, key=rank.get)
        member, how = tag[s], f"lineage->{s}" + (" (multi-split link)" if len(linked) > 1 else "")
    else:
        k = unl.index(rid)
        member = "CHALLENGE_HOLDOUT" if k < n_hold else ("CHALLENGE_DEV_VAL" if (k - n_hold) % 2 else "CHALLENGE_DEV_TRAIN")
        how = "unlinked->seeded"
    B = d["labels"]["B"]
    ends = collections.defaultdict(list)
    for b in B:
        seq = b.get("sequence") or []
        if len(seq) >= 2:
            e = dict(d["initial_cells"]); e.update({o: c for o, c in seq})
            okn = sum(1 for _ in __import__("itertools").takewhile(lambda s: s.get("object") != "t0" and s.get("ok"), b.get("steps", [])))
            ends[json.dumps(sorted((k, tuple(v)) for k, v in e.items()))].append({"role": b["role"], "relocations_complete": okn >= len(seq), "composite_y": b.get("y")})
    groups = [g for g in ends.values() if len(g) >= 2]
    rows.append({"root_id": rid, "observed_K": d.get("observed"), "member": member, "assignment": how, "same_endpoint_groups": len(groups),
                 "relocation_discordant_groups": sum(1 for g in groups if len({x["relocations_complete"] for x in g}) > 1),
                 "composite_discordant_groups": sum(1 for g in groups if len({x["composite_y"] for x in g}) > 1),
                 "stored_sequences": len([b for b in B if b.get("sequence")])})
res = {"cohort": "BWL_HARD_ORDER_CHALLENGE_V1 (proposal)", "roots": len(rows), "membership": dict(collections.Counter(r["member"] for r in rows)),
       "assignment": dict(collections.Counter(r["assignment"] for r in rows)), "by_K": dict(collections.Counter(r["observed_K"] for r in rows)),
       "same_endpoint_groups": sum(r["same_endpoint_groups"] for r in rows), "relocation_discordant_groups": sum(r["relocation_discordant_groups"] for r in rows),
       "composite_discordant_groups": sum(r["composite_discordant_groups"] for r in rows), "rule": __doc__.split("Rule", 1)[1].strip(), "rows": rows}
res["sha256_of_rows"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "rule")}, indent=1))
