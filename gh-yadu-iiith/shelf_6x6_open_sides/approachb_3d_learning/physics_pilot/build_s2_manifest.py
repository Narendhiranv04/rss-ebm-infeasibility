"""S2 resumable, root-sharded historical recovery work order (BD3DS66_X_V1 / BD3DS66_Y_REPLAY_V1).  Read-only; no physics.

    python3 -I build_s2_manifest.py RAW_DIR V3_DIR S0_HISTORICAL_PREFIXES.json(.gz) S1_PILOT_MANIFEST.json OUT.json

Units are whole roots, ordered train -> val -> test and by sha256('S2|'+root) inside each split, so a chunk = a contiguous run of roots and every
recovered root has ALL its Y-positive replays, executed-prefix X queries and relocation-failure reconfirmations together (decided (X, Y)
pairs per root).  Items already finished in S1 are skipped by item id (idempotent).  Negative re-confirmations run once (their determinism was
shown in S1); positives keep the independent replay.  Y for the 2,929 X-candidate endpoints WITHOUT any Y record needs a new target plan +
execute + replay protocol (BD3DS66_Y_PLAN_V1) that is NOT implemented here: listed as a separate, unscheduled package."""
import gzip, hashlib, json, sys, collections
from pathlib import Path

RAW, V3, PREF, S1, OUT = map(Path, sys.argv[1:6])
split = json.loads((V3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
qc = sorted(json.loads((V3 / "QC_ELIGIBLE_ROOTS.json").read_text())["roots"])
P = json.loads((gzip.open(PREF, "rt") if str(PREF).endswith(".gz") else open(PREF)).read())["H"]
s1 = json.loads(S1.read_text())["items"]
done_y = {(x["root_id"], json.dumps(x["edit"], sort_keys=True)) for x in s1 if x["kind"] == "Y"}
done_x = {(x["root_id"], x["x_key"]) for x in s1 if x["kind"] == "X"}
order = sorted(qc, key=lambda r: ({"train": 0, "val": 1, "test": 2}[split[r]], hashlib.sha256(f"S2|{r}".encode()).hexdigest()))
byroot = collections.defaultdict(list)
for k, v in P.items():
    byroot[k.split("::")[0]].append((k.split("::")[1], v))
items, noy = [], 0
for rank, rid in enumerate(order):
    d = json.loads((RAW / f"roots/{rid}.json").read_text())
    cells0 = {k: tuple(v) for k, v in d["initial_cells"].items()}
    ykeys = set()
    for fam, r in [("original", d["labels"]["original"])] + [("D", r) for r in d["labels"]["D"]]:
        e = r.get("edit", {})
        end = dict(cells0); end.update({o: tuple(c) for o, c in e.items()})
        ykeys.add("|".join(f"{o}@{c[0]},{c[1]}" for o, c in sorted(end.items())))
        if r["y"] == 1 and (rid, json.dumps(e, sort_keys=True)) not in done_y:
            items.append({"item": f"S2Y-{rid}-{len(items):05d}", "kind": "Y", "phase": f"CHUNK{rank // 30:03d}", "root_rank": rank, "root_id": rid,
                          "split": split[rid], "family": fam, "edit": e, "archived_y": 1, "negative_kind": None, "witness_id": r.get("witness"),
                          "grasp_id": r.get("grasp_id"), "action": "REPLAY_STORED_TARGET_WITNESS", "stratum": f"Y_POS_{fam}", "independent_replay": True})
    for key, v in sorted(byroot.get(rid, [])):
        if (rid, key) in done_x:
            continue
        if v["ykey"] not in ykeys:
            noy += 1
        items.append({"item": f"S2X-{rid}-{len(items):05d}", "kind": "X", "phase": f"CHUNK{rank // 30:03d}", "root_rank": rank, "root_id": rid,
                      "split": v["split"], "K": v["K"], "x_key": key, "m": v["m"], "status_in_store": v["status"], "negative_kind": v.get("negative_kind"),
                      "y_endpoint_key": v["ykey"], "endpoint_has_y_record": v["ykey"] in ykeys, "stratum": f"X_{v['status']}",
                      "action": "REPLAY_RELOCATIONS+CONNECTOR+INDEPENDENT_REPLAY" if v["status"] == "RELOCATIONS_EXECUTED" else "REPLAY_TO_FAILING_STEP+RECONFIRM",
                      "independent_replay": v["status"] == "RELOCATIONS_EXECUTED"})
c = collections.Counter((x["split"], x["kind"], x["action"]) for x in items)
man = {"manifest": "S2_HISTORICAL_RECOVERY_V1", "status": "WORK_ORDER (only chunks explicitly launched are executed)", "chunk_roots": 30,
       "chunks": 1 + (len(order) - 1) // 30, "roots": len(order), "items": len(items),
       "counts": {"|".join(k): n for k, n in sorted(c.items())}, "x_prefixes_whose_endpoint_has_no_y_record": noy,
       "unscheduled_package": {"BD3DS66_Y_PLAN_V1": "2,929 X-candidate endpoints without a Y record: plan target from q_ready in the ideal endpoint, execute, replay"},
       "skipped_already_done_in_S1": {"Y": len(done_y), "X": len(done_x)}, "items_list": items}
man["sha256_of_items"] = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
man["items"] = man.pop("items_list")
OUT.write_text(json.dumps(man))
print(json.dumps({k: v for k, v in man.items() if k != "items"}, indent=1))
