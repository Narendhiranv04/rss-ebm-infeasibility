"""R0: ONE canonical, deduplicated 3D X-prefix / Y-endpoint query ledger + the raw->unique->eligible->done->remaining reconciliation.

    python3 -I r0_ledger.py RAW_DIR V3_DIR S1_MANIFEST S1_SUMMARY S2_MANIFEST.json.gz S2_SUMMARY OUT_DIR

X key = (root, ordered moves) under BD3DS66_X_V1 (q0 = q_ref = q_ready, stop-point connector); Y key = (root, full endpoint arrangement) under
BD3DS66_Y_REPLAY_V1 / planning record.  Every stored B sequence (m >= 1) contributes each prefix it physically attempted: prefixes whose
relocations all executed -> X candidate (connector pending); the first failing prefix -> X=0 candidate (reconfirm); later prefixes of a failed
sequence were never attempted -> not a query.  The 9 attempted-but-unstored BFS tries carry no record -> MISSING (no key can be formed)."""
import gzip, hashlib, json, sys, collections
from pathlib import Path

RAW, V3 = Path(sys.argv[1]), Path(sys.argv[2])
S1M, S1S, S2M, S2S, OUT = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], Path(sys.argv[7])
split = json.loads((V3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
qc = sorted(json.loads((V3 / "QC_ELIGIBLE_ROOTS.json").read_text())["roots"])
ck = lambda c: f"{int(c[0])},{int(c[1])}"  # noqa: E731
xk = lambda mv: ".".join(f"{o}>{ck(d)}" for o, d in mv)  # noqa: E731
X, Y, raw = {}, {}, collections.Counter()
fail_seq_to_key, contra = [], []
for rid in qc:
    d = json.loads((RAW / f"roots/{rid}.json").read_text())
    c0 = {k: tuple(v) for k, v in d["initial_cells"].items()}
    ek = lambda mv: "|".join(f"{o}@{ck(c)}" for o, c in sorted({**c0, **{o: tuple(c) for o, c in mv}}.items()))  # noqa: E731
    for fam, r in [("original", d["labels"]["original"])] + [("D", r) for r in d["labels"]["D"]]:
        Y[(rid, ek(list(r.get("edit", {}).items())))] = {"split": split[rid], "K": d["verified_K"], "source": f"record:{fam}", "archived_y": r["y"],
                                                          "negative_kind": r.get("negative_kind"), "witness": r.get("witness"), "x_refs": 0}
    raw["sequence_tries_total"] += d["labels"]["blocker"].get("sequence_tries", 0) if isinstance(d["labels"].get("blocker"), dict) else 0
    for b in d["labels"]["B"]:
        seq = [(o, tuple(c)) for o, c in b.get("sequence") or []]
        if not seq:
            raw["empty_sequence_target_runs_K0"] += 1
            continue
        raw["stored_sequences"] += 1
        okn = 0
        for s in b.get("steps", []):
            if s.get("object") == "t0" or not s.get("ok"):
                break
            okn += 1
        complete = okn >= len(seq)
        raw["complete_sequences" if complete else "relocation_failed_sequences"] += 1
        for L in range(1, min(len(seq), okn + 1) + 1):
            key = (rid, xk(seq[:L]))
            st = "EXECUTED" if L <= okn else "RELOCATION_FAILED"
            e = X.setdefault(key, {"split": split[rid], "K": d["verified_K"], "m": L, "status": st, "y_key": ek(seq[:L]), "roles": [],
                                   "negative_kind": b.get("negative_kind") if st == "RELOCATION_FAILED" else None, "attempts": 0})
            if e["status"] != st:
                contra.append({"root": rid, "x_key": key[1], "statuses": [e["status"], st]})
            e["roles"].append(f"{b['role']}@{L}/{len(seq)}"); e["attempts"] += 1
            if st == "RELOCATION_FAILED":
                fail_seq_to_key.append((rid, b["role"], key[1]))
# outcomes already verified (S1 + S2 chunk 0), joined by key
done = {}
for mf, sf in ((S1M, S1S), (S2M, S2S)):
    man = json.loads((gzip.open(mf, "rt") if mf.endswith(".gz") else open(mf)).read())
    it = {x["item"]: x for x in man["items"]}
    for r in json.loads(Path(sf).read_text())["rows"]:
        x = it[r["item"]]
        if x["kind"] == "X":
            done[("X", x["root_id"], x["x_key"])] = r["final"]
        elif r["final"] == "Y1_REPLAY_VERIFIED":
            d = json.loads((RAW / f"roots/{x['root_id']}.json").read_text())
            c0 = {k: tuple(v) for k, v in d["initial_cells"].items()}; c0.update({k: tuple(v) for k, v in x["edit"].items()})
            done[("Y", x["root_id"], "|".join(f"{o}@{ck(c)}" for o, c in sorted(c0.items())))] = r["final"]
        elif r["final"].startswith("Y0_RECONFIRMED"):
            d = json.loads((RAW / f"roots/{x['root_id']}.json").read_text())
            c0 = {k: tuple(v) for k, v in d["initial_cells"].items()}; c0.update({k: tuple(v) for k, v in x["edit"].items()})
            done[("Y", x["root_id"], "|".join(f"{o}@{ck(c)}" for o, c in sorted(c0.items())))] = r["final"]
for k, e in X.items():
    e["verified"] = done.get(("X",) + k)
    if (k[0], e["y_key"]) in Y:
        Y[(k[0], e["y_key"])]["x_refs"] += 1
    else:
        Y.setdefault((k[0], e["y_key"]), {"split": e["split"], "K": e["K"], "source": "x_endpoint_no_record", "archived_y": None, "negative_kind": None,
                                          "witness": None, "x_refs": 0})["x_refs"] += 1
for k, e in Y.items():
    e["verified"] = done.get(("Y",) + k)
fails = [e for e in X.values() if e["status"] == "RELOCATION_FAILED"]
fk = collections.Counter((rr, k) for rr, _, k in fail_seq_to_key)
rec = {
    "x_relocation_negatives": {
        "raw_relocation_failed_sequences_stored": raw["relocation_failed_sequences"],
        "unique_failing_prefix_keys": len(fails),
        "collapsed_duplicates": raw["relocation_failed_sequences"] - len(fails),
        "collapse_explanation": "several stored sequences of one root share the SAME failing prefix (identical first move(s) from the identical S0 / q_ready: "
                                "deterministic, so one physical query); dominant case: noop_irrelevant [d] and redundant+1 [d, ...] both fail at [d]",
        "collapsed_keys_by_role_set": dict(collections.Counter("+".join(sorted({r for rr, r, kk in fail_seq_to_key if (rr, kk) == k})) for k, n in fk.items() if n > 1)),
        "collapsed_sequences_by_role_set": dict(sum((collections.Counter({"+".join(sorted({r for rr, r, kk in fail_seq_to_key if (rr, kk) == k})): n - 1}) for k, n in fk.items() if n > 1), collections.Counter())),
        "keys_with_multiple_sequences": sum(1 for n in fk.values() if n > 1),
        "already_reconfirmed": sum(1 for e in fails if e["verified"]),
        "remaining_eligible": sum(1 for e in fails if not e["verified"]),
        "ineligible_or_out_of_scope": 0,
        "unstored_attempted_tries_missing": "9 (audit: sequence_tries - stored minimal; no record, no key; MISSING)",
        "status_contradictions_same_key": contra,
        "by_split": dict(collections.Counter(e["split"] for e in fails)), "by_kind": dict(collections.Counter(e["negative_kind"] for e in fails))},
    "x_executed": {"unique_keys": sum(1 for e in X.values() if e["status"] == "EXECUTED"),
                   "verified": dict(collections.Counter(e["verified"] for e in X.values() if e["status"] == "EXECUTED" and e["verified"])),
                   "remaining": sum(1 for e in X.values() if e["status"] == "EXECUTED" and not e["verified"]),
                   "by_split_remaining": dict(collections.Counter(e["split"] for e in X.values() if e["status"] == "EXECUTED" and not e["verified"]))},
    "y": {"records": sum(1 for e in Y.values() if e["source"].startswith("record")),
          "record_positives": sum(1 for e in Y.values() if e["archived_y"] == 1),
          "record_positives_replay_verified": sum(1 for e in Y.values() if e["archived_y"] == 1 and e["verified"] == "Y1_REPLAY_VERIFIED"),
          "record_positives_remaining": sum(1 for e in Y.values() if e["archived_y"] == 1 and e["verified"] != "Y1_REPLAY_VERIFIED"),
          "record_negatives": sum(1 for e in Y.values() if e["archived_y"] == 0),
          "record_negatives_reconfirmed": sum(1 for e in Y.values() if e["archived_y"] == 0 and e["verified"]),
          "x_endpoints_without_record": sum(1 for e in Y.values() if e["source"] == "x_endpoint_no_record"),
          "x_endpoints_without_record_from_executed_prefixes": len({(k[0], e["y_key"]) for k, e in X.items() if e["status"] == "EXECUTED" and (k[0], e["y_key"]) in Y and Y[(k[0], e["y_key"])]["source"] == "x_endpoint_no_record"}),
          "x_endpoints_without_record_by_split": dict(collections.Counter(e["split"] for e in Y.values() if e["source"] == "x_endpoint_no_record"))},
    "raw": dict(raw)}
OUT.mkdir(parents=True, exist_ok=True)
ledger = {"X": [{"root_id": k[0], "x_key": k[1], **e} for k, e in sorted(X.items())], "Y": [{"root_id": k[0], "y_key": k[1], **e} for k, e in sorted(Y.items())]}
blob = json.dumps(ledger, sort_keys=True).encode()
(OUT / "R0_CANONICAL_QUERY_LEDGER.json.gz").write_bytes(gzip.compress(blob, 9, mtime=0))
rec["ledger_sha256_uncompressed"] = hashlib.sha256(blob).hexdigest()
rec["ledger_counts"] = {"X_keys": len(X), "Y_keys": len(Y)}
(OUT / "R0_QUERY_RECONCILIATION.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
