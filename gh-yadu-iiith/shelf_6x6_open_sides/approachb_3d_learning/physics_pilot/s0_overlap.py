"""S0: canonical keys + dedup overlap of (historical relocation prefixes) x (BWL-3 X pilot) x (7,012 Y records). Read-only."""
import gzip, json, sys, collections
from pathlib import Path
RAW, V3, PILOT, YROWS, OUT = map(Path, sys.argv[1:6])
split = json.loads((V3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
qc = set(json.loads((V3 / "QC_ELIGIBLE_ROOTS.json").read_text())["roots"])
ck = lambda c: f"{int(c[0])},{int(c[1])}"
xkey = lambda moves: ".".join(f"{o}>{ck(d)}" for o, d in moves)                      # canonical ordered move list
def ykey(cells0, moves):
    e = dict(cells0); [e.__setitem__(o, tuple(d)) for o, d in moves]
    return "|".join(f"{o}@{ck(c)}" for o, c in sorted(e.items()))                     # canonical endpoint arrangement
H, Hy, Hstat, Yrec, roots = {}, {}, collections.Counter(), {}, {}
for rid in sorted(qc):
    d = json.loads((RAW / f"roots/{rid}.json").read_text()); sp = split[rid]
    cells0 = {k: tuple(v) for k, v in d["initial_cells"].items()}; roots[rid] = (sp, d["verified_K"], cells0)
    for fam, r in [("original", d["labels"]["original"])] + [("D", r) for r in d["labels"]["D"]]:
        Yrec[(rid, ykey(cells0, list(r.get("edit", {}).items())))] = (fam, r["y"], r.get("negative_kind"), bool(r.get("witness")))
    for b in d["labels"]["B"]:
        seq = [(o, tuple(c)) for o, c in b.get("sequence") or []]
        if not seq: continue
        steps = b.get("steps", []); okn = 0
        for s in steps:
            if s.get("object") == "t0" or not s.get("ok"): break
            okn += 1
        for L in range(1, len(seq) + 1):                                            # every prefix whose relocations all executed
            k = (rid, xkey(seq[:L]))
            if L <= okn:
                H.setdefault(k, {"split": sp, "K": d["verified_K"], "m": L, "complete_seq": L == len(seq), "roles": set(), "status": "RELOCATIONS_EXECUTED",
                                 "terminal_q_recorded": None})["roles"].add(b["role"])
                if L == len(seq): H[k]["terminal_q_recorded"] = bool(b.get("y") == 1 or (b.get("failed_step") == "target" and len(b.get("witnesses", [])) > len(seq)))
                Hy[k] = ykey(cells0, seq[:L])
            elif L == okn + 1:
                H.setdefault(k, {"split": sp, "K": d["verified_K"], "m": L, "complete_seq": L == len(seq), "roles": set(), "status": "RELOCATION_FAILED",
                                 "negative_kind": b.get("negative_kind"), "terminal_q_recorded": False})["roles"].add(b["role"])
                Hy[k] = ykey(cells0, seq[:L])
cnt = collections.Counter((v["status"], v["split"]) for v in H.values())
man = json.loads(gzip.open(PILOT, "rt").read())
P = {}
for r in man["rows"]:
    moves = [(o, tuple(c)) for o, c in r["ordered_moves"]]
    P[(r["root_id"], xkey(moves))] = {"kind": r["query_kind"], "source": r["destination_source"], "also": r.get("also", []), "y": ykey(roots[r["root_id"]][2], moves), "m": r["m"]}
def ycls(k):
    v = Yrec.get(k)
    if v is None: return "Y_NO_RECORD"
    return "Y_POS_PLANNING_ONLY_NEEDS_REPLAY" if v[1] == 1 else f"Y_NEG_REUSABLE_{v[2]}"
ov = collections.Counter()
for k, p in P.items():
    h = H.get(k)
    xc = "X_NEW_FULL_EXECUTION" if h is None else ("X_RELOCATIONS_STORED_NEEDS_REPLAY+CONNECTOR" if h["status"] == "RELOCATIONS_EXECUTED" else "X_RELOCATION_FAILED_STORED")
    ov[(xc, ycls((k[0], p["y"])))] += 1
hy = collections.Counter()
for k, v in H.items():
    if v["status"] == "RELOCATIONS_EXECUTED": hy[ycls((k[0], Hy[k]))] += 1
uniq_pilot_y = {(k[0], p["y"]) for k, p in P.items()}
uniq_hist_y = {(k[0], Hy[k]) for k, v in H.items() if v["status"] == "RELOCATIONS_EXECUTED"}
res = {"canonical_keys": {"X": "(root_id, X_TEACHER=BD3DS66_X_V1 [BD3DS66_TEACHER_V2 relocation executor + CONNECTOR_TO_QREADY_V1 + independent replay], q_ref=q_ready(confined shelf), ordered moves 'obj>r,c.obj>r,c', stop-point connector)",
                          "Y": "(root_id, target t0 -> root goal, q_ref=q_ready, gripper free, endpoint arrangement 'obj@r,c|...' over ALL objects, Y_TEACHER=BD3DS66_Y_REPLAY_V1 [stored-witness replay from q_ref; plan only if no witness])"},
       "historical_prefixes": {"relocations_executed_unique": sum(v["status"] == "RELOCATIONS_EXECUTED" for v in H.values()),
                               "relocation_failed_unique": sum(v["status"] == "RELOCATION_FAILED" for v in H.values()),
                               "by_status_split": {f"{a}|{b}": n for (a, b), n in sorted(cnt.items())},
                               "complete_sequences": sum(v["status"] == "RELOCATIONS_EXECUTED" and v["complete_seq"] for v in H.values()),
                               "complete_without_recorded_terminal_q": sum(v["status"] == "RELOCATIONS_EXECUTED" and v["complete_seq"] and not v["terminal_q_recorded"] for v in H.values()),
                               "endpoint_Y_availability": dict(hy), "unique_endpoints_of_executed_prefixes": len(uniq_hist_y),
                               "of_which_without_any_Y_record": sum(1 for k in uniq_hist_y if k not in Yrec)},
       "bwl3_x_pilot": {"queries": len(P), "overlap_matrix": {f"{a}|{b}": n for (a, b), n in sorted(ov.items())},
                        "unique_y_endpoints": len(uniq_pilot_y), "y_endpoints_with_record": sum(1 for k in uniq_pilot_y if k in Yrec),
                        "y_endpoints_shared_with_historical_prefixes": len(uniq_pilot_y & uniq_hist_y)},
       "y_records": len(Yrec)}
OUT.write_text(json.dumps(res, indent=1))
json.dump({"H": {f"{k[0]}::{k[1]}": {**v, "roles": sorted(v["roles"]), "ykey": Hy[k]} for k, v in H.items()}}, open(str(OUT).replace(".json", "_prefixes.json"), "w"))
print(json.dumps(res, indent=1))
