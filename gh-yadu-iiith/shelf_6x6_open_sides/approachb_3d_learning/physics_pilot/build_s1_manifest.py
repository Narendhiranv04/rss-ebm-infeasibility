"""S1 predeclared physics pilot manifest (Y replay + historical X prefix recovery).  Read-only over the pinned 3D store; no physics.

    python3 -I build_s1_manifest.py RAW_DIR V3_DIR Y_REPLAY_MANIFEST.json S0_OVERLAP_prefixes.json OUT.json

Selection is fixed by sha256 order BEFORE any physics run.  Y: the BWL-3 120-record replay manifest (smoke = 2 per positive stratum + 1 per
negative stratum; pilot = all 72 positives + all 24 certified negatives + 6 of 24 planner-budget negatives).  X: train / val roots only, stored
relocation prefixes in strata (K1 complete, K2 AB/BA same-endpoint pairs, order-discordant pairs, K3 complete, proper prefixes, relocation
failures, complete sequences without a recorded terminal arm state).  No test-split X, no new destination, no BWL-3 fresh-execution query."""
import hashlib, json, sys, collections
from pathlib import Path

RAW, V3, YMAN, PREF, OUT = map(Path, sys.argv[1:6])
H = lambda tag, s: hashlib.sha256(f"{tag}|{s}".encode()).hexdigest()  # noqa: E731
split = json.loads((V3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
P = json.loads(PREF.read_text())["H"]
root = {}


def load(rid):
    if rid not in root:
        root[rid] = json.loads((RAW / f"roots/{rid}.json").read_text())
    return root[rid]


def seq_of(rid, key):
    """The stored B record(s) whose sequence has this ordered prefix; returns (record, prefix length)."""
    moves = key.split(".")
    for b in load(rid)["labels"]["B"]:
        s = [f"{o}>{int(c[0])},{int(c[1])}" for o, c in b.get("sequence") or []]
        if s[:len(moves)] == moves:
            return b, len(moves)
    raise KeyError(key)


# ---------------- Y
ym = json.loads(YMAN.read_text())["rows"]
smoke_quota = collections.Counter()
Y = []
for r in ym:
    pos = r["y"] == 1
    q = 2 if pos else 1
    smoke = smoke_quota[r["stratum"]] < q
    smoke_quota[r["stratum"]] += 1
    take = pos or r["negative_kind"] == "CERTIFIED" or smoke or smoke_quota[r["stratum"]] <= 3
    if not take:
        continue
    rec = load(r["root_id"])
    lab = rec["labels"]["original"] if r["family"] == "original" else next(d for d in rec["labels"]["D"] if d["edit"] == r["edit"])
    Y.append({"item": f"Y-{r['manifest_row']}", "kind": "Y", "phase": "SMOKE" if smoke else "PILOT", "stratum": r["stratum"], "root_id": r["root_id"],
              "split": r["split"], "family": r["family"], "edit": r["edit"], "archived_y": r["y"], "negative_kind": r["negative_kind"],
              "witness_id": lab.get("witness"), "grasp_id": lab.get("grasp_id"),
              "action": "REPLAY_STORED_TARGET_WITNESS" if pos else ("RESCREEN_CERTIFIED" if r["negative_kind"] == "CERTIFIED" else "REPLAN_BUDGET_NEGATIVE")})

# ---------------- X (train / val only)
tv = {k: v for k, v in P.items() if v["split"] in ("train", "val")}
order = lambda tag, keys: sorted(keys, key=lambda k: H(tag, k))  # noqa: E731
X, used = [], set()


def add(k, stratum, phase="PILOT"):
    if k in used:
        return
    used.add(k)
    rid, key = k.split("::")
    b, L = seq_of(rid, key)
    v = P[k]
    X.append({"item": f"X-{len(X):03d}", "kind": "X", "phase": phase, "stratum": stratum, "root_id": rid, "split": v["split"], "K": v["K"],
              "x_key": key, "m": L, "stored_role": b["role"], "status_in_store": v["status"], "negative_kind": v.get("negative_kind"),
              "full_sequence_length": len(b["sequence"]), "y_endpoint_key": v["ykey"],
              "action": "REPLAY_RELOCATIONS+CONNECTOR+INDEPENDENT_REPLAY" if v["status"] == "RELOCATIONS_EXECUTED" else "REPLAY_TO_FAILING_STEP+RECONFIRM"})


exe = {k: v for k, v in tv.items() if v["status"] == "RELOCATIONS_EXECUTED"}
fail = {k: v for k, v in tv.items() if v["status"] == "RELOCATION_FAILED"}
k1 = [k for k, v in exe.items() if v["K"] == 1 and v["complete_seq"] and v["m"] == 1 and "minimal" in v["roles"]]
for i, k in enumerate(order("S1X_K1", k1)[:6]):
    add(k, "K1_COMPLETE_SINGLE_MOVE", "SMOKE" if i == 0 else "PILOT")
# AB/BA same-endpoint pairs: both orders' relocations executed (K2 minimal vs wrong_order)
pairs = collections.defaultdict(list)
for k, v in tv.items():
    if v["m"] == 2:
        pairs[(k.split("::")[0], v["ykey"])].append(k)
both = [g for g in pairs.values() if len(g) == 2 and all(P[k]["status"] == "RELOCATIONS_EXECUTED" for k in g)]
disc = [g for g in pairs.values() if len(g) == 2 and sorted(P[k]["status"] for k in g) == ["RELOCATIONS_EXECUTED", "RELOCATION_FAILED"]]
for i, g in enumerate(sorted(both, key=lambda g: H("S1X_AB", min(g)))[:5]):
    for k in sorted(g):
        add(k, "K2_AB_BA_SAME_ENDPOINT_BOTH_EXECUTED", "SMOKE" if i == 0 else "PILOT")
for g in sorted(disc, key=lambda g: H("S1X_DISC", min(g)))[:3]:
    for k in sorted(g):
        add(k, "AB_BA_ORDER_DISCORDANT_RELOCATION")
for k in order("S1X_K3", [k for k, v in exe.items() if v["K"] == 3 and v["complete_seq"] and v["m"] == 3])[:5]:
    add(k, "K3_COMPLETE_THREE_MOVE")
for k in order("S1X_PREFIX", [k for k, v in exe.items() if not v["complete_seq"]])[:6]:
    add(k, "PROPER_PREFIX_INTERMEDIATE_STOP")
fk = collections.defaultdict(list)
for k, v in fail.items():
    fk[v.get("negative_kind")].append(k)
for kind, n in (("CERTIFIED", 3), ("NO_PLAN_FOUND", 1), ("OTHER_FAILURE", 2)):
    for i, k in enumerate(order(f"S1X_FAIL_{kind}", fk.get(kind, []))[:n]):
        add(k, f"RELOCATION_FAILED_{kind}", "SMOKE" if (kind == "CERTIFIED" and i == 0) else "PILOT")
for k in order("S1X_NOQ", [k for k, v in exe.items() if v["complete_seq"] and not v["terminal_q_recorded"]])[:5]:
    add(k, "COMPLETE_TERMINAL_ARM_STATE_NOT_RECORDED")

man = {"manifest": "S1_PHYSICS_PILOT_V1", "status": "PREDECLARED", "host": "gvlab (authorized 3D physics host; v4 producer)",
       "budget": {"workers": 2, "cpus": [16, 17], "nice": 19, "per_item_timeout_s": 900, "wall_cap_h": 3.0, "cpu_cap_h": 6.0,
                  "guard": "stop if coretemp package >= 75 C, MemAvailable < 16 GB or swap grows > 256 MB; never on CPUs 0-13 (Q2 workers + reserved)"},
       "x_teacher": "BD3DS66_X_V1 = BD3DS66_TEACHER_V2 relocation witnesses replayed with the unchanged run_plan from settle(S0) (carried state, no reset) "
                    "+ CONNECTOR_TO_QREADY_V1 (plan_free_move(q_now, q_ready), env RRT-Connect 5000 iters seed 0, executed; then gripper opened to "
                    "GRIPPER_OPEN; verified) + independent replay in a fresh process",
       "y_teacher": "BD3DS66_Y_REPLAY_V1 = settle(endpoint) at q_ready, stored target witness replayed with run_plan, teacher checks; replayed twice in fresh processes",
       "exclusions": ["test-split X prefixes (held out)", "BWL-3 fresh-execution queries (candidate-diverse stage comes after this protocol is validated)",
                      "non-cohort roots incl. the 75 DEPENDENT roots (challenge cohort, separate decision)"],
       "counts": {"Y": dict(collections.Counter((y["phase"], y["action"]) for y in Y)), "X": dict(collections.Counter((x["phase"], x["stratum"]) for x in X))},
       "items": Y + X}
man["counts"] = {k: {"|".join(kk): n for kk, n in v.items()} for k, v in man["counts"].items()}
man["sha256_of_items"] = hashlib.sha256(json.dumps(man["items"], sort_keys=True).encode()).hexdigest()
OUT.write_text(json.dumps(man, indent=1))
print(json.dumps({k: v for k, v in man.items() if k != "items"}, indent=1))
