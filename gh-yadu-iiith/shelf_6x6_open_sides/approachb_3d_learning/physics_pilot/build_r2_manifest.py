"""R2 preregistered missing-endpoint Y pilot (BD3DS66_Y_PLAN_V1).  Read-only; selection fixed by sha256 order BEFORE any outcome.

    python3 -I build_r2_manifest.py RAW_DIR R0_CANONICAL_QUERY_LEDGER.json.gz OUT.json

Population: Y endpoint keys of the canonical ledger with NO Y record (source x_endpoint_no_record), train / val roots only (test held out).
  A  48 endpoints reached by executed prefixes: K-stratified (12 per K0..K3 where available), at least 12 of them shared by >= 2 orders;
  B  12 endpoints reached only by relocation-failed prefixes (Y is defined independently of X);
  C  12 equivalence controls on EXISTING records (train / val): 6 planning positives (expect the IDENTICAL witness id and an executed
     pass) and 6 negatives (4 CERTIFIED, 2 NO_PLAN_FOUND; expect the same kind).
Eligibility mask (declared): every endpoint is an arrangement of distinct in-grid cells (true by construction of executed / attempted moves);
an occupied goal cell is NOT filtered (it is a legitimate Y = 0 case for the teacher to decide)."""
import gzip, hashlib, json, sys, collections
from pathlib import Path

RAW, LED, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
L = json.loads(gzip.open(LED, "rt").read())
H = lambda t, s: hashlib.sha256(f"{t}|{s}".encode()).hexdigest()  # noqa: E731
xs = collections.defaultdict(list)
for x in L["X"]:
    xs[(x["root_id"], x["y_key"])].append(x)
cells0 = {}


def edit_of(rid, ykey):
    if rid not in cells0:
        cells0[rid] = {k: tuple(v) for k, v in json.loads((RAW / f"roots/{rid}.json").read_text())["initial_cells"].items()}
    end = {p.split("@")[0]: tuple(map(int, p.split("@")[1].split(","))) for p in ykey.split("|")}
    assert len(set(end.values())) == len(end) and all(0 <= r < 6 and 0 <= c < 6 for r, c in end.values()), ykey
    return {o: list(c) for o, c in end.items() if c != cells0[rid][o]}


miss = [y for y in L["Y"] if y["source"] == "x_endpoint_no_record" and y["split"] in ("train", "val")]
exe = [y for y in miss if any(x["status"] == "EXECUTED" for x in xs[(y["root_id"], y["y_key"])])]
fonly = [y for y in miss if all(x["status"] == "RELOCATION_FAILED" for x in xs[(y["root_id"], y["y_key"])])]
multi = sorted([y for y in exe if len(xs[(y["root_id"], y["y_key"])]) >= 2], key=lambda y: H("R2M", y["root_id"] + y["y_key"]))
items, used = [], set()


def add(y, stratum, **kw):
    k = (y["root_id"], y["y_key"])
    if k in used:
        return False
    used.add(k)
    items.append({"item": f"R2-{len(items):03d}", "kind": "Y", "phase": "R2", "stratum": stratum, "root_id": y["root_id"], "split": y["split"], "K": y["K"],
                  "edit": edit_of(*k), "y_key": y["y_key"], "action": "PLAN_TARGET", "independent_replay": True,
                  "x_orders_reaching": len(xs[k]), "x_statuses": sorted({x["status"] for x in xs[k]}), **kw})
    return True


for y in multi[:12]:
    add(y, "A_EXECUTED_ENDPOINT_MULTI_ORDER")
for K in (0, 1, 2, 3):
    n = sum(1 for i in items if i["K"] == K)
    for y in sorted([y for y in exe if y["K"] == K], key=lambda y: H(f"R2A{K}", y["root_id"] + y["y_key"])):
        if n >= 9 or len(items) >= 48:
            break
        n += add(y, "A_EXECUTED_ENDPOINT")
for y in sorted(exe, key=lambda y: H("R2FILL", y["root_id"] + y["y_key"])):
    if len(items) >= 48:
        break
    add(y, "A_EXECUTED_ENDPOINT")
for y in sorted(fonly, key=lambda y: H("R2B", y["root_id"] + y["y_key"]))[:12]:
    add(y, "B_FAILED_PREFIX_ONLY_ENDPOINT")
rec = [y for y in L["Y"] if y["source"].startswith("record") and y["split"] in ("train", "val") and not y.get("verified")]
for want, n, tag in ((("pos", None), 6, "C_CONTROL_POSITIVE"), (("neg", "CERTIFIED"), 4, "C_CONTROL_NEG_CERTIFIED"), (("neg", "NO_PLAN_FOUND"), 2, "C_CONTROL_NEG_BUDGET")):
    pool = [y for y in rec if (y["archived_y"] == 1 if want[0] == "pos" else (y["archived_y"] == 0 and y["negative_kind"] == want[1]))]
    for y in sorted(pool, key=lambda y: H(tag, y["root_id"] + y["y_key"]))[:n]:
        add(y, tag, archived_y=y["archived_y"], negative_kind=y["negative_kind"], witness_id=y["witness"])
man = {"manifest": "R2_MISSING_Y_PILOT_V1", "status": "PREREGISTERED (selection by hash before outcomes)", "protocol": "BD3DS66_Y_PLAN_V1",
       "protocol_detail": "settle(ideal endpoint) at q_ready (bd_3dt.settle); bd_3dt.query(t0 -> goal): 32 grasp poses, RRT-Connect 5000 iterations, seed 0; "
                          "success -> witness stored with bd_3dt.witness into an additive store, executed with run_plan + seq() target checks, then replayed "
                          "in a fresh process (Y = 1 only if both pass); failure -> operational Y = 0 with the teacher's kind (attempted)",
       "budget_cpu_h": 6.0, "population": {"missing_train_val": len(miss), "from_executed_prefixes": len(exe), "failed_prefix_only": len(fonly)},
       "counts": dict(collections.Counter(i["stratum"] for i in items)), "by_K": dict(collections.Counter(i["K"] for i in items)), "items": items}
man["sha256_of_items"] = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
OUT.write_text(json.dumps(man, indent=1))
print(json.dumps({k: v for k, v in man.items() if k != "items"}, indent=1))
