# R2: Missing-endpoint Y protocol (BD3DS66_Y_PLAN_V1) and preregistered pilot

**Verdict: R2 PASS on labels; the preregistered *bitwise* witness-identity check FAILED, with structural equivalence instead (see §3).**

**Bounds:** pilot only, **0.73 of 6 CPU-h**. **No bulk missing-Y sweep was dispatched.**

**Evidence:**
- `results_r2/SUMMARY_R2.json`;
- `results_r2/R2_ITEMS_WITNESS_LOGS.tgz`, containing the per-item JSONs, the additive witness store `ystore/` and the worker logs;
- manifest `R2_PILOT_MANIFEST.json` (items sha256 `525e1087…`);
- code `bwlphys.py` (deployed sha256 prefix `259652f1`, action `PLAN_TARGET`).

## 1. Population (recomputed from the R0 canonical ledger)

Unique Y endpoint keys reached by recovered X prefixes **with no Y record**: **3,407** in total.

| Split | Endpoints |
|---|---|
| train | 2,460 |
| val | 463 |
| test | 484 |

Of these, **2,929** are reached by executed prefixes (the audit's figure) and 478 only by relocation-failed prefixes. Y is defined independently of X, so both kinds are valid queries.

The pilot draws from **train / val only (2,923)**; test endpoints are held out.

## 2. Protocol (pinned; same teacher as history)

1. Settle the **ideal endpoint**: `bd_3dt.settle`, i.e. reset to the home cells, `q_ready`, gripper `GRIPPER_OPEN`, nothing held, then teleport the edited objects.
2. Run `bd_3dt.query(t0 → goal)`: 32 grasp poses in the protocol's order, RRT-Connect with 5,000 iterations, seed 0, first feasible wins. These are the historical target-query budgets.
3. **On a plan:**
   - store the witness with `bd_3dt.witness` (the historical arrays and content-id rule) in an **additive** store;
   - **execute** it with `run_plan` and `bd_3dt.seq`'s target checks (goal cell, |z_err| ≤ 1 cm, nothing displaced);
   - **replay** it in a **fresh process** and require bitwise-identical end states.
   - Only then is Y = 1.
4. **No plan:** an **attempted** operational Y = 0, keeping the teacher's kind (`CERTIFIED` = every grasp of the finite set geometrically blocked; `NO_PLAN_FOUND` = budget).
5. A job cancellation or timeout would be **MISSING**. None occurred.

## 3. Pilot (72 preregistered items; selection by sha256 before outcomes)

| Stratum | n | Y = 1 (planned, executed, replay-verified) | Y = 0 CERTIFIED | Y = 0 NO_PLAN_FOUND |
|---|---|---|---|---|
| A: executed-prefix endpoint shared by ≥ 2 orders | 12 | **12** | 0 | 0 |
| A: executed-prefix endpoint | 36 | 19 | 14 | 3 |
| B: endpoint reached only by failed prefixes | 12 | 4 | 5 | 3 |
| **New endpoints** | **60** | **35** | **19** | **6** |
| C: controls, archived positive | 6 | 6 (labels agree) | | |
| C: controls, archived certified negative | 4 | | 4 (labels and kind agree) | |
| C: controls, archived budget negative | 2 | | | 2 (labels and kind agree) |

**Label equivalence on controls: 12 / 12.**

**Preregistered strict check: identical witness id on positive controls: 0 / 6 (FAILED).** Diagnosis:
- every replanned control used the **same grasp**, the **same waypoint count** (86–134) and identical segments and carry points;
- joint values differ by at most **5.3e-5 rad** (three controls at about 1e-15, three at about 1e-5).

The planner's iterative IK and RRT are therefore **structurally reproducible but not bit-reproducible** across processes and environments. That changes the content hash, not the trajectory or the label. Y = 1 is granted on each new witness's *own* execution plus bitwise fresh-process replay, which is unaffected. The strict criterion is reported as failed, not relaxed retroactively.

**Order invariance.** The 12 multi-order endpoints have **one** Y key each, so AB and BA share the same Y by construction. No endpoint was queried twice.

**Cost:**

| Query outcome | Mean CPU-s | p90 CPU-s |
|---|---|---|
| Positive (plan + execute + replay) | 23.8 | 40.6 |
| Negative | 58.2 | 84.0 |

The new-endpoint positive rate in the pilot is 0.58.

## 4. Remaining work and bulk proposal (NOT authorized)

- **Remaining missing endpoints:** 3,407 − 60 = **3,347** (train / val 2,863; test 484).
- **Estimated cost:** about **36 CPU-h** single-core for all of them at the pilot's measured costs and positive rate, about 18 h wall on 2 workers.
- **Proposal:** run it as resumable root-sharded chunks with the same summarizer gates, train / val first.
- It needs a new explicit budget.
