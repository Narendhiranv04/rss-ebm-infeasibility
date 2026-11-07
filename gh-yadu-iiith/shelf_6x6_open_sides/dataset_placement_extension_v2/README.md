# SHELF66V3 placement-only extension, v2 (sidecar; bounded mock + guarded production preparation)

This directory adds Approach B **placement labels** to the 6×6 open-sided shelf dataset (`../dataset_bd_classquota_without_home_position_no_overpass`, SHELF66V3, xArm7 / MuJoCo 3.16.0).
- It never writes into that dataset.
- v1 (`../dataset_placement_extension_v1`) stays immutable and replayable against `yadu/dataset-3d@8032ad0b`, which is commit `6f042be1` on this branch.
- **No production collection exists or has been started.**

Verdicts are in `3D_MOCK_V2_DECISION.json`. The full report is `3D_MOCK_V2_REPORT.md`, the source sync is in `source_sync/LIVE_3D_SOURCE_SYNC_REPORT.md`, branch history is in `BRANCH_PROVENANCE.md`, and push verification is in `PUSH_AUDIT.md`.

## Event (unchanged from v1): `V_place(S_hyp, o, d; PI_PLACE_3D)`, conditional on an already-held object

- **Held state.** Restore the settled hypothetical scene and set the arm to an IK branch `q_g` of one of the 32 SHELF66V3 level grasp poses. Pre-shape the fingers and require zero pre-contacts, then close in place (400 steps).
  - The hold is certified when both finger sides contact the object, the object moved ≤ 10 mm, its neighbours moved ≤ 4 mm, and there is no other robot contact.
- **Trial.** Lift 2.5 cm → carry along the source 2-D hand-footprint RRT-Connect route (seed 0) → lower → release → straight retreat → settle. The no-overpass columns and arm ceiling apply, and the fingers stay pre-shaped in every object check.
- **Verification.** 13 MuJoCo gates must hold: placement cell, xy, z, tilt, settled, attached, neighbours, released, no robot–structure / robot–object / held–object contact, and the margin-0 envelope.
- **Excluded from the label:** pick access, approach, grasp acquisition, inter-action configuration, target-task success, sequencing and return to ready.

## What v2 changes (teacher `SHELF66V3_PLACE_V2`, protocol hash in every unit JSON)

| v1 | v2 |
|---|---|
| Physical hold certified for IK branch 0 only; if it failed, the whole pose became `HELD_INVALID_PHYSICAL` | Every eligible branch (structure- and ceiling-clear, ≥ 5 mm from other objects) gets its own hold certificate, lazily in candidate order and cached. A unit is a pose with at least one held branch |
| Only the branch that **planned** first was executed; a hold or execution failure there ended the unit | A unit test tries every eligible branch in order: hold → `plan_branch` → re-instantiated hold → execute → verify, until one verifies (early-stop policies) or all are exhausted (reference) |
| One witness per positive | Every distinct verified (unit, branch) trajectory met within the attempted policy is stored, with rank, grasp and branch provenance (`branch` array + unit `witness_index`). The reference keeps all of them |
| Certificate tested branch 0's hold | `NEG_CERTIFIED_GEOMETRY` needs all 32 poses screened, footprint-only geometry reasons and every eligible branch of the `HELD_INVALID_PHYSICAL` poses hold-tested. It is scoped to the declared model (32 poses × ≤ 4 source IK candidates × the 2-D footprint carry model) |
| Resume checked the JSON hash only | Resume validates the JSON SHA-256 and NPZ SHA-256 and re-hashes every witness against the ledger. Absent, truncated, mismatched or orphan files are quarantined and recomputed |
| No source pin | The run manifest freezes per-root source SHA-256 (`mock_v2/configs/source_pin.json`); drift is refused at start and checked in every worker |
| Missing sensor: monitor kept going | Unreadable temperature sensor: refuse to start, or stop (even while paused); the sensor identity is logged |
| | Monitor watches CPU % per process (persistent objects, AMENDMENT_V2_001), the affinity union of the whole own tree (at most 18) and descendant identity. Production is gated by `guard3d.py` |

The route is planned once per unit test and shared by its branches. `plan_carry_xy` takes no branch argument and runs on the restored scene with a fixed seed; `test_route_is_branch_independent` checks this. Physics, tolerances, finger conventions and geometry are the source's. The **structure** check keeps the source's fully open fingers (a conservative choice), quantified by `code/finger_diag.py`.

## Files

| file | role |
|---|---|
| `code/placement3d.py` | physics: held state, `screen` (branch inventory), `plan_branch`, `route_state`, execution, verification, counters, witness pack |
| `code/policy3d.py` | ranking, lazy screening, per-branch hold certificates, unit tests over branches, labels, mode + branch accounting, `virtual` (branch-aware), `motion_work` |
| `code/plan3d.py` | v1 split/roster rules, `extend` (split extension with `SPLIT_CONFLICT`), `pin`, enriched holdout (`pick_holdout`, `roster_holdout`), `leak`, `freeze2` |
| `code/run3d.py` | collector v2 (pinned workers, atomic NPZ→JSON→ledger, validated resume, source pin, fail-closed monitor); production only through `guard3d` |
| `code/guard3d.py`, `code/prod_template.py`, `mock_v2/launch_full_3d.sh` | production authorization (recorded user authorization + token over current hashes); inert launcher |
| `code/replay3d.py` | fresh-process replay of every stored witness (parallel, pinned), provenance checks, GIFs |
| `code/source_sync3d.py` | live snapshot cut / audit / selection (reproduces the previous `selected.json`) / manifest / `pin-live` |
| `code/fixtures_v2.py` | natural branch fixtures mined from v1 records (frozen in `mock_v2/configs/fixtures_v2.json`) |
| `code/report3d.py`, `code/figs3d.py`, `code/finger_diag.py` | analysis, figures, structure-check finger diagnostic |
| `code/test_mock3d_v2.py` | regression tests (v1 tests adapted, plus branch, storage, sensor, source-drift, split, launcher and dry-run tests) |
| `mock_v2/configs/` | frozen selection rule v2, run plan, source pin, fixtures, amendment, subsets |
| `mock_v2/roster_v2.jsonl.gz` | 42 train roots / 405 physical queries: the 205 v1 queries (22 roots) plus an independent enriched holdout (20 roots, 200 queries) |
| `mock_v2/runs/` | `reference_v2`, `adaptive_v1_{A,B}`, `fixed_{4x1500,4x5000,8x5000}`, `adaptive_v1_interrupt`, `serial_adaptive`, `sweep_{8,12,16,18}` |
| `mock_v2/validation/`, `mock_v2/figures/` | analysis, replays, tests, integrity, finger diagnostic; plots, scenes, GIFs |
| `production_v2/` | live split extension, live production roster design, live source pin, production dry run |
| `source_sync/` | live snapshot sync report, per-file manifest, second-pass hashes |
