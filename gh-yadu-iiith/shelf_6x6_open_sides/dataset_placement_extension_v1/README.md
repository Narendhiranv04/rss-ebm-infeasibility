# SHELF66V3 placement-only extension, v1 (sidecar; mock only)

This directory adds Approach B **placement labels** to the 6×6 open-sided shelf dataset
(`../dataset_bd_classquota_without_home_position_no_overpass`, SHELF66V3, xArm7 / MuJoCo 3.16.0). It never writes into that dataset.
Its contents are code, a frozen mock design, and audited mock runs. **No production collection exists or is enabled here.**

**Mock v1 outcome (2026-11-07):** `MOCK_PASS`. Recommended policy: `adaptive_v1` at 12 workers, pending your confirmation. See `mock_v1/MOCK_REPORT.md`.

## Action semantics: `V_place(S_hyp, o, d; PI_PLACE_3D)`

The label is conditional on a valid **post-grasp** state of object `o` in a (possibly hypothetical) scene `S_hyp`. One trial is:

1. **Held state** (`placement3d.hold`).
   - The settled scene is restored and the arm is set to `q_g`, an IK branch of one of the 32 SHELF66V3 level grasp poses. The branch is clear of the structure and the arm ceiling, and keeps ≥ 5 mm from every other object.
   - The fingers are pre-shaped to the grasp's open width (object width + 2 × 6 mm).
   - The pre-closure state must touch nothing. The fingers then close in place (`run_plan`'s `close`: GRIPPER_CLOSED, 400 steps).
   - Certified if:
     - both finger sides contact `o`;
     - `o` moved ≤ 10 mm;
     - no other object moved > 4 mm;
     - the robot touches nothing else.
2. **Lift 2.5 cm → carry → lower → release → straight retreat → settle (200 steps).**
   - Planned with the source teacher's own code (`_plan_carry` and `_plan_retreat`): the 2-D hand-footprint RRT-Connect route, IK chains, no-overpass columns, arm ceiling = tallest object + 10 cm, and fingers pre-shaped in every object check.
   - Servo-executed exactly as `shelf_scene.execute`.
3. **Verification** (`placement3d.verify`, all gates required):
   - no lift slip, and the object stays attached during the carry (drift ≤ 10 mm);
   - final placement: end cell = `d`, xy ≤ 20 mm, |z| ≤ 10 mm, tilt ≤ 10°, settled;
   - no neighbour moved > 4 mm;
   - the hand is released from the object;
   - no physics contact at any step between robot and structure, robot and other objects, or held object and other objects;
   - every recorded arm configuration stays outside the structure, the arm ceiling and the other objects' no-overpass columns (margin 0).

**Excluded from the label:** free move to the object, hover, approach, the pick test, sequencing, return to ready, and target-goal success.

A pick that is inaccessible but whose object is held-placeable can be labeled 1. The fixture `pick_inaccessible_held_placeable` tests this.

## Modes, policy, labels (`policy3d.py`)

- **Nominal modes:** 32 poses, ranked by a predeclared diverse order: the k-th nearest azimuth to the robot, with grip heights alternating, then the other heights.
- **Scheduling unit:** a VALID pose that is physically held. Its ≤ 4 IK branches are tried inside the unit and are never counted as extra modes.
- **Stages `(M units, B RRT iterations)`:**
  - `adaptive_v1`: (4, 500) → (8, 1,500) → (16, 5,000);
  - `reference_v1`: all valid units at 5,000, no early stop;
  - `fixed_MxB`: virtual only.
- **Labels:**
  - `1` (`VERIFIED`);
  - `0` = `NEG_CERTIFIED_GEOMETRY` (every physically held pose fails the deterministic footprint certificate) or `NEG_UNDER_POLICY`, of kind `BOUNDED_SEARCH`, `EXECUTION` or `DETERMINISTIC_PLAN`;
  - unlabeled: static ineligible, `NO_VALID_POST_GRASP_STATE`, `INTERRUPTED`.
- A CPU guard or admin stop never yields 0.
- Full mode accounting per query covers nominal / screened / IK-invalid / held-invalid (kinematic, physical) / geometry-invalid / valid / motion-tested / unresolved / plan-failed / exec-failed / verified, plus attempted configurations.

## Determinism and budgets

- All work is counted: RRT outer iterations, extensions, edge checks, shortcut checks, IK calls, object and structure checks, simulation steps. No wall clock decides any outcome.
- Seeds are fixed (route seed 0, as the source).
- IK branches are solved at the source cache key's 0.1 mm-rounded pose. The cache is shared only within a root lineage, so it is a pure function of its key.
- Counter snapshots at 500 and 1,500 iterations make nested budgets exact; `policy3d.virtual` replays any smaller or adaptive policy from a reference record.

## Witness contract (per root: `units/<root>.json` + `units/<root>.npz`, keys `<witness_id>.<array>`)

- `witness_id`: sha256 over the sorted arrays (`bd_core.Store.witness` scheme, 24 hex characters).
- **Planned waypoints:** `q (N,7)`, `segment`, `seg_phase`, `seg_off`, `seg_grip`, `carry_xy` — source-compatible.
- **Held state:** `q_g`, `grasp` (id, azimuth, level index, open width, width), `grasp_pos`, `grasp_R`, `hold_ev`.
- **Executed MuJoCo samples, every 10 steps:** `q_exec (M,7)`, `gap_exec`, `obj_exec (M,7)`, `phase_exec`, `step_exec`.
- **Source B/D witnesses** (`q`, `segment`, `carry_xy`, `phases`) are whole pick-and-place kinematic plans. `replay3d.convert` is the documented converter to the placement-only segments; nothing assumes the source arrays are placement trajectories.

## Files

| file | role |
|---|---|
| `code/placement3d.py` | physics: held state, plan, execution, verification, counters, witness pack |
| `code/policy3d.py` | ranking, screening, schedules, labels, accounting, virtual budgets |
| `code/plan3d.py` | lineage split, geometry-only roster, frozen rule (`freeze`) |
| `code/run3d.py` | bounded coordinator (pinned workers, resume ledger, fail-closed monitor) |
| `code/replay3d.py` | fresh-process witness replay, source baseline replay + converter, GIFs |
| `code/audit3d.py` | source/host inventory, SHA-256 manifest, integrity re-check |
| `code/fixtures3d.py`, `code/f1_search.py`, `code/f1_recheck.py` | unit-fixture discovery and freeze (train roots outside the mock roster; source pick policy as a diagnostic only) |
| `code/report3d.py` | determinism comparisons, virtual budget table, coverage, negative audit, sweep, lineage-bootstrap forecast |
| `code/figs3d.py` | scene panels, failure GIF, plots (MUJOCO_GL=egl) |
| `code/finger_diag.py` | structure-check finger-width diagnostic |
| `code/test_mock3d.py` | 15 focused regression tests |
| `LIVE_SOURCE_AUDIT.md` | what exists on nav@gvlab vs the live generator host |
| `mock_v1/MOCK_REPORT.md`, `mock_v1/LAUNCH_READINESS_DECISION.json`, `mock_v1/RUNBOOK.md` | mock outcome (`MOCK_PASS`), decision, future-only runbook |
| `audit/` | `live_source_inventory.json`, `source_manifest.json.gz` |
| `mock_v1/` | split, rosters, configs, runs, figures, reports |

## Source provenance

The source is `yadu/dataset-3d` @ `8032ad0bf842a490897a8247ee46312036c12546`:
- 1,492 selected roots (NONE 3, K0 511, K1 357, K2 357, K3 264);
- protocol `XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022`;
- teacher config `8d521b1f6d23db23`.

`LIVE_SOURCE_AUDIT.md` explains why this snapshot is the latest data reachable from `nav@gvlab`.
