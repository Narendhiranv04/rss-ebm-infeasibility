# 2D placement-label sidecar: smoke test report (6 Nov 2026)

**Result: `SMOKE_FAIL`.** One acceptance criterion failed: reproducible output under the declared smoke caps. Every other functional check passed. The pipeline works:
- **Labels:** binary labels, with no observed `UNRESOLVED` class.
- **Witnesses:** every one re-certifies from disk.
- **Mode accounting:** exact.
- **Resume:** idempotent.
- **Original dataset:** unchanged.

Under 28 SMT workers, though, the declared 2 s per-pair and 12 s per-query wall-clock caps bind often enough that one label and 12 witness sets changed between repeated runs. The cause and the proposed fix are in [Outcome against the acceptance tests](#outcome-against-the-acceptance-tests). No retuning was done, as the contract requires.

## Source and provenance

| Item | Value |
|---|---|
| Repository | `rss-ebm-infeasibility`, branch `yadu/dataset-3d` at `9e64069038928dfae1e7f89391587e2f595374fc` |
| Work location | git worktree `/home/naren/Desktop/rss-placement-smoke`, new local branch `naren/placement-smoke-2d`. Nothing is committed or pushed. |
| New files (only these) | `planar_2d/dataset_relocations_v1/code/placement_smoke.py` (collection, 402 lines)<br>`placement_smoke_check.py` (audit and figures, 264 lines)<br>`test_placement_smoke.py` (49 lines)<br>this `smoke_20261106/` folder |
| Teacher | the dataset's own `code/latent-causal-tamp` (`skill`, `motion`, `planner`, `geometry`, `config.toml`), imported unmodified. `bd_2d.env()` reproduces the roots' `config_hash` `90d3eb68e8a42108` and robot profile `sha256:2d7986…ae858`. |
| Python | 3.13.11 venv with NumPy 2.4.6 (2.4.0 has a `.flat` memory leak), Matplotlib 3.11.2, Pillow 13.0, SciPy, psutil |
| Budget hash | `3708dc9d6ab09535`, identical in every run |
| `placement_protocol_id` | includes the script hash, so it differs per run:<br>Pass A `add4eeb596f1b40e`<br>main Pass B `544a5c7b9b1bdcb6`<br>replicas `5ce92ab81032f727`<br>resumed half of the interrupt test `d7cdf572fcd25621`<br>The budgets and event definition are identical across all of them. |
| Split | The 2D dataset defines no train/validation/test split (none in the root records or `selected.json`). Recorded as `split: null` and not regenerated. |

## Audit of the checkout (before coding)

- **Scenes:** 5,000 selected roots (NONE 1,000; K0 1,250; K1 800; K2 800; K3 1,150), 7,218 root files and 39,959 witness files.
- **Grid:** 8 × 8, with all 64 cells legal (`domain_policy = ALL_SUPPORTED`).
- **Modes:** confirmed in code. `[skill2d] approach_dirs = 24` and `skill.directions` give 24 tool axes × 2 elbow branches, so 48 pick and 48 place modes and 2,304 nominal pairs per object–destination.
- **Archived witnesses:** they already carry the achieved state between moves (step 2 starts at step 1's retreat, with no return home). Each move has the phases `pregrasp, approach, close_attach, carry, release, retreat`.

## Declared protocol (PLACEMENT_SMOKE_V1)

**Event.** Object `o` has already been grasped in scene `S`. Can carry → release → straight retreat to cell `d` be certified? It does not include pick access from the start pose (no pregrasp or approach), sequence X, or target Y.

**Components:**

| Part | Definition |
|---|---|
| Post-grasp mode | `skill.screen_pick`'s checks (IK, open-finger grasp pose clear, certified closure) without its approach line, because the approach is pick access |
| Place mode | `skill.screen_place` (IK, held pose, release, retreat) |
| Pair rejection | a pair with a definitely invalid mode is geometrically rejected |
| Carry | valid pairs are planned same elbow branch first, then by joint distance; `planner.plan` tries direct → guide → link-0 certificate → RRT |
| Witness | `close_attach, carry, release, retreat`, re-certified by `motion.certify_trajectory`, with place error ≤ the teacher's `release_tol` |

**Smoke budgets:**
- at most 8 new carry plans per query;
- RRT at most 150 nodes and 2 s per pair;
- 12 s per query, enforced in Pass B as "do not start a pair that cannot finish inside 12 s";
- 40 s hard per-query alarm (never triggered);
- 240 s global watchdog.

**Labels:**
- **1:** at least one verified witness.
- **0:** the budget ran without one, recorded with `negative_kind`, `failure_phase` and `is_certificate`. A certificate is possible only when all 2,304 pairs fall to definite screens; that never happened here.
- **`null`:** not attempted (a statically ineligible destination, or an engineering abort). It is never counted as 0.

**Archived witness reuse.** An archived first-move witness of the same root scene, object and destination is reused only if:
- its `close_attach…retreat` suffix re-certifies under this protocol;
- its grasp and release configurations equal declared modes that pass this protocol's screens.

## Roster (frozen before any teacher call)

[`roster_predeclared.json`](roster_predeclared.json), sha256 `02297bfd…99a0`, has 35 queries:

| Root | Class |
|---|---|
| `2d-plus2d-srv-w0230-s011` | K2 |
| `2d-plus2d-srv-w0408-s006` | K3 |
| `2d-conv-plus2d-srv-w0068-s006` | K1 |
| `2d-plus2d-srv-w0111-s001` | K0 |

How the roster was chosen:
- **Roots:** per class, the smallest hash among selected roots with at least 2 movable objects. That filter was added before any run, because the first K0 by hash holds T alone.
- **Objects:** two per root (the archived minimal repair's first two, topped up by hash).
- **Destinations:** four per object (the archived parking cell, plus one near, mid and far cell outside every historical destination).
- **Intermediate queries (2):**
  - K2: after o01 → 4, move o00 → 3.
  - K3: after o00 → 63, move o02 → 45, the cell o00 vacated. The same move at the root is statically ineligible (occupied) and is logged as excluded.

## Results (main Pass B, `smoke_20261106/`)

| | Count |
|---|---|
| Queries predeclared / attempted / not attempted | 35 / 34 / 1 (`DEST_OCCUPIED`) |
| Label 1 / label 0 | 28 / 6 (all `BUDGET_EXHAUSTED`, failure phase `CARRY`, none a certificate) |
| Mode pairs enumerated | 78,336 (34 × 2,304) |
| Geometrically rejected (definite screens) | 70,825 |
| Not motion-tested | 7,278 (6,818 valid pairs beyond the budget + 460 with an uncertain screen) |
| **Sent to real carry planning or re-certification** | **233** (190 verified + 43 without a witness) |
| Verified witnesses stored | 190 (185 new + 5 reused archived suffixes), content-addressed, no duplicates |

**Labels by root and object** (destinations in roster order; parking cells are the archived ones):

| Root (class) | Object | Labels |
|---|---|---|
| K2 | o00, o01 | all eight are 1 |
| K2 | o00 after o01 → 4 | 1 |
| K3 | o00, o02 | all eight are 1 |
| K3 | o02 → 45 | excluded at the root (occupied); 1 after o00 → 63 |
| K1 | o00 | all four are 1 |
| K1 | o01 | 1 for parking cell 23 and near cell 35; 0 for mid cell 26 and far cell 0 |
| K0 | o00 | all four are 1 |
| K0 | o01 | 0 for all four, including its parking cell 31 |

Both objects with zeros have only 3 valid post-grasp modes out of 48. All 30 of their carry plans were cut by the 2 s clock, so the zeros are budget outcomes, not physical impossibility.

**Pass A (serial).** One query: root K2, o01 → archived parking cell 4. Result: label 1 with 9 witnesses (8 new + the reused archived minimal-repair suffix, grasp mode 6, place mode 0). Replay and the audit passed (58 checks). It took 13.47 s, over the 12 s cap, because the cap was then checked only between pairs. That was fixed before Pass B; Pass B's maximum was 11.58 s.

## Outcome against the acceptance tests

**Passed** (audits re-run on every output folder from the saved files only: 1,242 / 1,252 / 1,212 / 1,236 checks, 0 failures):
- **Records:** every attempted query is exactly 0 or 1, every unattempted one is `null` with a reason, and no `UNRESOLVED` label exists.
- **Labels and witnesses:** every 1 has at least one witness file, and every 0 has none.
- **Witness files:** each reopens, matches its content id, has no NaN, starts with `close_attach` and ends with `retreat`, re-certifies with `motion.certify_trajectory` in the scene it was found in, and ends within tolerance of the destination.
- **Mode counts:** reconcile per query (2,304 = rejected + not tested + attempted, and attempted = verified + failed), equal the stored `pair_category` arrays, and `attempted_mask` is exact.
- **Hypothetical scenes:** re-hash to the recorded `hypothetical_scene_hash`, are marked `parent_reachability: HYPOTHETICAL`, and differ from their root.
- **Original folder:** `dataset_bd_classquota/` is unchanged against git, and every used root file equals its git blob.
- **Archived replays:** the 5 reused archived witnesses also replay in full in their original scenes (74–169 segments, all CLEAR).
- **Re-run of the finished pass:** 35 queries skipped, 0.1 CPU-s, and no file added or changed (sha256 over all records and trajectories).
- **Interrupt test** ([`interrupt_test/`](interrupt_test/)): SIGKILL of all 28 workers and the parent at 4 s, then resume:
  - 11 queries had finished before the kill, and the resume left all 11 byte-identical;
  - no torn files remained, and no npz existed without its JSON;
  - the resume completed 35/35.
- **Tests:** `test_placement_smoke.py` passes (3). The dataset's `test_bd_dataset.py` gives 6 passed, 1 skipped (no 3D repository), 1 failed. That failure is unrelated to this change: `test_existing_json_npz_readers_still_load` reads `runs/p16/…` and `runs/p14/…`, which are not part of this dataset's code copy, so it fails the same way on the untouched checkout.

**Failed:**

1. **Reproducible output under the declared caps.** Two complete repeats ([`replica_run/`](replica_run/), [`replica_run2/`](replica_run2/)) and the interrupt test were compared with the main run:
   - **Labels:** identical on 33 of 34 attempted queries. Query `87d50fb58df58ed2` (K1 o01 → 26) was 0 in the three full runs and 1 in the interrupt test, whose resumed half ran fewer concurrent queries.
   - **Witness sets:** differ in 12 of 35 queries.
   - **Clock cuts:** each run had 61–64 carry plans cut by the 2 s wall clock and 16–18 queries stopped by the 12 s cap. Serially, the same plans took 1.2–1.8 s.
   - **Cause:** the caps are wall time, and 28 SMT-paired workers slow each plan. Every affected record is flagged (`wall_clock_cut`, `query_wall_cut`), so nothing is silently non-deterministic, but the criterion is not met.
   - **Proposed fix (needs your approval; not applied):** make the node budget the binding limit and keep wall time only as a generous safety cap (for example 60 nodes with a 10 s cap). Otherwise, use 16 workers, one per physical core.

**Engineering bugs found and fixed during the smoke:**

| Bug | Effect | Fix |
|---|---|---|
| Missing `pool.close()` | The main Pass B crashed after all 35 queries were written, so that run's resource-monitor log was lost | Fixed. Resource figures below come from the identical replica runs. |
| No orphan clean-up on resume | The interrupted run left one orphan trajectory, from a query killed mid-flight that later found different witnesses | Fixed: trajectories no finished record references are removed on start-up. The audit is now clean. |
| Bytecode written into the original code folders | My first imports wrote 17 untracked `cpython-313.pyc` files into `dataset_bd_classquota/code/` | Removed; the script now never writes bytecode |

## Resources (local machine only; no SSH and no GPU)

- **CPU pinning:** 28 logical CPUs `[0–13, 16–29]` (cores 0–13 with their SMT siblings), leaving cores 14–15 free.
  - One BLAS/OpenMP thread per worker, `nice 10`, at most 28 processes, `maxtasksperchild=4`.
  - A cgroup `CPUQuota` could not be set: `systemd-run` does not exist inside the VS Code flatpak sandbox. Affinity to 28 CPUs bounds total CPU use to 28 CPU-equivalents.
- **Guard:** 1 s sampling, aborting if MemAvailable < 2 GB, swap grows by more than 512 MB, CPU temperature exceeds 95 °C, disk free < 1 GB, or 240 s pass. It never fired.

Per full 28-worker run (replicas and the resumed interrupt test):

| Measure | Value |
|---|---|
| Wall time | 12.3–12.8 s |
| CPU | 245–259 CPU-s |
| Min MemAvailable | 12.8 GB (14.4 GB before) |
| Swap | 0 |
| Workers' RSS / PSS | at most 2.0 GB / 1.18 GB |
| Single worker peak RSS | 72.8 MB |
| Peak CPU temperature (`k10temp`) | **87.8–91.5 °C**, close to the 95 °C cutoff |
| Leftover processes | 0 after every run |

**Total CPU spent:** about 1,270 CPU-s of query work (Pass A 13.5; main 244.7; replicas 258.6 + 257.1; interrupt test at most about 112 before the kill + 245.2 after), plus serial audits and rendering.

## Files

- **Main folder:** [`smoke_20261106/`](./) (12 MB):
  - `records/<root>.json` (4), `modes/<root>.npz` (4), `trajectories/<id>.npz` (190);
  - `parts/` (per-query commit files);
  - `visuals/`: 4 placement maps and the witness GIF with its filmstrip;
  - `summary.json`, `manifest.json` (sha256 and size of every file, array conventions, category codes), `run_*.json` (resource logs) and this report.
- **Pass A:** [`passA_serial/`](passA_serial/).
- **Repeats:** [`replica_run/`](replica_run/), [`replica_run2/`](replica_run2/), [`interrupt_test/`](interrupt_test/).

**Figures:** each is drawn from the reopened files, with the full arm bodies of `geometry.bodies`.
- Placement maps: green = 1, red = 0, plain = legal but not attempted, hatched = occupied. Example: [`visuals/2d-plus2d-srv-w0408-s006_placements.png`](visuals/2d-plus2d-srv-w0408-s006_placements.png) shows the vacated-cell contrast.
- Witness GIF: [`visuals/2d-plus2d-srv-w0230-s011_witness.gif`](visuals/2d-plus2d-srv-w0230-s011_witness.gif) is the intermediate-state query, with a filmstrip alongside.

## Limitations

- **Zeros are budget outcomes.** All six are `BUDGET_EXHAUSTED` under small smoke caps. In every one, all 5 carry plans were cut by the 2 s clock, and the 12 s cap stopped the query with 19–79 valid pairs still untested. None is evidence of physical impossibility.
- **Nominal attachment.** As in `skill.plan_skill`, place modes are screened with the nominal attachment. Witnesses use the actual grasp offset and are re-certified.
- **Untimed witnesses.** The 2D teacher's witnesses are untimed joint waypoints. Witness files store the waypoint index, the derived object position per waypoint, phase runs, start and end configurations, and joint travel. The GIF's `s` is normalised path length, not time.
- **Hypothetical scenes.** These place prefix objects at cell centres. They are object-pose edits, never a claimed robot state, even where an archived witness of that move exists.
- **Post-grasp choice.** The definition excludes the approach line. This is a protocol choice; it should be confirmed before any production run.
- **Script length.** `placement_smoke.py` is 402 lines, above the 120–280 guideline, because of the mode accounting, the reuse check and the resource guard.

**STOP.** No further roots, no 3D labels, no sampler or X/Y training, no budget changes, and no server use. The next step needs your authorization.
