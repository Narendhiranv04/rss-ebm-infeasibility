# Source and fixed-object QC audit for the first K0–K3 cohort

**Result:**

| Status | Meaning |
|---|---|
| `SOURCE_SNAPSHOT_INTEGRITY_PASS` | the published snapshot verifies and is the source's final K0–K3 selection |
| `FIRST_COHORT_QC_PASS` | all 1,582 selected K0–K3 roots have per-root fixed-object QC evidence |
| `FULL_2500_SOURCE_INCOMPLETE` | K4 and NONE are not part of this run; does not block the first cohort |

The first cohort is all 1,582 selected K0–K3 roots of the published snapshot: K0 511, K1 357, K2 357 and K3 357. Every one has per-root fixed-object QC evidence, and none is excluded:

- **1,004 author-approved roots:** the author's own per-object results cover every non-target object, and all pass.
- **116 roots with no non-target object:** the rule holds trivially, which is provable from the root file.
- **462 roots certified here:** the author's unmodified check tool was re-run on gvlab. These roots are kept as a **separate status** (`SIDECAR_REPRO_QC_PASS`).

Machine-readable files:
- [`QC_ELIGIBLE_ROOTS.json`](QC_ELIGIBLE_ROOTS.json) gives the status, provenance and root SHA-256 of every root.
- [`qc/qc_objects.jsonl`](qc/qc_objects.jsonl) gives the verdict of every object.
- [`source/SOURCE_FINAL_STATE_AUDIT.json`](source/SOURCE_FINAL_STATE_AUDIT.json) is the live-store audit.

## 1. The live source has finished its K0–K3 run

I audited the store read-only at **2026-11-07 10:46:31 UTC**: the store `user1@10.4.25.63:yadu_bd/runs/…/dataset_2500_raw_shelf66v3`, reached from the laptop by key. Every remote command was a read, and hashing ran at nice 19 with idle I/O priority.

| Check | Result |
|---|---|
| K-quota generator (`run_shelf66_full.py --name srv3`) | **stopped**. Its last ledger write was 07:37:29 UTC. The author's `QUOTA.json` (07:36:55 UTC) sets **K4 = 0 and NONE = 0** for this run. |
| Other generation on the host | `run_shelf66_none.py --name srvN` (NONE) writes to a **separate** directory (`…_none_dense`), and K4 experiments run under `diag66/k4_split`. Neither touches this store. Nothing was signalled. |
| Ledger append-only | yes. The published 1,539,645-byte ledger is a byte prefix of the final 1,847,012-byte, 3,331-line ledger; the host-side prefix SHA-256 is `8ec5d3b7…3713` on both sides. |
| Every published root, witness and `EXCLUDED_FIXED_OBJECT.json`, re-hashed **on the host** now | **20,574 / 20,574 identical** |
| Generator and scene code (12 files) vs `yadu/dataset-3d@8032ad0b`, compared by git blob SHA-1 | all identical |
| The author's selection rule on the **final** ledger and **final** `QUOTA.json` | K0 511, K1 357, K2 357, K3 357: **the same ordered lists as the published `selected.json`**. NONE 0 and K4 0 in this run. 2,617 accepted roots. |
| `EXCLUDED_FIXED_OBJECT.json` | unchanged since 5 Nov (SHA-256 `b068673e…`) |

**No new snapshot was published.** The run stopped with its K0–K3 quotas already met, and its final K0–K3 selection equals the published one. The roots accepted after the cut are surplus: none of them is selected, so they would add nothing to the first cohort. The published snapshot (cut 05:28:58 UTC, commit `32c3194f`) therefore stays the pinned source, byte for byte. The final ledger was staged on the laptop for this audit only.

## 2. The fixed-object rule and the author's check

The rule, from the author's `raw/EXCLUDED_FIXED_OBJECT.json`:

> A root is NOT counted / selected when one of its non-target objects cannot be picked from where it stands even with every other object gone, under the scene's own arm ceiling (tallest object + 10 cm).

The author's check lives in `~/yadu_bd/diag66` on the host. I copied it verbatim to [`qc/author_evidence/`](qc/author_evidence). Its SHA-256s were recomputed **on the host** after copying, and all 12 files match (`source_host_sha256_and_mtime.txt`). It ran in three steps:

1. **Scan 1** (`obj_pickable2.py`) put each non-target object alone on the empty shelf under its *own* height plus 10 cm. That is a lower and stricter ceiling than the scene's. It covered 2,917 objects of 1,023 roots: 2,507 pickable, 410 not.
2. **Scan 2** (`obj_pickable3.py`, SHA-256 `f301812e…`) used the **scene's** ceiling, which is the exact rule:
   - `F:` jobs re-ran all 410 scan-1 failures. Exactly 3 failed: the 3 excluded roots, each with object `o00` in the back-right cell.
   - `N:` jobs ran 125 objects of newer roots directly; all passed.
   - 20 `R:` reference jobs were run as well.
3. Each object was planned pick-and-place to cell (1,2) or (1,3), trying every grasp pose, then to (2,4) if some pose failed for a reason other than reach.

I rebuilt the author's per-object verdict as "scan-1 pass, or scene-ceiling pass". It reproduces `EXCLUDED_FIXED_OBJECT.json` exactly, and every job's shape, size, height, cell and ceiling matches the root it names (0 mismatches).

### Per-root author coverage: no timestamp cutoff is treated as a QC pass

The published `selected.json` lists 488 selected roots as pending, defined by acceptance order (after the first 1,147 accepted roots). I measured coverage root by root from the job tags instead. Of the 1,582 selected K0–K3 roots:

| Author evidence | Roots |
|---|---|
| every non-target object has an author result, all pass | 1,004 (K0 355, K1 354, K2 251, K3 44) |
| no non-target object | 116 (all K0) |
| at least one non-target object without an author result | 462 (K0 40, K1 3, K2 106, K3 313) |

Of the 488 "pending" roots:
- 5 actually have complete author evidence (`N:` jobs);
- 21 have no non-target object;
- 462 lack evidence;
- the rest are the 3 NONE roots, outside this cohort.

## 3. Re-running the author's check here

**What was run.** The author's `obj_pickable3.py` ran **unmodified**, pinned by SHA-256, against the published snapshot's native code (`code/shelf66_v3`, blob-identical to the host). The environment matched the host:

| | Host | gvlab |
|---|---|---|
| MuJoCo | 3.16.0 | 3.16.0 |
| NumPy | 2.2.6 | 2.2.6 |
| Robot model `xarm7.xml` | SHA-256 `ac6b369e…` | same file |

The planner is deterministic: seeded RRT-Connect with a 5,000-iteration cap and no wall-clock limit.

**Jobs:** 4,759 non-target objects of all 1,582 selected K0–K3 roots (tag `V:`), plus the author's own 535 `F:` and `N:` jobs, used for the equivalence test. One process ran per CPU, pinned to CPUs 0–11 at nice 10, with the fail-closed monitor (`code/qc3d.py run`). It took 24.3 minutes, with 0 monitor violations and a peak of 45 °C.

| Equivalence and determinism | Result |
|---|---|
| Author's 535 scene-ceiling verdicts, re-run here | **535 / 535 identical** |
| Author's 3 exclusions | reproduced exactly (`srv-w0004-s021`, `srv-w0030-s007`, `srv-w0044-s020`) |
| First feasible grasp and poses tried | 530 / 535 identical. In 5 jobs the first feasible grasp differs by one azimuth step (157.5° vs 180° or 202.5°); the object is pickable in all 5. |
| Repeat on gvlab: those 5 plus 55 hash-chosen jobs | **60 / 60 identical** in every field. The check is deterministic on each host; the 5 differences are a cross-host numerical tie-break in which grasp is found first, and they never change a verdict. |
| Author scan-1 passes in the selected roots, re-checked under the scene ceiling | 2,353 / 2,353 pickable |

**Results:**
- **0 of the 4,759 non-target objects fail.**
- All 1,582 roots are eligible: 1,004 `AUTHOR_QC_PASS`, 116 `VACUOUS_PASS_NO_NON_TARGET`, 462 `SIDECAR_REPRO_QC_PASS`.
- No root has a disagreement between the author's evidence and the re-check (`QC_DISAGREE_EXCLUDED` 0), and none is unchecked (`NOT_CHECKED` 0).

**What this does and does not claim.** The 462 `SIDECAR_REPRO_QC_PASS` roots are certified by re-running the **author's own tool** under the author's rule. They are **not** author-approved, and the status records that distinction for every root.

The author may still run their own final check. If it ever excludes one of these roots, two things would happen:
- that root's status changes in a new cohort version;
- the collected rows from that root are dropped by root id, with no other row affected.

The selection is never edited to hide anything: the native `selected.json` is untouched. The cohort is a versioned sidecar manifest ([`frozen/cohort_manifest.json`](frozen/cohort_manifest.json)).

## 4. Lineage, split and roster (summary; details in [`SPLIT_AND_ROSTER_AUDIT.json`](SPLIT_AND_ROSTER_AUDIT.json))

**Split.** Every assignment is inherited from the extended live split (`e9c389c4…`): 0 changed. Closure was recomputed by union-find over the parent groups, plus `parent_family_id`, plus identical layout across **all** 1,585 selected roots. It gives 1,045 components, and **0 span two splits**. The resulting split has train 1,120, val 246 and test 216 roots (SHA-256 `c0f8fb5a…`).

**Roster.** The frozen V1 production rule (`plan3d.roster_for`, geometry only) was applied to the cohort roots:
- **13,651 physical queries** from 1,466 roots; the 116 roots with no non-target object have nothing to place;
- by split: train 9,664, val 2,066, test 1,921;
- by depth: 0 → 5,822, 1 → 4,221, 2 → 3,608;
- by band: near 2,027, mid 4,761, far 4,557, vacated 2,306;
- plus **2,768 static contrasts** (OCCUPIED / OUT_OF_BOUNDS), which never go to the physical teacher.

The roster is **line-for-line identical** to the V2 live roster restricted to these roots. It has 30 fewer queries because the 3 NONE roots are out.

**Leakage.** 0 cross-split layouts among the 4,234 derived scenes (root, 1-edit and 2-edit).

**Coverage.** Every K, depth and band appears. The forecast backs off 307 `depth-2 / near` queries to the K × depth mean, because the mock had fewer than 3 samples in that stratum.

## 5. Not covered

- **K4 (0 / 510) and NONE:** generated separately by the source's author and **not in the first cohort**. No K4 or NONE example was manufactured.
- **Future selection change:** if the author later changes the K0–K3 selection (for example after their own fixed-object pass), the published snapshot and this cohort stay pinned. A later cohort version must be cut from a new audited snapshot.
