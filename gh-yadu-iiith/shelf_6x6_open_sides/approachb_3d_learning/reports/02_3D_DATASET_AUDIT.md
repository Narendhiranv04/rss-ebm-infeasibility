# 02: 3D v4 dataset snapshot and audit

**Snapshot `v4full_82926152fa9cc345`, kind `FULL_COMPLETE_V4`.** It is read-only on the training server at `~/approachb_3d_learning/snapshots/`. Machine-readable versions: `02_3D_DATASET_AUDIT.json` and `<snapshot>/SNAPSHOT_MANIFEST.json`.

## Provenance and verification

**Collection status.**
- gvlab `production_v4/run_q2_parity` is **complete**: 1,466 / 1,466 `unit_done` events, and the last ledger `end` is complete.
- `collect4 --audit`: complete, 1,466 validated, 59,972 queries, 40,822 witness associations.
- The 100 % witness replay verified 40,822 / 40,822 (38,833 unique witnesses), with a maximum joint-execution difference of 0 and a maximum object end difference of 0 m.

**Transfer path.**
- Source: the laptop's off-host mirror, which was verified against the gvlab backup index at 2026-11-08T00:26Z.
- Route: laptop → server. gvlab → server SSH is not configured and was not changed.

**Checks applied at build** (`data3d.build` refuses the whole snapshot on any failure):
- Every unit JSON sha256 matches the gvlab backup index (1,466 files).
- The mirror's `run_manifest.json` is byte-identical to gvlab's.
- Every unit's record `qid`s equal the run manifest's list.
- Every record joins the frozen roster by `qid`, with the same split, object and destination.
- No ghost witnesses: every `witness_id` is present in its unit's `witness_index`.
- Every source root's sha256 equals the v4 source pin (1,466 roots).
- Every run-manifest input sha256 matches the upstream files: roster, contexts manifest, QC roots, source pin, split manifest and forecast.
- The production-policy sha256 matches the run manifest. The policy was sanitized before transfer: authorization and token fields were removed, and the original's sha256 was kept.
- Label counts equal the gvlab audit exactly.

**Bound into the snapshot id:**
- the upstream commit `6c72eb4be17b`, which contains the v4 freeze `83c1135f`;
- the roster sha `80134f63…`, all input hashes, the policy `adaptive_v1` with stages `(4,500)→(8,1500)→(16,5000)`, and the teacher protocol `89cbabed5d7e4847`;
- every unit JSON hash, every witness NPZ hash (1,465, from the gvlab index), and every source-root hash.

**Witness layer.** The 1,465 NPZ files form a separate read-only layer, `<snapshot>_witnesses`. `attach-witnesses` verifies each file against the bound hash. One unit has no NPZ because it has no positive witness; this was checked. Status: see `07`.

**Immutability.**
- The snapshot and the `_derived` feature cache are `chmod a-w`; a write test was denied.
- `SHA256SUMS` covers every file.
- Builds refuse to overwrite an existing snapshot.

## Counts

| | Train | Val | Test | Total |
|---|---|---|---|---|
| Physical queries | 42,612 | 8,752 | 8,608 | **59,972** |
| `1 VERIFIED` (independently replayed witness) | 29,175 | 6,025 | 5,622 | 40,822 |
| `0 NEG_CERTIFIED_GEOMETRY` (finite declared scope) | 5,274 | 1,089 | 1,209 | 7,572 |
| `0 NEG_UNDER_POLICY` (BOUNDED_SEARCH 2,492 / DETERMINISTIC_PLAN 4,132 / EXECUTION 18; **operational**, not impossibility) | 4,703 | 974 | 965 | 6,642 |
| `None UNLABELED_NO_VALID_POST_GRASP_STATE` (excluded from binary negatives) | 3,460 | 664 | 812 | 4,936 |
| Query-bearing roots | 1,036 | 226 | 204 | 1,466 |

**Additional counts:**
- QC-eligible roots in the cohort: 1,582. The 116 roots without any non-target object have no physical query.
- Static contrasts: **4,616** (3,150 OCCUPIED, 1,466 OUT_OF_BOUNDS). They are a separate auxiliary stream, never physical negatives, and are not used in this kickoff.
- Contexts (unique root × hypothetical scene × held object): 10,610.
- Train generator support: 29,175 positive (context, destination) pairs over 6,513 contexts, 4.48 per context.

## Split hygiene

- Roots in more than one split: **0**.
- Full physical scene layouts (every object's cell, shape, size and height) in more than one split: **0**.
- The inherited lineage split is unchanged, and every `qid` keeps its split.
- No v3 smoke or validation record is included.
- TEST was evaluated only after `FROZEN_FOR_TEST.json` was written.

## Labelled positive rate by stratum (all splits; reporting only)

| | |
|---|---|
| Shape | box 0.734 · cylinder 0.749 |
| Height tertile | short 0.692 · medium 0.763 · tall 0.765 |
| Q2 block | root 0.759 · prefix1 0.732 · prefix2 0.737 · prefix3 0.666 |
| Destination band | near 0.818 · mid 0.700 · far 0.660 · vacated 0.761 |
| Objects in scene | 2: 0.966 · 3: 0.882 · 4: 0.792 · 5: 0.678 · 6: 0.675 · 7: 0.659 · 8: 0.606 |
| K (reporting field only, never an input) | K0 0.705 · K1 0.781 · K2 0.745 · K3 0.739 |

## Limits of Q2 selection (important for interpreting sampler metrics)

- Destinations are a **stratified subsample**: at most 8 per object, interleaved near / mid / far, with inclusion probability k / n recorded. Cells that were never tested are **UNKNOWN**, never negative.
- A positives-only generator therefore learns `p(d | VERIFIED, tested)`. That mixes feasibility with the roster's selection design.
- Hit and recall of *recorded* positives reward matching the roster's sampling. This is why the band-stratified control, which mimics that design, scores high on hit@1. Known-label precision is less sensitive to this effect.
- An unbiased precision estimate for proposals outside the tested support needs a later, separate teacher evaluation on an authorized physics host. None was run here.

## Loader

- Features come from `data3d.featurize`: 14 label-blind per-object features, an occupancy grid, the legality mask, and strata. TRAIN-only scalers are in `scalers.json`.
- Whole-snapshot GPU tensors, no DataLoader workers, no MuJoCo or IK at training or inference.
- Features sha256: `4a37b2edb6c572d2146467eb97044ab13c66ff4fb8cc2263ed36f0daec0592f7`.
