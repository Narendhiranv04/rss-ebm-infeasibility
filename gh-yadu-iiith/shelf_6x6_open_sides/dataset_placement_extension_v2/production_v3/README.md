# Production v3: the first K0–K3 SHELF66V3 placement collection, ready pending the user's authorization

This directory freezes and validates the first production collection of the SHELF66V3 placement-only dataset. The physical event is V_place(S_hyp, o, d), conditional on a certified post-grasp hold:

> already held → lift → carry → lower → release → retreat → settle, with 13 MuJoCo gates.

It contains:
- the QC-proven source cohort;
- the frozen roster, split and pin;
- the guarded production launcher, run on `nav@gvlab`;
- the integration evidence and the readiness decision.

**Nothing here has been launched.** The policy template is unauthorized.

| | |
|---|---|
| Decision (mechanical) | [`PRODUCTION_READINESS_DECISION.json`](PRODUCTION_READINESS_DECISION.json) (from `code/decide_v3.py`) |
| First cohort | 1,582 QC-eligible roots (K0 511, K1 357, K2 357, K3 357). 13,651 physical queries: train 9,664, val 2,066, test 1,921. 2,768 static contrasts. |
| Policy | `adaptive_v1` (4 × 500 → 8 × 1,500 → 16 × 5,000), locked by the V2 mock decision. Teacher protocol `89cbabed5d7e4847`, unchanged. |
| Workers | 12 single-threaded workers pinned to CPUs 0–11, under the 18-logical-CPU ceiling |
| Forecast | 35.2 CPU-h (32.8–37.8); ≈ 3.3 h wall time expected and **5.35 h at p95 × 1.5**; 0.64 GB of output (0.96 GB with the 1.5× buffer) |
| How to launch | [`FIRST_COHORT_RUNBOOK.md`](FIRST_COHORT_RUNBOOK.md); step 4 is the user's authorization |

## Contents

| File | What it is |
|---|---|
| [`SOURCE_QC_AUDIT.md`](SOURCE_QC_AUDIT.md) | the live store's final state, the fixed-object QC (author evidence + unmodified re-check), lineage |
| [`QC_ELIGIBLE_ROOTS.json`](QC_ELIGIBLE_ROOTS.json) | per-root QC status, provenance and root SHA-256; equivalence with the author's check |
| [`SPLIT_AND_ROSTER_AUDIT.json`](SPLIT_AND_ROSTER_AUDIT.json) | inherited split, closure, leakage, roster counts, coverage, label semantics |
| `frozen/` | `first_cohort_roster.jsonl.gz`, `split_manifest_first_cohort.json`, `source_pin_first_cohort.json` (roots + source code / model / selection files), `cohort_manifest.json`, `forecast_first_cohort.json` |
| [`production_policy.TEMPLATE.json`](production_policy.TEMPLATE.json) | the **unauthorized** frozen policy (authorization = null) |
| [`launch_first_cohort.sh`](launch_first_cohort.sh) | the inert launcher: preflight → guard → tests → collector |
| [`FINGER_WIDTH_QUERY_SENSITIVITY.md`](FINGER_WIDTH_QUERY_SENSITIVITY.md) | the alternate pre-shaped structure check changes 0 of 338 labels; no protocol change |
| [`PRODUCTION_INTEGRATION_TEST_REPORT.md`](PRODUCTION_INTEGRATION_TEST_REPORT.md) | code changes; physical smoke, kill + corruption recovery, replay, refusals, lock, preflight, tests |
| `qc/` | the author's evidence, copied verbatim with host-side hashes; the full re-check (jobs, per-process results, monitor); the determinism re-run; per-object verdicts |
| `source/` | `SOURCE_FINAL_STATE_AUDIT.json`: the read-only audit of the stopped live store |
| `integration/` | smoke roster, test fixtures, smoke runs A and B (marked `INTEGRATION_TEST`), replays, preflight, pytest logs, and the earlier attempts with their notes |
| `finger_width/` | static screens, targeted and cohort-sample runs under both identities, comparisons, replays, summary |
| `figures/` | `first_cohort_composition.png`, `finger_width_sensitivity.png`, `smoke_positive_witness.gif` (a replayed K3 one-edit placement into a vacated cell) |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 of every file in this directory except itself (check: `sha256sum -c --quiet SHA256SUMS`) |

**What stays unchanged.** The V2 sidecar is byte-identical and still verifies against `../ARTIFACT_MANIFEST.sha256`; that covers `mock_v2/`, `production_v2/` and the V2 code. So do the V1 sidecar (`../../dataset_placement_extension_v1/`), the native snapshot and the 2D trees.

**What is superseded.** `production_v2/` and `mock_v2/launch_full_3d.sh` were built from an in-progress roster that included unchecked and NONE roots. Production v3 supersedes them.
