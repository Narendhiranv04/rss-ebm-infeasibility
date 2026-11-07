# Production v4: the 3D Q2-parity placement training dataset, ready pending a new user authorization

v4 corrects the training-query coverage of the 3D placement dataset. The v3 roster kept the sparse 10-slot mock design, 13,651 queries. v4 applies the finalized **2D Q2 procedure** to the same 1,582 QC-eligible first-cohort roots:
- **root scene:** up to 4 objects × 8 destinations;
- **contexts:** two independent one-edit contexts and one two-edit context, each with up to 2 unmoved objects × 4 destinations.

The 6×6 shelf's geometry sets the source-distance and destination bins. The physical teacher, its 13 MuJoCo gates, the source-exact collision protocol and `adaptive_v1` are **unchanged** (protocol `89cbabed5d7e4847`). **Nothing has been launched.**

| | |
|---|---|
| Decision (mechanical) | [`V4_READINESS_DECISION.json`](V4_READINESS_DECISION.json), from `code/decide_v4.py` |
| Workload | **59,972 physical queries** (train 42,612, val 8,752, test 8,608); root 35,064, prefix1 9,224, prefix2 9,224, prefix3 6,460; 4,616 static contrasts; 1,466 query-bearing roots |
| Policy and resources | `adaptive_v1` (4 × 500 → 8 × 1,500 → 16 × 5,000); 12 workers on CPUs 0–11 (≤ 18 logical CPUs) |
| Time and storage | ≈ 15.6–17 h expected, **33.4 h buffered**; 2.6 GB (3.9 GB buffered) |
| Authorization | a **new**, v4-specific authorization: [`V4_RUNBOOK.md`](V4_RUNBOOK.md) §4. The v3 authorization does not apply. |

## Contents

| File | What it is |
|---|---|
| [`Q2_PARITY_SPEC_AND_COVERAGE.md`](Q2_PARITY_SPEC_AND_COVERAGE.md) | the procedure (2D Q2 → 3D), exact counts, reductions, coverage, the 9 gates, label semantics, the loader schema for the sampler |
| [`COVERAGE_REPORT_V4.json`](COVERAGE_REPORT_V4.json) | every histogram and joint table, the gates, the count reconciliation |
| [`V4_VALIDATION_REPORT.md`](V4_VALIDATION_REPORT.md) | job states with evidence, code changes, physical integration, guards, forecast, monitoring, one incident |
| [`V4_RUNBOOK.md`](V4_RUNBOOK.md) | validation, status / backup service install, authorization, launch, monitoring, stop / resume / rollback, post-run audit / replay / utility, data sync |
| `frozen/` | the roster, contexts, semantic-duplicate map, roster manifest, forecast |
| [`production_policy_v4.TEMPLATE.json`](production_policy_v4.TEMPLATE.json) | the **unauthorized** frozen v4 policy |
| [`launch_v4.sh`](launch_v4.sh) | the inert launcher |
| `integration/` | smoke roster, test fixtures, smoke runs A / B, replays, training-utility report, production preflight, pytest logs |
| `status_service/` | the measured state of the 10-minute service and backups |
| `figures/q2_parity_coverage.png` | coverage by block × band and by split × K |
| `push_verification/` | the fresh-checkout verification of the pushed commit |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 of every file in this directory except itself |

**Unchanged inputs from v3:** `production_v3/QC_ELIGIBLE_ROOTS.json`, `frozen/source_pin_first_cohort.json` and `frozen/split_manifest_first_cohort.json`.
