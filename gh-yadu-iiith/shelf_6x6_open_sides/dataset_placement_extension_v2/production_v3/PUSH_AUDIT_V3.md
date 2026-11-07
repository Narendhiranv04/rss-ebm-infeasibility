# Push audit: production v3 on `origin/naren/3d-dataset-relocations`

**Remote:** `https://github.com/Narendhiranv04/rss-ebm-infeasibility.git`.
- All pushes were normal: no force push, no rebase of published history, nothing merged to `main`.
- `naren/2d-object-relocations` and `naren/2d-dataset-relocation` still point at `3baf50ab`, and `main` at `82ebf6e6`.

## Commits (on top of `b943dff3`)

| SHA | Content |
|---|---|
| `43bfda4a8` | source final-state audit and fixed-object QC (author evidence verbatim, unmodified re-check, `QC_ELIGIBLE_ROOTS.json`) |
| `156f9d290` | first-cohort freeze: split, roster, source pin, forecast, split / roster audit |
| `bd0c038fc` | guarded collector (`collect3d.py`), guard + lock + template (`prod3d.py`), inert launcher, unauthorized template, runbook, tests |
| `af6f35cfe` | integration smoke, finger-width sensitivity, reports, figures, decision tooling, pre-push decision (GitHub gate pending). **Pushed and remote HEAD verified.** |
| *this commit* | the final `PRODUCTION_READINESS_DECISION.json`, `push_verification/` (fresh-clone evidence for `af6f35cfe`), this audit, regenerated `SHA256SUMS`. A file cannot contain its own commit SHA: the SHA and the remote HEAD after this push are in the session's final report. |

## Repository checks (`b943dff3..af6f35cf`)

| Check | Result |
|---|---|
| Merge commits | 0 |
| Files added / modified / deleted | 1,217 / **0** / **0** (only new files) |
| Paths outside `dataset_placement_extension_v2/{code,production_v3}/` (new files) | 0 |
| `planar_2d/`, `open_tabletop/`, `dataset_placement_extension_v1/`, the native snapshot `dataset_bd_classquota_without_home_position_no_overpass/`, `mock_v2/`, `production_v2/` | tree SHA unchanged |
| V2 sidecar `ARTIFACT_MANIFEST.sha256` | verifies: the V2 teacher and collector code is byte-identical |
| Largest file | 2.5 MB (a GIF); none ≥ 50 MB |
| Secrets (tokens, private keys, AWS keys, password patterns) | 0 hits |
| Caches (`__pycache__`, `*.pyc`, `.pytest_cache`) | 0 |
| Excluded on purpose | the final live ledger staged for the source audit (not needed: the published snapshot stays the pinned source; its hash is in `source/SOURCE_FINAL_STATE_AUDIT.json`); the live host's control files; no data was left out |

## Fresh-checkout verification of `af6f35cfe` (gvlab, `push_verification/fresh_clone_verification.json`)

The checkout was an independent `git clone --single-branch --filter=blob:none` from GitHub, sparse on `gh-yadu-iiith/shelf_6x6_open_sides/`, in a separate directory. Script: `push_verification/verify_clone_v3.py`.

| Check | Result |
|---|---|
| Clone HEAD | `af6f35cfe08c5f949b433dab8b7a867b5108ae52` |
| `production_v3/SHA256SUMS` | 1,206 / 1,206 OK |
| V2 sidecar manifest | OK |
| Published source snapshot | 20,577 / 20,577 files equal; the selection rule reproduces `selected.json` |
| QC verdicts rebuilt from the committed per-process results | `QC_ELIGIBLE_ROOTS.json` byte-identical |
| Cohort freeze rebuilt in the clone | roster, split, forecast, cohort manifest and split / roster audit byte-identical; source pin equal |
| Production template regenerated in the clone | frozen block identical except `SHELF_SRC` (the checkout path) |
| Metadata preflight from the clone (all 13,651 queries) | rc 0; 0 planner calls; 0 teacher-entry calls; nothing created; no refusal other than the missing authorization |
| `test_prod3d_v3.py` from the clone | 23 passed |
| Smoke A witnesses replayed from the clone | 8 / 8 verified, executed joints identical |

After that, the canonical production checkout `~/approachb3d_v3/repo` was fast-forwarded to `af6f35cf`. Its 1,216 development files were confirmed byte-identical to the commit before being replaced. `code/decide_v3.py` was then run there and gives **`FIRST_COHORT_READY_PENDING_USER_AUTHORIZATION`** with no blockers.
