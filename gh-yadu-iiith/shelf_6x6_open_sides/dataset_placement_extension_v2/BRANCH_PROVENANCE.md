# Branch provenance: `naren/3d-dataset-relocations`

Repository: `https://github.com/Narendhiranv04/rss-ebm-infeasibility.git` (remote `origin`). Every push was a normal `git push`: no force push, no merge commit, nothing merged to `main`, and no rewritten history.

## Remote state before any change (fetched 2026-11-07 ~05:25 UTC)

| ref | SHA |
|---|---|
| `origin/naren/2d-dataset-relocation` | `3baf50ab4ffb07ce056da7203e17bdc94e428d15` (2D v2 mock `0ba79986` + 3D v1 mock `3baf50ab`) |
| `origin/naren/PoC` | `386ddef792817ef7c2a276916faff54be80d3a75` (untouched) |
| `origin/yadu/dataset-3d` | `8032ad0bf842a490897a8247ee46312036c12546` (the 1,492-root SHELF66V3 snapshot; untouched) |
| `origin/naren/2d-object-relocations` | **absent** |
| `origin/naren/3d-dataset-relocations` | **absent** |

## Branches created

```bash
git fetch origin --prune
git branch naren/2d-object-relocations 3baf50ab4ffb07ce056da7203e17bdc94e428d15     # requested base: absent, so created as an alias
git push -u origin naren/2d-object-relocations                                        # new branch, points at 3baf50ab (unchanged since)
git branch naren/3d-dataset-relocations naren/2d-object-relocations
git push -u origin naren/3d-dataset-relocations
```

`naren/2d-object-relocations` is a non-destructive alias of the verified 2D branch. `naren/2d-dataset-relocation` was not modified.

## Commits on `naren/3d-dataset-relocations` (oldest first)

| SHA | content |
|---|---|
| `3baf50ab` | base (= `naren/2d-object-relocations` = `naren/2d-dataset-relocation`) |
| `6f042be1` | `git cherry-pick -x 8032ad0b`: the previous 1,492-root SHELF66V3 snapshot and the SHELF66V3 code switch. It is a single, inspected commit that touches only `gh-yadu-iiith/shelf_6x6_open_sides/`, and its trees are byte-identical to `8032ad0b` for `dataset_bd_classquota_without_home_position_no_overpass/` (`40a2992f`), `scene/` (`e1f097eb`), `README.md` and `requirements.txt`. No merge of `yadu/dataset-3d`. Author preserved (`git log --format=%an`) |
| `32c3194f` | live SHELF66V3 source snapshot (IN PROGRESS cut 2026-11-07T05:28:58Z) in the native dataset path, plus `dataset_placement_extension_v2/source_sync/` and `production_v2/` metadata |
| `991694b2` | v2 branch-aware teacher, hardened collector, guard module, regression tests, frozen v2 design (`selection_rule_v2`, roster, source pin, fixtures, AMENDMENT_V2_001) |
| `a3956bd2` | bounded v2 mock: all runs, validation, figures, `3D_MOCK_V2_REPORT.md`, `3D_MOCK_V2_DECISION.json` |
| `f734d226` | guarded full-collection preflight: `launch_full_3d.sh`, unauthorized production template, runbook (pushed; remote HEAD verified; fresh clone verified) |
| next commit | `PUSH_AUDIT.md`, this file, `ARTIFACT_MANIFEST.sha256`, fresh-clone evidence |

Why the cherry-pick was needed: the 2D branch descends from `9e640690`, which holds only the shelf generator's README and code. The 3D v1 mock pushed there referenced data that existed only on `yadu/dataset-3d`. On this branch, v1 and v2 replay natively against `6f042be1`, and the live snapshot sits on top as a separate, auditable commit.

## Invariants checked

- `git merge-base --is-ancestor 3baf50ab HEAD` passes.
- `gh-yadu-iiith/planar_2d/dataset_relocations_v1/{code_v2,launch_ready_2d_v2}` are unchanged: same tree SHA at `3baf50ab` and at `HEAD`.
- `dataset_placement_extension_v1/` is unchanged: same tree SHA at `3baf50ab` and at `HEAD`.
- The commit diff touches no path outside `gh-yadu-iiith/shelf_6x6_open_sides/`.

The final numbers are in `PUSH_AUDIT.md`.
