# Push audit: `origin/naren/3d-dataset-relocations`

**Remote:** `https://github.com/Narendhiranv04/rss-ebm-infeasibility.git`.
- All pushes were normal; there was no force push and nothing was merged to `main`.
- `git log --merges 3baf50ab..HEAD` is empty.
- `naren/2d-dataset-relocation` and `naren/2d-object-relocations` both still point at `3baf50ab`.

## Commits (oldest first)

| SHA | milestone | pushed |
|---|---|---|
| `6f042be12de3384f884a6f831e1e8c76d8516c5a` | cherry-pick of `8032ad0b`: the previous 1,492-root snapshot and the SHELF66V3 code switch, tree-identical under the shelf path (author preserved) | yes |
| `32c3194fc9a9da59a123e0f3a22723451238c3dd` | live source snapshot (IN PROGRESS cut 2026-11-07T05:28:58Z), sync report, manifest, live split / roster / pin | yes |
| `991694b2…` | v2 branch-aware teacher, hardened collector, guard module, tests, frozen v2 design | yes |
| `a3956bd2…` | bounded v2 mock runs, validation, figures, report, decision (`MOCK_V2_PASS`) | yes |
| `f734d22625aceac64f39116a12e5c77429849d39` | guarded full-collection preflight (launcher, unauthorized template, runbook) | yes; **remote HEAD verified** with `git ls-remote` |
| *this commit* | `PUSH_AUDIT.md`, final `BRANCH_PROVENANCE.md`, `ARTIFACT_MANIFEST.sha256`, fresh-clone evidence (`mock_v2/validation/push_verification/`, `code/verify_clone.py`) | its SHA and the remote HEAD after the push are reported in the session's final message (a file cannot contain its own commit SHA) |

## Repository-level checks (`3baf50ab..f734d226`)

| check | result |
|---|---|
| `git merge-base --is-ancestor 3baf50ab HEAD` | pass |
| `gh-yadu-iiith/planar_2d/dataset_relocations_v1/{code_v2, launch_ready_2d_v2}`, all of `planar_2d/` and `open_tabletop/` | tree SHA unchanged (`17ec75fc…`, `504a7472…`) |
| `dataset_placement_extension_v1/` (immutable v1 mock) | tree SHA unchanged (`00e4be6f…`) |
| paths changed outside `gh-yadu-iiith/shelf_6x6_open_sides/` | 0 |
| files added or modified | 22,567, totalling 431 MB. Largest: 4.2 MB (a GIF). No file ≥ 50 MB, so nothing needed LFS, chunking or omission |
| secret scan (tokens, private keys, AWS keys, password/API-key patterns, sensitive file names) over every changed text file | 0 hits |
| caches / virtualenvs (`__pycache__`, `*.pyc`, `.pytest_cache`, `venv/`) | 0 |
| omitted data | **none**. Every selected root, every referenced witness and every unreferenced pre-cut witness of the snapshot is in Git. Only the live `viz/`, the live control knobs and 125 witness files written after the cut were left out, by design (see `source_sync/LIVE_3D_SOURCE_SYNC_REPORT.md` §5) |

## Fresh-clone verification (independent checkout of the pushed `f734d226` on gvlab, MuJoCo 3.16.0)

`git clone --single-branch --branch naren/3d-dataset-relocations --filter=blob:none` from GitHub, with a sparse checkout of `gh-yadu-iiith/shelf_6x6_open_sides/`. Script: `code/verify_clone.py`. Outputs: `mock_v2/validation/push_verification/`.

| check | result |
|---|---|
| clone HEAD | `f734d22625aceac64f39116a12e5c77429849d39` |
| live snapshot files vs `SOURCE_SNAPSHOT_MANIFEST.json.gz` (SHA-256) | 20,577 / 20,577 equal, 0 missing, 0 unlisted raw files |
| selected roots present | 1,585 / 1,585 (NONE 3, K0 511, K1 357, K2 357, K3 357, K4 0) |
| witnesses referenced by the selected roots: present / re-hash equals id | 12,051 / 12,051, 0 corrupt |
| `selected.json` reproduced from the cloned ledger + `EXCLUDED_FIXED_OBJECT.json` | yes |
| v2 sidecar bytes vs the gvlab working copy | 1,970 / 1,970 identical (the only working-copy extra was this commit's `BRANCH_PROVENANCE.md`) |
| native source-witness replay (unchanged `shelf_scene.run_plan`) on 6 newly added K3 roots | 6 / 6 agree with the recorded outcome |
| v2 witness replay from the clone (`adaptive_v1` run A; serial subset) | 247 / 247 and 29 / 29 verified, 0 difference |
| v1 witness replay from the clone (v1 code, v1 reference run) | 26 / 26 verified, 0 difference |
| clone's own test suite (`pytest code/test_mock3d_v2.py`) | 25 passed (7 min 18 s) |

## Checksums

- `ARTIFACT_MANIFEST.sha256` holds the SHA-256 of every file in `dataset_placement_extension_v2/` except itself.
- `source_sync/SOURCE_SNAPSHOT_MANIFEST.json.gz` holds the SHA-256 of every published source file.
- `mock_v2/configs/teacher_code_at_freeze.sha256` and `mock_v2/configs/AMENDMENT_V2_001.json` hold the frozen and amended code hashes.

To check the sidecar from a checkout:

```bash
cd gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2 && sha256sum -c --quiet ARTIFACT_MANIFEST.sha256
```
