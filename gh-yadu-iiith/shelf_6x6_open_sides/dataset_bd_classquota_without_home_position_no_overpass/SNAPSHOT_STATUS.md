# Snapshot status: IN PROGRESS, live cut of 2026-11-07 05:28:58 UTC (10:58:58 IST)

The `raw/` store and `selected.json` here are a consistent copy of the live SHELF66V3 generator store. That store is on the lab server `user1@10.4.25.63`, at `~/yadu_bd/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3`.

Generation was still running at the cut (wave 113, 18 workers, K4 requests under way). **This is not the final dataset.**

| Class | Selected now | Quota | Previous snapshot (commit `8032ad0b`, 6 Nov 23:24 IST) |
|---|---|---|---|
| NONE | 3 | 408 | 3 |
| K0 | 511 ✔ | 511 | 511 |
| K1 | 357 ✔ | 357 | 357 |
| K2 | 357 ✔ | 357 | 357 |
| K3 | 357 ✔ | 357 | 264 |
| K4 | **0** | 510 | 0 |
| **Total** | **1,585** | 2,500 | 1,492 |

- **What changed since `8032ad0b`:**
  - **New files:** 3,804, all of them `raw/` additions: 362 new root files and 3,442 new witness files.
  - **Changed files:** 3.
    - `raw/ledger.jsonl` gained 567 appended attempt lines. Its first 2,125 lines are the previous snapshot's lines, same set and byte-identical, in the live host's order.
    - `raw/PROGRESS.json` was updated.
    - `selected.json` was regenerated.
  - **Removed or modified:** none. 16,770 files are byte-identical to `8032ad0b`, and every previously selected root is still selected.
- **Ledger order:** the previous snapshot's ledger came from the other side of the laptop/server ledger merge (`code/merge_with_server_3ds66_v3.sh`), so its line order differs. The selection result is identical for K0–K2 under both orders.
- **Selection:** `selected.json` follows the previous snapshot's rule. It is `run_shelf66_full.select` with the scenes in `raw/EXCLUDED_FIXED_OBJECT.json` skipped, and that procedure reproduces the previous `selected.json` exactly.
  - The generator's own `PROGRESS.json` counts NONE = 4 because it does not apply the exclusion; `selected.json` counts 3.
- **Fixed-object check is pending** for 488 selected roots. `EXCLUDED_FIXED_OBJECT.json` covers only the first 1,147 accepted roots (up to 5 Nov 09:20); the 488 roots were accepted after that. They are listed in `selected.json` → `fixed_object_check_pending`. The final selection can still change when the generator's author runs that check.
- **Same protocol:** all 2,072 accepted roots carry `BD_GEN_V1+ROUTE_V15+SHELF66V3`, `BD3DS66_TEACHER_V2`, `XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022` and config `8d521b1f6d23db23`.
  - The live generator code is byte-identical (git blob SHA-1) to this directory's `code/` and `../scene/` as of `8032ad0b`.
- **Not copied:** the live `viz/` directory (the previous snapshot did not include it either), and the live control knobs `K4_MAX_RUNNING` and `PROGRESS_reteach_srv.json`. The 125 witness files written after the cut are also excluded; they belong to scenes not yet in the ledger prefix.

The audit, consistency method, checksums and comparison are in [`../dataset_placement_extension_v2/source_sync/LIVE_3D_SOURCE_SYNC_REPORT.md`](../dataset_placement_extension_v2/source_sync/LIVE_3D_SOURCE_SYNC_REPORT.md). The machine-readable manifest, with SHA-256 for every file, is `../dataset_placement_extension_v2/source_sync/SOURCE_SNAPSHOT_MANIFEST.json.gz`.
