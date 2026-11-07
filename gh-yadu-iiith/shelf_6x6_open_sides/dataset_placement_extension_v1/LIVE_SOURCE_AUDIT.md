# LIVE_SOURCE_AUDIT: SHELF66V3 source as seen from `nav@gvlab` (2026-11-07)

Machine-readable version: `audit/live_source_inventory.json`. SHA-256 manifest of every audited source file:
`audit/source_manifest.json.gz` (12,424 entries, digest `4c5d5d3f…33aa`).

## 1. Where the live dataset is, and what that means here

- **The live generator does not run on gvlab.**
  - `ps -eo user,pid,pcpu,etime,args` showed no `run_shelf66*`, `bd_3dt*`, `run_3ds66*`, `shelf_scene` or `bd_shelf` process for any user.
  - A filesystem search of `/` (`-xdev`) and `/ssd_scratch` found no copy of `dataset_2500_raw_shelf66v3`, `yadu_bd`, `run_shelf66_full.py` or any `*3ds66*` data outside this mock's own clone.
  - The accounts `user1`, `yadu` and `ayush` do not exist on gvlab.
  - `/home/narendhiran` exists but is not readable by `nav` (permission denied), so it could not be checked.
- **The generator is on a different host.** The source's launcher `code/run_3ds66_v3_server.sh` places it on `user1@10.4.25.63` at `$HOME/yadu_bd/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3`. gvlab is `10.4.25.26`.
  - A laptop part (`lap3`, 5 workers) is merged into that store.
  - I have no access to 10.4.25.63 and no authorization to log in there, so the live store was **not** inspected.
- **The newest data reachable from gvlab is GitHub `yadu/dataset-3d` @ `8032ad0bf842a490897a8247ee46312036c12546`.**
  - Branch HEAD equals this commit, checked when the work started.
  - The clone is sparse, detached, and read-only in use: `~/approachb3d_mock/src`.
  - Its own status line reads: *"IN PROGRESS: snapshot of 2026-11-06 23:24 IST; the quotas are not met yet and generation continues on the lab server"*.
- **Consequence 1: no CPU conflict on gvlab.** No project job ran on gvlab before the mock, so the 18-logical-CPU ceiling belonged to the mock alone. `MOCK_BLOCKED_BY_ACTIVE_SOURCE_GENERATION` does not apply.
- **Consequence 2: any roster built later must be rebuilt from the live store.** The live selection will differ: K3 and NONE are still growing, and roots accepted after 2026-11-05 09:20 still need the fixed-object check (`EXCLUDED_FIXED_OBJECT.json` rule). Every number below describes the snapshot.

## 2. Snapshot inventory, computed independently (`audit3d.py inventory`)

| item | value |
|---|---|
| selected roots | **1,492**: NONE 3, K0 511, K1 357, K2 357, K3 264, K4 0 (quota NONE 408 / 511 / 357 / 357 / 357 / 510) |
| duplicate selected IDs | 0 |
| selected class = root `verified_K` | 1,492 / 1,492 |
| teacher / generator / protocol / config mix | single: `BD3DS66_TEACHER_V2`, `BD_GEN_V1+ROUTE_V15+SHELF66V3`, `XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022`, `8d521b1f6d23db23` |
| parent families among selected | 1,025 (329 lineage groups hold more than one root) |
| raw root files / ledger lines | 1,710 / 2,125 (1,710 accepted, 415 construction rejections), 0 root files outside the ledger, 0 duplicate ledger IDs |
| ledger generator mix | 2,125 × `SHELF66V3`. 1,088 accepted roots were re-taught from V2 (`retaught_from`), and all of them carry V3 identity |
| accepted but not selected | 218: DEPENDENT 50, INCONSISTENT 22, NONE_SOLVABLE 28, NONE_UNAUDITED 1, NONE 1, and K0 29 / K1 48 / K2 39 over quota |
| `EXCLUDED_FIXED_OBJECT` | 3 roots (`srv-w0044-s020`, `srv-w0004-s021`, `srv-w0030-s007`); **none selected** |
| witnesses referenced by selected roots | **10,927**: 0 missing, 0 corrupt (every `q` is `(N,7)` and finite); schema `carry_xy, phases, q, segment` × 10,927 |
| witness files on disk | 15,059 (4,132 not referenced by the selection, from re-taught or unselected roots) |
| `raw/PROGRESS.json` | **older than `selected.json`**: 20:28 vs 23:24, with NONE 4 vs 3 and K3 243 vs 264. `selected.json` is authoritative for the snapshot |

## 3. SHELF66V3 conventions verified (loaded through the source's own `bd_shelf.env()`)

- **Shelf:** 6×6 cells, 7.5 cm pitch, x0 = 0.42 m, floor 0.55 m, clear height 0.60 m, no side walls.
- **Robot and motion settings:** xArm7 (7 DoF, MuJoCo Menagerie). `NO_OVERPASS=True`. `ARM_CEILING=True` (tallest object + 0.10 m). `RETURN_TO_READY=False`. `CARRY_LIFT` 2.5 cm.
- **Grasps:** **32 grasp poses** = 16 azimuths (every 22.5°) × level tilt 90° × {middle, top} (`GRASP_AT=middle,top`). Up to 4 IK branches per pose.
- **Margins:** structure 3 mm, objects 5 mm, free move 2 cm.
- **Planner:** RRT-Connect, 5,000 iterations, no wall clock.
- **Source file hashes:** `shelf_scene.py 440ac53766f8`, `rrt_connect.py 01972f63842d`, `rrt_star.py bc46b48d5e68`. These reproduce the protocol hash `cd7f78f44022` and the config hash `8d521b1f6d23db23`.
- **Finger pre-shaping (V3):** every *object* check (`_object_hits`) sets the finger joints to the grasp's open width, which is the object width + 2 × 6 mm (`_check_width`).
  - The *structure* check (`static_ck`) sets only the 7 arm joints, so its fingers stay at the model default: **pad gap 0.0889 m, i.e. fully open** (calibrated range 0.005–0.0888 m).
  - This is conservative: it rejects more against the shelf, never fewer.
  - The placement teacher keeps it unchanged to preserve protocol identity, and reports it as a known inherited behaviour (MOCK_REPORT §8).

## 4. Host facts (gvlab)

| item | value |
|---|---|
| CPU | Intel Xeon Gold 5317 @ 3.00 GHz. 24 logical CPUs = 12 cores × 2 SMT (CPU k and k+12 share core k), 1 socket |
| affinity / cgroup | the session sees CPUs 0–23. cgroup v2 `user.slice/user-1012.slice`, delegated controllers **`memory pids` only** |
| hard CPU quota | **none available**. `systemd-run --user --scope -p CPUQuota=200% -p AllowedCPUs=0-1` created a scope, but no `cpu.max` or cpuset appeared and affinity stayed 0–23. Enforcement is therefore per-worker affinity (one logical CPU each), `nice 10`, 1 numerical thread, a coordinator pinned to the same mask, and a monitor that checks the union of affinities and test-owned CPU% every 2 s and fails closed. Affinity and nice are not quotas |
| memory | 62.4 GiB total, ~60 GiB available; swap 8 GiB (≈ 0.1 GiB used before the mock) |
| disk | `/` ext4 (`usrquota`, no quota set for `nav`), 438 GB, **64 GB free (85% used)**, inodes 11%. `/ssd_scratch` 77 GB free (96% used); not used |
| sensors | `coretemp` package temperature (24–29 °C idle); also `pch_lewisburg`, `eno1`. `hwmon2` has **no `name` file**: the first launch's monitor crashed on it (fixed, `configs/AMENDMENT_001.json`) |
| GPUs | 2 × RTX 4090, idle. MuJoCo **EGL headless rendering works** (`MUJOCO_GL=egl`). Physics stays on CPU; GPUs are used only to render figures and GIFs |
| software | Python 3.10.12, NumPy 2.2.6, MuJoCo 3.16.0 (pinned in `requirements.txt`), in a private conda prefix `~/approachb3d_mock/env`. No system package was touched |
| other users | 9 logged-in sessions (abhishek, furqan, sajid.ansari, nav). Load ≈ 0.2–0.3 before the mock. No CPU-heavy foreign process |

## 5. What the mock wrote

The mock writes only to:
- `gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v1/` inside the private clone `~/approachb3d_mock/src`, on the local branch `naren/shelf66v3-placement-mock-v1` (not pushed);
- `~/approachb3d_mock/{env,scratch}`.

The source `raw/`, `code/`, `selected.json`, ledger, and the live generator were never written. Before and after re-hash: MOCK_REPORT §10.
