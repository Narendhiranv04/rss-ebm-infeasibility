# Live SHELF66V3 3D source sync: `LIVE_SOURCE_SYNC_PASS` (consistent in-progress snapshot)

**What this certifies.** This is a consistent, read-only, checksummed copy of the live SHELF66V3 store as it stood at **2026-11-07 05:28:58 UTC**. It was audited against the previous GitHub snapshot (`yadu/dataset-3d@8032ad0b`) and published in the dataset's native path on `naren/3d-dataset-relocations`. The push audit is in [`../PUSH_AUDIT.md`](../PUSH_AUDIT.md).

**What it does not certify.** It does not mean the generator has finished. K4 has 0 of 510 selected roots and NONE has 3 of 408. 488 selected roots still await the generator's fixed-object check.

## 1. Access and the live host

- **Source:** `user1@10.4.25.63`, hostname `server`, reached by key-based SSH from the user's laptop.
  - `nav@gvlab` has no key or known host for it ("Host key verification failed"). That explains the v1 audit's "inaccessible" finding: v1 only tried from gvlab.
  - No password or key was requested, printed or stored.
- **Read-only commands only:** `ps`, `stat`, `cat`, `tar` to stdout, `rsync` from the host, `sha256sum` under `nice -n 19 ionice -c3`, and blob hashing through the host's own Python. Nothing was written on the host and no process there was signalled.
- **Live state when observed (10:55 IST):**
  - `tmux` session `yadu3d` is running `shelf66_v3/run_shelf66_full.py --out …/dataset_2500_raw_shelf66v3 --name srv3 --workers 18` (wave 112, 18 running, 15.3 h elapsed). The host had a load of about 18 on 24 CPUs.
  - `K4_MAX_RUNNING=2` and no `QUOTA.json`, so the generator uses its built-in quota. Other researchers' tmux sessions were left alone.
- **Code identity:** every generator file was compared by git blob SHA-1 against `8032ad0b`, and all are **identical**:
  - `shelf66_v3/{bd_3dt,bd_3dt_plus,bd_shelf,none_audit66_v3,reteach_shelf66_v3,run_shelf66_full}.py`;
  - the scene code `shelf_scene.py`, `rrt_connect.py`, `rrt_star.py`, `xarm7_cabinet_env.py`;
  - `dataset-bd/bd_core.py`, `run_3dt_plus.py`, `merge_ledgers.py`, `append_lines.py`.

## 2. Consistency method

- **Why the cut is consistent:** a generator worker writes its witness NPZs (temp file, then `os.replace`, content-addressed) and its root JSON (temp file, then `os.replace`) **before** the coordinator appends and fsyncs that root's ledger line (`bd_core.Store`, `run_3dt_plus`). So every root listed as accepted in a complete-line ledger prefix is complete on disk.
- **Cut:**
  - One remote invocation read the host clock (`2026-11-07T05:28:58.636Z`), then `ledger.jsonl` (1,539,645 bytes, 2,692 complete lines, no partial tail), `PROGRESS.json` and `EXCLUDED_FIXED_OBJECT.json`.
  - Then `rsync -a --partial` copied `roots/` and `witnesses/` into a laptop staging directory (7 min 16 s).
- **Restriction to the cut:**
  - All 2,072 accepted ledger roots had root files; no root file was outside the prefix.
  - 125 witness files were not referenced by any snapshot root and had modification times after the cut. They were moved to `meta/after_cut/` and are not published.
  - The 3,054 unreferenced witness files from before the cut are kept. The previous snapshot also kept its whole store, which had 2,458 unreferenced witness files.
- **Second pass on the source host:** after staging, `sha256sum` ran on the host over every snapshot file under `roots/` and `witnesses/`, plus `EXCLUDED_FIXED_OBJECT.json`.
  - **All 20,574 files are byte-identical** to the staged copies.
  - The host's live ledger, truncated to the cut length, hashes to `8ec5d3b7…3713`, the same as the staged ledger. The ledger is append-only.

## 3. Audit results

| Check | Result |
|---|---|
| Ledger lines / accepted / rejected in construction | 2,692 / 2,072 / 620 |
| Accepted by observed class | K0 680, K1 452, K2 433, K3 362, NONE 4, NONE_SOLVABLE 40, NONE_UNAUDITED 1, DEPENDENT 75, INCONSISTENT 25 |
| Root files: present / missing / unparseable | 2,072 / 0 / 0 |
| Witnesses referenced by snapshot roots: present / missing / corrupt (re-hash ≠ id) | 15,447 / 0 / 0 |
| Witness files published | 18,501 (15,447 referenced, 3,054 unreferenced pre-cut) |
| Protocol of every root | `BD_GEN_V1+ROUTE_V15+SHELF66V3`, `XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022`, config `8d521b1f6d23db23`, `BD_BINARY_BUDGET_V1`, teacher `BD3DS66_TEACHER_V2` (2,072 / 2,072) |
| Selection rule reproduces the previous `selected.json` exactly | yes (same counts, same ordered id lists) |
| Selected now | NONE 3, K0 511, K1 357, K2 357, K3 357, K4 0 (1,585) |
| Previously selected roots still selected / dropped / newly selected | 1,492 / 0 / 93 (all K3) |
| Files vs `8032ad0b`: identical / changed / removed / new | 16,770 / 3 (`raw/ledger.jsonl`, `raw/PROGRESS.json`, `selected.json`) / 0 / 3,804 (362 roots + 3,442 witnesses) |
| Old root or witness bytes changed | **0** |
| Fixed-object check pending (selected roots beyond the first 1,147 accepted) | 488 |

**Ledger order.** The previous snapshot's 2,125 ledger lines are all present unchanged in the live ledger, with the same multiset and no duplicates. The 567 new lines are all appended after them. The first 2,125 lines are in a different order, though: the previous snapshot came from the other side of the laptop/server merge (`merge_with_server_3ds66_v3.sh`). Selection is in ledger order, but both orders give the same K0–K2 selections, and all 1,492 old selections persist.

**Mixed protocols.** None. Every root is SHELF66V3 / V3 teacher.

**Abnormal classes.** NONE stays tiny (3 counted, 40 NONE_SOLVABLE). K4 has 0 accepted roots.

## 4. Lineage splits extended without leakage

`plan3d.py extend` takes the frozen v1 split (`768437e0…`, 1,492 roots) as its parent:
- it inherits every old assignment;
- it closes the old groups;
- it links new roots by `parent_family_id` and identical layout;
- it fails with `SPLIT_CONFLICT` if any group bridges two old splits.

| | |
|---|---|
| Old assignments changed | **0** |
| `SPLIT_CONFLICT`s | **0** |
| New roots / inheriting an existing group's split / in new groups (v1 hash rule) | 93 / 68 / 25 |
| Extended split (1,585 roots) | train 1,121, val 246, test 218; sha `e9c389c4…` |
| Derived-scene layouts (root + 1-edit + 2-edit) shared across splits in the live production roster | **0** of 4,243 |

Files: `../production_v2/split_manifest_live.json`, `../production_v2/production_roster_live.jsonl.gz` (13,681 physical queries plus 2,774 static contrasts over 1,585 roots), and `../production_v2/source_pin_live.json`.

## 5. What was published

These go in `gh-yadu-iiith/shelf_6x6_open_sides/dataset_bd_classquota_without_home_position_no_overpass/`, the native path, so `code/view_scene66.py` and the placement sidecars read it unchanged:
- `raw/{ledger.jsonl, PROGRESS.json, EXCLUDED_FIXED_OBJECT.json, roots/, witnesses/}` and `selected.json`;
- `SNAPSHOT_STATUS.md`, the in-progress marker;
- a short banner at the top of `README.md` pointing to it.

The README body, written by the generator's author, is unchanged and still describes the 6 Nov snapshot. The previous snapshot stays reproducible as commit `6f042be1`, a byte-identical cherry-pick of `8032ad0b`.

These were not published:
- the live `viz/` directory (not in the previous snapshot either);
- `K4_MAX_RUNNING` and `PROGRESS_reteach_srv.json` (live control and status files; kept in the staging `meta/`);
- the 125 witness files written after the cut.

No file is over 3.1 MB, the published `raw/` holds 20,577 files totalling 216 MB, and no credentials are included.

The machine-readable record is `SOURCE_SNAPSHOT_MANIFEST.json.gz`. It holds the host, path, cut time, method, the SHA-256 and size of every published file, every root's witness list, and the summary above. Alongside it are `source_snapshot_summary.json` and the second-pass evidence (`second_pass_server_sha256.txt`, `server_ledger_prefix_sha.txt`).

## 6. Taking the next snapshot (exact, read-only)

```bash
# on the laptop (key access); never on the active store itself
R=yadu_bd/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3
TS=$(date -u +%Y%m%dT%H%M%SZ); ST=~/Desktop/shelf66v3_live_snapshots/$TS; mkdir -p $ST/raw $ST/meta
ssh user1@10.4.25.63 "cd $R && date -u +%FT%T.%NZ && tar cf - ledger.jsonl PROGRESS.json EXCLUDED_FIXED_OBJECT.json" > $ST/meta/cut.tar.withhdr
#   (first line = the cut time; keep only complete ledger lines, as source_sync3d.py expects)
rsync -a --partial user1@10.4.25.63:$R/roots user1@10.4.25.63:$R/witnesses $ST/raw/
python code/source_sync3d.py audit $ST/raw <previous snapshot dataset dir> <cut ISO> $ST/audit   # restricts to the cut, re-selects, manifests
#   second pass: sha256sum on the host over meta/second_pass_paths.txt; compare with the manifest (must be identical)
python code/plan3d.py extend $ST <previous split> split_manifest_live.json                        # SPLIT_CONFLICT -> stop
```

Refresh after the generator stops (K4 / NONE quotas met or abandoned) and after its author's fixed-object check. A mid-run copy is only ever a labelled `IN PROGRESS` snapshot.
