# SHELF66V3 placement extension v2: runbook (for a future, separately authorized full collection)

**Status:** `FULL_COLLECTION_NOT_STARTED`. Nothing here was run except the refusal tests and the metadata-only dry runs in §3.

Paths are relative to `gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/`. On gvlab that directory is `~/approachb3d_mock/src/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2`, the Python environment is `~/approachb3d_mock/env` (MuJoCo 3.16.0, NumPy 2.2.6), and the tmux session is `approachb3d`.

## 0. Before anything: re-validate on the current host

```bash
cd dataset_placement_extension_v2; PY=~/approachb3d_mock/env/bin/python; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
ps -eo user,pid,pcpu,args --sort=-pcpu | head            # no other heavy job; NEVER kill other users' processes
$PY -m pytest -q -p no:cacheprovider code/test_mock3d_v2.py        # must pass (runs on CPU 0 + a 1-worker tiny run)
(cd ../dataset_placement_extension_v1 && $PY code/audit3d.py integrity audit/source_manifest.json.gz)   # pinned source unchanged
```

## 1. Source to collect on

- Do not collect on the in-progress snapshot by default. Take a fresh read-only snapshot after the generator finishes, following `source_sync/LIVE_3D_SOURCE_SYNC_REPORT.md` §6, and after its author has run the fixed-object check on the 488 pending roots.
- Then rebuild the split, roster and pin:

```bash
$PY code/plan3d.py extend <snapshot dir> production_v2/split_manifest_live.json production_v2/split_manifest_next.json   # SPLIT_CONFLICT -> STOP
$PY code/plan3d.py roster <snapshot dir> production_v2/split_manifest_next.json production_v2/production_roster_next.jsonl.gz --production
$PY code/plan3d.py leak production_v2/production_roster_next.jsonl.gz production_v2/split_manifest_next.json            # must print 0
$PY code/source_sync3d.py pin-live <snapshot dir> production_v2/source_pin_next.json
```

- `SHELF_SRC` must be a dataset directory with `code/` (SHELF66V3, byte-identical to `8032ad0b`), plus `raw/` and `selected.json` from the pinned snapshot. Either of these works:
  - a checkout of `naren/3d-dataset-relocations`, using its native dataset path;
  - the gvlab symlink layout `~/approachb3d_mock/production_src_live` (`code` → the pinned code, `raw` and `selected.json` → the snapshot).
- The collector refuses any root whose bytes differ from the pin.
- K4 and NONE have no measured placement cost (K4 has 0 roots; NONE has 3, never sampled), so their part of any forecast is a back-off estimate.

## 2. Freeze the production policy (user action)

1. Choose the policy recorded in `3D_MOCK_V2_DECISION.json`, field `policy.recommended`. The policy file must name it explicitly.
2. Regenerate the template from the current files:

   ```bash
   $PY code/prod_template.py <policy> production_v2/production_policy.json
   ```

   Edit `roster` / `source_pin` / `split_manifest` if you rebuilt them (§1), then regenerate so `frozen` matches.
3. **The user** fills `authorization.authorized_by`, `authorization.authorized_utc` and `authorization.statement` (for example "I authorize the full SHELF66V3 placement collection with policy X on roster sha Y").
4. Compute the confirmation token for the CURRENT hashes:

   ```bash
   $PY code/guard3d.py token production_v2/production_policy.json
   ```

   Any later change to the roster, source pin, split, teacher or collector code, policy, workers or CPU mask changes the token, and the guard then refuses.

## 3. Pre-launch checks (all must pass; these are the only ones exercised so far)

```bash
$PY code/guard3d.py check production_v2/production_policy.json <TOKEN> 12 0-11       # exit 0 only if every condition holds
SHELF_SRC=<dataset dir> $PY code/run3d.py production_v2/run --roster production_v2/production_roster_live.jsonl.gz --policy <policy> \
    --workers 12 --cpus 0-11 --source-pin production_v2/source_pin_live.json --production --dry-run      # planner_calls must be 0
```

Exercised during the mock:
- **Refusals:**
  - no arguments: `REFUSED`, exit 2;
  - `--execute-full` with a wrong token and the unauthorized template: `REFUSED`, exit 2;
  - `run3d.py --production` without an authorization file: `REFUSED (production)`;
  - `guard3d.check` on the template, which lists the missing authorization, the token mismatch and any running collector.
- **All-splits dry run:** on the live snapshot, 1,469 units, 13,681 queries, `planner_calls 0`, nothing written (`production_v2/dry_run_production_live.json`).

## 4. Launch (ONLY after the user's explicit, recorded authorization)

```bash
tmux new-window -t approachb3d -n full3d "SHELF_SRC=<dataset dir> bash mock_v2/launch_full_3d.sh --execute-full --confirm <TOKEN> --policy-file production_v2/production_policy.json"
```

`launch_full_3d.sh` re-runs `guard3d.check`, then the metadata dry run, then the full test suite. Only if all three pass does it `exec` `run3d.py --production` with 12 pinned workers on CPUs 0–11, at `nice` 10, with one thread each. The guard itself refuses any mask of more than 18 logical CPUs.

## 5. While running

- **Progress:** `production_v2/run/ledger.jsonl` (one `unit_done` line per root lineage, holding the JSON and NPZ SHA-256) and `monitor.jsonl` (every 2 s: own-tree CPU %, temperature and sensor identity, RSS, memory, swap, disk, affinity union, unexpected descendants).
- **Automatic stops (fail closed, own process group only):**
  - temperature sensor unreadable, even while paused;
  - package temperature ≥ 95 °C (pause at 90 °C, resume at ≤ 82 °C);
  - own-tree CPU above 100 × |mask| + 50 %;
  - affinity outside the mask or a mask of more than 18 CPUs;
  - an unexpected descendant process;
  - available memory under 4 GB, swap growth over 1 GB, free disk under 5 GB, a worker RSS over 2 GB;
  - a worker death, source drift, or a unit failing validation.
- **Stopping by hand:** `kill -KILL -- -<coordinator pid>` (the coordinator is its own process group leader). Never use `pkill -f`. Interrupted queries are unlabeled, never 0.
- **Resume:** rerun the identical command. A unit counts as done only if its ledger line, JSON SHA-256, NPZ SHA-256 and every witness re-hash agree. Anything else is moved to `production_v2/run/quarantine/<time>/` and recomputed. A changed policy, roster, code, protocol, worker policy or source pin is refused.

## 6. After the run

```bash
$PY code/replay3d.py ours production_v2/run production_v2/production_roster_live.jsonl.gz production_v2/replay.json --workers 12 --cpus 0-11   # 100% must verify
```

Then re-run the integrity check against the pinned snapshot manifest and report label counts by K, depth, band and split. The val/test splits are evaluation-only: never tune on them.
