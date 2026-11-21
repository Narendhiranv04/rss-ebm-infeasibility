# Fresh K1-K3 shelf roots on the home server (name `hx1`)

**What this is:** a guide to make new 6 x 6 shelf scenes ("roots") for the confirmatory test. It runs the team's own generator, unchanged:
`dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_v3/run_shelf66_full.py` (SHELF66V3).
**What you'll do:** copy the code to the server, check it, start the run, watch it, then copy the results back.

Terms (each explained once):

| Term | Plain meaning |
|---|---|
| Root | One starting shelf scene. |
| K1 / K2 / K3 | How many objects must be moved (class). K0 and K4 are not used. |
| Quota | How many roots of each class we want. |
| Ledger | The run's append-only list of every attempt. |
| Post-filter (QC) | A check run after generation that drops bad roots. |
| Worker | One generator process. |

- **Host:** `<home-server>` (user@host, reached over SSH). An i9-11900K (8 cores / 16 threads), 32 GB RAM, Linux. Its RTX 2060 is not used.
- **Server folder:** `G=~/Documents/RRC/rss_gen`
  - `$G/code/`: the generator. A copy of the dataset's `code/`, made with `rsync -aL`, so the scene symlink becomes a real folder.
  - `$G/qc/`: the post-filter.
  - `$G/.venv`: Python 3.12, mujoco 3.16.0, numpy 2.2.6.
  - `$G/launch_hx1.sh`: starts or resumes the run.
  - `$G/runs/hx1/`: the output.

## 0. Why `hx1` cannot repeat an existing scene

Every root is built from `construction_seed = int(sha256(["BD_GEN_V1", "3ds66", name, wave, slot])[:8], 16)` (`bd_core.ids`).
The root id is `3ds66-<name>-w<wave>-s<slot>`. The worker index plays no part.
The request plan is seeded the same way, with `sha([name, wave])`.

The existing names are `lap`, `lap3`, `srv` and `srv3`. A new name changes the hash input, so it gives new seeds. This was checked:
- the formula reproduces the stored seeds;
- across the 2,692 existing seeds, `hx1` and `lapx` (waves 0-999, slots 0-23) share no seed with them or with each other.

A repeat would need a 32-bit hash collision. As an extra guard, `qc_new_roots.py` also rejects exact layout duplicates of the existing raw roots.

**Warning: Never reuse `lap*` or `srv*`, and never run two `hx1` processes on one `$OUT`.** Two processes would plan the same wave from one ledger
and produce the same ids. `launch_hx1.sh` refuses to start while one is running.

## 1. Copy the code over (from the laptop)

Do this once. Run all of it on the laptop.

```bash
cd /Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides
C=dataset_bd_classquota_without_home_position_no_overpass/code
ssh <home-server> 'mkdir -p ~/Documents/RRC/rss_gen/code ~/Documents/RRC/rss_gen/qc/selftest_roots'
# generator (-L: shelf66_v3/shelf_scene_6x6 is a symlink to ../../../scene; copy it as a real folder)
rsync -aL --exclude='__pycache__' --exclude='.claude' $C/run_3dt_plus.py $C/dataset-bd $C/shelf66_v3 <home-server>:Documents/RRC/rss_gen/code/
# sha256 manifest of exactly that code, checked on the server below
(cd $C && find -L run_3dt_plus.py dataset-bd shelf66_v3 -type f -not -path '*__pycache__*' -not -path '*/.claude/*' | sort | xargs shasum -a 256) > /tmp/code_sha256.txt
rsync -a /tmp/code_sha256.txt <home-server>:Documents/RRC/rss_gen/code/
# launcher, post-filter, the team's pinned pick-check tool, 2 roots for the QC selftest
rsync -a hybrid/gen/launch_hx1.sh <home-server>:Documents/RRC/rss_gen/
rsync -a hybrid/gen/qc_new_roots.py hybrid/gen/README_homeserver.md \
  dataset_placement_extension_v2/production_v3/qc/author_evidence/obj_pickable3.py <home-server>:Documents/RRC/rss_gen/qc/
rsync -a dataset_bd_classquota_without_home_position_no_overpass/raw/roots/3ds66-lap-w0000-s004.json \
  dataset_bd_classquota_without_home_position_no_overpass/raw/roots/3ds66-srv-w0004-s021.json <home-server>:Documents/RRC/rss_gen/qc/selftest_roots/
```

Use rsync, not macOS `tar`. macOS `tar` adds `._*` AppleDouble files that are not JSON.
The code is about 7 MB, including the xArm7 model. Do not edit anything in it.

## 2. Check the server (once)

Run these on the server. They confirm the code and Python setup match the laptop.

```bash
ssh <home-server>
G=~/Documents/RRC/rss_gen
cd $G/code && sha256sum -c --quiet code_sha256.txt && echo CODE_OK          # same generator, teacher, scene, planners and robot model as the laptop
# venv: reuse $G/.venv if present. Otherwise, without sudo (keep uv's files inside $G, not in your own ~/.local/share/uv):
#   export UV_PYTHON_INSTALL_DIR=$G/.uv/python UV_CACHE_DIR=$G/.uv/cache
#   uv venv -p 3.12 $G/.venv && uv pip install -p $G/.venv/bin/python mujoco==3.16.0 numpy==2.2.6
$G/.venv/bin/python -c "import mujoco, numpy; assert (mujoco.__version__, numpy.__version__) == ('3.16.0', '2.2.6'); print('VENV_OK')"
export QC_SHELF66=$G/code/shelf66_v3 QC_TOOL=$G/qc/obj_pickable3.py QC_EXISTING=$G/qc/selftest_roots
$G/.venv/bin/python $G/qc/qc_new_roots.py --selftest     # ~40 s; expect: selftest OK: lap-w0000-s004 QC_PASS, srv-w0004-s021 QC_FAIL_FIXED_OBJECT [o00]
```

You should see `CODE_OK`, `VENV_OK`, then the selftest line.

## 3. Quota (K1-K3 only)

`$OUT/QUOTA.json` is re-read on every plan, so you can edit it live without a restart.
- NONE, K0 and K4 are 0. The runner never counts them, and it stops by itself when K1, K2 and K3 are full.
- `K4_MAX_RUNNING = 0` re-plans every K4 draw into another K. K4 constructions never finish on this shelf.
- A few cheap K0 requests remain: about 7 % at first, about 1 min each, and about 9 % of them turn out K1.
- NONE audits triggered by K1-K3 requests cost about 0.3 % of the time in the team's ledger.

`launch_hx1.sh` writes `{"NONE": 0, "0": 0, "1": 115, "2": 115, "3": 115, "4": 0}`, but only if no `QUOTA.json` exists.
- A smaller margin is enough: in the team's data the fixed-object QC excluded 3 of 1,147 roots. So, before the first start, use:
  `mkdir -p $G/runs/hx1 && echo '{"NONE": 0, "0": 0, "1": 105, "2": 105, "3": 105, "4": 0}' > $G/runs/hx1/QUOTA.json`
- For a VAL-like mix (53/67/40, scaled to 300, +5 %), use `{"NONE": 0, "0": 0, "1": 104, "2": 132, "3": 79, "4": 0}`.

**Selection rule (fix it before the run):** the test set is the first 100 usable roots of each class, in ledger order.

## 4. Run

> **WARNING: set the CPU power cap first.** The server runs on a UPS that crashed twice under full load.
> 1. Run: `echo 120000000 | sudo tee /sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw`
> 2. A reboot resets the cap. Set it again after every reboot.
> 3. `launch_hx1.sh` refuses to start if the cap is above 130 W.

Then start the run:

```bash
cd ~/Documents/RRC/rss_gen && bash launch_hx1.sh                # WORKERS=14 by default; nohup, survives logout
grep -h '"config_hash"' runs/hx1/roots/*.json | sort | uniq -c  # after the first root: must show only 8d521b1f6d23db23
```

The second command checks that every root has the same config hash. It must show only `8d521b1f6d23db23`.

The launcher does the following:
- checks the venv versions and that no `hx1` process is running;
- creates `QUOTA.json` and `K4_MAX_RUNNING=0` if they are absent;
- runs `nice -n 10 .venv/bin/python shelf66_v3/run_shelf66_full.py --out runs/hx1 --name hx1 --workers 14` from `$G/code`,
  with single-threaded BLAS and `MUJOCO_GL=disable`.

About the 14 workers:
- They leave 2 of the 16 hardware threads free for the OS and SSH. RAM is no limit (about 0.3-0.5 GB per worker).
- 14 workers on 8 physical cores share hyper-threads, so each runs at about 0.6-0.7 of a full core.
- The K3 construction has a 45 min **wall-clock** cap (a real-time limit, not CPU time), so slower workers hit it more often.
- If `ROUTE_TIME_CAP` lines clearly outnumber accepted K3 roots after the first 3-4 h, run `echo 8 > runs/hx1/MAX_RUNNING`.
  That gives one root per physical core, live. In the team's ledger the ratio was 0.42 capped to 0.33 accepted per K3 request.

## 5. Monitor, stop, resume

```bash
G=~/Documents/RRC/rss_gen; OUT=$G/runs/hx1
cat $OUT/PROGRESS.json                     # "selected" vs "quota", attempts, elapsed_h (updated after each finished root)
grep '^root ' $OUT/run.log | tail          # requested K -> verified class or reject reason, seconds
grep -c ROUTE_TIME_CAP $OUT/run.log; grep -c -- '-> 3 (' $OUT/run.log
echo 8 > $OUT/MAX_RUNNING                  # live: fewer roots at once; rm the file to go back to 14
```

- **Clean stop:** `touch $OUT/STOP`. No new roots start, and running roots finish.
  - That can take over an hour: the worker alarm is 70 min, and the construction cap overshoots 45 min under load (up to 59 min seen).
    A root in a NONE audit takes up to 1 h more.
  - The process exits after writing `selected.json`.
- **Hard stop:** `pkill -f 'run_shelf66_full.py --out'`. Only the roots that are running are lost.
- **Resume:** `FORCE=1 bash launch_hx1.sh`. It removes STOP and refuses while the old process is still draining (`pgrep`).
  The ledger is append-only, and the waves continue from it.

## 6. Post-filter (on the server, after the run has exited)

```bash
G=~/Documents/RRC/rss_gen
export QC_SHELF66=$G/code/shelf66_v3 QC_TOOL=$G/qc/obj_pickable3.py QC_EXISTING=$G/qc/selftest_roots
$G/.venv/bin/python $G/qc/qc_new_roots.py $G/runs/hx1 --workers 14     # -> runs/hx1/QC_NEW_ROOTS.json ("usable"); resumable, ~15 min
```

A root is **usable** when all of these hold:
- its ledger class is 1, 2 or 3;
- the root file loads and has every field of the existing raw roots (`labels.B/D/original` with a successful minimal sequence,
  `verified_K`, `goal`, `initial_cells`, `objects`, `target_src`, ...);
- it has the same provenance: `config_hash` `8d521b1f6d23db23`, teacher `BD3DS66_TEACHER_V2`, generator `...SHELF66V3`, label protocol;
- it has at most 7 non-target objects (MAXOBJ = 8 including the target);
- it passes the fixed-object QC;
- it is not an exact duplicate layout.

The fixed-object QC is the rule of Approach-B's `QC_ELIGIBLE_ROOTS.json`. It runs the team's pinned, unmodified `obj_pickable3.py`
(sha256 checked) on every non-target object: the object alone on the shelf, under the scene's arm ceiling.
In `qc3d.py` terms, a pass is the `SIDECAR_REPRO_QC_PASS` tier (the tool re-run by us), not `AUTHOR_QC_PASS`.

"Usable" means the roots are generated and QC-checked. Any extra labels the models need (for example Q2-parity labels) are not produced here.

## 7. Copy the results back (to the laptop)

```bash
# can be re-run, and is safe mid-run: it copies only what changed
rsync -a <home-server>:Documents/RRC/rss_gen/runs/hx1 /Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/
cd /Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/gen
/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python qc_new_roots.py ../../approachb_3d_learning/runs/hx1 --workers 2
```

- The laptop pass reuses every pick check already in `hx1/qc/part*.jsonl`. Only missing objects are checked.
- It adds the duplicate check against all 2,072 existing raw roots.
- `approachb_3d_learning/runs/` is gitignored.

## 8. Expected wall time for ~300 usable K1-K3 roots

The basis is the team's SHELF66V3 ledger (`raw/ledger.jsonl`: 1,604 new attempts, 734 worker-hours, mostly the lab server's 18 workers).
The laptop pilot is too small to give a rate (see the note below).

Cost per request, from the ledger (mean wall time of an attempt, including rejected ones, and what it yields):

| Requested | Mean time | Yields K1 / K2 / K3 |
|---|---|---|
| K1 | 6.6 min | 0.83 / 0 / 0 |
| K2 | 16.0 min | 0.11 / 0.72 / 0 |
| K3 | 37.6 min | 0.04 / 0.11 / 0.33 |

55 % of the K3 hours are lost to the 45 min construction cap.

Filling the quota in the order K3, then K2, then K1 (counting the by-products) costs:

| Quota | Worker-hours |
|---|---|
| 100 / 100 / 100 | 228 |
| 105 / 105 / 105 | **239** |
| 115 / 115 / 115 (launcher default) | 263 |
| VAL-like 104 / 132 / 79 | about 202 |

- The planner fills K1/K2 partly before K3's by-products arrive, so add about 6 %.
- K3 is about 85 % of the cost.

Home server (i9-11900K, 8 cores / 16 threads, 14 workers):
- **Assumption:** one 11900K core equals one ledger worker. This is not measured: the 11900K is fast single-threaded, but the server
  workers may also have shared hyper-threads.
- With hyper-threading, 8 cores give about 10 core-equivalents. That gives 239 x 1.06 / 10 ≈ **25 h** for 105/105/105.
- **Main risk:** each of the 14 workers runs at about 0.65 of a core, so the 45 min wall cap is effectively about 29 min of ledger work.
  Half of the ledger's accepted K3 roots took longer than 31 min, so K3's yield could drop well below 0.33.
  - The 8-worker setting (`MAX_RUNNING=8`) avoids that, at about 239 x 1.06 / 8 ≈ 32 h.
- **Plan for 25-45 h (1-2 days).** Decide 14 vs 8 workers from the capped/accepted K3 ratio after 3-4 h (section 4).

The post-filter adds about 1,100 pick checks (about 3.5 objects per root). At about 10 s each on 14 workers, that is about 15 min.

**Laptop pilot (`lapx`, 2 workers, 2026-11-21).** STOP was created after 33 min, and the runner drained by 56 min.
- Both requests were K3. One was accepted as K3 (34.5 min, 3 movable objects, QC pass, provenance identical to the team's data).
  The other hit the construction cap (`ROUTE_TIME_CAP`) after 55.6 min, not 45, because the cap is checked between steps.
- The laptop's load average was 80-105 from other MuJoCo jobs, and n = 2.
- The pilot confirms the pipeline end to end; it does not measure a rate.
