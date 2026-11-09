# Q2 2D handoff V2: drain the laptop remainder → gvlab (plan step 5, Nov 9 ~14:00 IST)

`spill2d_v2.py` is a versioned layer on top of the unchanged `spill2d.py`: it imports `spill2d` and never edits it.

**Owner map V2:**
- ORIGINAL and GVLAB are unchanged from V1.
- V1 laptop-owned roots **packed** at the drain become `LAPTOP`.
- V1 laptop-owned roots **not packed** (PARTIAL, with hash-verified parts, or UNTOUCHED) become `GVLAB2`.

**Execution:** GVLAB2 runs in the separate gvlab run directory `runs/spillover2_gvlab`, with 4 workers on CPUs 14–17 and the coordinator on 12–13.
- Q2 uses CPUs 0–13, so the total is 18 logical CPUs, which is the project ceiling. CPUs 18–23 stay free.
- CPUs 14–17 are the SMT siblings of Q2 workers 2–5.

## Tested on 2026-11-09, before the drain

All tests ran on gvlab against a scratch copy:
- 3 packed and 7 partial laptop roots;
- the real V1 handoff;
- the real gvlab staged plan and code.

| Check | Result |
|---|---|
| `server-run-v2 --dry-run`, pre-drain | Every gate passes: frozen identity, staged source identical to the laptop manifest, server profile, 3D gate, only the V1 SPILLOVER run active, CPUs 14–17 idle, 47 °C |
| `partition-v2` | Exact union: 4,620 roots / 249,965 tasks / 238,160 teacher queries, 0 duplicates; V1 ORIGINAL and GVLAB unchanged; re-running is refused (no overwrite) |
| `server-run-v2 --dry-run`, test handoff | 2,228 roots and 7 partial roots verified |
| Tamper test | A one-byte change to a copied part is refused |

## Steps

The paths assume the laptop's 2D repo, with `LPY=/home/naren/Desktop/rss3d_env/venv/bin/python`.

1. **Drain the laptop.** Run `touch …/q2_spillover/runs/remainder_laptop/DRAIN`. In-flight queries finish, packed roots are kept, and the coordinator exits.
   - Wait until `status.json` reports `final: true, stop: null`.
   - The laptop `spill2d tick` cron (MONITOR) then flags `LAPTOP_REMAINDER_STOPPED_INCOMPLETE`. It never restarts anything.
2. **Freeze the V2 map on the laptop.** Run `$LPY spill2d_v2.py partition-v2`. It refuses if the coordinator lock is held or the run did not end in a clean drain.
3. **Stage on gvlab from the laptop.** Run `$LPY spill2d_v2.py stage-v2`. It copies `handoff_v2/` and the partial parts.
4. **Start the gvlab run.** Start a tmux session `q2_spillover2` in `$HOME/<GSTAGE>`, then:
   1. run `OMP_NUM_THREADS=1 $HOME/approachb3d_v3/2d_q2_spillover/venv313/bin/python ~/approachb3d_v3/2d_q2_handoff_v2/spill2d_v2.py server-run-v2 --dry-run`;
   2. if it shows no refusals, run the same command without `--dry-run`.
   - Resume, if ever needed, is the same command; packed roots are kept.
5. **Merge, when the user is back and both gvlab runs are complete.**
   1. Rsync both gvlab run directories to the laptop.
   2. Run `$LPY spill2d_v2.py merge-v2 GVLAB_COPY GVLAB2_COPY`. It writes a NEW `merged/q2_canonical_v2` and runs a full `launch_check` audit into `MERGED_Q2_V2_AUDIT.json`.

**After step 3, nothing of the 2D pipeline runs on the laptop.**
