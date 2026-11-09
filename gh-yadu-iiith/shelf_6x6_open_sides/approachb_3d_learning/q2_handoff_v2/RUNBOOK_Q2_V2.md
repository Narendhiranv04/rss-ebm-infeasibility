# Q2 2D runbook: V2/V3 three-host cut and final merge (Nov 9, 2026)

`spill2d_v2.py` is a versioned layer on top of the unchanged `spill2d.py`: it imports `spill2d` and never edits it. The 2D teacher, plan, policy (ADAPTIVE_FULL 8→32), seeds and audit are imported unchanged. CPU and sensor profiles change execution only, never labels.

## Owner maps (never overwritten)

| Map | File | Cut |
|---|---|---|
| V1 | `q2_spillover/handoff/HANDOFF_ROOT_OWNER_MANIFEST.json` | ORIGINAL / LAPTOP remainder / GVLAB spillover |
| V2 | `q2_spillover/handoff_v2/HANDOFF_ROOT_OWNER_MANIFEST_V2.json` | 10:35 IST drain of the laptop remainder and gvlab V1. Packed roots stay with LAPTOP / GVLAB; partial and untouched roots split into GVLAB2 / LAPTOP2 / BLACKWELL |
| V3 | `q2_spillover/handoff_v3/HANDOFF_ROOT_OWNER_MANIFEST_V3.json` | 13:39 IST re-cut of the BLACKWELL share (2 partial + 776 untouched, nothing packed) into LAPTOP3 / GVLAB3 / BLACKWELL3 |

Frozen copies (gzipped, with sha sums and classification files) are in `handoff_v2_frozen/` and `handoff_v3_frozen/`.

**Union invariant, checked at every cut:** 4,620 roots / 249,965 tasks / 238,160 teacher queries, 0 duplicates.

**Final (V3) owners and run directories:**

| Owner | Roots | Host | Run directory | Status at freeze |
|---|---:|---|---|---|
| ORIGINAL | 201 | laptop | `dataset_relocations_v1/full_2d_v2` | frozen |
| LAPTOP | 1,585 | laptop | `q2_spillover/runs/remainder_laptop` | frozen (drained 10:35) |
| GVLAB | 1,520 | gvlab | `launch_ready_2d_v2_gvlab/runs/spillover_gvlab` | frozen (drained 10:35) |
| LAPTOP2 | 255 | laptop | `q2_spillover/runs/remainder2_laptop` | run to completion |
| GVLAB2 | 281 | gvlab | `launch_ready_2d_v2_gvlab/runs/spillover2_gvlab` | run to completion |
| LAPTOP3 | 85 | laptop | `q2_spillover/runs/remainder3_laptop` | run to completion |
| GVLAB3 | 520 | gvlab | `launch_ready_2d_v2_gvlab/runs/spillover3_gvlab` | chained after GVLAB2 (tmux `q2_spillover3`) |
| BLACKWELL3 | 173 | Blackwell | `~/approachb_3d_learning/q2bw/src/…/launch_ready_2d_v2_gvlab/runs/spillover3_blackwell` | autolauncher, E-cores 8–23 |

**Blackwell standard:** `LABEL_EQUIVALENT_NOT_BITWISE` (user-approved). The portability canary gave labels 848/848 identical and fingerprints 206/848, because there is no AVX-512. Every merged root records its provenance standard: `LABEL_EQUIVALENT_BLACKWELL` or `BITWISE_PORTABLE`.

## Commands

The paths assume `spill2d_v2.py` is in each host's deploy dir:
- gvlab: `~/approachb3d_v3/2d_q2_handoff_v2/`;
- Blackwell: `~/approachb_3d_learning/q2bw/`;
- laptop: this repo, run through `flatpak-spawn --host /home/naren/Desktop/rss3d_env/venv/bin/python`.

```
classify-export RUN_DIR OWNER OUT.json          # per host, after a clean DRAIN (final: true, stop: null)
partition-v2 / partition-v3 …                   # laptop; refuses to overwrite an existing map
copy-partials SHARE                             # hash-verified partial parts into the new run dir
run SHARE [--dry-run] [--coexist] [--cpus a-b]  # host gate, identity gate, then the unchanged 2D runner
canary / canary-check                           # Blackwell portability canary (separate audit process)
merge FINAL_OWNER_MAP OWNER=DIR ...             # NEW merged/q2_canonical_v3 + full launch_check audit
```

- **Resume** of any share is the same `run SHARE` command; packed roots are kept.
- **Blackwell:** `q2bw_autolaunch.sh BLACKWELL3 6` starts the run only when all of these hold:
  - sajid's CPU queue has nothing pending;
  - his jobs use < 1.5 cores;
  - yadu's generator is absent;
  - the package is < 65 °C.

  `q2bw_coexist.py` then PAUSEs the run whenever fewer than 4 of CPUs 0–9 are idle. Stop the launcher with `touch STOP_AUTOLAUNCH` in the q2bw dir.

## Final merge (on gvlab, once GVLAB3 and BLACKWELL3 are both `final: true, stop: null`)

1. **Laptop sources:** ORIGINAL, LAPTOP, LAPTOP2 and LAPTOP3 are copied to gvlab `~/approachb3d_v3/2d_q2_merge_src/<OWNER>/`, sha256-verified file by file. The merge re-checks every frozen root against its snapshot hash.
2. **Blackwell source:** relay it through the laptop, because the servers cannot ssh to each other:
   ```
   rsync -a user1@10.4.25.63:approachb_3d_learning/q2bw/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2_gvlab/runs/spillover3_blackwell/ /tmp/bw3/
   rsync -a /tmp/bw3/ gvlab:approachb3d_v3/2d_q2_merge_src/BLACKWELL3/
   ```
   Verify with sha256 sums on both ends.
3. **Merge on gvlab:**
   ```
   cd ~/approachb3d_v3/2d_q2_handoff_v2
   G=~/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1
   M=~/approachb3d_v3/2d_q2_merge_src
   OMP_NUM_THREADS=1 ~/approachb3d_v3/2d_q2_spillover/venv313/bin/python spill2d_v2.py merge $G/q2_spillover/handoff_v3/HANDOFF_ROOT_OWNER_MANIFEST_V3.json \
     ORIGINAL=$M/ORIGINAL LAPTOP=$M/LAPTOP LAPTOP2=$M/LAPTOP2 LAPTOP3=$M/LAPTOP3 \
     GVLAB=$G/launch_ready_2d_v2_gvlab/runs/spillover_gvlab GVLAB2=$G/launch_ready_2d_v2_gvlab/runs/spillover2_gvlab \
     GVLAB3=$G/launch_ready_2d_v2_gvlab/runs/spillover3_gvlab BLACKWELL3=$M/BLACKWELL3
   ```
   The merge refuses on any of:
   - a missing owner;
   - an unpacked root;
   - a stray root;
   - a snapshot-hash change;
   - a trial-id mismatch or collision;
   - an existing output directory.

   It writes `q2_spillover/merged/q2_canonical_v3` and `q2_spillover/MERGED_Q2_V3_AUDIT.json`. `pass` requires:
   - 0 audit failures;
   - 249,965 tasks;
   - 238,160 teacher queries.

   The audit uses 16 processes, so run it while GVLAB3 is not running (gvlab ceiling: 18 logical CPUs).
4. **If BLACKWELL3 cannot run** (collaborator load persists): drain it, `classify-export` it on Blackwell, and re-cut its untouched roots to gvlab as a **new** V4 map. Never edit V3. This needs a `partition-v4` step that does not exist yet; ask before adding it.

**2D Q2 intake into training happens only after `MERGED_Q2_V3_AUDIT.json` reports `pass: true`.**
