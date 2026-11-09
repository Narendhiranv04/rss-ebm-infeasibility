# Q2 2D runbook: V2/V3 three-host cut and final merge (Nov 9, 2026)

`spill2d_v2.py` is a versioned layer on top of the unchanged `spill2d.py`: it imports `spill2d` and never edits it. The 2D teacher, plan, policy (ADAPTIVE_FULL 8→32), seeds and audit are imported unchanged. CPU and sensor profiles change execution only, never labels.

## Owner maps (never overwritten)

| Map | File | Cut |
|---|---|---|
| V1 | `q2_spillover/handoff/HANDOFF_ROOT_OWNER_MANIFEST.json` | ORIGINAL / LAPTOP remainder / GVLAB spillover |
| V2 | `q2_spillover/handoff_v2/HANDOFF_ROOT_OWNER_MANIFEST_V2.json` | 10:35 IST drain of the laptop remainder and gvlab V1. Packed roots stay with LAPTOP / GVLAB; partial and untouched roots split into GVLAB2 / LAPTOP2 / BLACKWELL |
| V3 | `q2_spillover/handoff_v3/HANDOFF_ROOT_OWNER_MANIFEST_V3.json` | 13:39 IST re-cut of the BLACKWELL share (2 partial + 776 untouched, nothing packed) into LAPTOP3 / GVLAB3 / BLACKWELL3 |
| **V4 (final)** | `q2_spillover/handoff_v4/HANDOFF_ROOT_OWNER_MANIFEST_V4.json` | 15:50 IST, at the user's request ("queue everything to gvlab, do not depend on Blackwell"): all 173 BLACKWELL3 roots become **GVLAB4**. The Blackwell attestation (`BW3_ATTESTATION.json`) showed BLACKWELL3 never ran: 0 records, no run manifest, autolauncher stopped. Its 2 partial roots (`2d-plus2d-srv-w0218-s008`, `2d-plus2d-w0173-s009`) are **recomputed from scratch** on gvlab; their Blackwell parts are not used |

Frozen copies (gzipped, with sha sums and classification files) are in `handoff_v2_frozen/`, `handoff_v3_frozen/` and `handoff_v4_frozen/`. V4 was frozen on gvlab, because the 2D code is never loaded on the laptop while a laptop run is live; it was copied back sha-verified.

**Union invariant, checked at every cut:** 4,620 roots / 249,965 tasks / 238,160 teacher queries, 0 duplicates.

**Final (V4) owners and run directories:**

| Owner | Roots | Host | Run directory | Status at freeze |
|---|---:|---|---|---|
| ORIGINAL | 201 | laptop | `dataset_relocations_v1/full_2d_v2` | frozen |
| LAPTOP | 1,585 | laptop | `q2_spillover/runs/remainder_laptop` | frozen (drained 10:35) |
| GVLAB | 1,520 | gvlab | `launch_ready_2d_v2_gvlab/runs/spillover_gvlab` | frozen (drained 10:35) |
| LAPTOP2 | 255 | laptop | `q2_spillover/runs/remainder2_laptop` | run to completion |
| GVLAB2 | 281 | gvlab | `launch_ready_2d_v2_gvlab/runs/spillover2_gvlab` | run to completion |
| LAPTOP3 | 85 | laptop | `q2_spillover/runs/remainder3_laptop` | run to completion |
| GVLAB3 | 520 | gvlab | `launch_ready_2d_v2_gvlab/runs/spillover3_gvlab` | chained after GVLAB2 (tmux `q2_spillover3`) |
| GVLAB4 | 173 | gvlab | `launch_ready_2d_v2_gvlab/runs/spillover4_gvlab` | chained after GVLAB3 (tmux `q2_spillover4`, `chain_gvlab4.sh`) |

**No root is owned by Blackwell in V4**, so every merged root is `BITWISE_PORTABLE`. The Blackwell canary (labels 848/848 identical, fingerprints 206/848) is kept for the record only.

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
- **Blackwell:** no longer used. The autolauncher was stopped with `STOP_AUTOLAUNCH` at 15:48 IST, and no Q2 process of ours runs there.

## Final merge (on gvlab only, once GVLAB3 and GVLAB4 are both `final: true, stop: null`)

1. **Laptop sources:** ORIGINAL, LAPTOP, LAPTOP2 and LAPTOP3 are copied to gvlab `~/approachb3d_v3/2d_q2_merge_src/<OWNER>/`, sha256-verified file by file. The merge re-checks every frozen root against its snapshot hash. **No relay is needed:** after this, the merge uses nothing from the laptop or Blackwell.
2. **Merge on gvlab** (`spill2d_v2.py` sha256 `7c569b90…`):
   ```
   cd ~/approachb3d_v3/2d_q2_handoff_v2
   G=~/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1
   M=~/approachb3d_v3/2d_q2_merge_src
   R=$G/launch_ready_2d_v2_gvlab/runs
   OMP_NUM_THREADS=1 ~/approachb3d_v3/2d_q2_spillover/venv313/bin/python spill2d_v2.py merge $G/q2_spillover/handoff_v4/HANDOFF_ROOT_OWNER_MANIFEST_V4.json \
     ORIGINAL=$M/ORIGINAL LAPTOP=$M/LAPTOP LAPTOP2=$M/LAPTOP2 LAPTOP3=$M/LAPTOP3 \
     GVLAB=$R/spillover_gvlab GVLAB2=$R/spillover2_gvlab GVLAB3=$R/spillover3_gvlab GVLAB4=$R/spillover4_gvlab
   ```
   The merge refuses on any of:
   - a missing owner;
   - an unpacked root;
   - a stray root;
   - a snapshot-hash change;
   - a trial-id mismatch or collision;
   - an existing output directory.

   It writes `q2_spillover/merged/q2_canonical_v4` and `q2_spillover/MERGED_Q2_V4_AUDIT.json`. `pass` requires:
   - 0 audit failures;
   - 249,965 tasks;
   - 238,160 teacher queries.

   The audit uses 16 processes, so run it after GVLAB4 has finished (gvlab ceiling: 18 logical CPUs).

**2D Q2 intake into training happens only after `MERGED_Q2_V4_AUDIT.json` reports `pass: true`.**
