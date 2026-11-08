# Execution plan, user-approved 2026-11-08 ~16:20 IST (checked every 20 min by the status tick)

**Approved:**
- (a) 3D physics on Blackwell;
- (b) drain the laptop Q2 remainder and move it to gvlab under a new versioned owner map.

**Not approved:** (c) the bulk missing-Y job. It needs an explicit "c".

**Standing rule:** keep **≥ 4 CPU cores unoccupied** on every host I use (gvlab, Blackwell). The laptop is exempt.

**Hard deadline:** the laptop must be free of my jobs and the session wrapped up by **Nov 9 ~17:00 IST**. The Claude session itself runs on the laptop, so nothing is synced after shutdown.

## Order and host assignment

| # | When (IST) | Host | Step | Core rule check |
|---|---|---|---|---|
| 1 | now → Blackwell free | laptop → Blackwell | Copy the pinned 3D teacher tree (37 source-pin files + 2,072 roots + witnesses) and verify every hash. Build the physics venv (mujoco 3.16.0, numpy 2.2.6) locally in the workspace, no global install | none (transfer only) |
| 2 | Blackwell free (yadu's generator exited) **and the package cooled below 70 °C** | Blackwell | Re-check policy parity (**done** 11:10Z: identical hashes and q_ready). Replay the 18-item parity sample of verified gvlab items (`PARITY_MANIFEST.json`). Labels must match, or stop. A first attempt at 11:15Z was refused by the guard (package 91 °C under yadu); nothing ran | 1 worker |
| 3 | after 2 passes | gvlab + Blackwell | gvlab: `STOP_REQUESTED`, so its workers finish their current item and exit; copy gvlab `out_r1` items to Blackwell. Blackwell: resume the remaining R1 chunks. Start with **12 workers on CPUs 0–6, 8–12**; add up to 19 (CPUs 0–6, 8–19) only while the package stays below 80 °C. The guard stops at 85 °C. Thermal finding 11:18Z: the package already reads **91 °C** under yadu's 19 workers. CPU 7 is kept for the GPU job, **CPUs 20–23 stay free**. The Blackwell guard yields if yadu's generator or sajid's queue reappear | Blackwell: 4 free (20–23) + CPU 7 |
| 4 | R1 done (~2.5–3 h after step 3) | Blackwell GPU 0 + CPU 7 | R1 summary and sidecar, then the predeclared gate, then a single-seed development smoke of D / E / DIRECT-GEO / G (≤ 6 GPU-h) | physics finished; ≥ 4 free |
| 5 | Nov 9 ~14:00 IST | laptop → gvlab | `DRAIN` the laptop Q2 remainder (in-flight queries finish, packed roots kept). Freeze a new versioned owner map moving its remaining roots to gvlab. Run them on gvlab **CPUs 14–17** (4 workers): Q2 uses 0–13, so 18 busy and **6 free (18–23)** | gvlab: 6 free |
| 6 | Nov 9 ~17:00 IST | laptop | Final report and last GitHub push; **no job of mine left on the laptop**. gvlab Q2 keeps running on its own guards; the merge and audit happen when the user is back | |

## gvlab core usage now

- Q2: CPUs 0–11 workers + 12–13 reserved.
- Mine: CPUs 16–17.
- **8 free**, which satisfies the rule.

## Progress log

- **12:22Z, step 2 DONE.** Blackwell free (yadu 0 procs, 27 °C). The parity sample (18 items, 34 teacher/replay records) gave **PASS**: 34/34 labels and 34/34 per-move steps agree with gvlab. Max z_err diff 7.4e-7 m; max final-state diff 2.7e-5, which is cross-CPU floating point. Result: `results_parity/PARITY_BLACKWELL_VS_GVLAB.json` (`compare_parity.py`).
- **12:26Z, step 3 DONE.**
  - gvlab `STOP_REQUESTED`. Workers finished their items; CHUNK009 stopped at 178/191 on gvlab, the expected handover INCOMPLETE.
  - `out_r1` copied to Blackwell (4,589 files, sha256 verified). The partial record moved to `handover_partial_records` so finals are not double-counted.
  - Blackwell resumed with order `9-12,37-45,13-36,46-52`. CHUNK009 and CHUNK010 finished with 0 discrepancies, at about 2 min/chunk.
- **12:30Z, thermal finding.**
  - The Core Ultra 9 285K P-cores (CPUs 0–7) push the package to 85–87 °C within seconds. The TEMP guard stopped CHUNK011 cleanly at 157/222 (no item lost). The partial record was archived, along with `guard_events/thermal_1_pcores/`.
  - **Restarted at 12:33Z on E-cores 8–19 (12 workers):** package 59–63 °C. P-cores 0–6 idle, CPU 7 kept for the GPU job, CPUs 20–23 free. ≥ 12 cores are idle, so the core rule holds.
  - **12 E-core workers is the ceiling.** Adding P-cores trips the guard.
- gvlab now runs only Q2 (my CPUs 16–17 released), so it has 10 free cores.
- **12:57Z, discrepancy stop at CHUNK045 (21/52 done).**
  - Item `S2X-…-08710`: the X0 label agrees, but the negative subtype does not. The record says NO_PLAN_FOUND; the pinned teacher's re-query says CERTIFIED (all 32 grasps fail geometric checks).
  - Reproduced identically on gvlab and on Blackwell P- and E-cores (`results_r1_diag/`).
  - **User approved ~18:35 IST:** such subtype-only mismatches are non-blocking (`bwlsuper.subtype_only`). The item stays MISSING and is listed under `nonblocking_negative_subtype_mismatch`. Every other discrepancy still stops the run.
  - Resumed at 13:05Z with `13-36,46-52` on E-cores 8–19.
- **13:27Z, yield stop.**
  - The guard yielded to sajid's `bigbox_cpu.q` (6 pending `a7_sgt` jobs) at CHUNK015, 203/205 items, 23/52 chunks done. The stop was clean.
  - `guard_events/yield_1_sajid_queue/` archived; the partial record was moved.
  - The queue was clear by ~14:00Z. Resumed at 14:03Z with `15-36,46-52`.
  - The local wait-probe self-matched the yadu pattern, so it was replaced by `yield_probe.py`, which applies the same rule as the guard.
- **14:29Z, discrepancy stop at CHUNK026 (35/52 done).**
  - Item `05116`: the archived CERTIFIED negative can now be planned by the pinned teacher. gvlab and Blackwell agree.
  - **User approved ~20:05 IST:** the non-blocking category `nonblocking_negative_not_reconfirmed` (`bwlsuper.negative_not_reconfirmed`). The item stays MISSING.
  - Resumed at 14:53:40Z with `27-36,46-52`.
