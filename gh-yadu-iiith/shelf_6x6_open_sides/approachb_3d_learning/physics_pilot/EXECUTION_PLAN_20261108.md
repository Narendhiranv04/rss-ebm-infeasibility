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
| 2 | Blackwell free (yadu's generator exited) | Blackwell | Re-check policy parity, then replay a sample of already-verified gvlab items. They must match the gvlab results, or stop | 1 worker |
| 3 | after 2 passes | gvlab + Blackwell | gvlab: `STOP_REQUESTED`, so its workers finish their current item and exit; copy gvlab `out_r1` items to Blackwell. Blackwell: resume the remaining R1 chunks, **19 workers on CPUs 0–6, 8–19**. CPU 7 is kept for the GPU job, **CPUs 20–23 stay free**. The Blackwell guard yields if yadu's generator or sajid's queue reappear | Blackwell: 4 free (20–23) + CPU 7 |
| 4 | R1 done (~2.5–3 h after step 3) | Blackwell GPU 0 + CPU 7 | R1 summary and sidecar, then the predeclared gate, then a single-seed development smoke of D / E / DIRECT-GEO / G (≤ 6 GPU-h) | physics finished; ≥ 4 free |
| 5 | Nov 9 ~14:00 IST | laptop → gvlab | `DRAIN` the laptop Q2 remainder (in-flight queries finish, packed roots kept). Freeze a new versioned owner map moving its remaining roots to gvlab. Run them on gvlab **CPUs 14–17** (4 workers): Q2 uses 0–13, so 18 busy and **6 free (18–23)** | gvlab: 6 free |
| 6 | Nov 9 ~17:00 IST | laptop | Final report and last GitHub push; **no job of mine left on the laptop**. gvlab Q2 keeps running on its own guards; the merge and audit happen when the user is back | |

## gvlab core usage now

- Q2: CPUs 0–11 workers + 12–13 reserved.
- Mine: CPUs 16–17.
- **8 free**, which satisfies the rule.
