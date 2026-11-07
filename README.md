# SHELF66V3 3D v4 / 2D Q2 telemetry (status only, not data)

Small status files written every 10 minutes by the machine-side service `code/status4.py` (branch `naren/3d-dataset-relocations`):

- `status/3d_v4_latest.json`: the gvlab snapshot (3D v4 Q2-parity run and 3D v3 state, labels so far, bytes, throughput, ETA, temperature / CPU / RSS / disk, collector PID, errors, manifest ids, local backup summary).
- `status/2d_q2_latest.json`: the laptop's read-only observation of the independent 2D Q2 job.
- `status/history/<date>_{3d,2d}.jsonl`: every snapshot, append-only and idempotent.
- `status/offhost_mirror_latest.json`: hash verification of the laptop's off-host copy of the backed-up units.

Fast-forward pushes only, from one clone under a git lock. **A status commit is not a dataset backup.**
