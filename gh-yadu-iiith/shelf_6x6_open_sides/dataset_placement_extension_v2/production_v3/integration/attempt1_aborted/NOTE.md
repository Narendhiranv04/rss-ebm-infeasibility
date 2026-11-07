# Integration attempt 1 (aborted, superseded)

Smoke A completed normally (4 units, 50 s). Smoke B was started as a **session leader** (start_new_session) and its collector died
right after the guard passed: collect3d called os.setpgrp() (the V2 collector's call), which raises EPERM for a session leader. The
exclusive lock it had just taken was released by the kernel when the process died (prod3d check passed immediately afterwards).
The orchestrator's poll loop did not notice the early exit and was stopped by PID.

Fix (production v3): collect3d.own_group() accepts the session-leader case (a session leader already leads its own process group),
with a regression test (test_collector_process_group_as_session_leader); the orchestrator now polls the process every 0.2 s.
The same defect could have hit production when tmux's shell exec's the launcher. The integration run was then repeated from scratch.
