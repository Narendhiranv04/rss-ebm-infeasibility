# CONTEXT: working state for the hybrid study (read this first)

Last updated: 2026-11-23 ~14:00 IST. ROOT of a tree: keep THIS file under ~9,000 chars (SessionStart hook truncates there).
Details live in branch files under `context/` (word-for-word history, nothing dropped). When an entry here gets old, MOVE it
(verbatim) into the right branch and leave a one-line summary here. Open a branch only when you need its detail.

## Branches (open on demand)
- `context/log_2026-11-23.md`: Nov 23 log: server results (E-M4M 600 s, timing, belief noise, noisy labels), encoder seeds, abstain tests, test 1 (gain by K), test 2 (short->long), hybrid dropped.
- `context/log_2026-11-22.md`: Nov 22 log: VAL -> prereg -> TEST confirm, strict ARS review, nophase/setnet/encoders, eta/gate/calibration, rank check, M-sweep, robustness + noise curve, hardware/G2/height checks, E-M4M runs, unseen-worlds finding, fresh-set design, server/home-server setup.
- `context/setup_v34_and_val.md`: V3.4 data/code/env paths, team numbers + baselines, VAL runs, prereg history, DA reviews, older (pre-V3.4) 3D work.
- Also: `RESULTS.md`, `docs/EXPERIMENT_LOG.md`, `reports/plan_scorer_report_source.html` (+ PDF), `docs/HZ_V34_PREREG.md`.

## Who / what
- **User:** nav (new; wants plain explanations; keep Claude usage low). Teammates: Naren (Approach B + datasets), Ayush/Yadu (3D baselines), Abhinav (2D analytical baselines).
- **Task:** robot obstacle relocation (3D 6x6 shelf, xArm7; 2D 8x8 grid, 3R arm): which blockers to move, order, where; physics teacher verifies.
- **Study:** per-step hazard scorer replaces XYNet (whole-plan scorer) inside V3.4 (cVAE 4 spots/blocker, all 1-3 move plans, frozen rule noop gate 0.5 / eta 0.3). **Main = 3D**; 2D = ablations.
- **Git:** branch `nav/hybrid-xy`, last pushed 13cde24 (user pushes; push fails in sandbox). Keep the user's unrelated `M READMe.md` out of commits. Plain-English messages; never reword 8e18429 / 3a76365 / d94d512.

## Standing rules (also in memory)
1. Never touch TEST without a committed pre-registration; VAL first. Label results exploratory/confirmatory + split.
2. Look before any delete/move/overwrite. Never assume a file is ours.
3. One ARS Devil's Advocate review per substantial work unit (sonnet subagent); don't re-review the reviewer's own fixes.
4. Explain simply. Mac: max 7 of 8 cores, AC power only. No polling idle jobs.
5. Never run jobs on gvlab; gvlab2 copy-only. Home server: no heavy jobs (UPS). The second server's name/IP only in local memory, never in git/reports.
6. Safety guard: no `$VAR` paths in rm/cp/redirects/remote writes (use literals); if it blocks, ask the user. Remote scripts need sha256 in `harness/hooks/REVIEWED_REMOTE_SCRIPTS.txt`.
7. Reports = PDF (Chrome headless), not artifacts. Don't write the user's paper text. No file/folder renames. Reserved V3.4 cohort: never use.
8. User decisions (11-23): keep the PURE scorer (no hybrid / simulator check). Fresh-set prereg ON HOLD (don't draft; nothing scored on the fresh set without one).

## Key results so far (details in branches)
- 3D TEST confirmatory (prereg 8e18429, d94d512): SUPPORTED. Task 90.5 vs XYNet 80.8 (+9.7, lb 5.9); noop +7.5 (lb 0.5); abstain +4.3 (lb -4.2). 2D NOT supported (abstain lb -12.0 < -10), fixed post-hoc by eta 0.5-0.6.
- vs E-M4M (classical, 600 CPU-s, 3D TEST): task +14.5 (lb 8.2); noop -3.5 / abstain -9.7 (lb -16.1): our weak spot = giving up. ~1000x faster.
- Gain is ~half training objective, ~half encoder (setnet 87.0 vs ours 90.8 vs XYNet 78.5); transformer encoder ~ ours. Robust to shifts/friction/mass/belief noise; lead kept.
- Abstain fixes (3D VAL, post-hoc): 3-seed ensemble (min plan, median noop) -> abstain 100 with task ~92.
- Test 1: lead does not grow with K. Test 2 (no long examples): both drop on K3; ours keeps lead (+23 / +14).

## NOW / NEXT
- 14:00 (11-23) RUNNING on Mac: `explore/short_long/run_fullnostep.sh` = ours WITHOUT step-number input on FULL data (SL_VARIANT=fulldata_nostep, same snapshot abf_v32 as frozen scorer), 3D VAL seeds 0-2, log `approachb_3d_learning/runs/explore/short_long/logs/run_fullnostep.log` -> FULLNOSTEP_DONE (~20-30 min). Compare vs `fulldata_same_seed.hz` in each metrics file (= frozen scorer, same seed).
- USER ASKED (11-23 ~13:50): when that finishes -> ONE DA review of test 2 + nostep + fullnostep results -> add to report (PDF) -> commit + push (user pushes if sandbox fails). Uncommitted: explore/abstain_tests.py, explore/short_long/ (short_long.py, run_all.sh, run_nostep.sh, run_fullnostep.sh), CONTEXT tree.
- Test 2 + strict nostep (3D VAL, 3-seed means; task K1/K2/K3 | all task/noop/abstain): noK3scenes ours 72.4/79.1/63.8 | 72.6/85.5/97.7, XYNet 49.1/69.1/40.5 | 55.2/82.4/84.5. no3moveplans ours 74.9/75.9/61.0 | 70.9/85.1/59.2, XYNet 56.6/68.7/47.2 | 59.0/83.6/3.4. no3moveplans_nostep 73.6/77.7/62.4 | 71.8/85.9/82.2. Full-data ref ours 85.5/93.8/90.5, XYNet 71.1/83.9/76.7. No CIs yet.
- SERVER (13:50): tmux 'fresh' generating fresh 3D scenes, wave 10, 4.5 h in, 182 accepted. Selected vs quota: K0 49/49, K1 39/39, K2 63/63, K3 14/75 (K3 attempts ~45 min each, ~1 in 3 come out K3). none_dense: 55 attempts logged (unchanged since 12:04). At ~4 K3/h -> ~15 h more (~05:00 11-24). User (11-23 ~14:10): let it run overnight, workers unchanged, no K3 cap. Why slow: 40 of 72 K3 attempts hit ROUTE_TIME_CAP (~49 min each, 33 of ~50 CPU-h); 14 came out K2 or less. ~14:00 user set gen/k/MAX_RUNNING = 14 (one per physical core; runner reads it live, run_3dt_plus.py:155). Check K3 yield tomorrow vs 0.18 before (team ledger 0.33).
