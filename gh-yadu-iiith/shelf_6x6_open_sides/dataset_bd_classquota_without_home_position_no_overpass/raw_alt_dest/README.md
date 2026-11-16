# Alternative minimal sets retried with other parking cells

Second pass after `raw_alt_sets/`: the 150 alternative sets that did not execute with the first stored parking cell of each object were retried with the
other stored cells (`alt_parking`, up to 5 per object) and every order (up to 6) by `code/shelf66_v3/alt_dest66.py`. The first chain that executes in MuJoCo
with the dataset's checks is stored. Nothing in `raw/` or `raw_alt_sets/` is modified.

- `roots/<root_id>.json`: `alt_dest`, one entry per retried set: `set`, `y`, `dest` (object -> cell actually used), `sequence`, `steps`, `witnesses`, `independent`
  (each object moves once, any order), `assignments_tried`, `plan`/`exec` budgets used. If `y` = 0: `stopped_by` (exhausted / time / exec_budget).
- `witnesses/`, `ledger.jsonl` as in `raw_alt_sets/`.

Result: 150 retried, 61 executed (60 of them with a non-first parking cell), 89 not executable (85 exhausted every stored assignment). Of the 61, 43 are
order-dependent (`independent` = false). RRT seeds and a 8x larger iteration budget were checked on a 36-set sample and change nothing: the failures are
deterministic (out of reach, arm/hand hits an object, hand overlaps a neighbour).

## Label conflicts (important)

`label_conflicts.json`: 7 scenes (3 stored K2, 4 stored K3) now have an EXECUTED repair with fewer relocations than their stored K (e.g. one object moved to
another parking cell clears a K2 scene). The teacher only tried the first parking cell, so its K is the minimum for that cell only. These scenes' K labels in
`raw/` are therefore not the true minimum; exclude or relabel them before using K as ground truth.
