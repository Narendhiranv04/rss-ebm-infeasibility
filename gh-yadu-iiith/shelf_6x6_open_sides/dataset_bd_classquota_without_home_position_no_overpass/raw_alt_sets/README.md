# Alternative minimal sets (executed)

For each counted K1-K3 scene with more than one minimal set (289 scenes), every minimal set other than the stored repair
(`labels.B`, role `minimal`) was executed in MuJoCo with the same teacher, the same parking cells (first stored cell per
object) and the same checks as the dataset (each relocation ends in its cell, nothing else moves, T ends at its goal).
Produced by `code/shelf66_v3/alt_sets66.py`. Nothing in `raw/` is modified.

- `roots/<root_id>.json`: `executed_minimal_set` (the stored one), `minimal_sets` (all, from the T-only lattice) and `alt`, one entry per other set:
  `set`, `size`, `y` (1 = executed and verified), `sequence`, `steps`, `witnesses`, `independent` (bd_3dt_plus.independent: each object moves once, any order),
  `orders_tried` (up to 6 orders are tried; the first that executes is stored). If `y` = 0, `failures` lists why each order failed.
- `witnesses/`: trajectories (same format as `raw/witnesses`). `ledger.jsonl`: one line per scene.

Result: 317 alternative sets, 167 executed, 150 not executable. A set can clear T in the planner's T-only lattice (objects teleported to their
parking cells) and still not be executable by real relocations. Of the executed sets, 77 are order-dependent (`independent` = false), so they do not
satisfy the dataset's independent-moves rule; use `independent` to filter. Only one order per set is stored, and only the first parking cell per object.
