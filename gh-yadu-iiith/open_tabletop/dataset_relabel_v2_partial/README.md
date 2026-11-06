# 3D open-tabletop relabel for teacher V2: PARTIAL, paused

This is work in progress. It is not a finished dataset and it is not merged into `dataset_bd_classquota/raw/`.

**What V2 changes.** In an executed B sequence the arm returns to the ready pose only **once**, after the last relocation, and then
moves T. V1 returned to the ready pose after every relocation. The code changes are in `../dataset_bd_classquota/code/`:
- `dataset-bd/bd_3dt.py`, `make_combined_3dt.py`, `view_3dt.py` and `viz_3dt.py` are modified;
- `relabel_3dt_v2.py`, `finalize_v2.py` and `dataset-bd/tests/test_bd_3dt_v2.py` are new.

Each script's docstring says exactly what is recomputed and what is kept byte-identical from V1.

## `output/`: the relabel run's output, as it stood when stopped

The run was `python relabel_3dt_v2.py <raw> <out> 12`, resumable.

- **Progress:** 194 of the 1,590 accepted roots listed at the start of the run (`run.log`) were relabelled.
- **Stopped:** on the user's instruction ("stop the dataset", 3 Nov 2026).
- **Contents:**
  - `RELABEL.jsonl`: one line per relabelled root.
  - `roots/`: V2 records.
  - `witnesses/`: V2 trajectories.
  - `run1_aborted*`: an earlier aborted start, kept as evidence.
- **Verified K, V1 → V2, over the 194 roots:**

  | V1 → V2 | Roots |
  |---|---|
  | 4 → NONE | 83 |
  | 3 → NONE | 31 |
  | 3 → 3 | 31 |
  | 4 → 4 | 28 |
  | NONE → NONE | 13 |
  | 3 → 4 | 7 |
  | 2 → 2 | 1 |

  Without the intermediate ready pose, many K4 and K3 repairs are no longer verified within budget. This is a teacher-policy
  change, not new scenes.
- **Reproduction check:** the kept fields (single-relocation B entries, D edits, blocker lattice) reproduce for every relabelled
  root (`repro` field).
- **Compute:** about 12.3 CPU-hours.

`finalize_v2.py` (merging into `raw/` and redoing the class selection) has **not** been run.

## `stage2_draft/`: draft only, never run

`stage2_3dt.py` would produce, for each base scene, up to five verified alternative solutions and three redundant-move variants,
each executed with the V2 teacher. The folder holds the copies of `bd_3dt.py`, `bd_3dt_plus.py`, `bd_core.py` and `k2v2.json` that
the draft was written against.
