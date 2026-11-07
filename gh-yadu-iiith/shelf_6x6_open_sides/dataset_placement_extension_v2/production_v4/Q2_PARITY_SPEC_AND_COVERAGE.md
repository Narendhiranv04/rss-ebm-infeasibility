# Q2_3D_PARITY_V1: the 3D placement training roster, specification and coverage

**What changed and why.** The v3 production roster carried the sparse V1 mock design into production: `plan3d.roster_for` gives 10 slots per root over one root scene, one 1-edit scene and one 2-edit scene, 13,651 queries in all. That under-samples the training distribution compared with the finalized **2D Q2** methodology. This was a sampling error, not a teacher failure.

`Q2_3D_PARITY_V1` replaces the v3 roster. It applies the 2D Q2 procedure (`planar_2d/dataset_relocations_v1/code_v2/launch_plan.py`, quota Q2) to the same 1,582 QC-eligible first-cohort roots. Only the 6×6-specific details change.

**Result:**
- **59,972 physical placement queries**, exactly the theoretical Q2 maximum for this cohort's object counts;
- 4,616 static contrasts;
- 0 exact semantic duplicates;
- all 9 predeclared coverage gates pass.

Rule code: `code/q2roster3d.py`. Machine-readable results:

| File | What it is |
|---|---|
| `COVERAGE_REPORT_V4.json` | coverage report and gates |
| `frozen/q2_parity_roster_v4.jsonl.gz` | the roster |
| `frozen/q2_contexts_v4.jsonl.gz` | every context, including NOT_APPLICABLE ones with their reason |
| `frozen/semantic_duplicates_v4.json` | the duplicate map (empty) |
| `frozen/roster_manifest_v4.json` | the manifest and its hashes |

## 1. Procedure (identical to 2D Q2 except where marked 3D)

| Step | 2D Q2 | 3D Q2-parity (this roster) |
|---|---|---|
| Root objects | up to 4 non-target objects: round robin over source bins near / mid / far, hash order inside a bin | same base order; then **3D:** a greedy diversity pick over (source bin, shape, height tertile, source clutter), ties broken by the base order |
| Source bins | base-distance tertiles of the 8×8 table | **3D:** tertiles of the 36 shelf cells' horizontal distance from the xArm base at the world origin (cell centre = (0.42 + (r−2.5)·0.075, (c−2.5)·0.075) m) |
| Root destinations | up to 8 legal empty cells per object, stratified by Chebyshev distance from the source, hash order inside a stratum, interleaved near / mid / far (`allocate`: k = n/3 per bin, topped up from the others), inclusion probability k / n_bin | same, with **3D bins near = 1, mid = 2–3, far = 4–5** (instead of 1–2 / 3–4 / 5–7). Central cells have no far cell on a 6×6 grid; `allocate` then tops up near and mid, which is genuine geometric availability. |
| Prefix 1 | one-edit hypothetical scene: mover = the first hash-ordered non-target, to a hash-chosen free cell | same |
| Prefix 2 | an independent one-edit scene: mover = the second, to a hash-chosen free cell | same, preferring a cell at Chebyshev ≥ 2 from prefix 1's cell (geometrically distinct). It is **not** a continuation of prefix 1. |
| Prefix 3 | one two-edit scene: mover 1 then mover 2 (distinct), never back into mover 1's source | same |
| Prefix queries | up to 2 **unmoved** objects nearest the edited objects' original cells × 4 destinations of the edited scene, the first object's first destination being the vacated cell | same; free cells and legality are computed in **each hypothetical scene** |
| Static contrasts | (first prefix object, vacated cell) in the root scene | same (OCCUPIED, deduplicated), plus one OUT_OF_BOUNDS root contrast per root. Never sent to the physical teacher. |
| Inapplicable contexts | not generated | recorded NOT_APPLICABLE with the reason; never manufactured |

**Label-blind by construction.** `allowed()` passes the rule exactly these fields:
- object id, shape, size and height;
- initial cells;
- shelf geometry;
- the goal cell, **only** to flag `dest_is_t_goal`: never a veto, never a preference; it appears as an ordinary destination 2,255 times;
- K, as a reporting field only.

Everything else is never read: roles, blockers, parking, mechanism pools, B/D labels, repairs, construction spec, teacher outcomes. `test_rule_ignores_protected_metadata` scrambles 14 protected fields in 50 roots and checks the rows are identical. All randomness is SHA-256 of (rule, root, slot), so the roster is fully deterministic.

**Recorded per query:**
- edit depth and block;
- full hypothetical cells, parent scene and edits;
- object shape, size and height (and height tertile);
- source bin and distance, and source clutter;
- destination band and Chebyshev distance, and the clutter of the source–destination box;
- object and destination ranks;
- stratum k / n and the **inclusion probability**, for later weighting;
- the semantic key.

**Semantic key.** The key is the SHA-256 of:
- the snapshot manifest hash;
- the full hypothetical object state (id, shape, size, height, cell for every object);
- the queried object and destination;
- the teacher protocol `89cbabed5d7e4847`;
- the policy (`adaptive_v1`, its stages).

Identical keys are labelled once, and every context reference is kept. This cohort has **none**: the key is computed on the full object state, and no two contexts coincide.

## 2. Exact counts

| | Value |
|---|---|
| Non-target objects per root (0 / 1 / 2 / 3 / 4 / 5 / 6 / 7) | 116 / 164 / 298 / 393 / 383 / 121 / 66 / 41 roots |
| Q2 maximum per root (0 / 1 / 2 / 3 / ≥4 non-target) | 0 / 8 / 24 / 44 / 56 |
| Theoretical maximum (before destination availability) | **59,972** |
| Realized before deduplication | 59,972 (every root has ≥ 28 free cells, so destinations never run short) |
| Exact semantic duplicates removed | 0 |
| **Physical queries** | **59,972** (1,466 roots; the 116 with no non-target object have nothing to place) |
| Static contrasts | 4,616 (3,150 OCCUPIED, 1,466 OUT_OF_BOUNDS) |
| Distinct hypothetical scenes | 5,074 (1,466 root + 1,302 + 1,302 one-edit + 1,004 two-edit) |
| Distinct root objects queried | 4,632 |
| Distinct destination cells per root (mean) | 22.7 |

**Every reduction from 56 per root is structural, never filtered.** The NOT_APPLICABLE contexts are:
- 116 root contexts, which have no non-target object;
- prefix1 in 280 roots: 116 have no object and 164 have one object, so no unmoved object is left to query;
- prefix2 in 280 roots, which have fewer than 2 movers;
- prefix3 in 578 roots, which have fewer than 3 movers.

## 3. Coverage

| By block | root 35,064 · prefix1 9,224 · prefix2 9,224 · prefix3 6,460 |
|---|---|
| By depth | 0 → 35,064 · 1 → 18,448 · 2 → 6,460 |
| By destination band | near 23,390 · mid 20,370 · far 12,604 · vacated 3,608 |
| By split | train 42,612 · val 8,752 · test 8,608 (inherited lineage split, unchanged) |
| By K | K0 13,680 · K1 11,916 · K2 15,620 · K3 18,756 |
| By shape | box 28,616 · cylinder 31,356 |
| By source distance | near 19,240 · mid 20,776 · far 19,956 |
| By height tertile | short 19,968 · medium 20,044 · tall 19,960 (edges 0.129 / 0.155 m) |
| By source clutter (occupied 8-neighbours) | 0 → 18,980 · 1–2 → 38,132 · 3+ → 2,860 |

The full joint tables are in `COVERAGE_REPORT_V4.json`: K × split × block × band, block × shape × source, split × block. **No K × split × block × band cell has fewer than 20 queries.**

Figure: `figures/q2_parity_coverage.png`.

**Geometric limits, by design:**
- Far destinations are scarcer than near and mid, because a 6×6 grid has no cell at Chebyshev ≥ 4 from its central cells.
- Source-clutter 3+ is rare in the scenes themselves.
- Both are reported; neither is oversampled.

## 4. Predeclared gates (mechanical, all PASS)

| Gate | Result |
|---|---|
| Every Q2 block appears in every split for every K where it is structurally representable | PASS |
| Split inherited unchanged (every root = the frozen v3 / extended live split) | PASS |
| Every semantic key in one split | PASS |
| Every scene layout (root + edits) in one split; no cross-split duplicate layout | PASS |
| Edit states valid: inside free shelf cells, no duplicate occupied cell, target never moved, distinct movers, queried objects unmoved | PASS |
| Every physical destination legal and free **in its own hypothetical scene**; static rows really occupied / out of bounds | PASS |
| near / mid / far (and vacated for depth ≥ 1) present in every split × depth | PASS |
| Physical semantic keys unique | PASS |
| Count reconciles: realized ≤ theoretical, every difference explained | PASS |

## 5. Labels the teacher will attach (unchanged)

The labels come from SHELF66V3_PLACE_V2, protocol `89cbabed5d7e4847`, under `adaptive_v1`. The physical event is the placement of an already-held object: lift, carry, lower, release, retreat and settle, checked by 13 MuJoCo gates. It is not pick accessibility and not target-task success.

| Label | Status and meaning |
|---|---|
| 1 | `VERIFIED`: a stored witness that replays in a fresh process |
| 0 | `NEG_CERTIFIED_GEOMETRY`: a finite-scope certificate over 32 poses × ≤ 4 IK candidates × the footprint model |
| 0 | `NEG_UNDER_POLICY`: `BOUNDED_SEARCH` / `EXECUTION` / `DETERMINISTIC_PLAN`, with its scope |
| None | `UNLABELED_*` / `INTERRUPTED`: never a negative |

Static contrasts are a separate stream, never physical negatives.

## 6. Loader schema for the placement sampler

There is one sample per physical query. Join the roster row with its unit record by `qid`; witnesses come from the unit NPZ by `witness_id`.

| Group | Fields |
|---|---|
| Identity / split | `qid`, `root_id`, `split`, `K` (reporting), `block`, `depth`, `prefix_kind`, `semantic_key`, `scene_hash`, `parent_scene` |
| Hypothetical state | `cells` (every object's cell), `edits`, object dimensions from the root (`shape`, `size`, `height`), `n_obj` |
| Query | `object`, `src`, `dest`, `band`, `cheb`, `source_bin`, `source_dist_m`, `height_bin`, `src_clutter`, `clutter{nbr_src, nbr_dst, between}`, `dest_is_t_goal` |
| Sampling design | `obj_rank`, `dest_rank`, `stratum_k`, `stratum_n`, `incl_prob` (use 1 / `incl_prob` to undo the stratified destination design) |
| Label | `label` ∈ {1, 0, None}, `status`, `negative_kind`, `certificate` / `scope`, `verified_rank`, `verified_branch`, `witnesses[{witness_id, rank, az, level, grasp_id, branch}]` |
| Witness arrays (NPZ) | grasp, `q_g`, hold evidence, packed plan, executed joints and object poses, branch |

**Training use:**
- Train on label ∈ {0, 1}.
- Keep None out of the negative class.
- Keep static contrasts as a separate auxiliary signal.
- For held-out evaluation, report per block × band × K; every cell has ≥ 20 queries in each split.

`python code/q2roster3d.py utility RUN_DIR OUT.json` produces the training-utility report of any partial or finished run: labels by block, band, shape, K, split and depth, effective distinct queries, scenes and context references. For the 32-query integration run, see `integration/training_utility_smoke_A.json`.
