# Finger-width sensitivity at query level

**Result: `NO_PROTOCOL_CHANGE`.** The alternate structure check changed **0 of 338** labels compared:
- 194 targeted mock-V2 queries, which include **every** certified-negative and unlabeled query and every query whose static screen changes;
- 144 first-cohort TRAIN queries, a label-blind sample of 12 per K × depth stratum.

The frozen source-exact teacher stays the first-cohort protocol.

Machine-readable results: [`finger_width/finger_width_summary.json`](finger_width/finger_width_summary.json). Figure: [`figures/finger_width_sensitivity.png`](figures/finger_width_sensitivity.png). Code: `code/finger3d.py`, which the production collector never imports.

## The question

The source's IK candidates come from `shelf_scene.ik_candidates`. Its **structure** check (shelf and arm ceiling) keeps the fingers at the model's default, fully open, qpos. Every **object** check instead pre-shapes the fingers to the grasp's open width (SHELF66V3).

The V2 pose-level diagnostic counted poses that are structure-clear only under one finger setting:
- 72 of 12,960 clear only when pre-shaped;
- 31 clear only when fully open.

So open fingers are not uniformly conservative. The question here is whether that changes **labels**.

## Two protocol identities, never mixed

| Identity | Structure check | Teacher protocol |
|---|---|---|
| `SOURCE_EXACT` (frozen V2, the production protocol) | default (fully open) finger qpos | `89cbabed5d7e4847` |
| `ALT_PRESHAPED` (sensitivity only) | fingers pre-shaped to the grasp's open width whenever the teacher has set a check width: screening (`ik_candidates`) **and** every planning chain | `fb172ca477f267d3` |

The alternate identity is installed at class level in its own processes, under its own protocol hash. Its runs live only under `finger_width/` and are never part of any production roster or run.

## Method and results

**1. Static screen.** For every pose and both identities: IK candidates, branches eligible against objects (5 mm, pre-shaped), and the pose's static class (IK_INVALID / HELD_INVALID / to the carry test).

| | Mock V2 (405 queries, 12,960 poses) | Cohort TRAIN sample (144 queries, 4,608 poses) |
|---|---|---|
| Poses whose IK candidates differ | 142 (1.1%) | 53 (1.2%) |
| Queries with at least one pose-class change | 85 (21%) | 31 (21.5%) |
| Pose-class transitions | IK→HELD 48, HELD→IK 27, **IK→carry test 24**, **carry test→IK 4** | IK→HELD 20, HELD→IK 8, **IK→carry test 10**, **HELD→carry test 1** |

Only transitions into or out of the carry test can open or close a route to a placement.

**2. Real MuJoCo teacher runs** (`adaptive_v1`, both identities, 12 pinned workers).

*Targeted mock set (194 queries):*
- all 85 screen-affected queries;
- all 57 certified negatives;
- all 33 unlabeled queries;
- 24 hash-ranked controls with no screen change (12 positives, 12 policy negatives).

*Cohort sample (144 queries).*

| | Targeted mock (194) | Cohort sample (144) |
|---|---|---|
| Harness check: source-exact re-run vs V2 run A | **194 / 194 identical** (label, status, kind, rank, branch, witnesses) | n/a (no earlier labels) |
| Label changes, ALT vs SOURCE | **0 / 194** | **0 / 144** |
| … certified negatives | 0 / 57 | 0 / 20 |
| … unlabeled (`NO_VALID_POST_GRASP_STATE`) | 0 / 33 | 0 / 10 |
| … policy negatives | 0 / 27 (one kind changes, `DETERMINISTIC_PLAN` → `BOUNDED_SEARCH`, still label 0) | 0 / 16 |
| … positives | 0 / 77 | 0 / 98 |
| Fresh-process MuJoCo replay of every ALT positive | **77 / 77 verified** | **98 / 98 verified** |
| Wilson 95% upper bound on the label-change rate | 4.3% (the 85 screen-affected queries) | 2.6% |

**Why labels don't move (observed, query by query).**

*Queries where the alternate check unlocks a pose (into the carry test):*
- mock: 24 queries; 17 were already positive under source-exact, and the other 7 (3 deterministic-plan, 3 bounded-search, 1 certified negative) stay negative when the full teacher tries the unlocked poses;
- cohort sample: 10 queries; 9 already positive, and 1 certified negative that stays negative.

*Queries where the alternate check removes a pose (out of the carry test):*
- mock only: 4 queries. 3 are positives whose witness uses a different grasp: grasp ids 0, 4 and 0, against removed poses 15, 15 and 13. The fourth stays a bounded-search negative.

## Decision

- **Protocol:** the first cohort uses the frozen source-exact protocol `89cbabed5d7e4847`, the one the V2 mock validated.
- **Rate limit:** an alternate checker would change labels on at most a few percent of queries (95% bounds above), and no change was observed.
- **Future change:** should a later study prefer pre-shaped structure checks, it must be registered as its own protocol (here `fb172ca477f267d3`) and collected as a separate dataset version. It must never be mixed into this one.
