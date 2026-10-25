# mamo: M4M, E-M4M, I-M4M, DOGAR, RRT, KPIECE and SELSIM in MuJoCo and Python

A reimplementation of

* **M4M**: Saxena & Likhachev, *Planning for Complex Non-prehensile Manipulation Among Movable Objects by Interleaving Multi-Agent Pathfinding and Physics-Based Simulation*, ICRA 2023 (`papers/2303.13352v1.pdf`)
* **E-M4M**: Saxena & Likhachev, *Planning for Manipulation among Movable Objects: Deciding Which Objects Go Where, in What Order, and How*, ICAPS 2023 (`papers/2303.13385.pdf`)
* **I-M4M**: Saxena & Likhachev, *Improved M4M: Faster and Richer Planning for Manipulation Among Movable Objects in Cluttered 3D Workspaces*, ICRA 2024 (`papers/Improved_M4M_...pdf`)
* **DOGAR** (baseline): Dogar & Srinivasa, *A Framework for Push-Grasping in Clutter*, RSS 2011 (`papers/dogar.pdf`)
* **RRT** (baseline): LaValle & Kuffner, *Randomized Kinodynamic Planning*, IJRR 2001 (`papers/RRT.pdf`)
* **KPIECE** (baseline): Sucan & Kavraki, *A Sampling-Based Tree Planner for Systems With Complex Dynamics*, IEEE T-RO 2012 (`papers/kpiece_preprint.pdf`)
* **SELSIM** (baseline): Saleem & Likhachev, *Planning with Selective Physics-based Simulation for Manipulation Among Movable Objects*, ICRA 2020 (`papers/selective simulation.pdf`)

RRT, KPIECE and SELSIM are set up as the M4M / E-M4M papers use them as baselines (reference: `archive/ros_src/em4m-main/src/sampling/ompl.cpp`, `test/run_ompl.cpp`), but written here from their papers in Python instead of calling OMPL. Everything runs with MuJoCo in place of Bullet, a uFactory xArm7 (MuJoCo Menagerie model, with gripper) in place of the PR2, and plain Python in place of ROS/SMPL/C++. The reference C++ code (`archive/zips/m4m-main.zip`, `em4m-main.zip`, `im4m-main.zip`) was used for every detail the papers leave open; where paper and code disagree, the paper wins and the difference is noted below.

This package replaces the earlier port in `../code/m4m`, which is kept unchanged for comparison.

It also generates MAMO problems in the five rearrangement environments of **REAP** (Swoboda & Geffner, 2026, `papers/required environments.pdf`, App. B) for the xArm7, with retrieve and relocate tasks and easy / medium / hard difficulty, and has a benchmark web app to compare your own method with all of the above on them (see *Environments* and *Benchmark tool*).

## Install and run

```bash
cd mamo
pip install -e .
python -m pytest -q                                   # unit tests (seconds)

# label reference scenes Easy / Medium / Hard by NGR overlap (paper protocol)
python scripts/classify_scenes.py --levels 5 10 15 --out results/scene_labels.csv
# run planners
python scripts/run_benchmark.py --labels results/scene_labels.csv --algos m4m em4m \
    --per-difficulty 10 --workers 6 --execute --out results/benchmark.csv
# look at a scene: 2D top view + 3D views, with the NGR (or --interactive for the MuJoCo viewer)
python scripts/view_scene.py --scene 10/plan_200003_SCENE.txt --out results/figures/200003
# plan, replay and film a solution: <out>.mp4, <out>_plan.png (one panel per push), <out>.pkl
python scripts/render_solution.py --scene 10/plan_200003_SCENE.txt --algo em4m --out results/videos/200003_em4m
```

On Windows, MuJoCo's DLL fails to load (WinError 1114) if pandas was imported first; `import mamo` loads MuJoCo itself, so import `mamo` (or `mujoco`) before pandas in your own scripts.

Scenes are read from `../archive/ros_src/em4m-main/dat/clutter_scenes/{5,10,15}` (the reference release: 176 / 179 / 202 scenes with 5 / 10 / 15 movable objects).

## Benchmark tool (web app)

```bash
pip install -e ".[app]"
python -m mamo.bench app            # opens http://localhost:8501 (this machine only)
```

* **Datasets**: the reference scenes (5 / 10 / 15 movables) are built in. **Generate** a dataset in the REAP
  environments (choose environments, tasks, difficulties and problems per combination), or add any folder of
  scene files in the same text format (sections `O` and `ooi`, optional `meta`), label it Easy / Medium / Hard
  (movables overlapping the initial NGR), and look at any scene in 2D (one panel per surface) and 3D with its
  NGR and goals.
* **Methods**: every planner in this package plus plugins. **Your method** is a `.py` file in `plugins/`
  (or another folder added on the Methods page) with a function registered by `@register(..., tasks=...)`;
  start from `plugins/example_planner.py`, which lists everything a planner can use and must return, with a
  retrieve example (`example_greedy`) and a relocate example (`example_relocate`).
* **New experiment**: methods x datasets x (environments x tasks) x difficulties x scenes per difficulty x
  seeds, with a budget and parameters per method. A method only runs on scenes of the tasks it declares (the
  M4M family, DOGAR, RRT, KPIECE and SELSIM: retrieve). It runs in the background (each run in its own process, killed if it
  hangs) and can be resumed.
* **Experiments**: progress; solved / replay-success tables per difficulty, min / median / max planning
  and simulation time (as in the papers), success-rate and anytime plots, failure reasons; per run: stats,
  plan figure and execution video. CSVs from `scripts/run_benchmark.py` can be imported.
* **Compare**: several experiments side by side on their common scenes.

Every solution, whatever the method, is replayed from the initial scene by `mamo.execute` with the same
interaction constraints, so success means the plan really retrieves the OoI (retrieve) or leaves every
target at its goal (relocate) in simulation.

Command line equivalents: `python -m mamo.bench list | label <dataset> | new ... [--run] | run <folder> |
generate <name> --envs ... --tasks ... --difficulties ... --per N`.
Experiments, labels and cached images live in `workspace/` (override with `MAMO_BENCH_HOME`).

## Layout

One folder per method in `mamo/`, next to the shared machinery:

| Folder | Method |
|---|---|
| `m4m/` | `planner.py`: M4M Algorithm 1 (and M4M-hat, single CBS call) |
| `em4m/` | `planner.py`: E-M4M Algorithm 1 (best-first search, invalid and hallucinated goals, push database, ablations); `priority.py`: Naive Bayes priority |
| `im4m/` | `planner.py`: I-M4M (depth-first E-M4M, pick-and-place edges, eagerly lazy evaluation); `pickplace.py`: the prehensile action (known grasps, approach, grasp, lift 2 cm, transport, lower 1 cm, release, retract 10 cm); `robust.py`: Robust I-M4M (6 x 2 simulations with sampled masses and friction) and I~-M4M |
| `dogar/` | `planner.py`: DOGAR, Dogar & Srinivasa's backward recursive Reconfigure with this domain's push actions |
| `rrt/` | `planner.py`: RRT over arm + object states, physics-based motions |
| `kpiece/` | `planner.py`: KPIECE (grid of cells on the TCP projection, interior / exterior, importance, progress scores, goal bias) |
| `selsim/` | `planner.py`: SELSIM (Lazy Weighted A*, relevant objects, tracking); `heuristic.py`: 3D BFS heuristic |

Old import paths (`mamo.m4m`, `mamo.em4m`, `mamo.im4m`, `mamo.dogar`, `mamo.pickplace`) still work, so solutions saved before the move load unchanged.

| Shared module | Paper concept |
|---|---|
| `scene.py` | MAMO problem: objects, shelf, OoI; reader for the reference scene format |
| `world.py` | MuJoCo model: static shelf, free-body objects with scene mass/friction (Bullet's product friction rule), xArm7 with position servos and gripper |
| `robot/arm.py` | kinematics, damped-least-squares IK, collision checks against a chosen obstacle set |
| `robot/motion.py` | joint-space planning (RRT-Connect + shortcutting), straight-line tool motion |
| `retrieval.py` | PLANRETRIEVAL: grasp sampling, approach, reach, grasp, lift, extract |
| `ngr.py` | NGR V(gamma_OoI): 1 cm voxels of the arm (and carried OoI) swept along the retrieval |
| `mapf/agent.py`, `mapf/cbs.py` | MAPF abstraction: movables as agents on a 4-connected 1 cm grid; CBS with space-time Dijkstra to a pseudo-goal |
| `push.py` | PLANPUSH: shortcut path, AABB contact point, sampled push start, approach, IK push |
| `simulate.py` | SIMULATEPUSH / ISVALID: physics rollout with the interaction constraints |
| `kinodynamic.py` | the full MAMO state space of the sampling-based baselines: arm + object poses, STEERACTION (1 deg interpolation, physics simulation of motions touching movables), the OoI pre-grasp goal with a simulated grasp check, TCP projection |
| `solution.py` | `Solution`: actions in execution order, final retrieval, statistics (every planner returns one) |
| `envs.py` | the five REAP environments for the xArm7, retrieve / relocate problem generator, feasibility check |
| `bench/` | benchmark tool: planner registry and plugins, datasets and labels, experiment runner, reports, media, Streamlit app (`bench/app`) |
| `execute.py` | end-to-end replay of a solution with a real grasp (not part of either paper's planning); replays pushes, pick-and-places and arm motions |
| `viz.py` | 2D top views (matplotlib) and 3D MuJoCo views of scenes, NGRs and solutions |
| `config.py` | every parameter, tagged `[paper]`, `[cpp]` or `[xarm]` |

## Environments (REAP, App. B) for your dataset

`mamo/envs.py` builds REAP's five environments with the paper's dimensions and the xArm7. The paper's robot faces +y; here it faces +x, so every environment opens towards the robot (-x) like the reference shelf, and the base is placed so the cells can be reached (`scripts/env_reachability.py`: share of cells from which PLANRETRIEVAL succeeds with nothing else on the grid).

| Environment | Paper | Here | Reachable cells |
|---|---|---|---|
| `open_tabletop` | 40x28 cm table, 10x7 cells of 40 mm, cylinders r 12.5 / 17.5 / 20 mm | same; table 30 cm above the base, 20 cm in front | 70/70 |
| `confined_shelf` | cubicle 79x44x34 cm open at the front, 9x4 cells of 70 mm (2 unreachable), cylinders r 30 mm h 20 cm | same; placed like the reference shelf | 34/34 |
| `multi_level_shelf` | open shelf, decks 80x45 cm 24 cm apart, floor compartment split by a separator; 67 cells of 70 mm; boxes 6x6x9 cm | same; floor 20 cm above the base, front 30 cm (paper 35.5); floor columns 10 cm either side of the separator (not given in the paper) | 65/67 |
| `access19` | bookcase compartment 39x27.8x24 cm, roof = top deck, 5x7 cells of 40x36 mm per level, 18 blockers in a fixed pattern; OoI deep inside, goal = top cell (2, 3) | same, but rows 40 mm deep and the compartment 30 cm deep (see below); floor 40 cm above the base (paper 34 cm, Panda) | 35/35 |
| `stacking` | parts table in front and stack table behind the arm (35x35 cm), 10x10 cells of 30 mm; cubes, oblongs, longs | same, flat blocks on the parts table; level 0 of the stack table as goals (stacking itself is not one of our tasks); blocks grasped from above | 100/100 |

**Tasks** (`Scene.task`, stored in a scene file's `meta` line):

* `retrieve`: take the OoI out towards the robot. This is M4M's problem, so M4M, E-M4M, I-M4M (all variants), DOGAR, RRT, KPIECE and SELSIM run on it unchanged, on the OoI's surface: objects on other decks or tables are fixed obstacles for them.
* `relocate`: move 1-4 targets to goal cells, possibly on another surface (other deck, Access-19's roof, the stack table). Success: every target's centre within half a cell of its goal, 2 cm in height, upright. No baseline solves it; it is for your method (`example_relocate` shows the API).

**Difficulty** = the preset a problem is sampled from (per environment, shown on the Datasets page): ranges of movable and immovable objects, of targets (relocate) and of *blockers*, movables placed in a target's or goal's access corridor (the cells in front of it, following REAP's feasibility rules, directly in front first; for stacking's grasps from above, the cells within two columns and one row, where the open fingers come down). Immovables never stand in a corridor, and in easy and medium problems neither do the other movables. With the feasibility check on (default), a **retrieve** problem is kept only if the OoI can be retrieved with only the immovables as obstacles *and* exactly 1 (easy), 2 (medium) or 3-6 (hard) movables overlap that retrieval's NGR, which is the E-M4M / I-M4M papers' labelling (`retrieval_blockers`, recorded as `ngr_blockers` in the scene's meta). A relocate problem is kept if every target can be retrieved from where it stands and from its goal once the movables are removed.

The `reap-retrieve` dataset (`python -m mamo.bench generate reap-retrieve --tasks retrieve --per 4`) is retrieve-only: 4 problems per environment and difficulty, plus the 30 reference shelf scenes of the v6-v8 comparison as a sixth environment (`reference_shelf`), so every method runs in every environment.

Adaptations, all for these environments only (the reference shelf scenes and the planners' behaviour on them are unchanged):

| Item | Why |
|---|---|
| Object centres on the 1 cm MAPF lattice, at least 1 mm apart (so 30 mm blocks and 40 mm cylinders never fill neighbouring cells) | M4M rounds poses to its 1 cm grid; touching objects off the lattice overlap there and no CBS solution exists. Access-19's rows are therefore 40 mm deep (paper 36 mm) |
| Several support surfaces per scene (`Scene.surfaces`, from every horizontal static part); the planning surface is the one under the OoI | multi-level shelf, Access-19 roof, two tables |
| Robot base and per-environment parameter overrides in the scene's `meta` (stacking: grasps pitched 85-90 deg, 2 cm above the table) | xArm7 placement per environment; 3 cm blocks can only be grasped from above |
| `grasp_top_margin` (default 2 cm as before; 1 cm for stacking) | the grasp height cap below the object top was hard-coded |
| Pick-and-place options `goal_z`, `turns`, `via`, `reach_first` (all off by default) | placements on another surface; sliding straight out of / into compartments (the 1 s RRT rarely threads two narrow compartments); trying every reachable grasp |
| Relocate replay: a carried (welded) object is exempt from the tilt / speed / off-surface checks | it swings with the hand while carried between surfaces |

**Baselines on the retrieve problems** (`workspace/experiments/*_reap-retrieve-baselines`, dataset `reap-envs`: 2 problems per environment and difficulty, 300 s, solved and replayed / 6 per environment):

| Environment | M4M | E-M4M | I-M4M | I-M4M-PnP | Lazy I-M4M-PnP | DOGAR |
|---|---|---|---|---|---|---|
| Open tabletop | 0 | 0 | 0 | 1 | 1 | 0 |
| Confined shelf | 1 | 1 | 1 | 1 | 1 | 1 |
| Multi-level shelf | 2 | 4 | 4 | 4 | 4 | 2 |
| Access-19 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3D stacking | 5 | 5 | 5 | 5 | 5 | 4 |
| **Total / 30** | **8** | **10** | **10** | **11** | **11** | **7** |

Push-based rearrangement struggles here: these environments were designed for prehensile manipulation (dense grids, tall thin cylinders that tip when pushed 3 cm up, Access-19's boxes 1 cm apart). Most stacking problems need no rearrangement (grasps from above), so they are solved in about a second. Three solved plans failed on replay (a thin cylinder tipped by the arm, a block pushed off the table).

## What follows the papers

### I-M4M (ICRA 2024, `papers/Improved_M4M_...pdf`; reference `archive/zips/im4m-main.zip`)

I-M4M is E-M4M above, unchanged, plus the paper's four improvements:

* **Depth-first search**: OPEN ordered by (more valid actions, fewer re-expansions); ties by the E-M4M priority, then random (reference `OPENCompare`).
* **Pick-and-place** (`im4m_pnp`): for every graspable object that moves in the MAPF solution, a pick-and-place edge to the last cell of its MAPF path is evaluated next to the push. Grasps are the reference's known grasps (face normals of boxes that fit the gripper, 8 directions around cylinders); approach and transport are planned collision-free with every object; only the grasp and placement are simulated. As in the reference, failed pick-and-places do not become invalid goals.
* **Eagerly lazy** (`lazy_im4m`, `lazy_im4m_pnp`): expanding a vertex adds unevaluated edges to OPEN; an edge is planned and simulated only when it reaches the top. Unevaluated entries are not hashed. The released code has no lazy variant; this follows the paper's description.
* **Robust** (`robust_im4m`, baseline `tilde_im4m`): masses and friction coefficients are uncertain; pushes are accepted if 10 of 12 sampled simulations (6 threads x 2 copies) are valid. Robust I-M4M validates every push in those 12 sampled simulations; I~-M4M plans in a single world with one sampled copy of each object; both search over that sampled world's kinematics, and solutions are executed with the true values.

### DOGAR (Dogar & Srinivasa, RSS 2011, `papers/dogar.pdf`)

Algorithm 1 of the paper (backward recursive `Reconfigure`: NGR accumulation, `FindPenetrated`, monotone `avoid` list, search over orderings, GoTo between actions), as the M4M and E-M4M papers re-implement it as a MAMO baseline: the same OoI retrieval trajectory as M4M, the M4M push planner validated in simulation as the action library, and push goals sampled outside the latest NGR (closest free cell plus noise, M4M paper, `dogar`; or uniformly, E-M4M paper, `dogar_uniform`). A push is valid only if the object ends completely outside the NGR. The paper's push-grasp/sweep capture regions and pose-uncertainty regions are not used: the simulator validates actions and poses are known, as in the other planners. The sampled search restarts until the budget is spent. The reference code releases contain no DOGAR implementation.

### RRT, KPIECE and SELSIM: the M4M papers' other baselines

The M4M paper compares against "standard implementations of sampling-based algorithms KPIECE and RRT from OMPL that search the entire MAMO state space by randomly sampling robot motions" and against SELSIM; the E-M4M paper does too, "with the full interaction constraint set that E-M4M considers". The reference release has the RRT / KPIECE setup (`sampling/ompl.cpp`, `test/run_ompl.cpp`, `Robot::SteerAction`, `ComputeGraspTraj`, `CheckGraspTraj`) but not SELSIM's code. Shared by all three (`mamo/kinodynamic.py`):

* **State**: arm joints + pose of every object; distance, sampling and interpolation use the joints only (`PushplanStateSpace`).
* **Motion check** (`SteerAction`): straight joint-space line, interpolated as SMPL's `fillMotionInterpolation` does with the reference's `res = DEG5/5`: ceil(sum_j R_j |dq_j| / 1.745 cm) + 1 waypoints, R_j the radius of the sphere around joint j's anchor that holds every robot geom it moves; self-collision invalidates it; touching the shelf, an immovable obstacle or the OoI ends it (RRT: invalid; KPIECE: valid up to the last free waypoint, unless a movable was touched first); a motion touching a movable is simulated (2 s settle) and is valid if E-M4M's interaction constraints hold, its end state carrying the simulated object poses; one touching nothing is not simulated. The gripper is open throughout.
* **Goal** (`PushplanPoseGoal`): the OoI pre-grasp pose of one grasp sampled at the start (here: PLANRETRIEVAL with only the immovables, which is stricter than `ComputeGrasps`' IK checks), within 1.5 cm per axis and 0.05 rad, and the grasp must succeed in simulation from that state (`CheckGraspTraj`: to the exact pre-grasp, grasp, lift, then out of the shelf where the reference retreats 30 cm; every IK state free of the immovables). Goal samples are IK solutions from random seeds (`RunIK`); the goal distance used by KPIECE's goal bias is the joint distance to the nearest of a fixed set of IK solutions computed at the start (the reference draws a new random one per call).
* **Solution**: one `MotionAction` per edge, replayed by `mamo.execute` like any other action, then the grasp trajectory. The whole wall time counts as planning time (simulation time is reported separately).

**RRT** (`rrt`, `mamo/rrt/`): Fig. 5 of the paper, BUILD_RRT / EXTEND from the start only. The bidirectional planner of Fig. 7 needs backward propagation, which a physics simulator cannot do (KPIECE paper Sec. II-C). RANDOM_STATE is uniform, or the goal with probability 0.05; NEW_STATE steers toward it by at most 0.2 x the joint space's extent (OMPL's goal bias and range: 12.7 rad for the xArm7, whose four +-360 deg joints are bounded; about 4.1 rad for the reference's PR2, whose continuous joints count pi each, so ours truncates fewer extensions). Deviation from the reference: its OMPL wrapper copied the *sampled* state into the tree and dropped the simulated object poses; here a vertex keeps them (what NEW_STATE means in the paper).

**KPIECE** (`kpiece`, `mamo/kpiece/`): Algorithm 2 of the paper: cells on the TCP projection (the reference's `ProjectionEvaluatorFK`) of 10% of its bounding box over 1000 random states (Sec. III-G's initial guess; OMPL, which the authors ran, uses 1/20 of a 10% padded box over 100 states; the paper's re-tuning of cell sizes between runs is not done); exterior cells (fewer than 2k = 6 neighbours) chosen 75% of the time, then the cell of highest Importance = log(I) score / (S (1 + |neighbours|) coverage); a motion of that cell by a half-normal over recency, a state along it uniformly; motions split so every part lies in one cell, coverage = states per cell; progress score update P = 0.7 + 5 x coverage increase / time, score *= min(P, 1); goal biasing of Sec. V-B (a new cell's score is 1 / (1 + joint distance to the goal), and 5% of expansions continue from one of the 30 motions closest to the goal, one per cell, picked as OMPL's control KPIECE1 picks from its close samples: the closest, re-inserted with distance (closest + farthest) x 0.55; that cell's score is then updated too). As in the paper's kinematic experiments (Sec. IV-F), "the sampling of controls was replaced with sampling of random states and connecting to the random states using a local planner": a state within the OMPL range of the selected one, or the goal with probability 0.05 (OMPL goal bias, as the reference used). States inside a simulated motion are known only at its end; expanding from an earlier one re-simulates the prefix (Sec. III-E). Importance uses the paper's log(I) (the root cell, I = 1, never wins against others), and scores that underflow are raised by 1 + log(I) as in OMPL's `Discretization::selectMotion`.

**SELSIM** (`selsim`, `mamo/selsim/`): Algorithms 1 and 2 of the paper with its Lazy Weighted A*. C_R starts empty; PLAN searches the lattice of arm configurations plus the poses of the relevant movables; GETSUCCS collision-checks an action against the environment and C_R only (irrelevant objects are absent from both the collision checker and the simulator), rejects contact with the shelf or a relevant fixed object (immovables, OoI), and defers actions touching relevant movables: they enter OPEN unevaluated and are simulated, with only C_R present, when popped. TRACK replays the plan with every object (actions touching nothing are collision-checked, the rest simulated), records the interacting pairs at every simulation step (contacts already present before an action are not interactions caused by it), and on a violation adds the first object not in C_R found by a BFS from the object the constraint is about. Extension: if the violation concerns no object (the robot touching the shelf, pushed there by an irrelevant object) or that BFS finds nothing, the BFS starts at the robot and may pass through it; SELSIM gives up only if that fails too. Action space and search parameters are the lab's SMPL arm planner as the reference configures it: 1 deg lattice, +-7 deg on joints 1-4 (far from start and goal) and +-4 deg on all joints (within 10 cm of either, measured as SMPL does: BFS cells to the goal, Manhattan cells to the start), IK snap to the goal within 1 cm (BFS distance, i.e. the goal cell and its neighbours), epsilon 100, OPEN ordered by f only, edges collision-checked at SMPL's sphere-motion resolution, and the paper's 3D BFS heuristic from the goal over the immovables (1 cm voxels, walls within 2 cm Euclidean of them and on the grid boundary, 100 per cell). The action cost is the paper's path length, 1000 per degree of joint-space distance (SMPL charges 1000 per action). States hash the relevant objects' full poses at E-M4M's 1 cm / 5 deg. Adaptation: the paper's goal is the pre-grasp pose; here, as for RRT and KPIECE, the grasp must also succeed in simulation (with the OoI and the relevant objects present) and is tracked with the plan. Without this, an object the extraction knocks over can never be fixed: the plan to the pre-grasp pose does not change when that object becomes relevant. Tracking skips the simulation of actions that touch no object (nothing would move), so SELSIM's simulation times are lower than if every action were simulated.

### DOGAR additions

* Pushes are simulated with the gripper closed, as they are planned and replayed.
* Push goals are among the cells the object can reach; "closest cell" is the closest reachable one. Push paths are the cheapest paths under M4M's MAPF step cost (the single-agent MAPF solution M4M's push planner takes its paths from).
* The search draws `samples = 10` push goals per object per Reconfigure call (the paper's action library is finite; ours is sampled) and restarts until the budget is spent; neither paper says how often to sample. DOGAR's preference for actions that penetrate fewer objects is not used: the M4M paper's DOGAR "naively samples pushes".
* The GoTo from home to the first executed push is planned inside the search (a failure keeps searching); the budget is wall-clock time.

### Fidelity audit (2026-10-25)

Each method was checked line by line against its paper and the official code (the M4M, E-M4M and I-M4M releases, OMPL for RRT / KPIECE, SMPL for SELSIM's action space). What changed:

* **All M4M-family methods**: CBS no longer reports a conflict between two objects that are both still at their start cells (two objects a few mm apart can share a rounded cell; CBS then had no solution and M4M ended with "CBS failed") -- `cbs.cpp` checks from t = 1 and skips such pairs. PLANRETRIEVAL keeps the grasp found by PLANRETRIEVAL(O_I) for every later call (`ComputeGrasps` once; the paper's goal is one grasp pose). The NGR voxelises the robot shrunk by 1 cm (`voxeliseTrajectory`: sphere radius - RES); M4M's NGR is the arm alone ("the volume occupied by the robot arm"), E-M4M's and I-M4M's the arm and the carried OoI. Push points include the object's start x_1 (Fig. 5: i = 0..n). Contacts present when a simulation starts are no longer exempt.
* **M4M**: objects are tried in random order (`chooseObjDTS` with all priors at Beta(1, 1)); M4M-hat keeps pushing towards its single CBS solution until the budget is spent (`coop_plan_no_iterate.cpp`).
* **E-M4M / I-M4M**: unit MAPF step costs and the reference's penalty weights (table below); TIEBREAK is lexicographic; E-M4M's noops counts expansions with a failed push (`mamo_node.cpp`), I-M4M's reexpands counts re-expansions (paper); a duplicate that has already been expanded keeps its parent (its children's approaches start from its arm configuration); a cached push is not reused if the pushed objects would land on an irrelevant one (paper: irrelevant objects "will not be affected"); the budget is wall-clock time (`Solve`).
* **I-M4M**: the pick-and-place transport keeps the tool's roll and pitch fixed (yaw free), so the carried object stays upright (`addPathConstraint`: 5 deg); a pick-and-place is valid without a minimum displacement (`SimPickPlace`); the carried object is exempt from the pose checks; Robust I-M4M and I~-M4M keep the scene's robot base and overrides (they lost them on the REAP environments).
* **DOGAR, RRT, KPIECE, SELSIM**: as described in their sections above.

Not changed, still open: the Naive Bayes coefficients and the shelf friction (table below). The overnight and validation results above were obtained before this audit.

**Dataset labels.** The `ngr_blockers` stored with the `reap-retrieve` problems were computed with the old (unshrunk, OoI-inclusive) NGR. Recomputing them for the same retrieval with the 1 cm shrink lowers the count in 12 of 58 problems (11 by one, `stacking_retrieve_hard_0002` by two: 3 -> 1), mostly stacking problems, where blockers sit just below the descending open fingers. Those 12 problems may now be in the wrong difficulty, or below the hard preset's 3-blocker floor, so regenerate (or re-label) the dataset before comparing methods by difficulty.

**Post-audit sanity run** (`workspace/experiments/20261025-191635_post-audit-sanity`): all nine planners, one problem per difficulty on the reference, confined and stacking shelves, 60 s each, 81 runs. Nothing crashed; solved and replayed / 9: M4M 0, E-M4M 1, I-M4M 1, I-M4M-PnP 2, Lazy I-M4M-PnP 3, DOGAR 1, RRT 4 (3 replay), KPIECE 5, SELSIM 3. The 60 s budget is a smoke test, not a comparison (M4M ends 4 times with "no MAPF-suggested push left").

### M4M and E-M4M

* **Interaction constraints**: no contact with immovable obstacles by the robot *or any object*, no movable tilting past 25 deg, moving faster than 1 m/s or leaving the shelf; robot-shelf contact forbidden. Robot contact with other movables is allowed (multi-object pushes).
* **NGR** is the 3D swept volume of a planned retrieval trajectory, computed once from PLANRETRIEVAL(O_I).
* **Termination** (M4M L9, E-M4M DONE) is PLANRETRIEVAL with all objects as obstacles; E-M4M also stops when the rearrangement clears the initial NGR.
* **MAPF**: all movables are agents; goals are cells where the object's footprint, over its height, misses the NGR; CBS; step cost (1 + 2^(1 - d/0.15)) for M4M (m4m-main) and 1 for E-M4M and I-M4M (em4m-main, im4m-main).
* **Push planner**: shortcut path x_1..x_n, x_aabb from the backward ray, push start x^0_push ~ N(x_aabb, 2.5 cm), later waypoints x^i_push ~ N(x_i, 2.5 cm), tool 3 cm above the shelf, approach planned with all objects as obstacles, IK along every segment. A push is valid if no constraint is violated and at least one object moved (> 1 cm); 2 s settle.
* **M4M**: Algorithm 1 as written: PLANRETRIEVAL(O_I U O_M) before every CBS call, objects tried in random order (as the reference) with one PLANPUSH each, greedy commit to the first valid push, CBS re-run only after a valid push.
* **E-M4M**: Algorithm 1 as written: one PLANPUSH per moving object per expansion; a failed push adds its goal to INVALIDGOALS(v); vertices stay in OPEN and are re-expanded with kappa(v) = all invalid goals, or, when no new invalid goal exists, also the goals of all valid pushes from v; v is closed when CBS fails or kappa cannot grow. Invalid goals are hard-excluded and goals near them penalised through a pseudo-goal (negative cache); valid pushes are cached under (O_m, pi_m) and reused when the moved objects are unchanged and the rest stay clear (positive cache); Naive Bayes priority on (phi1, phi2, phi3); NEG-DB / POS-DB / NO-DB / TIEBREAK ablations.

## Overnight benchmark: every environment as OoI retrieval, 900 s budget

`results/overnight/` (report `report.md`, gallery `index.html`, raw rows `results.csv`); experiment `workspace/experiments/20261024-234342_overnight-900s`. Dataset `reap-retrieve`: the five REAP environments plus the reference shelf, 3 problems per environment and difficulty (Access-19 has 2 hard ones: only 2 of 4 generated met the 3-6 blocker cap), 53 problems; all eight planners with **900 s** of planning each (3x the E-M4M / I-M4M papers' 300 s); 424 runs, every solved plan replayed. A planner's search does not depend on its budget, so success within 60 / 120 / 300 / 600 s is read off the same runs. Run with MuJoCo 3.10 (the project now requires 3.17; see the last notes below).

| Solved and replayed (of 53) | <= 60 s | <= 120 s | <= 300 s | <= 600 s | <= 900 s |
|---|---|---|---|---|---|
| M4M | 16 | 20 | 20 | 20 | 20 |
| E-M4M | 21 | 24 | 24 | 24 | 26 |
| I-M4M | 23 | 24 | 27 | 28 | 28 |
| I-M4M-PnP | 21 | 25 | 26 | 28 | 28 |
| Lazy I-M4M | 23 | 25 | 26 | 26 | 26 |
| Lazy I-M4M-PnP | 22 | 25 | 29 | 30 | **31** |
| DOGAR | 7 | 7 | 9 | 9 | 11 |
| DOGAR (uniform goals) | 3 | 6 | 6 | 8 | 11 |

| Environment (within 300 s -> 900 s) | M4M | E-M4M | I-M4M | I-M4M-PnP | Lazy I-M4M | Lazy I-M4M-PnP | DOGAR | DOGAR (uniform) |
|---|---|---|---|---|---|---|---|---|
| Reference shelf (/9) | 6 -> 6 | 6 -> 6 | 7 -> 7 | 6 -> 6 | 7 -> 7 | 7 -> 7 | 2 -> 2 | 2 -> 2 |
| Open tabletop (/9) | 0 -> 0 | 1 -> 1 | 1 -> 1 | 1 -> 2 | 1 -> 1 | 1 -> 2 | 0 -> 0 | 0 -> 0 |
| Confined shelf (/9) | 5 -> 5 | 6 -> 6 | 6 -> 6 | 5 -> 5 | 6 -> 6 | 6 -> 6 | 2 -> 2 | 1 -> 3 |
| Multi-level shelf (/9) | 4 -> 4 | 5 -> 6 | 5 -> 6 | 5 -> 6 | 5 -> 5 | 6 -> 6 | 2 -> 3 | 1 -> 3 |
| Access-19 (/8) | 0 -> 0 | 0 -> 0 | 0 -> 0 | 3 -> 3 | 0 -> 0 | 3 -> 3 | 0 -> 0 | 0 -> 0 |
| 3D stacking (/9) | 5 -> 5 | 6 -> 7 | 8 -> 8 | 6 -> 6 | 7 -> 7 | 6 -> 7 | 3 -> 4 | 2 -> 3 |

* **More time helps little.** Tripling the budget (the E-M4M and I-M4M papers use 300 s, the M4M paper 120 s) adds 14 solves over all methods (167 -> 181 of 424 runs). Most solves come early: 23 of I-M4M's 28 within 60 s. The M4M-family searches mostly *end* before the budget (E-M4M / I-M4M: "OPEN exhausted", M4M: "CBS failed"), so extra time cannot help them; DOGAR restarts until the budget is spent and gains the most (uniform goals: 6 -> 11).
* **Pick-and-place matters in REAP's environments**: only I-M4M-PnP and Lazy I-M4M-PnP solve any Access-19 problem (3 of 8) or more than one tabletop problem; Lazy I-M4M-PnP solves the most overall (31 of 53).
* **By difficulty** (within 900 s, best method): easy 16/18, medium 12/18, hard 4/17.
* 12 solved plans failed on replay (5 DOGAR), each counted as a failure. One run (I-M4M-PnP, `confined_shelf_retrieve_medium_0001`) crashes deterministically inside MuJoCo 3.10's native convex collision (`mj_collision`, access violation); with `mjDSBL_NATIVECCD` set on the planning model it runs normally (and solves the problem after 702 s; not counted, as the other runs use the default collision code), so it is a MuJoCo bug, recorded as an unsolved CRASH rather than changing the collision code for one run. On one medium scene (`plan_200126`, reference shelf) none of the six M4M-family planners found the initial retrieval (sampling-based PLANRETRIEVAL; the generator's own attempt had succeeded).
* **MuJoCo 3.17** (the project's minimum version since 2026-10-25): the crashing run above completes with the default collision code and solves the problem (about 13 min wall time on a loaded machine). Re-running the 121 overnight runs that solved within 15 s or failed within 60 s under 3.17 (120 s budget) gave the same outcome and number of actions for 114; the 7 that differ are all equal or better: on `plan_200126` E-M4M and Lazy I-M4M now find the initial retrieval and solve it, M4M solves `confined_shelf_retrieve_medium_0000` (was "CBS failed"), E-M4M / I-M4M / DOGAR solve `confined_shelf_retrieve_easy_0002` with 1 push instead of 2 (DOGAR's plan now also replays), and M4M fails `stacking_retrieve_medium_0002` the same way after 2 pushes instead of 3. Collision checks and push simulation changed between the versions, so the tables above are 3.10 numbers; re-run the experiment to get 3.17 ones.
* GIFs: `results/overnight/scenes/<env>/` (all 88 dataset scenes: the camera circles the scene, then the OoI's retrieval with its NGR in purple and the movables in its way in orange) and `results/overnight/solutions/<env>/` (all 193 solved runs replayed, iso + top view; 181 succeed, 12 fail on replay).

## Results so far

`results/benchmark_v8_all.csv`: all eight planners on the same 30 reference scenes (10 per difficulty, chosen at random from `results/scene_labels.csv`, as in v6/v7), one seed, 300 s planning budget each, every solved plan replayed from the initial scene (`--execute`). Solved and replayed / 10, median planning time of the solved runs:

| Difficulty | M4M | E-M4M | I-M4M | I-M4M-PnP | Lazy I-M4M | Lazy I-M4M-PnP | DOGAR | DOGAR (uniform goals) |
|---|---|---|---|---|---|---|---|---|
| Easy | 9 | 10 | 9 | 10 | 10 | 10 | 7 | 6 |
| Medium | 6 | 6 | 7 | 6 | 6 | 5 | 4 | 1 |
| Hard | 5 | 5 | 3 | 4 | 4 | 3 | 0 | 0 |
| **Total** | **20** | **21** | **19** | **20** | **20** | **18** | **11** | **7** |
| Median planning time (s) | 17 | 21 | 20 | 25 | 14 | 10 | 11 | 69 |

* Three solved plans failed on replay (E-M4M and I-M4M on hard scene 200093, I-M4M on easy 100086), each by a pushed object or the arm touching an immovable a few millimetres off the simulated plan.
* The M4M family is within two problems of itself on these 30 scenes; with one seed per scene the differences between them are not significant. DOGAR, which moves each object once and only out of the accumulated NGR, solves about half as many, as in both papers' comparisons.
* Earlier runs (`benchmark_v3.csv`, `benchmark_v5_sidepush.csv`, `benchmark_v6_paper.csv`, `benchmark_v7_im4m.csv`) predate shared fixes (the CBS low level, grasp in simulation and arm tracking rows of the deviations table) and are not directly comparable.

## Deviations and why

| Item | Reference | Here | Why |
|---|---|---|---|
| Robot | PR2 right arm, Bullet | xArm7 + gripper, MuJoCo | project requirement |
| Arm planners | SMPL MHA* / ARA* | RRT-Connect + shortcutting | no SMPL in Python; sampling planner over the same collision model |
| Grasp | horizontal at shelf + 2.5-7.5 cm, any width (Bullet constraint) | pitch 0-30 deg, height 3-12 cm, gripper width enforced | the xArm7 wrist hangs below the tool axis and its stroke is 8.4 cm |
| Lift | 7.5 cm | 7.5 cm capped by ceiling clearance | tall objects otherwise hit the shelf ceiling |
| Push tool pose | pitch 30 deg, tool pointing along the push | that, plus 30 deg rolled 180 deg, 45 deg, and **side-face pushes** (tool yawed +-90 deg, the closed fingers' outer face pushes) | inside the 32 cm tall shelf the xArm7's 25 cm gripper+wrist cannot point along a sideways or -x push without hitting a side wall or the ceiling. No base placement fixes this (`results/reachability_sweep.txt`); side-face pushes raise push coverage from 40 % to 68 % of shelf cells x directions, and -x pushes from 22 % to 69 % (`results/reachability_sweep_sidepush.txt`) |
| Push start | paper: N(x_aabb, 2.5 cm); C++: N(x_aabb - 5 cm, 2.5 cm) with starts within 1 cm of any object rejected, up to 5 samples per MAPF goal | paper (`backoff=0`, `start_clearance_check=False`, `samples=1`); C++ values are config switches | paper. With the xArm7 about 9 in 10 push starts then collide or are unreachable, and in E-M4M each failure costs a new CBS call |
| Push waypoints | paper: x^i_push sampled around each path point x_i; C++: one straight push | paper (`mode="paper"`); `mode="cpp"` gives the C++ push | paper |
| Push retract | paper: next approach planned from the previous push's end; C++: push reversed back to its start | reversed, as in C++ | the push ends touching the object it pushed, so no collision-free approach can start there |
| Valid push | paper: no constraint violated and at least one object moved; C++: the pushed object moved > 1 cm | paper, with a 1 cm threshold | paper |
| Invalid goals | paper: per vertex (kappa(v)); C++: kinematic failures invalid at every vertex, hallucinate one valid goal at a time | paper | paper |
| Push database key | paper: (O_m, pi_m); C++: (object, pose, goal) | paper | paper |
| E-M4M DONE retrieval | paper: PLANRETRIEVAL(v.M U I) whenever v is at the top of OPEN | once per vertex (as in C++) | the paper's MHA* is deterministic, so repeat calls on the same v.M give the same answer; ours is sampling-based and each failing call costs seconds |
| Invalid-goal penalty | em4m-main: unit MAPF steps; pseudo-goal edge += weight / ceil(d) in a +-3 cell box, weight = failed push samples at that goal (SAMPLES = 5 per failed goal) and 1 for a hallucinated (valid) goal | the same, with steps and penalty both scaled by 1000 and weight 5 per failed goal (we sample one push per goal, as the paper) | reference; the paper only says "proportional to the respective penalty" |
| MAPF step cost | m4m-main: (1 + 2^(1 - d/0.15)) per move or wait; em4m-main and im4m-main: 1 | M4M the former, E-M4M and I-M4M the latter (`mapf.step_cost`) | reference; E-M4M paper: the MAPF solver ignores immovable obstacle poses "other than for collision checking" |
| Push database | C++ keeps only the last push per key | keeps all | bug in reference |
| CBS high level | C++: focal search with suboptimality weight 2 | optimal CBS: best-first on cost, ties on conflicts | both papers say CBS, and E-M4M relies on "CBS is an optimal MAPF solver" |
| CBS low level | space-time Dijkstra (h = 0) | space-time A* with the exact static cost-to-go (consistent, so the same optimal paths); agent-agent collisions cached per cell offset | speed only: in Python the root CBS call timed out (30 s) on 7 of 30 scenes; I-M4M's release also runs optimal CBS (weight 1) |
| Priority model | Beta / exponential Naive Bayes fitted by MLE on 2400 self-supervised datapoints (paper); the release has the data collection (`SaveNBData`, `scripts/nb_files.py`) but not the fit | the release's PR2 coefficients | **open**: refitting needs the paper's data collection on xArm7 scenes (breadth-first E-M4M on 20 scenes for 10 min, each vertex re-run for 60 s: about 4 h) |
| Shelf friction | Bullet multiplies frictions; shelf parts get `objects/friction_min`, which `sim.yaml` sets to 0.001 in the M4M and E-M4M releases and 0.5 in the I-M4M release | 0.8 (product rule kept) | **open**: one physics serves every method here, and the two releases disagree; 0.8 is neither (it is `sim.py`'s default, overridden by `sim.yaml`) |
| Budgets | M4M paper 120 s, E-M4M and I-M4M papers 300 s | benchmark default 300 s for every method (`m4m_budget = 120` when `m4m()` is called directly) | one budget per experiment so methods are compared at equal time |
| Interaction pairs exempt at the start of a simulation | none (`checkInteractions`) | none (an earlier version exempted contacts present when a simulation started; no scene starts with one, so it only hid violations in planned states) | reference |
| Carried object during pick-and-place | exempt from the pose and velocity checks (`grasp_at`) | exempt | reference |
| Object velocity check | C++: roll/pitch rate > 2 rad/s | paper: 1 m/s linear | paper |
| E-M4M DONE when the NGR is clear | v.M does not intersect V(gamma_OoI) | the same test on the exact swept volume: gamma_OoI is re-checked against every object | the 1 cm voxels only approximate V(gamma_OoI); a movable just outside them can still graze gamma_OoI |
| Grasp in simulation | Bullet fixed constraint; grasped mass set to 50 g (sim.py) | stiff MuJoCo weld (`solref` 2 timesteps) switched on at the planned in-hand pose *before* the fingers close; grasped mass 50 g as in the reference | MuJoCo's soft default let a tall OoI sag up to 3 cm; closing the fingers first shoved a 1 g OoI 1.4 cm and the weld then flung it into its neighbours |
| Arm tracking | Bullet velocity control along the trajectory | position servos with velocity feedforward | plain PD servos lag 0.085 rad at 0.8 rad/s, enough to clip the shelf on approach |
| Gripper while carrying | n/a | planned with the fingers stopped on the OoI (`Arm.attach`), not fully closed | the partly open knuckles sit elsewhere; a near-ceiling extract planned with closed fingers touched the shelf |

Reference scenes were generated for the PR2. With the xArm7 some have no feasible retrieval at all (`classify_scenes.py` labels them `infeasible`); almost all of these are OoIs wider than the xArm gripper from every approach the shelf allows.
