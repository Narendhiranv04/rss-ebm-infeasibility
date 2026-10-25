"""All tunable parameters in one place, each annotated with its source.

Source tags:  [paper]  stated in the M4M / E-M4M papers
              [cpp]    value used by the reference implementation (file noted)
              [xarm]   adaptation required by the xArm7 + MuJoCo substitution
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhysicsConfig:
    timestep: float = 0.002                 # [xarm] MuJoCo step; reference uses Bullet at 140 Hz
    shelf_mu: float = 0.8                   # [cpp] sim.py AddObject: shelf parts get friction_min (0.8 default)
    robot_mu: float = 1.0                   # [cpp] Bullet default lateral friction for robot links
    combine_friction_as_product: bool = True  # [cpp] Bullet multiplies friction coefficients
    settle_time: float = 2.0                # [cpp] sim.py SimPushes: 480 steps at 240 Hz
    gravity_compensation: bool = True       # [xarm] arm bodies compensated, tracked by position servos
    grasped_mass: float = 0.05              # [cpp] sim.py: a grasped object's mass is set to 50 g while held


@dataclass
class ConstraintConfig:
    """Interaction constraints (paper Sec. III; E-M4M 'Problem Formulation')."""
    max_tilt_deg: float = 25.0              # [paper] movable objects may not tilt beyond 25 deg
    max_speed: float = 1.0                  # [paper] or move faster than 1 m/s
    immovable_contact_forbidden: bool = True  # [paper] neither robot nor any object may touch immovables
    robot_shelf_contact_forbidden: bool = True  # [cpp] sim.py checkTableCollision
    contact_depth_tol: float = 1e-4         # [xarm] penetration that counts as contact (cpp: 1e-9 in Bullet)
    check_every: int = 2                    # [xarm] check constraints every N physics steps


@dataclass
class RobotConfig:
    base_pos: tuple[float, float, float] = (0.30, -0.50, 0.43)  # [xarm] best of scripts/reachability.py sweep
    base_yaw: float = 0.0
    # [xarm] folded outside the shelf (all robot geoms at x < 0.45), TCP at (0.40, -0.5, 1.05) facing +x
    home_q: tuple[float, ...] = (0.0, -1.922, 0.0, 0.636, 0.0, 0.987, 0.0)
    max_joint_speed: float = 0.8            # [xarm] rad/s for free-space motion
    push_speed: float = 0.10                # [cpp] robot/speed: 0.1 m/s
    collision_check_step: float = 0.04      # [xarm] max joint-space step between collision checks (rad)
    planning_margin: float = 0.005          # [xarm] clearance required by the motion planner (m)


@dataclass
class RetrievalConfig:
    """PLANRETRIEVAL: approach, grasp, lift and extract the OoI (M4M Alg. 1 L4/L9, E-M4M Alg. 1 L7/L28)."""
    grasp_tries: int = 20                   # [cpp] robot/grasping/tries (100) capped for Python speed
    # [cpp] robot.cpp ComputeGrasps samples table + U(2.5, 7.5) cm for the PR2. [xarm] The xArm7 wrist
    # hangs below the tool axis, so grasps are sampled higher (capped 2 cm below the OoI top) and pitched.
    grasp_height_range: tuple[float, float] = (0.03, 0.12)
    grasp_top_margin: float = 0.02          # [xarm] grasp at least this far below the OoI top
    approach_pitch_range_deg: tuple[float, float] = (0.0, 30.0)  # [xarm]
    ik_restarts: int = 12                   # [xarm]
    pregrasp_backoff: float = 0.10          # [xarm] distance of the pre-grasp pose before the grasp pose
    approach_yaw_range_deg: float = 60.0    # [xarm] grasp approach yaw sampled around +x
    lift: float = 0.075                     # [cpp] robot/grasping/lift
    gripper_max_width: float = 0.084        # [xarm] xArm gripper stroke
    plan_time: float = 5.0                  # [cpp] 2 s retrieval, 2x for retract (pr2.launch); Python allowance
    max_rrt_iters: int = 4000


@dataclass
class PushConfig:
    """PLANPUSH (M4M Sec. IV-B, E-M4M 'Nonprehensile Push Planner')."""
    mode: str = "paper"                     # "paper": x^i_push ~ N(x_i, sigma I) around every shortcut point
                                            # "cpp": single straight push front->back + overshoot (robot.cpp)
    sigma: float = 0.025                    # [paper] x_push ~ N(x, sigma I), sigma = 2.5 cm
    height_above_shelf: float = 0.03        # [paper] push z fixed 3 cm above the shelf
    backoff: float = 0.0                    # [paper] x^0_push ~ N(x_aabb, sigma I); robot.cpp backs off 5 cm
    start_clearance_check: bool = False     # [cpp] robot.cpp rejects push starts within 1 cm of any object
    overshoot: float = 0.05                 # [cpp] used only in "cpp" mode
    tool_pitch_deg: float = 30.0            # [cpp] robot.cpp: tool pitched 30 deg down
    # [xarm] tool orientations tried in random order: (pitch deg, gripper rolled 180 deg,
    # tool yaw relative to the push direction deg). The first is the reference pose. With a
    # yaw offset the hand points across the push and the closed fingers' outer face pushes:
    # the 25 cm gripper+wrist cannot point along a sideways or -x push inside the 32 cm
    # tall shelf without hitting a wall or the ceiling (scripts/reachability.py).
    tool_alternatives: tuple[tuple[float, bool, float], ...] = (
        (30.0, False, 0.0), (30.0, True, 0.0), (45.0, False, 0.0), (45.0, True, 0.0),
        (30.0, False, 90.0), (30.0, False, -90.0))
    samples: int = 1                        # [paper] one PLANPUSH per moving object per expansion
                                            # (robot.cpp: robot/pushing/samples = 5)
    plan_time: float = 1.0                  # [cpp] 0.2 s approach planning; Python allowance
    max_rrt_iters: int = 1500
    min_displacement: float = 0.01          # [paper] valid push: at least one object moved (here > 1 cm,
                                            # the MAPF grid); robot.cpp: the pushed object must move
    retract_by_reversing: bool = True       # [cpp] the push is reversed back to its start pose. The papers
                                            # plan each approach from the previous push's end, but that end
                                            # touches the pushed object, so a collision-free approach needs it.


@dataclass
class MAPFConfig:
    res: float = 0.01                       # [paper] 1 cm grid
    connectivity: int = 4                   # [paper] four-connected + wait
    cost_mult: int = 1000                   # [cpp] COST_MULT
    # Step cost: "proximity" = (1 + 2^(1 - d/0.15)) per move or wait ([cpp] m4m-main agent_lattice.cpp);
    # "unit" = 1 per move or wait ([cpp] em4m-main / im4m-main agent_lattice.cpp; E-M4M paper: the
    # MAPF solver ignores immovable obstacle poses "other than for collision checking").
    # M4M uses proximity; E-M4M and I-M4M switch to unit.
    step_cost: str = "proximity"
    obstacle_cost_scale: float = 0.15       # [cpp] agent_lattice.cpp: 2^(1 - d/0.15)
    cbs_time_limit: float = 30.0            # [cpp] cbs.cpp
    lowlevel_time_limit: float = 30.0       # [cpp] focal.cpp
    max_time_steps: int = 400               # [xarm] horizon for the space-time low-level search


@dataclass
class NGRConfig:
    res: float = 0.01                       # [cpp] 1 cm voxels (pr2_right_arm.yaml)
    shrink: float = 0.01                    # [cpp] robot spheres shrunk by RES before voxelising
    traj_step: float = 0.01                 # [xarm] TCP spacing when sampling the swept volume (m)


@dataclass
class SearchConfig:
    m4m_budget: float = 120.0               # [paper] M4M: 120 s planning timeout
    em4m_budget: float = 300.0              # [paper] E-M4M: 5 min planning timeout
    seed: int = 0
    # E-M4M ablations (paper Table 2)
    use_negative_cache: bool = True         # NEG-DB keeps this, drops positive
    use_positive_cache: bool = True         # POS-DB keeps this, drops negative
    priority: str = "learned"               # "learned" (Naive Bayes) or "tiebreak" (lexicographic features)
    hash_res_xy: float = 0.01               # [cpp] problem.yaml objects/discretisation
    hash_res_theta: float = 0.0872665       # [cpp] 5 deg
    invalid_goal_box: int = 3               # [cpp] agent_lattice.cpp checkNNCost: +-3 cells
    invalid_goal_weight: int = 5            # [cpp] agent_lattice.cpp: a goal whose SAMPLES = 5 pushes failed
    im4m_budget: float = 300.0              # [paper] I-M4M: 5 min planning timeout


@dataclass
class PickPlaceConfig:
    """I-M4M prehensile rearrangement (I-M4M paper Sec. IV-A, robot.cpp PlanPickPlace)."""
    # [cpp] graspable if the footprint fits the gripper along some axis (GRIPPER_WIDTH_2 = 4.5 cm
    # half-width on the PR2); [xarm] half of the 8.4 cm stroke, less 3 mm clearance per side.
    max_half_width: float = 0.039
    pregrasp_offset: float = 0.075          # [cpp] Object::initPregrasps: 7.5 cm from the object face
    lift: float = 0.02                      # [cpp] post-grasp lift 2 cm
    lower: float = 0.01                     # [cpp] place: lower 1 cm before release
    retract: float = 0.10                   # [cpp] retract 10 cm back along the tool direction
    pitches_deg: tuple[float, ...] = (0.0, 15.0, 30.0)  # [xarm] cpp grasps are horizontal; see retrieval
    # [xarm] the reference puts the PR2's end-effector at the object centre (robot.cpp PlanPickPlace);
    # the xArm7's fingertips reach this far past the near face (or to the centre of thin objects)
    finger_depth: float = 0.035
    # [xarm] grasp directions further than this from +x (towards the shelf back) are not reachable
    max_approach_yaw_deg: float = 90.0
    place_yaws: int = 8                     # [cpp] place yaw is free (2 pi tolerance): 8 tool yaws tried
    release_time: float = 0.5               # [xarm] gripper opening before the retract
    settle_time: float = 1.0                # [cpp] sim.py holds 1 s (HZ steps) after the action
    plan_time: float = 1.0                  # [cpp] 0.5 s pick/place planning; Python allowance
    max_rrt_iters: int = 1500
    # [ext] keep only grasps whose straight reach in is collision-free before planning the approach
    # (off: as I-M4M, the approach planner picks a pre-grasp and only that grasp is tried)
    reach_first: bool = False


@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    push: PushConfig = field(default_factory=PushConfig)
    mapf: MAPFConfig = field(default_factory=MAPFConfig)
    ngr: NGRConfig = field(default_factory=NGRConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    pick_place: PickPlaceConfig = field(default_factory=PickPlaceConfig)
