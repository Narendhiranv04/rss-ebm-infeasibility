"""Slim runner for grid_dataset.py: edit the settings below, then run  python run_task.py

Cells use 1-based numbering: [row, col] with 1..8 (row 1 = front, col 1 = left), or an id 1..64.
"""
import grid_dataset as G

# ---------------------------------------------------------------- your settings
START = [3, 7]                 # target's start cell
GOAL = [8, 7]                  # target's goal cell
PLAN_ONLY = False              # True: only decide feasible / infeasible, no viewer and no simulation
SEED = 0

# target: shape (box|cylinder|sphere|capsule), size (radius; box = half-size), height (m), colour
TARGET = dict(shape="cylinder", size=0.013, height=0.1, color="green")

# obstacles: shape, size (box: half-size or [hx, hy, hz]; others: radius), height (m, not for sphere),
# colour (name or [r, g, b, a]), cell ([row, col] or id), optional dx / dy offsets (m) from the cell centre
OBSTACLES = [
    dict(shape="box",   size=0.012,  height=0.2,   color="blue",   cell=[6, 1]),
    dict(shape="box",   size=0.012,  height=0.1,   color="blue",  cell=[4, 6]),
    dict(shape="box",   size=0.013,  height=0.24,  color="blue",  cell=[6, 8]),
    dict(shape="box",   size=0.013,  height=0.16,  color="blue",  cell=[2, 5]),
    dict(shape="box",   size=0.0125, height=0.13,  color="blue",  cell=[5, 2]),
    dict(shape="box",   size=0.0125, height=0.14,  color="blue",  cell=[3, 3]),
    dict(shape="box",   size=0.0125, height=0.14,  color="blue",  cell=[4, 5])
]
# ----------------------------------------------------------------------------


def main():
    start, goal = G.user_rc(START), G.user_rc(GOAL)
    obstacles = G.obstacles_from_items(OBSTACLES)
    env = G.build_env(start, obstacles, TARGET, seed=SEED)
    planner = G.GridPlanner(env, seed=SEED)
    print(f"{len(obstacles)} obstacles, task: cell {G.cell_id(start) + 1} -> {G.cell_id(goal) + 1}")
    for ob in obstacles:
        print(f"   {ob['name']:<10} {ob['shape']:<8} cell {G.cell_id(ob['cell']) + 1:>2}  "
              f"{ob['height']*100:4.1f} cm tall")
    if not PLAN_ONLY:
        env.launch_viewer()
    planner.place_target(start)
    res = planner.solve(start, goal)
    print(G.describe(res))
    if res["feasible"] and not PLAN_ONLY:
        planner.execute(res)
        ex = res["exec"]
        print(f"executed: {'VERIFIED' if res['verified'] else 'NOT VERIFIED'} (error {ex['error_cm']:.1f} cm, "
              f"obstacle contacts: {ex['obstacle_contacts'] or 'none'})")
    if env.viewer is not None:
        while env.viewer.is_running():
            env.step(10)
    return res


if __name__ == "__main__":
    main()
