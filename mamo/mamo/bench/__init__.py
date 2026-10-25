"""MAMO benchmark tool: planner registry, datasets, experiment runner, reports and a web app.

    python -m mamo.bench app          # open the web app
    python -m mamo.bench --help       # command-line use
"""

from mamo.bench.planners import REGISTRY, PlannerSpec, load_plugins, register

__all__ = ["REGISTRY", "PlannerSpec", "load_plugins", "register"]
