"""Planner registry: the built-in methods and user plugins.

A planner is a function ``fn(ctx, budget, **params) -> Solution``:

* ``ctx`` (MAMOContext) is a fresh planning context for one scene: MuJoCo world and
  simulator, planning arm, push planner, PLANRETRIEVAL, NGR, MAPF and timers
  (see plugins/example_planner.py)
* ``budget`` is the planning time limit in seconds
* the returned Solution lists the rearrangement actions (PushAction / PickPlaceAction)
  in execution order and the final RetrievalPlan, so every method is replayed and scored
  the same way (mamo.execute)
* ``tasks`` lists the scene tasks the method solves: "retrieve" (take the OoI out, the M4M
  problem) and/or "relocate" (bring targets to goal poses, mamo.envs); an experiment only
  runs a method on scenes of its tasks

Register with the decorator::

    from mamo.bench import register

    @register("my_method", description="...", budget=300, tasks=("retrieve", "relocate"))
    def my_method(ctx, budget, **params): ...

Plugins are the .py files in the plugins folder(s); they are imported by the app and by
every benchmark worker.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from mamo.config import Config
from mamo.context import MAMOContext
from mamo.scene import Scene


@dataclass
class PlannerSpec:
    name: str
    fn: Callable
    description: str = ""
    family: str = "user"
    budget: float = 300.0                      # default planning budget (s)
    params: dict = field(default_factory=dict)  # default keyword arguments for fn
    # Optional hooks for methods that need a special setup:
    configure: Callable[[Config], None] | None = None                      # edit the Config
    make_context: Callable[[Scene, Config, int], MAMOContext] | None = None  # build the context
    source: str = ""                           # file it was loaded from
    tasks: tuple[str, ...] = ("retrieve",)     # scene tasks it solves


REGISTRY: dict[str, PlannerSpec] = {}


def register(name: str, description: str = "", family: str = "user", budget: float = 300.0,
             configure=None, make_context=None, tasks=("retrieve",), **params):
    """Decorator registering ``fn(ctx, budget, **params) -> Solution`` under ``name``."""
    def deco(fn):
        src = getattr(sys.modules.get(fn.__module__), "__file__", "") or ""
        REGISTRY[name] = PlannerSpec(name, fn, description, family, budget, params, configure, make_context, src,
                                     tuple(tasks))
        return fn
    return deco


def load_plugins(dirs: list[str | Path]) -> list[str]:
    """Import every plugins/*.py (each registers its planners). Returns import errors."""
    errors = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_"):
                continue
            mod = f"mamo_plugin_{f.stem}"
            try:
                spec = importlib.util.spec_from_file_location(mod, f)
                m = importlib.util.module_from_spec(spec)
                sys.modules[mod] = m
                spec.loader.exec_module(m)
            except Exception as exc:  # a broken plugin must not take the tool down
                errors.append(f"{f.name}: {type(exc).__name__}: {exc}")
    return errors


# --------------------------------------------------------------- built-ins
def _m4m(ctx, budget, one_shot=False):
    from mamo.m4m.planner import m4m
    return m4m(ctx, budget, one_shot=one_shot)


def _em4m(ctx, budget):
    from mamo.em4m.planner import em4m
    return em4m(ctx, budget)


def _im4m(ctx, budget, pick_place=False, lazy=False):
    from mamo.im4m.planner import im4m
    return im4m(ctx, budget, pick_place=pick_place, lazy=lazy, robust=getattr(ctx, "robust", None))


def _dogar(ctx, budget, goal_sampling="closest", samples=10):
    from mamo.dogar.planner import dogar
    return dogar(ctx, budget, goal_sampling=goal_sampling, samples=samples)


def _rrt(ctx, budget, goal_bias=0.05, range_frac=0.2):
    from mamo.rrt.planner import rrt
    return rrt(ctx, budget, goal_bias=goal_bias, range_frac=range_frac)


def _kpiece(ctx, budget, goal_bias=0.05, range_frac=0.2):
    from mamo.kpiece.planner import kpiece
    return kpiece(ctx, budget, goal_bias=goal_bias, range_frac=range_frac)


def _selsim(ctx, budget):
    from mamo.selsim.planner import selsim
    return selsim(ctx, budget)


def _caches(neg: bool, pos: bool):
    def f(cfg: Config) -> None:
        cfg.search.use_negative_cache, cfg.search.use_positive_cache = neg, pos
    return f


def _tiebreak(cfg: Config) -> None:
    cfg.search.priority = "tiebreak"


def _uncertain_context(robust: bool):
    """I-M4M Sec. V-A second experiment: plan with sampled masses/frictions (execution uses the truth)."""
    def make(scene: Scene, cfg: Config, seed: int) -> MAMOContext:
        import numpy as np
        from mamo.im4m.robust import RobustSim, sample_scene
        rng = np.random.default_rng(1000 + seed)
        ctx = MAMOContext(sample_scene(scene, rng), cfg, seed=seed)
        ctx.robust = RobustSim(scene, cfg, rng) if robust else None
        return ctx
    return make


_P = "Saxena & Likhachev"
register("m4m", f"M4M ({_P}, ICRA 2023): greedy, CBS re-run after each valid push", "M4M")(_m4m)
register("m4m_hat", "M4M-hat: a single CBS call (M4M paper Table I)", "M4M", one_shot=True)(_m4m)
register("em4m", f"E-M4M ({_P}, ICAPS 2023): best-first search with invalid goals, push DB, learned priority",
         "E-M4M")(_em4m)
register("em4m_negdb", "E-M4M NEG-DB ablation: only failed pushes cached", "E-M4M",
         configure=_caches(True, False))(_em4m)
register("em4m_posdb", "E-M4M POS-DB ablation: only valid pushes cached", "E-M4M",
         configure=_caches(False, True))(_em4m)
register("em4m_nodb", "E-M4M NO-DB ablation: nothing cached", "E-M4M", configure=_caches(False, False))(_em4m)
register("em4m_tiebreak", "E-M4M TIEBREAK ablation: lexicographic priority", "E-M4M", configure=_tiebreak)(_em4m)
register("im4m", f"I-M4M ({_P}, ICRA 2024): depth-first E-M4M, pushes only", "I-M4M")(_im4m)
register("im4m_pnp", "I-M4M-PnP: pushes and pick-and-place", "I-M4M", pick_place=True)(_im4m)
register("lazy_im4m", "Lazy I-M4M: eagerly lazy edge evaluation, pushes only", "I-M4M", lazy=True)(_im4m)
register("lazy_im4m_pnp", "Lazy I-M4M-PnP: lazy, pushes and pick-and-place", "I-M4M", pick_place=True,
         lazy=True)(_im4m)
register("robust_im4m", "Robust I-M4M: pushes validated in 6x2 simulations with sampled mass/friction "
         "(plans with sampled parameters; replay uses the true ones)", "I-M4M",
         make_context=_uncertain_context(True))(_im4m)
register("tilde_im4m", "I~-M4M: plans with one sampled copy of each object's mass/friction", "I-M4M",
         make_context=_uncertain_context(False))(_im4m)
register("dogar", "DOGAR (Dogar & Srinivasa, RSS 2011): backward recursive rearrangement; goals near the "
         "closest free cell (M4M paper)", "Baseline")(_dogar)
register("dogar_uniform", "DOGAR with goals sampled uniformly outside the NGR (E-M4M paper)", "Baseline",
         goal_sampling="uniform")(_dogar)
register("rrt", "RRT (LaValle & Kuffner, IJRR 2001): physics-based RRT over arm + object states, motions "
         "simulated when they touch movables (M4M papers' OMPL baseline)", "Baseline")(_rrt)
register("kpiece", "KPIECE (Sucan & Kavraki, T-RO 2012): interior-exterior cell exploration on the TCP "
         "projection, physics-based motions (M4M papers' OMPL baseline)", "Baseline")(_kpiece)
register("selsim", "SELSIM (Saleem & Likhachev, ICRA 2020): Lazy WA* over single-joint primitives, simulating "
         "only relevant objects, found by tracking the plan with all objects", "Baseline")(_selsim)

BUILTINS = set(REGISTRY)
