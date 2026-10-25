"""Parallelised simulations for robustness to parameter uncertainty (I-M4M paper Sec. IV-D).

Each movable object's mass m and friction coefficient mu are uncertain. Every simulated
copy samples

    m_hat  = min(2 m,  max(m / 2,  N(m, 0.2 m)))
    mu_hat = min(2 mu, max(mu / 2, N(mu, 0.5 mu)))

Robust I-M4M evaluates an action in N simulators with M copies each (N M independent
samples) and accepts it if at least max(1, floor(delta N M)) samples are valid; the
successor state is taken from the first valid sample (sim.py getValidObjects). The paper
uses N = 6, M = 2, delta = 0.9 (10 of 12 samples).

The reference keeps M copies of each object in one Bullet world, separated by collision
groups. Here each sample is its own MuJoCo world (same thing: copies never interact),
stepped on N threads (MuJoCo releases the GIL while stepping, but the per-step Python work
does not, so the speed-up is smaller than the reference's processes). In the reference the
first copy in each simulator samples mu with standard deviation mu / 2 and further copies
with 0.2 mu (sim.py); the paper's N(mu, 0.5 mu) is used for every sample.

I~-M4M (the paper's baseline for this experiment) plans with a single sampled copy:
build its context on ``sample_scene(scene, rng)``.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from mamo.config import Config
from mamo.robot.arm import Arm
from mamo.scene import Scene
from mamo.simulate import SimResult, Simulator, simulate_push
from mamo.world import Snapshot, World


def sample_scene(scene: Scene, rng: np.random.Generator) -> Scene:
    """A copy of ``scene`` whose movable objects have sampled masses and friction coefficients."""
    objs = []
    for o in scene.objects:
        if o.movable and not o.is_shelf:
            m = float(min(2 * o.mass, max(o.mass / 2, rng.normal(o.mass, 0.2 * o.mass))))
            mu = float(min(2 * o.mu, max(o.mu / 2, rng.normal(o.mu, 0.5 * o.mu))))
            o = replace(o, mass=m, mu=mu)
        objs.append(o)
    return Scene(objs, scene.ooi_id, scene.name, scene.meta)  # meta: robot base, task, overrides


class RobustSim:
    def __init__(self, scene: Scene, cfg: Config, rng: np.random.Generator, n_sims: int = 6,
                 copies: int = 2, delta: float = 0.9) -> None:
        self.worlds = [World(sample_scene(scene, rng), cfg) for _ in range(n_sims * copies)]
        self.cfg = self.worlds[0].cfg  # with the scene's robot base and parameter overrides
        self.arms = [Arm(w) for w in self.worlds]
        self.sims = [Simulator(w, self.cfg) for w in self.worlds]
        self.needed = max(1, math.floor(delta * n_sims * copies))
        self.pool = ThreadPoolExecutor(n_sims)

    def simulate_push(self, snapshot: Snapshot, action) -> SimResult:
        def job(i: int) -> SimResult:
            self.worlds[i].restore(snapshot)
            return simulate_push(self.sims[i], self.arms[i], action, self.cfg)

        results = list(self.pool.map(job, range(len(self.worlds))))
        valid = [r for r in results if r.valid]
        if len(valid) >= self.needed:
            out = valid[0]
            out.reason = f"{len(valid)}/{len(results)} samples valid"
            return out
        bad = next(r for r in results if not r.valid)
        return SimResult(False, f"only {len(valid)}/{len(results)} samples valid (first failure: {bad.reason})",
                         bad.object_states, sim_time=bad.sim_time, arm_q=bad.arm_q)

    def close(self) -> None:
        self.pool.shutdown(wait=False)
