"""REAP feasibility checker backed by our 6x6 shelf (selected with CONFINED_CHECKER=yadu).

REAP calls check(state, action) -> True if INFEASIBLE. Each problem <stem>.pddl has a geometry sidecar <stem>.json
(roots66_to_pddl.py); one Shelf66Oracle (one MuJoCo env, the dataset teacher's) is built per problem, lazily, in each
worker process.
"""
from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shelf66_oracle import Shelf66Oracle  # noqa: E402

# YADU_TABLE=1 (training only): answers from the per-scene feasibility table (shelf66_table), teacher on a miss.
# Unset (every test-time evaluation): the plain teacher decides every query.
# YADU_LIB=1 (training only): the conflict library (lib66, REAP's library idea built from our teacher), teacher fallback
# for a class not in the library. Unset (every test-time evaluation): the plain teacher decides every query.
if os.environ.get("YADU_LIB") == "1":
    from lib66 import LibOracle as _Oracle  # noqa: E402
elif os.environ.get("YADU_TABLE") == "1":
    from shelf66_table import TableOracle as _Oracle  # noqa: E402
else:
    _Oracle = Shelf66Oracle


class YaduShelfChecker:
    def __init__(self) -> None:
        # LRU of per-problem oracles: training workers interleave problems, and an oracle holds the remembered pick
        # sources of its problem's search (dropping it mid-search would make later puts look infeasible)
        self._oracles: OrderedDict[str, Shelf66Oracle] = OrderedDict()
        self._max = int(os.environ.get("YADU_ORACLE_LRU", 4))

    def init(self, state=None) -> None:
        """Oracles are built on first use per problem."""

    def oracle(self, state) -> Shelf66Oracle:
        path = state.problem.filepath
        if path not in self._oracles:
            side = Path(path).with_suffix(".json")
            if not side.exists():
                raise FileNotFoundError(f"no geometry sidecar {side} for {path} (make problems with roots66_to_pddl.py)")
            self._oracles[path] = _Oracle(json.loads(side.read_text()))
            while len(self._oracles) > self._max:
                _, old = self._oracles.popitem(last=False)
                if hasattr(old, "save"):
                    old.save()                                  # keep what this worker added to the table
        self._oracles.move_to_end(path)
        return self._oracles[path]

    def check(self, state, action) -> bool:
        layout, held = {}, None
        for atom in state.fluent_atoms:
            pname, objs = atom.predicate.name, atom.objects
            if pname == "occupied" and len(objs) == 2:
                layout[objs[1].name] = objs[0].name
            elif pname == "holding" and len(objs) == 1:
                held = objs[0].name
        a = (action.name, *(o.name for o in action.objects))
        return not self.oracle(state).check(layout, held, a)
