"""MAMO benchmark - home page."""

from __future__ import annotations

from common import header, planners, runner, workspace

import streamlit as st

header("MAMO benchmark", "Manipulation among movable objects: M4M, E-M4M, I-M4M, DOGAR and your own planners "
       "on the reference shelf scenes and on REAP's five environments (retrieve and relocate tasks), "
       "xArm7 in MuJoCo")

ws = workspace()
reg, errors = planners()
from mamo.bench.datasets import list_datasets, load_labels  # noqa: E402

dss = list_datasets(ws)
exps = runner.list_experiments(ws)
c1, c2, c3 = st.columns(3)
c1.metric("Planners", len(reg))
c2.metric("Datasets", len(dss))
c3.metric("Experiments", len(exps))
for e in errors:
    st.error(f"Plugin failed to load: {e}")

st.markdown(f"""
**Workflow**

1. **Datasets** – the reference scenes are built in; generate a dataset in the REAP environments
   (open tabletop, confined shelf, multi-level shelf, Access-19, 3D stacking; retrieve / relocate;
   easy / medium / hard), or add a folder of your own scene files (same text format).
2. **Methods** – built-in planners (retrieve tasks) and your plugins (`.py` files in
   `{runner.DEFAULT_PLUGINS[0]}`; start from `example_planner.py`, which has a retrieve and a
   relocate example).
3. **New experiment** – pick methods, datasets, environments, tasks, difficulties, number of scenes,
   seeds and budgets; it runs in the background (you can close this page).
4. **Experiments** – progress, results tables in the style of the papers, plots, and per-run
   plan figures and videos.
5. **Compare** – put several experiments side by side.

Workspace: `{ws}`
""")

running = [e for e in exps if runner.read_status(e).get("state") == "running"]
if running:
    st.subheader("Running")
    for e in running:
        s = runner.read_status(e)
        st.progress(s.get("done", 0) / max(1, s.get("total", 1)), text=f"{e.name}: {s.get('done')}/{s.get('total')}")
