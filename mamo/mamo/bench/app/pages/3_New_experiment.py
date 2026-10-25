"""New experiment: methods x datasets x difficulties x seeds."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import header, planners, runner, workspace

from mamo.bench import datasets as D

header("New experiment", "Runs in the background; follow it on the Experiments page.")
ws = workspace()
reg, errors = planners()
dss = D.list_datasets(ws)

name = st.text_input("Name", "comparison")
fams = sorted({s.family for s in reg.values()})
c1, c2 = st.columns(2)
methods = c1.multiselect("Methods", list(reg), default=[m for m in ("m4m", "em4m", "im4m", "dogar") if m in reg],
                         format_func=lambda m: f"{m}  [{reg[m].family}; {'/'.join(reg[m].tasks)}]")
datasets = c2.multiselect("Datasets", list(dss), default=[d for d in ("reference-10",) if d in dss])

if methods:
    st.markdown("Budgets and parameters per method (seconds; parameters as JSON, merged with the defaults)")
    table = pd.DataFrame([{"method": m, "budget": reg[m].budget, "params": ""} for m in methods])
    table = st.data_editor(table, hide_index=True, disabled=["method"], use_container_width=True)

# environments / tasks present in the chosen datasets' labels (generated datasets)
envs_tasks = {(r.get("env") or "", r.get("task") or "") for d in datasets for r in D.load_labels(ws, dss[d])}
all_envs = sorted({e for e, _ in envs_tasks if e})
sel_envs = sel_tasks = None
if all_envs:
    c1, c2 = st.columns(2)
    sel_envs = c1.multiselect("Environments", all_envs, default=all_envs)
    all_tasks = sorted({t for _, t in envs_tasks if t})
    sel_tasks = c2.multiselect("Tasks", all_tasks, default=all_tasks)
c1, c2, c3, c4 = st.columns(4)
diffs = c1.multiselect("Difficulties", ["easy", "medium", "hard", "trivial", "infeasible", "all"],
                       default=["easy", "medium", "hard"])
per = c2.number_input("Scenes per difficulty (0 = all)", 0, 10000, 10,
                      help="for generated datasets: per environment, task and difficulty")
sample_seed = c3.number_input("Scene sampling seed", 0, 10 ** 6, 0)
seeds_txt = c4.text_input("Planner seeds", "0", help="comma separated, e.g. 0,1,2")
c1, c2, c3, c4 = st.columns(4)
workers = c1.number_input("Parallel runs", 1, 64, 6, help="each run is one process; robust_im4m uses 6 threads")
execute = c2.checkbox("Replay solved plans (scores success)", True)
save = c3.checkbox("Save solutions (plan figures, videos)", True)
timeout_factor = c4.number_input("Kill a run after budget x", 1.0, 10.0, 2.0, help="+180 s")

seeds = [int(s) for s in seeds_txt.replace(" ", "").split(",") if s != ""]
if methods and datasets and seeds:
    import json
    mlist = []
    for _, r in table.iterrows():
        try:
            params = json.loads(r["params"]) if str(r["params"]).strip() else {}
        except json.JSONDecodeError:
            st.error(f"Parameters of {r['method']} are not valid JSON")
            st.stop()
        mlist.append({"name": r["method"], "budget": float(r["budget"]), "params": params})
    cfg = {"methods": mlist, "datasets": datasets, "difficulties": diffs, "per_difficulty": per or None,
           "seeds": seeds, "execute": execute, "save_solutions": save, "sample_seed": sample_seed,
           "envs": sel_envs, "tasks": sel_tasks}
    skipped: list = []
    jobs = runner.build_jobs(ws, cfg, skipped)
    if skipped:
        by = pd.Series([m for m, _ in skipped]).value_counts()
        st.caption("Not run (scene task not supported by the method): "
                   + ", ".join(f"{m} on {n} scenes" for m, n in by.items()))
    if not jobs:
        st.warning("No scenes match (label the dataset first, or choose 'all').")
        st.stop()
    worst = sum(j["budget"] for j in jobs) / workers / 3600
    st.info(f"{len(jobs)} runs on {len({(j['dataset'], j['scene']) for j in jobs})} scenes ({len(methods)} methods, "
            f"{len(seeds)} seeds); at most ~{worst:.1f} h with {workers} parallel runs if nothing is solved early.")
    if st.button("Start", type="primary"):
        exp = runner.new_experiment(ws, name, mlist, datasets, diffs, per or None, seeds, int(workers), execute,
                                    save, sample_seed, float(timeout_factor), sel_envs, sel_tasks)
        runner.launch(ws, exp)
        st.success(f"Started {exp.name}. Open the Experiments page to follow it.")
