"""Experiments: progress, results tables, plots, per-run plan figures and videos."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from common import header, runner, workspace

from mamo.bench import media, report

header("Experiments")
ws = workspace()


with st.expander("Import a results CSV from scripts/run_benchmark.py"):
    csvs = {p.name: p for p in sorted((Path(runner.ROOT) / "results").glob("benchmark_*.csv"))}
    pick = st.selectbox("CSV", list(csvs))
    if pick and st.button("Import"):
        exp = runner.import_benchmark_csv(ws, csvs[pick], Path(pick).stem)
        st.success(f"Imported as {exp.name}")

exps = runner.list_experiments(ws)
if not exps:
    st.info("No experiments yet - create one on the New experiment page.")
    st.stop()


def label(e: Path) -> str:
    s = runner.read_status(e)
    return f"{e.name}  [{s.get('state')} {s.get('done', 0)}/{s.get('total', '?')}]"


by_name = {e.name: e for e in exps}
exp = by_name[st.selectbox("Experiment", list(by_name), format_func=lambda n: label(by_name[n]))]
cfg = runner.read_config(exp)
status = runner.read_status(exp)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
c1.progress(status.get("done", 0) / max(1, status.get("total", 1)),
            text=f"{status.get('state')}: {status.get('done', 0)}/{status.get('total', '?')} runs, "
                 f"{status.get('elapsed', 0) // 60} min")
if status.get("state") == "running":
    if c2.button("Cancel"):
        runner.request_cancel(exp)
    auto = c3.checkbox("Auto-refresh", True)
else:
    auto = False
    if status.get("state") not in ("done", "done (imported)") and c2.button("Resume"):
        runner.launch(ws, exp)
        st.rerun()
c4.button("Refresh")
with st.expander("Configuration"):
    st.json(cfg)
    if status.get("running"):
        st.write("Running now:", status["running"])

rows = runner.read_results(exp)
df = report.frame(rows)
if not df.empty:
    df = report.filter_env_task(df, st)
if df.empty:
    st.info("No finished runs yet.")
else:
    methods = [m["name"] for m in cfg["methods"]]
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Plots", "Failures", "Runs"])
    with tab1:
        st.markdown("**Solved problems** (replay success in the second table)")
        st.dataframe(report.pivot_solved(df, "solved"), use_container_width=True)
        st.dataframe(report.pivot_solved(df, "success"), use_container_width=True)
        env_t = report.by_environment(df)
        if not env_t.empty:
            st.markdown("**Per environment and task** (solved and replayed)")
            st.dataframe(env_t, use_container_width=True)
        st.markdown("**Per difficulty and method** - times over solved runs, min / median / max (as in the papers)")
        st.dataframe(report.summary(df, methods), hide_index=True, use_container_width=True)
        st.download_button("Download all runs (CSV)", df.drop(columns=["stats", "error"], errors="ignore")
                           .to_csv(index=False), file_name=f"{exp.name}.csv")
    with tab2:
        s = report.summary(df, methods)
        if not s.empty:
            st.plotly_chart(px.bar(s, x="difficulty", y="success %", color="method", barmode="group",
                                   title="Success rate (solved and replayed)"), use_container_width=True)
        a = report.anytime(df)
        if not a.empty:
            st.plotly_chart(px.line(a, x="t", y="solved fraction", color="method", facet_col="difficulty",
                                    title="Solved within t seconds of planning"), use_container_width=True)
        sol = df[df["solved"]]
        if not sol.empty:
            st.plotly_chart(px.box(sol, x="difficulty", y="planning_time", color="method", points="all",
                                   title="Planning time of solved runs (s)"), use_container_width=True)
    with tab3:
        st.dataframe(report.failure_reasons(df), hide_index=True, use_container_width=True)
    with tab4:
        cols = [c for c in ("method", "difficulty", "scene", "seed", "solved", "executed", "planning_time",
                            "num_actions", "num_pick_places", "reason", "exec_reason") if c in df]
        f_m = st.multiselect("Methods", methods, default=methods)
        view = df[df["method"].isin(f_m)][cols].sort_values(["scene", "method"])
        st.dataframe(view, hide_index=True, use_container_width=True, height=300)
        run_ids = [r["id"] for r in rows if r.get("method") in f_m]
        rid = st.selectbox("Run details", run_ids)
        r = next((x for x in rows if x["id"] == rid), None)
        if r:
            st.json({k: v for k, v in r.items() if k != "error"})
            if r.get("error"):
                st.code(r["error"])
            solf = exp / "solutions" / r.get("solution_file", "-")
            if r.get("solved") and solf.exists():
                c1, c2 = st.columns(2)
                if c1.button("Plan figure"):
                    with st.spinner("Replaying..."):
                        st.image(str(media.plan_figure(solf, exp / "media" / (solf.stem + "_plan.png"))))
                if c2.button("Render video"):
                    with st.spinner("Replaying and rendering (about a minute)..."):
                        st.video(str(media.video(solf, exp / "media" / (solf.stem + ".mp4"))))

if auto:
    time.sleep(5)
    st.rerun()
