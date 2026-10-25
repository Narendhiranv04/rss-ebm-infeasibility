"""Compare: several experiments in one table (only scenes common to all selected methods are kept)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from common import header, runner, workspace

from mamo.bench import report

header("Compare experiments", "Combine runs of several experiments, e.g. your method's experiment with the "
       "baselines' one on the same scenes.")
ws = workspace()
exps = {e.name: e for e in runner.list_experiments(ws)}
sel = st.multiselect("Experiments", list(exps))
if not sel:
    st.stop()
rows = []
for e in (exps[n] for n in sel):
    for r in runner.read_results(e):
        r = dict(r)
        r["experiment"] = e.name
        rows.append(r)
df = report.frame(rows)
if df.empty:
    st.stop()
df = report.filter_env_task(df, st)
methods = list(dict.fromkeys(df["method"]))
methods = st.multiselect("Methods", methods, default=methods)
df = df[df["method"].isin(methods)]
if st.checkbox("Only scenes every selected method ran on", True):
    per = df.groupby(["dataset", "scene"])["method"].nunique()
    common = set(per[per == len(methods)].index)
    df = df[[k in common for k in zip(df["dataset"], df["scene"])]]
st.caption(f"{df[['dataset', 'scene']].drop_duplicates().shape[0]} scenes, {len(df)} runs")
st.dataframe(report.pivot_solved(df, "success"), use_container_width=True)
env_t = report.by_environment(df)
if not env_t.empty:
    st.dataframe(env_t, use_container_width=True)
st.dataframe(report.summary(df, methods), hide_index=True, use_container_width=True)
s = report.summary(df, methods)
if not s.empty:
    st.plotly_chart(px.bar(s, x="difficulty", y="success %", color="method", barmode="group"),
                    use_container_width=True)
