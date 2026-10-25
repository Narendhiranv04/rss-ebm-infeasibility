"""Methods: built-in planners and plugins."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from common import header, planners, runner, workspace

header("Methods", "Planners available to experiments. Your own method is a plugin: a .py file in a plugins "
       "folder that registers a function plan(ctx, budget, **params) -> Solution. A method runs only on scenes "
       "of the tasks it declares (retrieve / relocate).")
ws = workspace()
if st.button("Reload plugins"):
    planners(reload=True)
reg, errors = planners()
for e in errors:
    st.error(f"Plugin failed to load: {e}")

st.dataframe(pd.DataFrame([{"name": s.name, "family": s.family, "tasks": ", ".join(s.tasks),
                            "default budget (s)": s.budget,
                            "parameters": json.dumps(s.params) if s.params else "", "description": s.description,
                            "file": Path(s.source).name if s.family == "user" else ""}
                           for s in reg.values()]), hide_index=True, use_container_width=True)

st.subheader("Plugin folders")
dirs = runner.plugin_dirs(ws)
for d in dirs:
    st.code(str(d))
with st.form("dirs"):
    extra = st.text_input("Add another plugin folder", placeholder=r"D:\my_method\plugins")
    if st.form_submit_button("Add folder") and extra:
        f = ws / "plugin_dirs.json"
        cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        if extra not in cur:
            cur.append(extra)
        f.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        planners(reload=True)
        st.rerun()

st.subheader("Writing a plugin")
example = runner.DEFAULT_PLUGINS[0] / "example_planner.py"
st.markdown("Start from the example (it is a working planner). The tool replays every returned solution "
            "from the initial scene, so all methods are scored identically.")
if example.exists():
    st.code(example.read_text(encoding="utf-8"), language="python")
