"""Shared state for the app pages: workspace, planner registry, cached results."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamo.bench import runner  # noqa: E402
from mamo.bench.planners import REGISTRY, load_plugins  # noqa: E402


def workspace() -> Path:
    ws = None
    if "--workspace" in sys.argv:
        ws = sys.argv[sys.argv.index("--workspace") + 1]
    ws = Path(ws or os.environ.get("MAMO_BENCH_HOME", runner.DEFAULT_WORKSPACE))
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def planners(reload: bool = False) -> tuple[dict, list[str]]:
    """Registry with plugins loaded (re-imported when ``reload``)."""
    if reload or "plugin_errors" not in st.session_state:
        for name in [n for n, s in REGISTRY.items() if s.family == "user" or "mamo_plugin_" in s.fn.__module__]:
            REGISTRY.pop(name, None)
        st.session_state["plugin_errors"] = load_plugins(runner.plugin_dirs(workspace()))
    return REGISTRY, st.session_state["plugin_errors"]


def header(title: str, caption: str = "") -> None:
    st.set_page_config(page_title=f"MAMO bench - {title}", layout="wide")
    st.title(title)
    if caption:
        st.caption(caption)
