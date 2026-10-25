"""Summaries and plots of benchmark results (tables in the style of the M4M / E-M4M / I-M4M papers)."""

from __future__ import annotations

import pandas as pd

ORDER = ["trivial", "easy", "medium", "hard", "infeasible", "unlabeled"]


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c, default in (("solved", False), ("executed", None), ("planning_time", None), ("sim_time", None),
                       ("num_actions", None), ("num_pick_places", 0), ("reason", ""), ("exec_reason", "")):
        if c not in df:
            df[c] = default
    df["solved"] = df["solved"].fillna(False).astype(bool)
    # "success" = solved and (if replayed) the replay retrieved the OoI
    df["success"] = df["solved"] & df["executed"].map(lambda v: v is None or v != v or bool(v))
    df["difficulty"] = pd.Categorical(df["difficulty"], [d for d in ORDER if d in set(df["difficulty"])], ordered=True)
    return df


def _mmm(s: pd.Series) -> str:
    s = s.dropna()
    return "-" if s.empty else f"{s.min():.1f} / {s.median():.1f} / {s.max():.1f}"


def summary(df: pd.DataFrame, methods: list[str] | None = None) -> pd.DataFrame:
    """One row per (difficulty, method): solved, replay success, min/median/max times of solved runs."""
    if df.empty:
        return df
    methods = methods or list(dict.fromkeys(df["method"]))
    out = []
    for d in df["difficulty"].cat.categories:
        for m in methods:
            g = df[(df["difficulty"] == d) & (df["method"] == m)]
            if g.empty:
                continue
            s = g[g["solved"]]
            replayed = s["executed"].notna() & (s["executed"] == s["executed"])
            out.append({
                "difficulty": d, "method": m, "runs": len(g),
                "solved": f"{len(s)}/{len(g)}",
                "solved %": round(100 * len(s) / len(g), 1),
                "replay ok": f"{int(s.loc[replayed, 'executed'].astype(bool).sum())}/{int(replayed.sum())}"
                if replayed.any() else "-",
                "success %": round(100 * g["success"].sum() / len(g), 1),
                "planning time (s) min/med/max": _mmm(s["planning_time"]),
                "sim time (s) min/med/max": _mmm(s["sim_time"]),
                "actions (mean)": round(s["num_actions"].mean(), 1) if len(s) and s["num_actions"].notna().any() else "-",
                "pick-and-place (mean)": round(s["num_pick_places"].fillna(0).mean(), 1) if len(s) else "-",
            })
    return pd.DataFrame(out)


def pivot_solved(df: pd.DataFrame, value: str = "solved") -> pd.DataFrame:
    """Difficulty x method table of counts (like the papers' 'solved problems' rows)."""
    if df.empty:
        return df
    t = df.groupby(["difficulty", "method"], observed=True)[value].sum().unstack("method").fillna(0).astype(int)
    n = df.groupby(["difficulty", "method"], observed=True)[value].size().unstack("method").fillna(0).astype(int)
    out = t.astype(str) + "/" + n.astype(str)
    out.loc["total"] = [f"{int(t[c].sum())}/{int(n[c].sum())}" for c in t.columns]
    return out


def by_environment(df: pd.DataFrame, value: str = "success") -> pd.DataFrame:
    """(environment, task, difficulty) x method table of counts, for generated datasets."""
    if df.empty or "env" not in df or not df["env"].notna().any():
        return pd.DataFrame()
    d = df.assign(env=df["env"].fillna("reference_shelf"), task=df["task"].fillna("retrieve"))
    keys = ["env", "task", "difficulty"]
    t = d.groupby(keys + ["method"], observed=True)[value].sum().unstack("method")
    n = d.groupby(keys + ["method"], observed=True)[value].size().unstack("method")
    return (t.fillna(0).astype(int).astype(str) + "/" + n.fillna(0).astype(int).astype(str)).where(n.notna(), "-")


def filter_env_task(df: pd.DataFrame, st) -> pd.DataFrame:
    """Environment / task multiselects (Streamlit) when the results have them."""
    for col in ("env", "task"):
        if col in df and df[col].notna().any():
            vals = sorted(df[col].fillna("").unique())
            keep = st.multiselect({"env": "Environments", "task": "Tasks"}[col], vals, default=vals, key=f"f_{col}")
            df = df[df[col].fillna("").isin(keep)]
    return df


def failure_reasons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    f = df[~df["solved"]].copy()
    f["reason"] = f["reason"].fillna("").str.replace(r"\(.*", "", regex=True).str.slice(0, 60)
    return f.groupby(["method", "reason"]).size().rename("runs").reset_index().sort_values(["method", "runs"],
                                                                                            ascending=[True, False])


def anytime(df: pd.DataFrame, grid: list[float] | None = None) -> pd.DataFrame:
    """Fraction of problems solved within t seconds of planning, per method (and difficulty)."""
    if df.empty:
        return df
    tmax = float(df["planning_time"].max() or 1.0)
    grid = grid or [tmax * i / 60 for i in range(61)]
    rows = []
    for (m, d), g in df.groupby(["method", "difficulty"], observed=True):
        n = len(g)
        t = g.loc[g["success"], "planning_time"].dropna()
        for x in grid:
            rows.append({"method": m, "difficulty": d, "t": x, "solved fraction": float((t <= x).sum()) / n})
    return pd.DataFrame(rows)
