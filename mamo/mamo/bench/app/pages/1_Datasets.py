"""Datasets: add scene folders, label them by difficulty, browse scenes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from common import header, runner, workspace

from mamo.bench import datasets as D
from mamo.bench import media

header("Datasets", "Folders of scene files in the reference format (sections O and ooi). Reference scenes are "
       "labelled by the number of movables overlapping the initial NGR (E-M4M / I-M4M papers); generated scenes "
       "(REAP environments) by the difficulty preset they were sampled from.")
ws = workspace()

if (Path(runner.ROOT) / "results/scene_labels.csv").exists():
    done = D.import_reference_labels(ws, Path(runner.ROOT) / "results/scene_labels.csv")
    if done:
        st.toast(f"Imported existing labels for {', '.join(done)}")

with st.expander("Add a dataset"):
    with st.form("add"):
        name = st.text_input("Name", placeholder="my-dataset")
        folder = st.text_input("Folder with scene .txt files", placeholder=r"D:\data\my_scenes")
        if st.form_submit_button("Add") and name and folder:
            try:
                ds = D.add_dataset(ws, name, folder)
                st.success(f"Added {name}: {len(ds.scenes())} scene files")
            except FileNotFoundError:
                st.error(f"No such folder: {folder}")

with st.expander("Generate a dataset in the REAP environments"):
    from mamo import envs as E
    st.caption("Open tabletop, confined shelf, multi-level shelf, Access-19 and 3D stacking (Swoboda & Geffner "
               "2026, App. B) with the xArm7. Retrieve: take the target out (the M4M family solves these). "
               "Relocate: bring 1+ targets to goal cells, possibly on another surface. Difficulty sets the numbers "
               "of movable / immovable objects, targets and blockers:")
    st.dataframe(pd.DataFrame([{"environment": n, "difficulty": d, "movables": f"{p.movables[0]}-{p.movables[1]}",
                                "immovables": f"{p.immovables[0]}-{p.immovables[1]}",
                                "targets (relocate)": f"{p.targets[0]}-{p.targets[1]}",
                                "blockers": f"{p.blockers[0]}-{p.blockers[1]}"}
                               for n, f in E.ENVIRONMENTS.items() for d, p in f().presets.items()]),
                 hide_index=True, use_container_width=True, height=220)
    with st.form("generate"):
        c1, c2 = st.columns(2)
        gname = c1.text_input("Dataset name", "reap-envs")
        genvs = c2.multiselect("Environments", list(E.ENVIRONMENTS), default=list(E.ENVIRONMENTS))
        c1, c2, c3, c4, c5 = st.columns(5)
        gtasks = c1.multiselect("Tasks", list(E.TASKS), default=list(E.TASKS))
        gdiffs = c2.multiselect("Difficulties", list(E.DIFFICULTIES), default=list(E.DIFFICULTIES))
        gper = c3.number_input("Problems per combination", 1, 1000, 10)
        gstart = c4.number_input("First problem index", 0, 10 ** 6, 0, help="problem k is always the same scene")
        gworkers = c5.number_input("Workers", 1, 32, 4)
        gcheck = st.checkbox("Keep only problems whose targets can be retrieved once the movables are removed", True)
        if st.form_submit_button("Generate") and gname and genvs and gtasks and gdiffs:
            runner.launch_generation(ws, gname, genvs, gtasks, gdiffs, int(gper), int(gstart), gcheck, int(gworkers))
            st.info(f"Generating {len(genvs) * len(gtasks) * len(gdiffs) * gper} problems in the background into "
                    f"{ws / 'datasets' / gname}; select the dataset below and refresh to follow it.")

dss = D.list_datasets(ws)
rows = []
for ds in dss.values():
    labels = D.load_labels(ws, ds)
    c = Counter(r["difficulty"] for r in labels)
    rows.append({"dataset": ds.name, "scenes": len(ds.scenes()), "labelled": len(labels),
                 **{d: c.get(d, 0) for d in D.DIFFICULTIES}, "folder": str(ds.root),
                 "kind": "built-in" if ds.builtin else "user"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

name = st.selectbox("Dataset", list(dss))
if not name:
    st.stop()
ds = dss[name]
labels = D.load_labels(ws, ds)
status_f = D.labels_path(ws, ds).with_suffix(".status.json")
status = json.loads(status_f.read_text()) if status_f.exists() else {}

c1, c2, c3 = st.columns([1, 1, 2])
workers = c1.number_input("Labelling workers", 1, 32, 6)
if c2.button("Label (re)compute", disabled=status.get("state") == "running"):
    runner.launch_labelling(ws, name, int(workers))
    st.info("Labelling started in the background; refresh to see progress.")
if not ds.builtin and c3.button("Remove from list (files are kept)"):
    D.remove_dataset(ws, name)
    st.rerun()
if status.get("state") == "running":
    st.progress(status["done"] / max(1, status["total"]), text=f"labelling {status['done']}/{status['total']}")
    st.button("Refresh")

if labels:
    df = pd.DataFrame(labels)
    c1, c2, c3 = st.columns(3)
    present = [d for d in D.DIFFICULTIES if d in set(df.difficulty)] + sorted(set(df.difficulty) - set(D.DIFFICULTIES))
    diff = c1.multiselect("Difficulties", present, default=present)
    df = df[df.difficulty.isin(diff)]
    for col, box in (("env", c2), ("task", c3)):
        if col in df and df[col].fillna("").astype(bool).any():
            vals = sorted(set(df[col].fillna("")))
            df = df[df[col].fillna("").isin(box.multiselect(col.capitalize(), vals, default=vals))]
    st.dataframe(df, hide_index=True, use_container_width=True, height=260)
    scenes = list(df.scene)
else:
    st.info("Not labelled yet: experiments can still run on all scenes ('all' difficulty).")
    scenes = [ds.rel(p) for p in ds.scenes()]

st.subheader("Scene viewer")
scene = st.selectbox("Scene", scenes)
if scene and st.button("Show 2D / 3D views (with NGR)"):
    with st.spinner("Planning the retrieval and rendering..."):
        cache = ws / "cache" / "views" / ds.name / scene.replace("/", "_")
        p2, p3 = media.scene_views(ds.path(scene), cache)
    st.image(str(p2), caption="Top view of each surface: red = NGR (volume swept by the target's retrieval), "
             "dashed = relocate goals, grey cells = not part of the grid")
    st.image(str(p3), caption="Iso / front / top (top view without the ceiling); bottom row with NGR voxels")
