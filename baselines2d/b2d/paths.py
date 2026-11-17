"""Where the 2D dataset, its checker code and our outputs live (container layout: D:\\RRC\\New folder mounted at /work)."""
from __future__ import annotations

import os
from pathlib import Path

WORK = Path(os.environ.get("B2D_WORK", "/work"))
DATA = WORK / "data2d"

# 2D class-quota dataset (rss-ebm-infeasibility @ yadu/dataset-3d): original scenes + the checker code it was labelled with
DS2 = DATA / "yadu-dataset-3d/gh-yadu-iiith/planar_2d/dataset_bd_classquota"
CODE = DS2 / "code"
LCT = CODE / "latent-causal-tamp"

# final selection, label manifest and the newer scene stores (rss-ebm-infeasibility @ naren/dataset-done)
FDG = DATA / "naren-dataset-done/final_dataset_generation"
SEL2 = FDG / "k3_hard/2D_K3_STRENGTH_SELECTION_V3.json"            # the selection Approach B evaluates on
XY2 = FDG / "FINAL_READINESS_GATE/manifests/2D_FINAL_XY_LABEL_MANIFEST.json"

# finalized X / Y teacher (latent-causal-tamp @ approach-b-xy-v1, identical to approach-b-universal-search)
WT = DATA / "lct-approach-b-xy-v1"

OUT = Path(os.environ.get("B2D_OUT", Path(__file__).resolve().parents[1] / "out"))     # next to the code (works for copies of the folder)
