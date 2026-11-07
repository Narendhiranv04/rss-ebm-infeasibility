#!/bin/bash
# FULL 3D PLACEMENT COLLECTION LAUNCHER: DISABLED BY DESIGN (mock v1). It never starts a collection.
#
# Why it cannot run:
#   1. run3d.py only accepts TRAIN rows (load_units filters split == "train"); production also needs val/test labels, which
#      requires an explicit, reviewed code change (a new run3d version and a new frozen hash).
#   2. The production roster must be rebuilt from the LIVE store on 10.4.25.63 after it is copied read-only to this host
#      (the GitHub snapshot 8032ad0b is incomplete: K3 264/357, K4 0, NONE 3). This mock has no access to that store.
#   3. A production policy file (mock_v1/configs/production_policy.json) does not exist. Only the user's explicit authorization
#      may create it, from LAUNCH_READINESS_DECISION.json.
#
# The checks a future (authorized, re-versioned) launcher must perform are listed in RUNBOOK.md §4; this stub only prints them.
echo "REFUSED: full 3D placement collection is not authorized (mock v1). See mock_v1/RUNBOOK.md section 4." >&2
exit 2
