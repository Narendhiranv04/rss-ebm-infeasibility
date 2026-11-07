"""Launch-readiness tests: frozen splits and plan, oracle-free candidates, teacher = repaired smoke, nested budgets, witness packs,
held-state semantics, emergency interruption, launch guard (refuses without any planner call), topology. Teacher calls here use
only dev-touched TRAIN roots (the smoke reference query and the unit fixture)."""
import copy, json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import launch_check as K
import launch_guard as LG
import launch_plan as LP
import launch_runner as R
import placement_teacher as T

P = T.P
L = Path(__file__).resolve().parents[1] / "launch_ready_2d_v2"
V2 = L.parent / "smoke_repro_v2"
FIX = {"root": "2d-plus2d-srv-w0408-s006", "trial_id": "fixture-o08-55", "prefix": [], "object": "o08", "dest": 55, "block": "root", "prefix_kind": None,
       "source_cell": 0, "source_bin": "", "dest_bin": "", "stratum_k": 0, "stratum_n": 0, "obj_rank": 0, "dest_rank": 0}
POL = {"pairs_per_query": 12, "coverage": "FULL", "witness_store_max": 12, "reuse_archived": True}
SPLIT = {r["root_id"]: r for r in json.loads((L / "split_manifest.json").read_text())["roots"]}


def test_splits_are_lineage_closed_balanced_and_dev_roots_train():
    s = json.loads((L / "plan_summary.json").read_text())
    assert s["cross_split_shared"] == {"conversion": 0, "family": 0, "layout": 0}
    by = Counter(r["split"] for r in SPLIT.values())
    assert by["train"] == 3500 and abs(by["val"] - 750) <= 5 and abs(by["test"] - 750) <= 5
    groups = {}
    for r in SPLIT.values():
        groups.setdefault(r["root_group_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in groups.values())
    assert all(SPLIT[x]["split"] == "train" for x in {t["root"] for t in json.loads((V2 / "roster.json").read_text())} | {FIX["root"]})
    assert all(SPLIT[x]["split"] == "train" and not SPLIT[x]["dev_touched"] for x in s["pilot_roots"]) and s["pilot_inside_every_quota"]


def test_candidates_ignore_protected_metadata():
    """Scrambling roles, blockers, parking, construction and every label leaves the root block and design edit unchanged."""
    G = P.A.env()
    for rid in sorted(SPLIT, key=lambda x: LP.H("test", x))[:60]:
        r = P.load_root(rid)
        s = copy.deepcopy(r)
        s.update(roles={}, intended_blockers=[], parking={}, alt_parking={}, construction={}, verified_K=None, observed=None)
        s["labels"] = {"B": [], "D": [], "original": {}, "blocker": {}}
        a = [x for x in LP.root_plan(rid, r, G, False) if x[3] != "ARCHIVED_FIRST_MOVE" and not x[1].endswith("prefix2")]
        b = [x for x in LP.root_plan(rid, s, G, False) if not x[1].endswith("prefix2")]
        assert a == b


def test_teacher_reproduces_the_repaired_smoke(tmp_path):
    t = json.loads((V2 / "roster.json").read_text())[0]
    old = json.loads((V2 / "serial" / "parts" / f"{t['trial_id']}.json").read_text())
    task = {**FIX, "root": t["root"], "trial_id": t["trial_id"], "prefix": t["prefix"], "object": t["object"], "dest": t["dest"]}
    r = T.run_query(task, tmp_path, {**POL, "pairs_per_query": 8, "witness_store_max": 8}, 1200)
    z = np.load(tmp_path / "parts" / t["root"] / f"{t['trial_id']}.npz")
    assert r["label"] == old["label"] and {k: r["mode_counts"][k] for k in old["mode_summary"]} == old["mode_summary"]
    assert [(a["pick"], a["place"]) for a in old["attempts"]] == list(zip(z["a_pick"].tolist(), z["a_place"].tolist()))
    assert np.array_equal(z["pair_category"], np.load(V2 / "serial" / "parts" / f"{t['trial_id']}.npz")["pair_category"])
    oq = [np.load(V2 / "serial" / "trajectories" / f"{w}.npz")["q"] for w in old["witness_refs"]]
    assert all(np.array_equal(a, T.witness(tmp_path, t["root"], w)["q"]) for a, w in zip(oq, r["witness_refs"])) and len(oq) == len(r["witness_refs"])


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    out = {}
    for name, pol in (("b12", POL), ("b4", {**POL, "pairs_per_query": 4}), ("fs", {**POL, "coverage": "FIRST_SUCCESS"})):
        d = tmp_path_factory.mktemp(name)
        r = T.run_query(FIX, d, pol, 1200)
        z = np.load(d / "parts" / FIX["root"] / f"{FIX['trial_id']}.npz")
        out[name] = (d, r, {k: z[k] for k in T.ATT})
    return out


def test_nested_budgets_and_first_success_equal_their_virtual_accounting(fx):
    _, r12, a12 = fx["b12"]
    _, r4, a4 = fx["b4"]
    sel = a12["a_rank"] < 4
    assert all(np.array_equal(a4[k], a12[k][sel]) for k in T.ATT if k != "a_stored")
    assert K.virtual(r12, a12, 4, "FULL")["label"] == r4["label"] and K.virtual(r12, a12, 4, "FULL")["kind"] == r4["negative_kind"]
    _, rf, af = fx["fs"]
    v = K.virtual(r12, a12, 12, "FIRST_SUCCESS")
    assert v["label"] == rf["label"] and v["planned"] == rf["work"]["pairs_planned"] == int((af["a_rank"] >= 0).sum())


def test_pick_blocked_object_is_placeable_and_witness_is_placement_only(fx):
    d, r, _ = fx["b12"]
    assert r["pick_access_modes_ok"] == 0 and r["held_modes_valid"] > 0 and r["label"] == 1
    G = P.A.env()
    w = T.witness(d, FIX["root"], r["witness_refs"][0])
    assert list(dict.fromkeys(str(x) for x in w["phase"])) == ["carry", "release", "retreat"] and w["gap"][0] < G["robot"].open_gap
    s = T.scene_for(FIX)
    assert P.verify_witness(G["robot"], s, "o08", w, np.asarray(s.lattice.center(55)), P.sk.params(G["cfg"]))[0]


def test_pack_roundtrip_keeps_ids_and_removes_parts(fx, tmp_path):
    import shutil
    d, r, _ = fx["b12"]
    shutil.copytree(d, tmp_path / "run")
    hdr = {"root_id": FIX["root"], "pair_protocol_id": T.pair_protocol()[0]}
    T.pack_root(tmp_path / "run", hdr, [FIX])
    s = T.scene_for(FIX)
    ident = f"{hdr['pair_protocol_id']}|{s.robot_profile_hash}|{r['hypothetical_scene_hash']}|o08"
    assert all(T.wid_of(ident, T.witness(tmp_path / "run", FIX["root"], w)) == w for w in r["witness_refs"])
    assert not (tmp_path / "run" / "parts" / FIX["root"]).exists() and (tmp_path / "run" / "records" / f"{FIX['root']}.json").exists()


def test_emergency_interruption_is_unlabeled_and_rerun(tmp_path):
    r = T.run_query({**FIX, "trial_id": "fixture-interrupt"}, tmp_path, POL, 0.05)
    assert r["completed"] is False and r["label"] is None and r["safety_interruption"] == "QUERY_EMERGENCY_CPU"
    assert not T.final(tmp_path / "parts" / FIX["root"] / "fixture-interrupt.json")


def test_static_contrast_is_occupied_and_never_planned(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "plan_carry", lambda *a, **k: pytest.fail("planner called"))
    p = next(x for x in R.plan_rows(L) if x["root_id"] in SPLIT and SPLIT[x["root_id"]]["pilot"])
    row = next(dict(zip(LP.COLS, q)) for q in p["queries"] if q[12])
    r = T.run_query({**row, "root": p["root_id"]}, tmp_path, POL, 1200)
    assert not r["eligible"] and r["label"] is None and r["excluded"] == "DEST_OCCUPIED"


def test_full_launch_guard_refuses_without_running_the_teacher(monkeypatch):
    monkeypatch.setattr(P, "plan_carry", lambda *a, **k: pytest.fail("planner called"))
    monkeypatch.setattr(R, "run", lambda *a, **k: pytest.fail("runner started"))
    target = Path(R.cfg_of(L)["full_run_dir"])
    for argv in ([], ["--execute-full"], ["--confirm", LG.token(L)], ["--execute-full", "--confirm", "LAUNCH-2D-wrong"]):
        with pytest.raises(SystemExit, match="REFUSED"):
            LG.full(L, argv)
    if not json.loads((L / "configs" / "frozen_hashes.json").read_text()).get("production_policy_sha256"):
        with pytest.raises(SystemExit, match="REFUSED"):           # right token, but no frozen production policy yet
            LG.full(L, ["--execute-full", "--confirm", LG.token(L)])
    assert not target.exists()


def test_topology_selection():
    C = R.cfg_of(L)
    w, res, topo = R.topology(25, 7)
    assert len(set(w)) == 25 and len(res) >= 7 and not set(w) & set(res) and w == C["execution"]["worker_cpus"]
    cores = list(topo["cores"].values())
    assert set(w[:len(cores)]) == {c[0] for c in cores}            # one thread per physical core before any SMT sibling
