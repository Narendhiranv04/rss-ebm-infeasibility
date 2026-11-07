"""Read-only final-state audit of the live SHELF66V3 store after its K-quota run stopped (production v3).  Runs on the laptop (the only
host with key access); every remote command is a read (ps / stat / cat / head / sha256sum under nice + ionice / python hashing).

  python source_final3d.py PUBLISHED_DATASET_DIR STAGE_DIR OUT.json

Checks: the K-quota generator is no longer running; the published ledger is a byte prefix of the final ledger (append-only); every
published root / witness / EXCLUDED_FIXED_OBJECT.json is byte-identical on the host now (host-side sha256 = second pass); the generator
code is blob-identical to yadu/dataset-3d@8032ad0b; the author's selection rule on the FINAL ledger with the FINAL QUOTA.json gives
exactly the published K0-K3 selection (so the first cohort needs no newer data); the fixed-object check file is unchanged.
"""
import gzip, hashlib, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_sync3d as SS  # noqa: E402

HOST = "user1@10.4.25.63"
R = "yadu_bd/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3"
CODE = ["shelf66_v3/bd_3dt.py", "shelf66_v3/bd_3dt_plus.py", "shelf66_v3/bd_shelf.py", "shelf66_v3/none_audit66_v3.py",
        "shelf66_v3/reteach_shelf66_v3.py", "shelf66_v3/run_shelf66_full.py", "dataset-bd/bd_core.py", "run_3dt_plus.py"]
SCENE = ["shelf_scene.py", "rrt_connect.py", "rrt_star.py", "xarm7_cabinet_env.py"]


def ssh(cmd, inp=None):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd], input=inp, capture_output=True, text=True, check=True).stdout


def main(pub, stage, out):
    pub, stage = Path(pub), Path(stage)
    (stage / "meta").mkdir(parents=True, exist_ok=True)
    man = json.loads(gzip.open(pub.parent / "dataset_placement_extension_v2/source_sync/SOURCE_SNAPSHOT_MANIFEST.json.gz", "rt").read())
    ent = man.get("files") or man.get("entries")
    res = {"host": HOST, "path": R, "observed_utc": ssh("date -u +%FT%T.%NZ").strip()}
    ps = ssh("ps -eo pid,etime,args | grep -E 'run_shelf66|run_3dt_plus' | grep -v grep || true")
    res["generator_processes"] = [x.strip() for x in ps.splitlines() if x.strip()]
    res["k_quota_run_stopped"] = not any("run_shelf66_full.py" in x for x in res["generator_processes"])
    res["stat"] = ssh(f"cd {R} && stat -c '%y %s %n' ledger.jsonl PROGRESS.json QUOTA.json EXCLUDED_FIXED_OBJECT.json").splitlines()
    for f in ("ledger.jsonl", "PROGRESS.json", "QUOTA.json", "EXCLUDED_FIXED_OBJECT.json"):
        (stage / "meta" / f).write_text(ssh(f"cat {R}/{f}"))
    res["log_tail"] = ssh(f"tail -n 5 {R}/../dataset_2500_raw_shelf66v3.log").splitlines()
    old = (pub / "raw/ledger.jsonl").read_bytes()
    new = (stage / "meta/ledger.jsonl").read_bytes()
    host_prefix = ssh(f"head -c {len(old)} {R}/ledger.jsonl | sha256sum").split()[0]
    res["ledger"] = {"published_bytes": len(old), "final_bytes": len(new), "published_sha256": hashlib.sha256(old).hexdigest(),
                     "host_prefix_sha256": host_prefix, "append_only": new[:len(old)] == old and host_prefix == hashlib.sha256(old).hexdigest(),
                     "final_lines": new.count(b"\n"), "final_sha256": hashlib.sha256(new).hexdigest()}
    paths = sorted(p for p in ent if p.startswith(("raw/roots/", "raw/witnesses/")) or p == "raw/EXCLUDED_FIXED_OBJECT.json")
    remote = [R + "/" + p[4:] for p in paths]
    t0 = time.time()
    got = ssh("cd ~ && xargs -d '\\n' nice -n 19 ionice -c3 sha256sum", "\n".join(remote) + "\n")
    hs = {ln[66:].strip(): ln[:64] for ln in got.splitlines()}
    diff = [p for p, rp in zip(paths, remote) if hs.get(rp) != ent[p]["sha256"]]
    res["published_files_rehashed_on_host"] = {"files": len(paths), "identical": len(paths) - len(diff), "differ_or_missing": diff[:20],
                                               "seconds": round(time.time() - t0, 1)}
    py = ("import hashlib,sys,json;print(json.dumps({p:hashlib.sha1(b'blob %d\\0'%len(d)+d).hexdigest() for p in sys.argv[1:] "
          "for d in [open(p,'rb').read()]}))")
    host_code = json.loads(ssh("cd yadu_bd/code_nooverpass/bd_samples && python3 -c \"" + py + "\" " + " ".join(CODE)))
    host_scene = json.loads(ssh("cd yadu_bd/code_nooverpass/bd_samples/shelf66_v3/shelf_scene_6x6 && python3 -c \"" + py + "\" " + " ".join(SCENE)))
    rel = "gh-yadu-iiith/shelf_6x6_open_sides/"
    tree = {}
    for p in CODE:
        tree[p] = subprocess.run(["git", "rev-parse", f"8032ad0b:{rel}dataset_bd_classquota_without_home_position_no_overpass/code/{p}"],
                                 cwd=pub, capture_output=True, text=True).stdout.strip()
    for p in SCENE:
        tree["scene/" + p] = subprocess.run(["git", "rev-parse", f"8032ad0b:{rel}scene/{p}"], cwd=pub, capture_output=True, text=True).stdout.strip()
    hostall = {**host_code, **{"scene/" + k: v for k, v in host_scene.items()}}
    res["generator_code_blob_identical_to_8032ad0b"] = {k: hostall.get(k) == v for k, v in tree.items()}
    led = SS.ledger(stage / "meta/ledger.jsonl")
    q = dict(SS.QUOTA); q.update(json.loads((stage / "meta/QUOTA.json").read_text()))
    exc = set(json.loads((stage / "meta/EXCLUDED_FIXED_OBJECT.json").read_text())["excluded"])
    SS.QUOTA.clear(); SS.QUOTA.update(q)
    got_, pick = SS.select(led, exc)
    sel = json.loads((pub / "selected.json").read_text())
    res["final_selection"] = {"quota_final": q, "counts_final": got_,
                              "k0_k3_identical_to_published": all(pick[k] == sel["selected"][k] for k in ("0", "1", "2", "3")),
                              "none_final": len(pick["NONE"]), "k4_final": len(pick["4"]),
                              "accepted_final": sum(1 for e in led if e.get("accepted"))}
    res["excluded_fixed_object_unchanged"] = hashlib.sha256((stage / "meta/EXCLUDED_FIXED_OBJECT.json").read_bytes()).hexdigest() == \
        ent["raw/EXCLUDED_FIXED_OBJECT.json"]["sha256"]
    res["new_snapshot_published"] = False
    res["why_no_new_snapshot"] = ("the K-quota run stopped after the published cut with its K0-K3 quotas already met: the final selection's K0-K3 "
                                  "lists are identical to the published ones and the author's fixed-object file is unchanged, so newer roots "
                                  "would add no first-cohort root (the published snapshot stays the pinned source)")
    res["pass"] = (res["k_quota_run_stopped"] and res["ledger"]["append_only"] and not diff and all(res["generator_code_blob_identical_to_8032ad0b"].values())
                   and res["final_selection"]["k0_k3_identical_to_published"] and res["excluded_fixed_object_unchanged"])
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in res.items() if k not in ("log_tail", "stat")}, indent=1)[:3000])


if __name__ == "__main__":
    main(*sys.argv[1:4])
