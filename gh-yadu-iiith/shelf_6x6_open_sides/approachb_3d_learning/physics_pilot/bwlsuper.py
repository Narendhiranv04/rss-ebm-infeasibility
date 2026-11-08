"""R1 supervisor: run S2 recovery chunks serially with 2 workers on CPUs 16-17, enforcing the AUTHORIZED caps and the discrepancy gate.

    PY=$HOME/approachb3d_mock/env/bin/python
    taskset -c 16 nice -n 19 $PY bwlsuper.py --manifest S2_RECOVERY_MANIFEST.json --out out_r1 --first 1 --last 52 --cpu-cap-h 60 --wall-cap-h 42

After every chunk: summarize (summarize_s1 rules), append to R1_BUDGET_LEDGER.json (measured process CPU of every item incl. retries), stop on
  * any discrepancy: MISSING_* / ENV_ERROR / CENSORED_TIMEOUT / X_REPLAY_DISAGREES / Y_REPLAY_FAILED / RELOCATION_MISMATCH / NEGATIVE_* other
    than reconfirmed  (legitimate outcomes such as X0_STOP_CONNECTOR_* are NOT discrepancies);
  * a worker guard stop (STOP_GUARD_*), a wall-cap stop, or cumulative CPU >= cap - reserve (no new chunk starts; workers also stop on
    STOP_BUDGET between items).  Writes SUPERVISOR_STATE.json for the owner; never touches any other process."""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OK = ("X1_VERIFIED", "X0_RELOCATION_FAILED_RECONFIRMED", "Y1_REPLAY_VERIFIED", "X0_STOP_CONNECTOR_NO_PLAN", "X0_STOP_CONNECTOR_FAILED")


def subtype_only(out, r):
    """User-approved 2026-11-08 ~18:35 IST: an X item whose stored attempt failed at PLANNING, whose re-query STILL finds no plan, and whose only
    difference from the record is the negative subtype (NO_PLAN_FOUND vs CERTIFIED) is non-blocking.  It stays MISSING (no label) and is listed
    in the ledger; every other discrepancy still stops the run."""
    if r["final"] != "MISSING_RELOCATION_MISMATCH_WITH_RECORD":
        return False
    try:
        st = json.loads((out / "items" / f"{r['item']}.teacher.json").read_text())["steps"][-1]
    except (OSError, ValueError, KeyError, IndexError):
        return False
    return (st.get("failure") == "PLANNING" and st.get("replan_found") is False and st.get("kind") != st.get("recorded_kind")
            and {st.get("kind"), st.get("recorded_kind")} == {"CERTIFIED", "NO_PLAN_FOUND"})


def cpu_used(out):
    t = 0.0
    for f in glob.glob(str(out / "items" / "*.json")):
        try:
            t += float(json.load(open(f)).get("cpu_s") or 0)
        except (OSError, ValueError):
            pass
    return t / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=1); ap.add_argument("--last", type=int, default=52)
    ap.add_argument("--order", help="explicit chunk order, e.g. 2-10,38-46,11-37 (membership unchanged; overrides --first/--last)")
    ap.add_argument("--cpu-cap-h", type=float, default=60.0); ap.add_argument("--reserve-h", type=float, default=1.5)
    ap.add_argument("--wall-cap-h", type=float, default=42.0); ap.add_argument("--cpus", default="16,17")
    ap.add_argument("--host-guard", default="gvlab")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ledger_f, state_f = out / "R1_BUDGET_LEDGER.json", out / "SUPERVISOR_STATE.json"
    ledger = json.loads(ledger_f.read_text()) if ledger_f.exists() else {"cap_cpu_h": a.cpu_cap_h, "wall_cap_h": a.wall_cap_h, "chunks": [],
                                                                        "started_utc": time.strftime("%FT%TZ", time.gmtime()), "t0": time.time()}
    cpus = a.cpus.split(",")
    py = sys.executable

    def state(s, **kw):
        state_f.write_text(json.dumps({"state": s, "utc": time.strftime("%FT%TZ", time.gmtime()), "pid": os.getpid(), **kw}, indent=1))

    order = [c for part in a.order.split(",") for c in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))] if a.order else range(a.first, a.last + 1)
    for c in order:
        phase = f"CHUNK{c:03d}"
        used = cpu_used(out)
        if used >= a.cpu_cap_h - a.reserve_h:
            (out / "STOP_BUDGET").write_text(json.dumps({"cpu_h": used})); state("STOPPED_BUDGET", cpu_h=used, next=phase); return
        if time.time() - ledger["t0"] > a.wall_cap_h * 3600:
            state("STOPPED_WALLCAP", next=phase); return
        if any(out.glob("STOP_*")):
            state("STOPPED_EXISTING_STOP_FILE", files=[p.name for p in out.glob("STOP_*")]); return
        if (out / "CPUS").exists():                                           # owner may rescale between chunks (thermal / idle-core rule)
            cpus = (out / "CPUS").read_text().split()[0].split(",")
        state("RUNNING", chunk=phase, cpu_h=used, cpus=cpus)
        t0, wall_left = time.time(), int(a.wall_cap_h * 3600 - (time.time() - ledger["t0"]))
        procs = [subprocess.Popen(["taskset", "-c", cpus[i], "nice", "-n", "19", "ionice", "-c3", py, str(HERE / "bwlphys.py"), "work", "--manifest", a.manifest,
                                   "--phase", phase, "--shard", f"{i}/{len(cpus)}", "--out", str(out), "--wall-cap-s", str(max(60, wall_left)),
                                   "--host-guard", a.host_guard],
                                  stdout=open(out / f"worker_{phase}_{i}.log", "a"), stderr=subprocess.STDOUT) for i in range(len(cpus))]
        state("RUNNING", chunk=phase, workers=[p.pid for p in procs], cpu_h=used, cpus=cpus)
        for p in procs:
            p.wait()
        summ = out / f"SUMMARY_{phase}.json"
        subprocess.run([py, "-I", str(HERE / "summarize_s1.py"), a.manifest, str(out), str(summ)], stdout=subprocess.DEVNULL, check=False)
        s = json.loads(summ.read_text())
        rows = [r for r in s["rows"] if r["phase"] == phase]
        bad, nonblock = [], []
        for r in rows:
            if r["final"] in OK or r["final"].startswith("Y0_RECONFIRMED"):
                continue
            (nonblock if subtype_only(out, r) else bad).append(r["item"] + ":" + r["final"])
        n_expect = sum(1 for x in json.loads(Path(a.manifest).read_text())["items"] if x["phase"] == phase)
        if len(rows) < n_expect:
            bad.append(f"INCOMPLETE_CHUNK:{len(rows)}/{n_expect}")
        rec = {"chunk": phase, "items": len(rows), "finals": {}, "wall_s": round(time.time() - t0, 1), "cpu_h_cumulative": round(cpu_used(out), 4),
               "discrepancies": bad, "nonblocking_negative_subtype_mismatch": nonblock, "utc_end": time.strftime("%FT%TZ", time.gmtime())}
        for r in rows:
            rec["finals"][r["final"]] = rec["finals"].get(r["final"], 0) + 1
        ledger["chunks"].append(rec); ledger_f.write_text(json.dumps(ledger, indent=1))
        guard = list(out.glob("STOP_GUARD_*")) + list(out.glob("STOP_WALLCAP_*"))
        if bad or guard:
            state("STOPPED_DISCREPANCY" if bad else "STOPPED_GUARD", chunk=phase, discrepancies=bad[:20], guard=[g.name for g in guard]); return
    state("DONE", last=a.last, cpu_h=cpu_used(out))


if __name__ == "__main__":
    main()
