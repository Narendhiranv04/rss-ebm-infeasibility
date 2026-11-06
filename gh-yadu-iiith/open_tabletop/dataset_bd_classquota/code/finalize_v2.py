"""Merge a finished relabel_3dt_v2.py output into the dataset's raw/ store (same layout as before) and redo the class selection.

    python finalize_v2.py <raw> <v2out> <selected.json>

  raw/roots/<id>.json      replaced by the V2 record (unchanged roots: only ids.teacher_version / config_hash, `relabel`, ends_at_ready)
  raw/witnesses/           + the V2 witnesses the records point to; witnesses no record points to any more are removed
  raw/RELABEL_V2_SUMMARY.json   class transitions, B outcome changes per role, reproduction check, CPU
  raw/RELABEL_V2.jsonl     one line per accepted root: V1 -> V2 class, per-B-entry outcome change, reproduction check, CPU
  raw/ledger.jsonl, PROGRESS.json   kept: the generation history (V1 classes)
  selected.json            the same rule as before (accepted roots in ledger order, per class, up to the quota) on the V2 classes
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def refs(r):
    L = r["labels"]
    out = [L["original"].get("witness")] + [d.get("witness") for d in L["D"]]
    out += [w for b in L["B"] for w in b.get("witnesses", [])]
    return {w for w in out if w}


def main(raw, v2, sel_path):
    raw, v2, sel_path = Path(raw), Path(v2), Path(sel_path)
    led = [json.loads(x) for x in (raw / "ledger.jsonl").read_text().splitlines() if x.strip()]
    acc = [e["root_id"] for e in led if e.get("accepted")]
    log = {}
    for x in (v2 / "RELABEL.jsonl").read_text().splitlines():
        line = json.loads(x)
        log[line["root_id"]] = line                                     # a resumed run: the last line of a root wins
    missing = [i for i in acc if log.get(i, {}).get("action") not in ("kept", "relabelled")]
    if missing:
        raise SystemExit(f"{len(missing)} roots not relabelled, e.g. {missing[:3]}")
    used = set()
    for i in acc:
        r = json.loads((v2 / "roots" / f"{i}.json").read_text())
        for w in refs(r):
            dst = raw / "witnesses" / w[:2] / f"{w}.npz"
            if not dst.exists():
                dst.parent.mkdir(exist_ok=True)
                shutil.copy2(v2 / "witnesses" / w[:2] / f"{w}.npz", dst)
            used.add(w)
        shutil.copy2(v2 / "roots" / f"{i}.json", raw / "roots" / f"{i}.json")
    gone = [p for p in (raw / "witnesses").glob("*/*.npz") if p.stem not in used]
    for p in gone:
        p.unlink()
    (raw / "RELABEL_V2.jsonl").write_text("".join(json.dumps(log[i], sort_keys=True) + "\n" for i in acc))
    sel = json.loads(sel_path.read_text())
    Q = sel["quota"]
    got, chosen = {k: 0 for k in Q}, {k: [] for k in Q}
    for i in acc:
        c = log[i]["observed_v2"]
        if c in got and got[c] < Q[c]:
            got[c] += 1
            chosen[c].append(i)
    sel_v1 = {r: k for k, v in sel["selected"].items() for r in v}
    sel.update(rule=sel["rule"] + "; V2 classes (raw/RELABEL_V2.jsonl observed_v2)", counts=got, selected=chosen,
               v1_counts={k: len(v) for k, v in sel["selected"].items()},
               v1_selected_still_selected=sum(sel_v1.get(i) is not None for k in chosen for i in chosen[k]))
    sel_path.write_text(json.dumps(sel, indent=1) + "\n")
    summary(acc, log, sel_v1, raw)
    print(json.dumps({"roots": len(acc), "witnesses_used": len(used), "witnesses_removed": len(gone), "counts": got,
                      "short_of_quota": {k: Q[k] - got[k] for k in Q if got[k] < Q[k]}}, indent=1))


def summary(acc, log, sel_v1, raw):
    """raw/RELABEL_V2_SUMMARY.json: class transitions V1 -> V2 (all accepted roots and the V1-counted ones), B outcome changes per
    role, the reproduction check of the protocol-invariant labels, CPU."""
    trans, trans_sel, flips, repro = Counter(), Counter(), Counter(), Counter()
    for i in acc:
        x = log[i]
        trans[f"{x['observed_v1']}->{x['observed_v2']}"] += 1
        if i in sel_v1:
            trans_sel[f"{x['observed_v1']}->{x['observed_v2']}"] += 1
        for b in x.get("B", []):
            flips[f"{b['role']}:{b['y_v1']}->{b['y_v2']}"] += 1
        for k, v in x.get("repro", {}).items():
            repro[f"{k}:{'ok' if v else 'MISMATCH'}"] += 1
    out = {"transitions_all_accepted": dict(sorted(trans.items())), "transitions_v1_counted": dict(sorted(trans_sel.items())),
           "B_outcomes_role_v1_to_v2": dict(sorted(flips.items())), "reproduction_of_unchanged_labels": dict(sorted(repro.items())),
           "actions": dict(Counter(log[i]["action"] for i in acc)), "cpu_s": round(sum(log[i].get("cpu_s") or 0 for i in acc), 1)}
    (raw / "RELABEL_V2_SUMMARY.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
