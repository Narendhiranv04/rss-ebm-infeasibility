"""EXPLORATORY bottleneck check for the 0/15 K4-K6 scaling result (no teacher calls: the certified witness plan is a VERIFIED success).
Per root (seed 0 sampler/scorer; scorers s0-s2 for p):
  (a) proposal coverage: along the witness path, is each witness destination among the task-conditioned cVAE's M proposals for that
      object in that state?  M = 4 (as searched) and M = 16.
  (b) scorer: p11 of the witness plan vs the p11 of the search's #1 and #5 plans (search_s{seed}.jsonl), per scorer seed.
  (c) oracle injection: rerun the same depth-6 search with the witness destination added to the proposals when the state is on the
      witness path; does the witness reach the top-1 / top-5?  -> top-5 yes = the proposal step is the bottleneck; no = the scorer/search.
    python scaling_diag.py   -> runs/explore/scaling_out/diag.json
"""
import json, math
import scaling_search as SS
import torch

H, OUT = SS.H, SS.OUT


def wit(root, lab):
    return [(root.ids.index(o), int(c)) for o, c in lab["witness"]]


def coverage(root, samp, w, M):
    gen, hit = torch.Generator(device=samp.dev).manual_seed(0), []
    for j, (o, c) in enumerate(w):
        seq = w[:j]
        moved = {i for i, _ in seq}
        todo = [x for x in root.movable if x not in moved]
        batch = samp.propose_many(root, root.cells(seq), todo, M, gen)
        hit.append(c in [int(d) for d in batch[o]])
    return hit


class Inject:
    """sampler wrapper: on the witness path, the next witness destination is added to that object's proposals."""
    def __init__(self, samp, w):
        self.s, self.w, self.dev = samp, w, samp.dev

    def propose_many(self, root, cells, todo, M, gen):
        out = self.s.propose_many(root, cells, todo, M, gen)
        for j in range(len(self.w)):
            if list(cells) == list(root.cells(self.w[:j])):
                o, c = self.w[j]
                if o in out and c not in [int(d) for d in out[o]]:
                    out[o] = list(out[o]) + [c]
        return out


def main():
    samp, _, _ = SS.BM.load_bundle(H.K3 / "stage/bundle", "2d", "cpu", 0)
    scor = {s: H.HZ.HazardScorer(H.OUT / f"model_2d_s{s}.pt") for s in (0, 1, 2)}
    top = {(r["root_id"], r["seed"]): r for f in OUT.glob("search_s*.jsonl") for r in map(json.loads, open(f))}
    rows = []
    for root, rec, p, lab in SS.roots():
        w = wit(root, lab)
        row = {"root_id": root.id, "K": lab["K_certified"], "witness": w,
               "cov_M4": coverage(root, samp, w, 4), "cov_M16": coverage(root, samp, w, 16)}
        for s, sc in scor.items():
            pw = sc.score(root, [[list(a) for a in w]])[0]["p11"]
            t = top.get((root.id, s), {}).get("top", [])
            row[f"s{s}"] = {"p_witness": round(pw, 6), "p_top1": t[0]["p11"] if t else None, "p_top5": t[-1]["p11"] if t else None}
        nodes, _ = SS.deep_search(root, Inject(samp, w), scor[0], 4, 6, 200, 0)
        ranked = sorted((n for n in nodes.values() if n["m"]), key=lambda n: -n["p11"])
        wk = [list(a) for a in w]
        rank = next((i + 1 for i, n in enumerate(ranked) if n["seq"] == wk), None)
        row["inject_s0"] = {"witness_found": rank is not None, "witness_rank": rank, "n_candidates": len(ranked)}
        rows.append(row)
        print(root.id, "K", row["K"], "cov4", sum(row["cov_M4"]), "/", len(w), "cov16", sum(row["cov_M16"]), "/", len(w),
              "p_wit/top1 s0", row["s0"]["p_witness"], row["s0"]["p_top1"], "inject rank", rank, flush=True)
    n = len(rows)
    summ = {"roots": n,
            "full_witness_covered_M4": sum(all(r["cov_M4"]) for r in rows), "full_witness_covered_M16": sum(all(r["cov_M16"]) for r in rows),
            "step_coverage_M4": round(sum(sum(r["cov_M4"]) for r in rows) / sum(len(r["cov_M4"]) for r in rows), 3),
            "step_coverage_M16": round(sum(sum(r["cov_M16"]) for r in rows) / sum(len(r["cov_M16"]) for r in rows), 3),
            "witness_scored_above_search_top1_s0": sum(r["s0"]["p_top1"] is not None and r["s0"]["p_witness"] > r["s0"]["p_top1"] for r in rows),
            "inject_witness_top1": sum(r["inject_s0"]["witness_rank"] == 1 for r in rows),
            "inject_witness_top5": sum((r["inject_s0"]["witness_rank"] or 99) <= 5 for r in rows)}
    out = {"label": "EXPLORATORY scaling2d K4-K6 bottleneck check (no teacher calls)", "summary": summ, "rows": rows}
    (OUT / "diag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    assert math.isfinite(1.0)
    main()
