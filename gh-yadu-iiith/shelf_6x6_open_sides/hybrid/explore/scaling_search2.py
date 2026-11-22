"""EXPLORATORY scaling fix (user 2026-11-22): deep plans failed (0/15) because search ranked partial plans by full-plan success p11,
which includes "can the target be picked NOW" (~0 until the last blocker moves), so correct partial paths were dropped.
Fix, settings fixed IN ADVANCE (no tuning on these 15 test roots):
  * beam search by depth, beam B = 64, M = 16 proposals per object;
  * partial-plan priority = sum log p(relocation steps so far) + W * log max(p_target_now, 1e-4) with W = 0.5
    (feasibility so far dominates; the target term still rewards progress);
  * every node generated is a candidate plan; final ranking = full p11 (unchanged scorer, s0-s2); top-5 to the teacher.
Also reports whether the certified witness ends up among the candidates (oracle diagnostic; not used for any choice).
    python scaling_search2.py search SEED      -> runs/explore/scaling_out2/search_s{SEED}.jsonl
    python scaling_search2.py teach WORKERS    -> runs/explore/scaling_out2/verify_2d.jsonl
    python scaling_search2.py report           -> runs/explore/scaling_out2/report.json
"""
import json, math, sys, time
import numpy as np
import scaling_search as SS
import torch

H = SS.H
OUT = SS.RUNS / "explore/scaling_out2"
B, M, W, KMAX, TOPN = 64, 16, 0.5, 6, 5


def parts(sc, root, seqs):
    """-> per plan (log p_reloc, p_target_now, full p11) from the hazard scorer's per-step cache."""
    full = sc.score(root, seqs)
    out = []
    for s, f in zip(seqs, full):
        st = H.HZ.steps_of(root, [tuple(a) for a in s])
        lr = sum(math.log(max(sc.cache[x[0]], 1e-12)) for x in st if x[5] == "reloc")
        pt = sc.cache[st[-1][0]]
        out.append((lr, pt, f["p11"]))
    return out


def beam(root, samp, sc, seed):
    gen = torch.Generator(device=samp.dev).manual_seed(seed)
    beam_, cand = [()], {}
    for depth in range(KMAX):
        kids = []
        for k in beam_:
            seq = [tuple(a) for a in k]
            moved = {i for i, _ in seq}
            todo = [o for o in root.movable if o not in moved]
            if not todo:
                continue
            batch = samp.propose_many(root, root.cells(seq), todo, M, gen)
            kids += [tuple(seq + [(o, int(d))]) for o in todo for d in dict.fromkeys(int(x) for x in batch[o])
                     if int(d) not in root.cells(seq)]
        kids = [c for c in dict.fromkeys(kids) if c not in cand]
        if not kids:
            break
        sc_ = parts(sc, root, [list(map(list, c)) for c in kids])
        for c, (lr, pt, p11) in zip(kids, sc_):
            cand[c] = {"seq": [list(a) for a in c], "p11": p11, "prio": lr + W * math.log(max(pt, 1e-4)), "m": len(c)}
        beam_ = sorted(kids, key=lambda c: -cand[c]["prio"])[:B]
    return cand


def search(seed):
    OUT.mkdir(parents=True, exist_ok=True)
    fo = OUT / f"search_s{seed}.jsonl"
    done = {json.loads(l)["root_id"] for l in open(fo)} if fo.exists() else set()
    samp, _, _ = SS.BM.load_bundle(H.K3 / "stage/bundle", "2d", "cpu", 0)
    sc = H.HZ.HazardScorer(H.OUT / f"model_2d_s{seed}.pt")
    with open(fo, "a") as f:
        for root, rec, p, lab in SS.roots():
            if root.id in done:
                continue
            t0 = time.time()
            cand = beam(root, samp, sc, seed)
            plans = sorted(cand.values(), key=lambda n: -n["p11"])[:TOPN]
            w = [[root.ids.index(o), int(c)] for o, c in lab["witness"]]
            wr = sorted(cand.values(), key=lambda n: -n["p11"])
            rank = next((i + 1 for i, n in enumerate(wr) if n["seq"] == w), None)
            f.write(json.dumps({"root_id": root.id, "K": lab["K_certified"], "seed": seed, "B": B, "M": M, "W": W,
                                "n_candidates": len(cand), "witness_rank": rank,
                                "top": [{"seq": n["seq"], "p11": round(n["p11"], 6), "m": n["m"]} for n in plans],
                                "s": round(time.time() - t0, 1)}) + "\n")
            f.flush()
            print(f"{root.id} K{lab['K_certified']} s{seed}: {len(cand)} plans, top p {plans[0]['p11']:.3f} len {plans[0]['m']}, "
                  f"witness rank {rank}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    assert B > 0 and M > 0 and 0 <= W <= 1 and np.isfinite(math.log(1e-4))
    c, a = sys.argv[1], sys.argv[2:]
    SS.OUT = OUT                       # teach / report of scaling_search, pointed at this folder
    {"search": lambda: search(int(a[0])), "teach": lambda: SS.teach(int(a[0])), "report": SS.report}[c]()
