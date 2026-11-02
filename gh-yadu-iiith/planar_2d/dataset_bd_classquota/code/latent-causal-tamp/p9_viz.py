"""P9 figures (no robot motion: P9 executes nothing). Every file gets a SHA-256 sidecar and an entry in visualizations/p9/index.json.

  factor_instances        L0: a tiny root as a real factor graph per coupling mode (circles = binary outcome variables,
                          squares = factor instances, dashed = observed context; one shared circle for one physical query)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

from viz import plt

OUT = Path("visualizations/p9")
FAM = {"F2": "#7a5195", "F3": "#2f6f9f", "F4": "#c26a1b", "F5": "#3b8a52", "F1": "#8f99a3"}


def _save(fig, name, what):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    h = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / (name + ".sha256")).write_text(h + "\n")
    idx = OUT / "index.json"
    files = [f for f in (json.loads(idx.read_text())["files"] if idx.exists() else []) if f["file"] != str(p)]
    idx.write_text(json.dumps({"files": files + [{"file": str(p), "sha256": h, "what": what}]}, indent=1) + "\n")
    return h


def factor_instances():
    V = {"z0": (1.0, 5.6), "zS[A]": (3.4, 5.6), "zS[B]": (5.8, 5.6), "zS[A+B]": (8.2, 5.6),
         "zR[r1]": (1.6, 1.7), "x": (5.0, 1.7), "zO[r1,1]": (8.4, 1.7)}
    label = {"x": "zO[r1,0]\n= zR[r2]"}
    # (family, position, touched variables, text, modes it exists in)
    U = ("full", "quadratic", "structured", "none")
    F = [("F3", (1.0, 6.75), ["z0"], "scene\nunary", U), ("F3", (3.4, 6.75), ["zS[A]"], "scene\nunary", U),
         ("F3", (5.8, 6.75), ["zS[B]"], "scene\nunary", U), ("F3", (8.2, 6.75), ["zS[A+B]"], "scene\nunary", U),
         ("F3", (2.2, 4.55), ["z0", "zS[A]"], "z0-zS\npair", U[:3]), ("F3", (3.6, 4.15), ["z0", "zS[B]"], "z0-zS\npair", U[:3]),
         ("F3", (5.0, 3.8), ["z0", "zS[A+B]"], "z0-zS\npair", U[:3]),
         ("F5", (4.6, 6.45), ["zS[A]", "zS[B]"], "generic", U[:2]), ("F5", (7.0, 6.45), ["zS[B]", "zS[A+B]"], "generic", U[:2]),
         ("F5", (5.8, 7.45), ["zS[A]", "zS[A+B]"], "generic", U[:2]),
         ("F5", (7.0, 3.5), ["zS[A+B]", "zR[r1]"], "scene-\nrepair", U[:2]), ("F5", (6.2, 2.75), ["zS[B]", "x"], "scene-\nrepair", U[:2]),
         ("F2", (1.6, 0.45), ["zR[r1]"], "seq unary\nr1", U), ("F4", (4.3, 0.45), ["x"], "omission\nunary (r1)", U),
         ("F2", (5.7, 0.45), ["x"], "seq unary\nr2", U), ("F4", (8.4, 0.45), ["zO[r1,1]"], "omission\nunary (r1)", U),
         ("F4", (3.3, 2.45), ["zR[r1]", "x"], "zR-zO\npair", U[:3]), ("F4", (5.0, 2.95), ["zR[r1]", "zO[r1,1]"], "zR-zO\npair", U[:3]),
         ("F5", (6.7, 1.05), ["x", "zO[r1,1]"], "generic", U[:2])]
    fig, axs = plt.subplots(2, 2, figsize=(16, 11.5))
    for ax, mode in zip(axs.flat, U):
        ax.set_xlim(-1.3, 11.4)
        ax.set_ylim(-0.3, 8.6)
        ax.axis("off")
        ax.set_title(f"B-{mode}", fontsize=13, loc="left")
        if mode == "full":                               # higher-order residuals: one factor on ALL bits of a block (hyperedge)
            for (x0, y0, w, h), txt in (((0.35, 5.0, 8.55, 1.2), "higher-order scene factor (all scene bits)"),
                                         ((0.95, 1.12, 8.1, 1.16), "higher-order\nrepair factor\n(all r1 bits\n+ scene bits)")):
                ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.08", fc=FAM["F5"], alpha=0.12, ec=FAM["F5"], ls=":", zorder=0))
                side = "\n" in txt
                ax.text(x0 + w + (0.2 if side else 0), y0 + (h / 2 if side else -0.14), txt, ha="left" if side else "right",
                        va="center" if side else "top", fontsize=6.6, color=FAM["F5"])
        for fam, (x, y), vs, txt, modes in F:
            if mode not in modes:
                continue
            for v in vs:
                ax.plot([x, V[v][0]], [y, V[v][1]], color=FAM[fam], lw=1.0, alpha=0.7, zorder=1)
            ax.text(x, y, txt, ha="center", va="center", fontsize=6.6, color="white", zorder=3,
                    bbox={"boxstyle": "square,pad=0.3", "fc": FAM[fam], "ec": "none"})
        for v, (x, y) in V.items():
            ax.add_patch(plt.Circle((x, y), 0.42, fc="#dbeafe", ec="#285a84", lw=2.6 if v == "x" else 1.4, zorder=2))
            ax.text(x, y, label.get(v, v), ha="center", va="center", fontsize=7.4, zorder=4)
        ax.text(-1.2, 7.9, "c0: initial scene, robot q0, goal,\nDINOv2 + geometry (observed)", fontsize=7.5, va="top",
                bbox={"boxstyle": "round", "fc": "white", "ec": "#8f99a3", "ls": "--"})
        ax.text(-1.2, 3.35, "repairs (observed inputs):\nr1 = (A->a, B->b)\nr2 = (B->b)", fontsize=7.5, va="top",
                bbox={"boxstyle": "round", "fc": "white", "ec": "#8f99a3", "ls": "--"})
        ax.text(9.0, 8.5, "F1 static (not a z factor):\nU(A@a) U(B@b) V(A,B) on the\nendpoints of r1 -> E_static(r1);\n"
                "cancels from P(z | r); trained by\nstatic BCE; enters E_rank only", fontsize=6.6, va="top", ha="left",
                bbox={"boxstyle": "round", "fc": "#f1f3f5", "ec": FAM["F1"], "ls": "--"})
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, ms=9, label=lab) for lab, c in
               (("F2 ordered sequence", FAM["F2"]), ("F3 scene intervention", FAM["F3"]), ("F4 repair omission", FAM["F4"]),
                ("F5 generic coupling", FAM["F5"]), ("F1 static (separate)", FAM["F1"]))]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.035, "z0 = zR[empty] (the empty repair has no separate slot).  Thick circle: ONE physical query - omitting A from r1 IS the repair r2 - "
             "so factors of both repair blocks touch the same variable (equality-aware root model).  Scene-repair pairs: 2 of 12 drawn.",
             ha="center", fontsize=8.5)
    fig.suptitle("ABC7 / ABC9 factor instances on a tiny root (circles: binary outcomes; squares: factor instances)", fontsize=13)
    return _save(fig, "P9_factor_instances.png", "L0: factor instances of a tiny root for B-full / quadratic / structured / none (diagram of the code)")


FAMROWS = [("F1 static unary", ["unary"]), ("F1 static pair", ["pairh"]), ("F2 sequence", ["order_s", "pairo", "mix", "gru", "h0"]),
           ("F3 scene unary", ["scene"]), ("F3 z0-zS pair", ["pair_scene"]), ("F4 zR-zO pair", ["pair_omit"]),
           ("F5 generic pairs", ["cpl"]), ("F5 higher-order", ["ho_u", "ho_s", "ho_q"]), ("travel J head", ["cost"]),
           ("shared DINOv2 projection", ["vproj"])]


def _fam(mods, keys):
    return sum(mods.get(k, 0.0) ** 2 for k in keys) ** 0.5


def l1_math():
    """L1: exact evidence vs brute force, cost of the equality-blind product, legacy copy disagreement vs one shared
    variable, and which factor families each ABC9 term reaches."""
    import numpy as np
    import torch

    import abc7_train as t7
    import abc8
    import joint_likelihood as jl
    rows = json.loads(Path("runs/p9/l1/proof_rows.json").read_text())
    grads = json.loads(Path("runs/p9/l1/term_gradients.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    a = ax[0, 0]
    bf = [(x["brute_force_ll"], x["exact_ll"]) for x in rows if "brute_force_ll" in x]
    a.scatter(*zip(*bf), s=12, color="#2f6f9f")
    lim = [min(min(p) for p in bf) - 0.5, 0.2]
    a.plot(lim, lim, color="#8f99a3", lw=1)
    a.set(xlabel="brute-force log P(D_c) (all variables enumerated)", ylabel="exact elimination log P(D_c)",
          title=f"(a) {len(bf)} saved P8c roots: max |diff| {max(abs(x - y) for x, y in bf):.1e}")
    a = ax[0, 1]
    d = [x["exact_ll"] - x["duplicated_product_ll"] for x in rows if x["shared_vars"]]
    a.hist(d, bins=30, color="#c26a1b")
    a.set(xlabel="equality-blind product NLL - exact NLL (nats per root)", ylabel="roots",
          title=f"(b) untied copies: {len(d)} roots, mean +{np.mean(d):.2f} nats")
    a = ax[1, 0]
    roots = t7.prepare(torch.load("runs/p8/c_training/dataset_sealed.pt", weights_only=False))
    m = abc8.build("B-full")
    m.load_state_dict(torch.load("runs/p8/c_training/ckpt_B-full#s0.pt", weights_only=True))
    k = 0
    with torch.no_grad():
        for r in roots:
            if len(r["cands"]) < 2:
                continue
            S, blocks, lab, outs = jl.tables(m, r)
            pe = jl.marginals(S, blocks)
            cp = {}
            for c, o in zip(r["cands"], outs):
                for s, q, p in zip(o["slots"], c["z_q"], o["marg"].tolist()):
                    if s[:2] in ("zR", "zO"):
                        cp.setdefault(q, []).append(p)
            for q, ps in cp.items():
                if len(ps) > 1:
                    a.plot([k, k], [min(ps), max(ps)], color="#c26a1b", lw=2.2)
                    a.scatter([k] * len(ps), ps, s=10, color="#c26a1b")
                    a.scatter([k], [pe[q]], s=26, marker="D", color="#2f6f9f", zorder=3)
                    k += 1
    a.set(xlabel="physical sequence query with >= 2 copies (disclosed P8c test roots)", ylabel="P(success)", ylim=(-0.03, 1.03),
          title="(c) legacy B-full copies (orange) vs one tied variable (blue)")
    a = ax[1, 1]
    cols = [(arm, t) for arm in ("B-full", "B-structured", "C-full", "A", "SF") for t in ("nll", "static", "travel")]
    M = np.array([[_fam(grads[arm]["init_seed0"].get(t, {}), keys) for arm, t in cols] for _, keys in FAMROWS])
    im = a.imshow(np.log10(M + 1e-6), cmap="Blues", vmin=-6, vmax=1.5, aspect="auto")
    a.set_xticks(range(len(cols)), [f"{arm}\n{t}" for arm, t in cols], fontsize=6.3, rotation=90)
    a.set_yticks(range(len(FAMROWS)), [n for n, _ in FAMROWS], fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == 0 and not (FAMROWS[i][0].startswith(("F3", "F4", "F5")) and cols[j][0] in ("A", "SF")):
                a.text(j, i, "0", ha="center", va="center", fontsize=6, color="#593b13")
    fig.colorbar(im, ax=a, label="log10 gradient norm (seed-0 init, 12 training roots)")
    a.set_title("(d) which factor families each ABC9 term reaches ('0' = no path)")
    fig.tight_layout()
    return _save(fig, "P9_L1_exact_likelihood.png", "L1: exact evidence vs brute force, cost of untied copies, legacy copy disagreement, per-term factor reach")


ORDER = ["B-full", "B-quadratic", "B-structured", "B-none", "C-full", "A", "SF"]
OBJ = {"P8 multi-loss": "#8f99a3", "ABC9 joint": "#2f6f9f"}


def _rows():
    s = json.loads(Path("runs/p9/l2/summary.json").read_text())
    return s, [v for v in s["rows"].values() if v["objective"] in OBJ]


def l2_nll():
    """L2: exact root NLL (validation, disclosed test), calibrated repair NLL and scene NLL, P8 multi-loss vs ABC9 joint, per seed."""
    import numpy as np
    s, rows = _rows()
    panels = [("val_root_nll", "validation: mean exact root NLL"), ("test_root_nll", "disclosed P8c test: mean exact root NLL"),
              ("root.test.nll", "disclosed test: calibrated repair NLL (root model)"), ("root.test.scene_nll", "disclosed test: scene-bit NLL (z0, zS)")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (k, title) in zip(axs.flat, panels):
        for i, arm in enumerate(ORDER):
            for j, (obj, col) in enumerate(OBJ.items()):
                v = [r[k] for r in rows if r["arm"] == arm and r["objective"] == obj and r.get(k) is not None]
                if not v:
                    continue
                x = i + (j - 0.5) * 0.32
                ax.bar(x, np.mean(v), 0.3, color=col, alpha=0.75)
                ax.scatter([x] * len(v), v, s=12, color="black", zorder=3)
        ax.set_xticks(range(len(ORDER)), ORDER, fontsize=8)
        ax.set_title(title, fontsize=10)
    axs[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75, label=o) for o, c in OBJ.items()], fontsize=8)
    fig.suptitle("P9-L2: same init, same root batches, 200 updates; dots = seeds 0-2 (lower is better; test is a disclosed legacy diagnostic)", fontsize=10)
    fig.tight_layout()
    return _save(fig, "P9_L2_nll_vs_legacy.png", "L2: exact root NLL and calibrated NLLs, P8 multi-loss vs ABC9 joint, per arm and seed")


def l2_gradients():
    """L2: per-factor-family gradient norms over the 200 ABC9 updates (seed 0) and median pre-clip norms vs P8."""
    import numpy as np
    runs = {a: json.loads(Path(f"runs/p9/l2/runs/{a}#s0.json").read_text()) for a in ("B-full", "B-structured", "C-full", "SF")}
    fams = {"F2 sequence": ["order_s", "pairo", "mix", "gru", "h0"], "F3 scene unary": ["scene"], "F3 z0-zS pair": ["pair_scene"],
            "F4 zR-zO pair": ["pair_omit"], "F5 generic pairs": ["cpl"], "F5 higher-order": ["ho_u", "ho_s", "ho_q"],
            "F1 static": ["unary", "pairh"], "travel J": ["cost"], "SF head": ["head"], "DINOv2 projection": ["vproj"]}
    cols = dict(zip(fams, ["#7a5195", "#2f6f9f", "#5fa8d3", "#c26a1b", "#3b8a52", "#7fbf7b", "#8f99a3", "#593b13", "#d45087", "#bbbbbb"]))
    fig, axs = plt.subplots(1, 5, figsize=(19, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
    for ax, (arm, r) in zip(axs, runs.items()):
        h = [x for x in r["history"] if "grad_by_module" in x]
        for f, mods in fams.items():
            if not any(m in h[0]["grad_by_module"] for m in mods):
                continue
            v = np.array([sum(x["grad_by_module"].get(m, 0) ** 2 for m in mods) ** 0.5 for x in h])
            ax.plot(np.convolve(v, np.ones(10) / 10, mode="valid"), color=cols[f], lw=1.3, label=f)
        ax.set_yscale("log")
        ax.set_title(f"ABC9 {arm} seed 0 (clip {r['clip']:.0f}, clipped {r['clip_fraction']:.0%})", fontsize=9)
        ax.set_xlabel("update (10-update mean)")
    axs[0].set_ylabel("gradient norm by factor family")
    axs[0].legend(fontsize=6.5, loc="lower left")
    s, rows = _rows()
    ax = axs[4]
    for i, arm in enumerate(ORDER):
        for j, (obj, col) in enumerate(OBJ.items()):
            v = [r["grad"]["grad_pre_median"] if obj == "ABC9 joint" else r["grad_pre_median"] for r in rows if r["arm"] == arm and r["objective"] == obj]
            if v:
                ax.bar(i + (j - 0.5) * 0.35, np.mean(v), 0.33, color=col, alpha=0.8, label=obj if i == 0 else None)
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=45, fontsize=8)
    ax.set_yscale("log")
    ax.set_title("median pre-clip total gradient norm", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "P9_L2_gradient_health.png", "L2: per-factor gradient norms over ABC9 training (seed 0) and median pre-clip norms vs P8")


def l2_predictions():
    """L2: reliability of root-model repair probabilities (disclosed test, seed 0), selection funnel, and root-cluster contrasts."""
    import numpy as np
    import torch

    import abc8
    import p9_run as pr
    s, rows = _rows()
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3)
    _, _, test = pr._data()
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1)
    for arm, ls in (("B-full", "-"), ("B-structured", "--"), ("SF", ":")):
        for obj, path, tau_key in (("ABC9 joint", f"runs/p9/l2/ckpt/ckpt_{arm}#s0.pt", "ABC9"), ("P8 multi-loss", f"runs/p8/c_training/ckpt_{arm}#s0.pt", "P8")):
            m = abc8.build(arm)
            m.load_state_dict(torch.load(path, weights_only=True))
            tau = s["rows"][f"{tau_key}|{arm}#s0"]["root.tau"]
            ps_, ys = [], []
            for r in test:
                for c in pr.root_eval(m, r)["root"]["cands"]:
                    if c["y"] is not None:
                        ps_.append(1 / (1 + np.exp(-c["u"] / tau)))
                        ys.append(c["y"])
            ps_, ys = np.array(ps_), np.array(ys)
            b = np.minimum((ps_ * 5).astype(int), 4)
            pts = [(ps_[b == i].mean(), ys[b == i].mean()) for i in range(5) if (b == i).sum()]
            ax.plot(*zip(*pts), ls=ls, marker="o", ms=4, color=OBJ[obj], label=f"{arm} {obj}")
    ax.set(xlabel="calibrated p (root model)", ylabel="observed success rate", title="(a) reliability, disclosed test, seed 0")
    ax.legend(fontsize=6.5)
    ax = fig.add_subplot(gs[0, 1:])
    cats = [("funnel_success", "success", "#3b8a52"), ("funnel_abstain", "abstain", "#bbbbbb"),
            ("funnel_false_feasible_empty", "false-feasible empty", "#c0392b")]
    labels, x = [], 0
    for arm in ORDER:
        for obj in OBJ:
            v = [r for r in rows if r["arm"] == arm and r["objective"] == obj]
            if not v:
                continue
            bottom = 0
            for k, lab, col in cats:
                m_ = np.mean([r[f"root.test.{k}"] for r in v])
                ax.bar(x, m_, 0.7, bottom=bottom, color=col, label=lab if x == 0 else None)
                bottom += m_
            ax.bar(x, 12 - bottom, 0.7, bottom=bottom, color="#f1c40f", alpha=0.5, label="other failure" if x == 0 else None)
            labels.append(f"{arm}\n{'ABC9' if obj.startswith('ABC9') else 'P8'}")
            x += 1
    ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=40)
    ax.set(ylabel="test roots (of 12), mean over seeds", title="(b) restricted-catalogue selection on disclosed test (validation-chosen tau, threshold; saved records)")
    ax.legend(fontsize=7, ncol=4)
    ax = fig.add_subplot(gs[1, :])
    con = s["root_nll_contrasts_negative_favours_first"]
    names = [k for k in con if k.startswith("validation|")]
    for i, k in enumerate(names):
        for j, split in enumerate(("validation", "test_disclosed")):
            c = con.get(k.replace("validation|", f"{split}|"))
            if c:
                y = i + (j - 0.5) * 0.3
                ax.plot(c["ci95"], [y, y], color=["#2f6f9f", "#c26a1b"][j], lw=2)
                ax.scatter([c["mean"]], [y], color=["#2f6f9f", "#c26a1b"][j], s=18, label=split if i == 0 else None)
    ax.axvline(0, color="#888888", lw=1)
    ax.set_yticks(range(len(names)), [k.split("|", 1)[1] for k in names], fontsize=7.5)
    ax.set(xlabel="per-root difference in exact root NLL (seed-averaged; root-cluster bootstrap 95 % CI); negative favours the first",
           title="(c) factor and objective contrasts (12 validation / 24 disclosed-test roots; not a fresh holdout)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "P9_L2_predictions_selection.png", "L2: reliability, selection funnel and root-cluster NLL contrasts (disclosed legacy diagnostics)")


def _save3(fig, name, what, sources):
    """_save plus a <name>.manifest.json of SHA-256 for every input data / code file of the figure."""
    h = _save(fig, name, what)
    man = {"figure": f"visualizations/p9/{name}", "sha256": h,
           "inputs": {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}}
    (OUT / f"{name}.manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return h


def l3a_factor_graph():
    """L3a: a REAL saved validation root drawn twice - P9 PoE (every block re-applies its slots' unaries, one normalizer per
    block) vs unique factor (one circle and one unary square per physical query, one relational square per distinct block)."""
    import collections

    import torch

    import joint_likelihood as jl
    import p9_l3
    import unique_factor as uf
    _, val, _ = p9_l3.data()
    m = p9_l3.load("B-full", "runs/p9/l2/ckpt/ckpt_B-full#s0.pt")
    best = None
    for r in val:
        mult = collections.Counter(q for c in r["cands"] if c["seq"] for s, q in zip(c["slots"], c["z_q"]) if s[:2] in ("zR", "zO"))
        if mult and max(mult.values()) > 1 and (best is None or len(mult) < len(best[1])):
            best = (r, mult)
    r, mult = best
    with torch.no_grad():
        inst = uf.instances(m, r)
        pu = uf.prior_marginals(inst)
        S, blocks, _, _ = jl.tables(m, r)
        pp = jl.marginals(S, blocks)
    qs = sorted(mult, key=lambda q: (q.count("."), q))
    xq = {q: 1.2 + i * 10.6 / max(len(qs) - 1, 1) for i, q in enumerate(qs)}
    sc = inst["S"]["queries"]
    xs = {q: 2.5 + i * 7.5 / max(len(sc) - 1, 1) for i, q in enumerate(sc)}
    short = lambda q: q.split(":", 2)[2].replace(".", "\n")  # noqa: E731
    sname = dict(zip(sc, inst["S"]["names"]))
    blist = [c for c in r["cands"] if c["seq"]]
    fig, axs = plt.subplots(1, 2, figsize=(19, 8.2))
    for ax, kind in zip(axs, ("PoE", "unique")):
        ax.set_xlim(-0.5, 12.8)
        ax.set_ylim(-1.6, 8.6)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((1.6, 7.25), 9.4, 0.7, boxstyle="round,pad=0.05", fc=FAM["F3"], alpha=0.85, ec="none"))
        ax.text(6.3, 7.6, "scene block P_S (F3 + F5 scene factors), ONCE per root", ha="center", va="center",
                color="white", fontsize=8.5)
        for q, x in xs.items():
            ax.plot([x, x], [6.3, 7.25], color=FAM["F3"], lw=1)
            ax.add_patch(plt.Circle((x, 6.0), 0.36, fc="#dbeafe", ec="#285a84", lw=1.3, zorder=3))
            ax.text(x, 6.0, sname[q], ha="center", va="center", fontsize=7.5, zorder=4)
        for i, c in enumerate(blist):
            bx = 1.2 + i * 10.6 / max(len(blist) - 1, 1)
            qc = [q for s, q in zip(c["slots"], c["z_q"]) if s[:2] in ("zR", "zO")]
            lab = ("normalized\nP_Q,r (pairs+HO\n+ its unaries)" if kind == "PoE" else "pairs + HO\nof block r")
            for q in qc:
                ax.plot([bx, xq[q]], [3.9, 1.5], color=FAM["F4"], lw=0.9, alpha=0.7)
            for x in xs.values():
                ax.plot([bx, x], [4.3, 5.64], color=FAM["F5"], lw=0.4, alpha=0.35)
            ax.text(bx, 4.1, f"r{i + 1}\n" + lab, ha="center", va="center", fontsize=6.3, color="white",
                    bbox={"boxstyle": "square,pad=0.3", "fc": FAM["F5"] if kind == "unique" else "#7f7f7f", "ec": "none"})
        for q, x in xq.items():
            ax.add_patch(plt.Circle((x, 1.2), 0.42, fc="#dbeafe", ec="#285a84", lw=2.4 if mult[q] > 1 else 1.3, zorder=3))
            ax.text(x, 1.2, short(q), ha="center", va="center", fontsize=5.6, zorder=4)
            k = mult[q] if kind == "PoE" else 1
            for j in range(k):
                ux = x + (j - (k - 1) / 2) * 0.32
                ax.plot([x, ux], [0.78, 0.25], color=FAM["F2"], lw=0.9)
                ax.text(ux, 0.05, "u", ha="center", va="center", fontsize=6.5, color="white", bbox={"boxstyle": "square,pad=0.25", "fc": FAM["F2"], "ec": "none"})
            p = pp[q] if kind == "PoE" else pu[q]
            ax.text(x, -0.55, f"p={p:.3f}", ha="center", fontsize=6.8)
        ttl = ("P9 PoE: one variable per query, but a query in k blocks gets k unary squares and k block normalizers"
               if kind == "PoE" else "Unique factor: one circle, ONE unary (F2 phi_seq = F4 omission unary) per query; one normalizer per scene state")
        ax.set_title(ttl, fontsize=9.5)
    fig.suptitle(f"Real saved validation root {r['key']} (B-full, frozen P9-L2 seed-0 PoE-trained weights; prior marginals, no labels used). "
                 "Thick circle = query reused by several candidates", fontsize=10)
    fig.text(0.5, 0.02, "squares 'u': repair-outcome unaries; grey/green squares: per-candidate blocks (F4 zR-zO pairs, F5 scene-repair and zO-zO pairs, "
             "F5 higher order) - edges to scene bits drawn faintly; F1 static and travel J are outside this conditional graph", ha="center", fontsize=8.5)
    return _save3(fig, "P9_L3a_real_root_factor_graph.png", "L3a: literal factor graph of a saved root, PoE vs unique-factor (variable circles, factor squares)",
                  ["runs/p8/c_training/dataset.pt", "runs/p9/l2/ckpt/ckpt_B-full#s0.pt", "unique_factor.py", "joint_likelihood.py", "p9_viz.py"])


def l3a_invariance():
    """L3a: invariance cases (max |change| of prior marginals of the original queries) for PoE / dedup-PoE / unique, the
    duplicated 0.8 expert, and PoE vs unique prior marginals by unary multiplicity (frozen PoE-trained weights)."""
    import collections

    import numpy as np
    a = json.loads(Path("runs/p9/l3/l3a1/artifacts.json").read_text())
    rows = json.loads(Path("runs/p9/l3/l3a1/marginal_rows.json").read_text())
    cases = ["permutation", "renamed_ids", "exact_duplicate", "disconnected_candidate", "alias_as_new_candidate", "related_candidate"]
    forms = [("P9-PoE", "#7f7f7f"), ("dedup-PoE diagnostic", "#c26a1b"), ("unique-factor", "#2f6f9f")]
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4)
    for col, arm in enumerate(("B-full", "B-structured", "C-full")):
        ax = fig.add_subplot(gs[0, col])
        aggs = {}
        for form, _ in forms:
            aggs[form] = collections.defaultdict(list)
            for x in a["sensitivity"][f"{arm}|{form}"]:
                aggs[form][(x["case"], x["expected"])].append(x["max_abs_change_repair_marginals"])
        keys = sorted({k for g in aggs.values() for k in g}, key=lambda k: (cases.index(k[0]), k[1]))
        for i, (case, exp) in enumerate(keys):
            if exp.startswith("EXPECTED"):
                ax.axvspan(i - 0.45, i + 0.45, color="#e8f0fa", zorder=0)
            for j, (form, colr) in enumerate(forms):
                y = max(aggs[form][(case, exp)])
                ax.bar(i + (j - 1) * 0.27, max(y, 1e-6), 0.25, color=colr, alpha=0.95 if exp.startswith("EXPECTED") else 0.45,
                       label=form if i == 0 else None)
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 1.5)
        lab = lambda c, e: c.replace("_", "\n") + ("\n(invariant)" if c.startswith("alias") and e.startswith("EXP") else "\n(new factor)" if c.startswith("alias") else "")  # noqa: E731
        ax.set_xticks(range(len(keys)), [lab(c, e) for c, e in keys], fontsize=6.5)
        ax.set_title(f"{arm}: max |change| of prior marginals", fontsize=9.5)
        if col == 0:
            ax.set_ylabel("max |change| (bar at 1e-6 = exactly 0)")
            ax.legend(fontsize=7.5)
    ax = fig.add_subplot(gs[0, 3])
    ax.bar([0, 1, 2], [0.8, 0.64 / 0.68, 0.8], color=["#bbbbbb", "#7f7f7f", "#2f6f9f"])
    for i, v in enumerate([0.8, 0.64 / 0.68, 0.8]):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1, 2], ["one expert", "PoE, expert\nduplicated", "unique factor,\nduplicated"], fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("toy: one Bernoulli expert P(x=1)=0.8", fontsize=9.5)
    ax = fig.add_subplot(gs[1, :2])
    k = np.array([x["multiplicity"] for x in rows])
    pp, pu = np.array([x["poe_prior"] for x in rows]), np.array([x["unique_prior"] for x in rows])
    sc = ax.scatter(pu, pp, c=k, cmap="viridis", s=14)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
    fig.colorbar(sc, ax=ax, label="blocks containing the query (PoE unary multiplicity)")
    ax.set(xlabel="unique-factor prior marginal", ylabel="P9 PoE prior marginal", title="same frozen weights: PoE sharpens reused queries")
    ax = fig.add_subplot(gs[1, 2:])
    mm = a["poe_vs_unique_prior_abs_diff_by_multiplicity"]
    ks = sorted(mm, key=int)
    ax.bar(range(len(ks)), [mm[x]["mean"] for x in ks], color="#7f7f7f")
    ax.plot(range(len(ks)), [mm[x]["max"] for x in ks], "o", color="#c0392b", label="max")
    ax.set_xticks(range(len(ks)), [f"k={x}\nn={mm[x]['queries']}" for x in ks], fontsize=8)
    ax.set(ylabel="|PoE - unique| prior marginal", title="mean (bars) and max (dots) by multiplicity")
    ax.legend(fontsize=8)
    fig.suptitle("P9-L3a: catalogue invariance (shaded = EXPECTED_INVARIANT; faded bars = ALLOWED_CHANGE). Saved roots, frozen P9-L2 seed-0 weights, no labels",
                 fontsize=10)
    fig.tight_layout()
    return _save3(fig, "P9_L3a_invariance.png", "L3a: invariance cases PoE / dedup-PoE / unique-factor; duplicated-expert toy; PoE vs unique by multiplicity",
                  ["runs/p9/l3/l3a1/artifacts.json", "runs/p9/l3/l3a1/marginal_rows.json", "p9_viz.py"])


L3B_ARMS = ["PoE-B-full", "U-B-full", "PoE-B-structured", "U-B-structured", "U-B-full-geo", "PoE-C-full", "U-C-full", "SF", "A"]
L3B_COL = {"PoE": "#7f7f7f", "U": "#2f6f9f", "SF": "#d45087", "A": "#c26a1b"}


def _l3b():
    return json.loads(Path("runs/p9/l3/l3b/summary.json").read_text())


def l3b_comparison():
    """L3b: own-likelihood root NLL (validation / disclosed test), calibrated repair NLL and accuracy, per seed; root-cluster contrasts."""
    import numpy as np
    s = _l3b()
    rows = list(s["rows"].values())
    keys = [("val_root_nll", "validation: exact root NLL (own joint)"), ("test_root_nll", "disclosed test: exact root NLL (own joint)"),
            ("root.val.nll", "validation: calibrated repair NLL (root prior)"), ("root.test.accuracy", "disclosed test: repair accuracy (root prior)")]
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4)
    for i, (k, title) in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        for j, arm in enumerate(L3B_ARMS):
            v = [r[k] for r in rows if r["arm"] == arm and r.get(k) is not None and r.get("updates", 200) == 200]
            if v:
                ax.bar(j, np.mean(v), 0.7, color=L3B_COL[arm.split("-")[0]], alpha=0.8)
                ax.scatter([j] * len(v), v, s=10, color="black", zorder=3)
        ax.set_xticks(range(len(L3B_ARMS)), L3B_ARMS, rotation=60, fontsize=7)
        ax.set_title(title, fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    con = s["contrasts"]
    names = sorted({k.split("|", 1)[1] for k in con}, key=lambda n: list(dict.fromkeys(k.split("|", 1)[1] for k in con)).index(n))
    for i, n in enumerate(names):
        for j, split in enumerate(("validation", "test_disclosed")):
            c = con.get(f"{split}|{n}")
            if c:
                y = i + (j - 0.5) * 0.3
                ax.plot(c["ci95"], [y, y], color=["#2f6f9f", "#c26a1b"][j], lw=2)
                ax.scatter([c["mean"]], [y], color=["#2f6f9f", "#c26a1b"][j], s=18, label=("validation (predeclared)" if j == 0 else "disclosed test (diagnostic)") if i == 0 else None)
    ax.axvline(0, color="#888888", lw=1)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set(xlabel="per-root difference in own exact root NLL (seed-averaged; root-cluster bootstrap 95 % CI); negative favours the first")
    ax.legend(fontsize=8)
    fig.suptitle("P9-L3b: PoE vs unique-factor, matched 200 updates, seeds 0-2 (saved data only; prediction metrics, no execution)", fontsize=10)
    fig.tight_layout()
    return _save3(fig, "P9_L3b_comparison.png", "L3b: PoE vs unique-factor vs SF / A / geometry-only / C-full, root NLL, calibrated metrics, contrasts",
                  ["runs/p9/l3/l3b/summary.json", "runs/p9/l3/l3b/L3B_plan.json", "p9_viz.py"])


def l3b_health():
    """L3b: per-factor gradient norms (seed 0), clip fraction, trained-model catalogue sensitivity; optional 400-update curves."""
    import numpy as np
    s = _l3b()
    fams = {"F2 sequence": ["order_s", "pairo", "mix", "gru", "h0"], "F3 scene unary": ["scene"], "F3 z0-zS pair": ["pair_scene"],
            "F4 zR-zO pair": ["pair_omit"], "F5 generic pairs": ["cpl"], "F5 higher-order": ["ho_u", "ho_s", "ho_q"], "F1 static": ["unary", "pairh"],
            "travel J": ["cost"]}
    cols = dict(zip(fams, ["#7a5195", "#2f6f9f", "#5fa8d3", "#c26a1b", "#3b8a52", "#7fbf7b", "#8f99a3", "#593b13"]))
    curves = s.get("curves", {})
    fig, axs = plt.subplots(1, 4 if curves else 3, figsize=(22 if curves else 16, 4.8))
    for ax, (name, path) in zip(axs[:2], (("PoE-B-full (P9-L2)", "runs/p9/l2/runs/B-full#s0.json"), ("U-B-full", "runs/p9/l3/l3b/runs/U-B-full#s0.json"))):
        h = [x for x in json.loads(Path(path).read_text())["history"] if "grad_by_module" in x]
        for f, mods in fams.items():
            if any(m in h[0]["grad_by_module"] for m in mods):
                v = np.array([sum(x["grad_by_module"].get(m, 0) ** 2 for m in mods) ** 0.5 for x in h])
                ax.plot(np.convolve(v, np.ones(10) / 10, mode="valid"), color=cols[f], lw=1.2, label=f)
        ax.set_yscale("log")
        ax.set(title=f"{name} s0: gradient by family", xlabel="update (10-update mean)")
    axs[0].legend(fontsize=6.5)
    ax = axs[2]
    cases = ["exact_duplicate|EXPECTED_INVARIANT", "permutation|EXPECTED_INVARIANT", "disconnected_candidate|EXPECTED_INVARIANT", "related_candidate|ALLOWED_CHANGE"]
    arms = [a for a in L3B_ARMS if a not in ("SF", "A")]
    for j, c in enumerate(cases):
        v = [np.mean([r["sensitivity"].get(c, np.nan) for r in s["rows"].values() if r["arm"] == a and r.get("sensitivity")]) for a in arms]
        ax.bar(np.arange(len(arms)) + (j - 1.5) * 0.2, np.maximum(np.nan_to_num(v), 1e-6), 0.2, label=c.split("|")[0])
    ax.set_yscale("log")
    ax.set_xticks(range(len(arms)), arms, rotation=45, fontsize=7)
    ax.set(title="trained: max |change| (1e-6 = exactly 0)")
    ax.legend(fontsize=6.5)
    if curves:
        ax = axs[3]
        for name, v in curves.items():
            ax.plot([u for u, _ in v], [x for _, x in v], label=name, color=L3B_COL[name.split("-")[0]], alpha=0.6, lw=1)
        ax.set(title="400-update runs: validation root NLL", xlabel="update")
        ax.legend(fontsize=6)
    fig.tight_layout()
    return _save3(fig, "P9_L3b_health.png", "L3b: per-factor gradients, trained-model invariance, optional 400-update curves",
                  ["runs/p9/l3/l3b/summary.json", "runs/p9/l2/runs/B-full#s0.json", "runs/p9/l3/l3b/runs/U-B-full#s0.json", "p9_viz.py"])



def l3c_factor_graph():
    """L3c: a real saved validation root drawn as the unique model (relations per listed parent and role) and the keyed model
    (one relation per physical scope). Squares show each relation's number of factor instances."""
    import collections

    import p9_l3 as l3
    import unique_factor as uf
    import unique_factor_keyed as ufk
    _, val, _ = l3.data()
    r = next(x for x in val if x["key"] == "p8_2026102510")
    m = l3.load("B-full", "runs/p9/l3/l3b/ckpt/ckpt_U-B-full#s0.pt")
    insts = {"unique ABC9U_UNIQUE_V1": uf.instances(m, r), "keyed ABC9K_KEYED_V1": ufk.instances(m, r)}
    sc = insts["unique ABC9U_UNIQUE_V1"]["S"]["names"]
    qs = sorted(insts["unique ABC9U_UNIQUE_V1"]["unary"], key=lambda q: (q.count("."), q))
    roles = collections.defaultdict(set)
    for c in r["cands"]:
        for sl, q in zip(c["slots"], c["z_q"]):
            if sl[:2] in ("zR", "zO"):
                roles[q].add(sl[:2])
    fig, axs = plt.subplots(1, 2, figsize=(19, 8.6))
    for ax, (name, inst) in zip(axs, insts.items()):
        ax.set_xlim(-0.6, 12.6)
        ax.set_ylim(-1.4, 9.0)
        ax.axis("off")
        xs = {n: 2.5 + i * 7.5 / (len(sc) - 1) for i, n in enumerate(sc)}
        xq = {q: 0.9 + i * 10.8 / (len(qs) - 1) for i, q in enumerate(qs)}
        ax.text(6.3, 8.6, "P_S: F3 scene unaries + z0-zS pairs, F5 scene pairs + ho_s - ONCE", ha="center", fontsize=8.5, color="white",
                bbox={"boxstyle": "square,pad=0.35", "fc": FAM["F3"], "ec": "none"})
        for n, x in xs.items():
            ax.plot([x, 6.3], [7.3, 8.45], color=FAM["F3"], lw=0.8)
            ax.add_patch(plt.Circle((x, 7.0), 0.34, fc="#dbeafe", ec="#285a84", lw=1.3, zorder=3))
            ax.text(x, 7.0, n, ha="center", va="center", fontsize=7, zorder=4)
        sq = collections.Counter()
        ho = []
        for b, blk in inst["blocks"].items():
            ho.append(blk["queries"])
            for q in (blk["queries"] if name.startswith("unique") else blk["queries"][:1]):
                sq[q] += 1
        for q, x in xq.items():
            ax.plot([x, x], [5.0, 6.65], color=FAM["F5"], lw=0.5, alpha=0.5)
            for sx in xs.values():
                ax.plot([x, sx], [5.0, 6.66], color=FAM["F5"], lw=0.25, alpha=0.35)
            ax.text(x, 4.75, f"scene-query\npairs x{sq[q]}\n({len(sc)} bits each)", ha="center", va="center", fontsize=6.2, color="white",
                    bbox={"boxstyle": "square,pad=0.25", "fc": FAM["F5"], "ec": "none"})
            alias = len(roles[q]) > 1 or sq[q] > 1
            ax.add_patch(plt.Circle((x, 1.4), 0.46, fc="#dbeafe", ec="#285a84", lw=2.6 if alias else 1.3, zorder=3))
            ax.text(x, 1.4, q.split(":", 2)[2].replace(".", "\n"), ha="center", va="center", fontsize=6, zorder=4)
            ax.text(x, 0.55, "unary x1", ha="center", fontsize=6.2, color="white", bbox={"boxstyle": "square,pad=0.2", "fc": FAM["F2"], "ec": "none"})
            ax.text(x, -0.15, "listed as " + "+".join(sorted(roles[q])), ha="center", fontsize=6.2)
            ax.plot([x, x], [1.85, 4.45], color=FAM["F5"], lw=0.6)
        for j, scope in enumerate(ho):
            hx = 0.9 + j * 10.8 / max(len(ho) - 1, 1)
            for q in scope:
                ax.plot([hx, xq[q]], [3.0, 1.86], color=FAM["F4"], lw=0.8, alpha=0.7)
            ax.text(hx, 3.0, f"HO + zR-zO\nof {scope[0].split(':', 2)[2]}", ha="center", va="center", fontsize=5.8, color="white",
                    bbox={"boxstyle": "square,pad=0.25", "fc": FAM["F4"], "ec": "none"})
        ax.set_title(f"{name}: {len(inst['blocks'])} relational blocks; scene-query instances {sum(sq.values()) * len(sc)}", fontsize=10)
    fig.suptitle(f"Saved validation root {r['key']} (frozen U-B-full seed-0 weights; structure only). Circles: binary outcomes; squares: factor instances. "
                 "Thick circle: query listed in several roles or parents", fontsize=10)
    return _save3(fig, "P9_L3c_factor_graph_unique_vs_keyed.png", "L3c: real-root factor instances, unique (per listed parent / role) vs keyed (per physical scope)",
                  ["runs/p8/c_training/dataset.pt", "runs/p9/l3/l3b/ckpt/ckpt_U-B-full#s0.pt", "unique_factor.py", "unique_factor_keyed.py", "p9_viz.py"])


def l3c_selection():
    """L3c: prior change per catalogue transformation (unique vs keyed, frozen weights), deployed-selector outcomes, and the
    validation calibration of the L3d models."""
    import collections

    import numpy as np
    u = json.loads(Path("runs/p9/l3c/l3c0/transformations.json").read_text())
    k = json.loads(Path("runs/p9/l3c/l3c1/transformations_keyed.json").read_text())
    summ = json.loads(Path("runs/p9/l3d/summary.json").read_text())
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.2), gridspec_kw={"width_ratios": [1.3, 1.3, 1.3, 1]})
    for ax, arm in zip(axs[:3], ("B-full", "B-structured", "C-full")):
        cases = []
        for rows in (u[arm], k[arm]):
            for x in rows:
                if (x["case"], x["expected"]) not in cases:
                    cases.append((x["case"], x["expected"]))
        for j, (rows, lab, col) in enumerate(((u[arm], "unique", "#7f7f7f"), (k[arm], "keyed", "#2f6f9f"))):
            agg = collections.defaultdict(list)
            for x in rows:
                agg[(x["case"], x["expected"])].append(x)
            for i, c in enumerate(cases):
                if c in agg:
                    v = max(x["max_abs_prior_change"] for x in agg[c])
                    sw = sum(x["deployed_choice_changed"] and not x["chose_new_unlabelled_candidate"] for x in agg[c])
                    ax.bar(i + (j - 0.5) * 0.38, max(v, 1e-6), 0.36, color=col, alpha=0.95 if c[1].startswith("EXP") else 0.5, label=lab if i == 0 else None)
                    if sw:
                        ax.text(i + (j - 0.5) * 0.38, max(v, 1e-6) * 1.3, f"{sw} sw", ha="center", fontsize=6)
        ax.set_yscale("log")
        ax.set_ylim(5e-7, 2)
        ax.set_xticks(range(len(cases)), [c[0].replace("alias: ", "").replace("_", " ")[:24] + ("" if c[1].startswith("EXP") else " *") for c in cases],
                      rotation=60, fontsize=6.3, ha="right")
        ax.set_title(f"{arm}: max |prior change| (1e-6 = exactly 0; * = allowed / decomposition)", fontsize=8.5)
        ax.legend(fontsize=7)
    ax = axs[3]
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1)
    for name, col in (("full-vision U-B-full (400)", "#2f6f9f"), ("visual-zero U-B-full (400)", "#c26a1b"), ("keyed K-B-full (200)", "#3b8a52"),
                      ("unique U-B-full (200)", "#7f7f7f")):
        b = summ["calibration_validation"].get(name)
        if b:
            ax.plot([x[0] for x in b], [x[1] for x in b], "o-", color=col, ms=4, label=name)
    ax.set(xlabel="calibrated root-prior p (validation)", ylabel="observed success rate", title="validation calibration (pooled seeds)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    return _save3(fig, "P9_L3c_selection_and_calibration.png", "L3c/L3d: prior and deployed-selection changes per transformation, unique vs keyed; validation calibration",
                  ["runs/p9/l3c/l3c0/transformations.json", "runs/p9/l3c/l3c1/transformations_keyed.json", "runs/p9/l3d/summary.json", "p9_viz.py"])


def l3d_visual():
    """L3d: full-vision vs visual-zero (same architecture), validation curves, grouped NLLs and the dominant validation roots."""
    import numpy as np
    s = json.loads(Path("runs/p9/l3d/summary.json").read_text())
    rows = s["rows"]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.2))
    ax = axs[0]
    for n, r in rows.items():
        if n.startswith(("U-B-full#s", "V0-")) and "u400" in n:
            ax.plot([u for u, _ in r["curve"]], [v for _, v in r["curve"]], color="#2f6f9f" if n.startswith("U") else "#c26a1b", alpha=0.75,
                    label=("DINOv2 + geometry" if n.startswith("U") else "visual-zero (same architecture)") if n.endswith("s0#u400") else None)
    ax.set(xlabel="update", ylabel="validation root NLL (own joint)", title="validation curves, seeds 0-2 (400 updates)")
    ax.legend(fontsize=8)
    ax = axs[1]
    keys = [("val_root_nll", "root NLL"), ("val_nll_per_query", "per query"), ("val_repair_nll", "repair NLL"), ("val_scene_nll", "scene NLL")]
    for j, (pre, col, lab) in enumerate((("U-B-full#s", "#2f6f9f", "DINOv2"), ("V0-B-full#s", "#c26a1b", "visual-zero"))):
        vals = [[r[k] for n, r in rows.items() if n.startswith(pre) and "u400" in n] for k, _ in keys]
        ax.bar(np.arange(len(keys)) + (j - 0.5) * 0.38, [np.mean(v) for v in vals], 0.36, color=col, alpha=0.85, label=lab)
        for i, v in enumerate(vals):
            ax.scatter([i + (j - 0.5) * 0.38] * len(v), v, s=10, color="black", zorder=3)
    ax.set_xticks(range(len(keys)), [b for _, b in keys])
    ax.set_title("validation (12 roots), seed dots", fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[2]
    c = s["contrasts_validation"]["visual: DINOv2 - zero (400 upd)"]
    ax.errorbar([0], [c["mean"]], yerr=[[c["mean"] - c["ci95"][0]], [c["ci95"][1] - c["mean"]]], fmt="o", color="#2f6f9f", capsize=5, label="root-cluster 95% CI")
    ax.plot([0, 0], c["leave_one_root_out"], color="#c26a1b", lw=6, alpha=0.5, label="leave-one-root-out range")
    for i, (kk, v) in enumerate(c["largest_root_differences"].items()):
        ax.scatter([0.25 + 0.1 * i], [v], s=20, color="#7f7f7f")
        ax.text(0.27 + 0.1 * i, v, kk.split("_", 1)[-1], fontsize=6.5)
    ax.axhline(0, color="#999999", lw=1)
    ax.set_xlim(-0.4, 0.8)
    ax.set_xticks([])
    ax.set(ylabel="per-root validation NLL difference (DINOv2 - zero)",
           title=f"contrast; top-3 share DINOv2 {c['top3_share']['a']:.0%}, zero {c['top3_share']['b']:.0%}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save3(fig, "P9_L3d_visual_control.png", "L3d: full-vision vs visual-zero B-full (same architecture), validation curves, grouped NLL, dominant roots",
                  ["runs/p9/l3d/summary.json", "runs/p9/l3d/L3D_plan.json", "p9_viz.py"])

if __name__ == "__main__":
    import sys
    print(globals()[sys.argv[1]]())
