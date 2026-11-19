# trainv4_all_solvable: train = all solvable V4 train roots; validation = all solvable V4 val roots (the split's own validation part); valsub = 60 of them
import json, os, sys, shutil, random
from pathlib import Path
B = Path.home() / "learned_reap"; H = B / "reap_shelf"
os.environ.update(DATASET66=str(B / "dataset66"), SHELF66_V3_DIR=str(B / "bd_samples/shelf66_v3"), YADU_LIB_DIR=str(B / "lib66v2"))
sys.path.insert(0, str(H))
import make_train66 as M, lib66
split = json.loads((H / "split_v4.json").read_text())
test = {r for ids in split["test"].values() for r in ids}
built = {d.name for d in lib66.LIB_DIR.iterdir() if len(list(d.glob("src_*.pkl"))) == 36}
def covered(rec):
    cc = lib66.ceil_class(rec["objects"])
    return all(lib66.combo_name(lib66.mover_class(o), cc) in built for o in rec["objects"])
out = H / "trainv4_all_solvable"; shutil.rmtree(out, ignore_errors=True)
for p in ("train", "validation"): (out / p).mkdir(parents=True)
shutil.copy(H / "data" / "domain.pddl", out / "domain.pddl")
sk = {}
for part, dst in (("train", "train"), ("val", "validation")):
    for c, ids in split[part].items():
        if c == "NONE": continue
        k = sk = 0
        for rid in sorted(ids):
            assert rid not in test
            rec = json.loads(M.find_root(rid).read_text())
            if not covered(rec): sk += 1; continue
            rec["_root_id"] = rid; M.write_problem(rec, c, out / dst); k += 1
        print(part, "class", c, "roots", len(ids), "written", k, "skipped (library does not cover)", sk)
(out / "DONE").touch()
# valsub: 60 validation scenes, class-proportional, for the snapshot evaluations (real search + teacher)
vs = H / "data" / "valsub"; shutil.rmtree(vs, ignore_errors=True); vs.mkdir(parents=True)
by = {}
for f in sorted((out / "validation").glob("*.json")): by.setdefault(json.loads(f.read_text())["cls"], []).append(f)
n = sum(len(v) for v in by.values())
for c, fs in by.items():
    for f in random.Random("valsub:" + c).sample(fs, round(60 * len(fs) / n)):
        shutil.copy(f, vs); shutil.copy(str(f)[:-5] + ".pddl", vs)
print({p: len(list((out / p).glob("*.json"))) for p in ("train", "validation")}, "valsub", len(list(vs.glob("*.json"))))
