"""Download the team's scaling2d release_v2 (branch naren/scaling2d-k12-test) into runs/explore/scaling2d. Never overwrites (mode 'x')."""
import json, subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "approachb_3d_learning/runs/explore/scaling2d"
API = "repos/Narendhiranv04/latent-causal-tamp/contents/scaling2d_k12/G7/release_v2"
REF = "?ref=naren/scaling2d-k12-test"


def gh(path, raw=True):
    cmd = ["gh", "api", API + path + REF] + (["-H", "Accept: application/vnd.github.raw"] if raw else [])
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def get(rel):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        print("exists, skipped:", rel)
        return
    with open(p, "xb") as f:
        f.write(gh("/" + rel))


for rel in ("SUMMARY.json", "private/LABELS.jsonl", "private/COVERAGE.jsonl"):
    get(rel)
for e in json.loads(gh("/public/roots", raw=False)):
    get("public/roots/" + e["name"])
print("files:", sum(1 for _ in OUT.rglob("*") if _.is_file()))
