"""Write production_policy.TEMPLATE.json: the exact frozen inputs a FUTURE full collection would need, with EMPTY authorization fields.
guard3d.check refuses it until the user records authorized_by / authorized_utc / statement AND passes --confirm token(current hashes).
  python prod_template.py POLICY OUT.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard3d, policy3d as Q  # noqa: E401
pol = sys.argv[1]
doc = {"version": "SHELF66V3_PLACE_PRODUCTION_V2", "status": "TEMPLATE -- NOT AUTHORIZED (full collection not started)",
       "policy": pol, "stages": (Q.POLICIES | Q.FIXED)[pol], "workers": 12, "cpus": "0-11", "run_dir": "production_v2/run",
       "roster": "production_v2/production_roster_live.jsonl.gz", "source_pin": "production_v2/source_pin_live.json",
       "split_manifest": "production_v2/split_manifest_live.json",
       "source_layout": "SHELF_SRC = a dataset directory whose raw/ and selected.json are the pinned live snapshot (e.g. a checkout of"
                        " naren/3d-dataset-relocations, gh-yadu-iiith/shelf_6x6_open_sides/dataset_bd_classquota_without_home_position_no_overpass)",
       "authorization": {"authorized_by": None, "authorized_utc": None, "statement": None}, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
cur = guard3d.current(doc)
doc["frozen"] = {k: cur[k] for k in ("roster_sha256", "source_pin_sha256", "split_sha256", "code_sha256", "teacher_protocol")}
doc["token_for_current_hashes"] = "computed at authorization time: python code/guard3d.py token <policy file>"
Path(sys.argv[2]).write_text(json.dumps(doc, indent=1, sort_keys=True)); print(json.dumps(doc["frozen"])[:600])
