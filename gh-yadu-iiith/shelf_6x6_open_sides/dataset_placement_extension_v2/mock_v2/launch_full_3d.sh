#!/bin/bash
# SHELF66V3 placement extension V2 -- FULL 3D placement collection launcher.  INERT unless the user has RECORDED an explicit authorization.
# Starts nothing unless ALL hold: --execute-full --confirm <token> --policy-file <production_policy.json carrying the user's authorization>;
# guard3d.check (every frozen hash == the current one, token == token(current hashes), <= 18 CPUs, sensor readable, no other collector);
# a metadata-only dry run of the exact command (0 planner calls); the full test suite.  A default / no-argument invocation is refused.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE"); PY=${PY:-$HOME/approachb3d_mock/env/bin/python}
refuse() { echo "REFUSED: $*" >&2; exit 2; }
[ $# -eq 0 ] && refuse "no arguments (this launcher never starts on a default invocation)"
EXEC=0; TOKEN=""; POL=""
while [ $# -gt 0 ]; do case "$1" in
  --execute-full) EXEC=1; shift;;
  --confirm) TOKEN="${2:-}"; shift 2;;
  --policy-file) POL="${2:-}"; shift 2;;
  *) refuse "unknown argument $1";; esac; done
[ "$EXEC" -eq 1 ] || refuse "--execute-full missing"
[ -n "$TOKEN" ] || refuse "--confirm <token> missing"
[ -f "$POL" ] || refuse "--policy-file missing or not a file"
"$PY" "$ROOT/code/guard3d.py" check "$POL" "$TOKEN" || refuse "guard3d.check failed (reasons above)"
# ---- reachable only with a recorded authorization and matching hashes ----
read -r RUN ROSTER POLICY WORKERS CPUS PIN < <("$PY" -c "import json,sys;p=json.load(open(sys.argv[1]));print(p['run_dir'],p['roster'],p['policy'],p['workers'],p['cpus'],p['source_pin'])" "$POL")
cd "$ROOT" || refuse "cannot enter $ROOT"
A=(--roster "$ROSTER" --policy "$POLICY" --workers "$WORKERS" --cpus "$CPUS" --source-pin "$PIN" --production --authorization "$POL" --confirm "$TOKEN")
"$PY" code/run3d.py "$RUN" "${A[@]}" --dry-run || refuse "metadata dry run failed"
"$PY" -m pytest -q -p no:cacheprovider code/test_mock3d_v2.py || refuse "test suite failed"
exec nice -n 10 "$PY" code/run3d.py "$RUN" "${A[@]}"
