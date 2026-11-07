#!/bin/bash
# SHELF66V3 placement collection -- Q2_3D_PARITY_V1 roster (production v4) launcher.  INERT unless the user has RECORDED an explicit
# authorization FOR V4 (a v3 authorization or token is refused: different version, roster, code and hashes).  Runs, in order:
#   collect4 --dry-run   metadata preflight: the roster regenerated from SHELF_SRC and compared line by line, 0 teacher calls
#   prod4 check          authorization + token + every frozen hash + schedule + QC roots + pin + lock + sensor + disk + idle CPUs 0-11
#   the V4 and V2 test suites
# then exec's the collector (which repeats the check and holds the exclusive lock).  A default invocation is refused.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE"); PY=${PY:-$HOME/approachb3d_mock/env/bin/python}
refuse() { echo "REFUSED: $*" >&2; exit 2; }
[ $# -eq 0 ] && refuse "no arguments (this launcher never starts on a default invocation)"
EXEC=0; TOKEN=""; POL=""
while [ $# -gt 0 ]; do case "$1" in
  --execute) EXEC=1; shift;;
  --confirm) TOKEN="${2:-}"; shift 2;;
  --policy-file) POL="${2:-}"; shift 2;;
  *) refuse "unknown argument $1";; esac; done
[ "$EXEC" -eq 1 ] || refuse "--execute missing"
[ -n "$TOKEN" ] || refuse "--confirm <token> missing"
[ -f "$POL" ] || refuse "--policy-file missing or not a file"
[ -n "${SHELF_SRC:-}" ] || refuse "SHELF_SRC is not exported"
POL=$(cd "$(dirname "$POL")" && pwd)/$(basename "$POL")
cd "$ROOT" || refuse "cannot enter $ROOT"
PRE=$("$PY" code/collect4.py --policy-file "$POL" --dry-run) || { echo "$PRE"; refuse "metadata preflight failed"; }
echo "preflight: $PRE" | cut -c1-400
"$PY" code/prod4.py check "$POL" "$TOKEN" || refuse "authorization / hash / lock / resource check failed (reasons above)"
"$PY" -m pytest -q -p no:cacheprovider code/test_prod4.py code/test_mock3d_v2.py || refuse "test suite failed"
exec nice -n 10 taskset -c 0-11 "$PY" code/collect4.py --policy-file "$POL" --confirm "$TOKEN"
