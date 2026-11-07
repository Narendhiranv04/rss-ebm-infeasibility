#!/bin/bash
# SHELF66V3 placement collection -- FIRST K0-K3 COHORT (production v3) launcher.  INERT unless the user has RECORDED an explicit authorization.
# Starts nothing unless ALL hold, in this order:
#   --execute --policy-file <the user's filled production_policy.json> --confirm <token>, SHELF_SRC exported;
#   collect3d --dry-run   metadata preflight of every roster row (0 teacher calls; every non-authorization condition holds);
#   prod3d check          authorization + token + every frozen hash + implemented schedule + QC roots + source pin + lock + sensor + disk + idle CPUs;
#   the V3 and V2 test suites;
# then exec's the collector, which repeats the check and holds the exclusive lock for the whole run.  A default invocation is refused.
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
PRE=$("$PY" code/collect3d.py --policy-file "$POL" --dry-run) || { echo "$PRE"; refuse "metadata preflight failed"; }
echo "preflight: $PRE" | cut -c1-400
"$PY" code/prod3d.py check "$POL" "$TOKEN" || refuse "authorization / hash / lock / resource check failed (reasons above)"
"$PY" -m pytest -q -p no:cacheprovider code/test_prod3d_v3.py code/test_mock3d_v2.py || refuse "test suite failed"
exec nice -n 10 taskset -c 0-11 "$PY" code/collect3d.py --policy-file "$POL" --confirm "$TOKEN"
