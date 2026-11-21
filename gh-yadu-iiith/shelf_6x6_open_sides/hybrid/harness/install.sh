#!/usr/bin/env bash
# Install the research harness globally for Claude Code (run it yourself; Claude's own guard blocks it from writing to ~/.claude).
#   1. copies the hooks to ~/.claude/research-harness/
#   2. adds the three hooks to ~/.claude/settings.json (backup first; existing settings kept; no duplicates)
#   3. appends the research rules to ~/.claude/CLAUDE.md (skipped if already there)
#   4. removes this project's old project-local copies of the same hooks from .claude/settings.local.json (so they don't run twice)
# Re-running is safe.  Uninstall: delete ~/.claude/research-harness, the 3 hook entries, and the marked block in CLAUDE.md.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/.claude/research-harness"
mkdir -p "$DST"
cp "$SRC"/hooks/{harness_config,guard_destructive,require_da_review,context_inject,init_project}.py "$DST"/
for f in guard_destructive require_da_review context_inject; do python3 "$DST/$f.py" --selftest; done

python3 - "$DST" <<'EOF'
import json, os, shutil, sys, time
dst = sys.argv[1]
home = os.path.expanduser("~/.claude")
sp = os.path.join(home, "settings.json")
s = json.load(open(sp)) if os.path.exists(sp) else {}
if os.path.exists(sp):
    shutil.copy(sp, sp + time.strftime(".bak-%Y%m%d-%H%M%S"))
want = {
    "PreToolUse": {"matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit", "script": "guard_destructive.py", "timeout": 10},
    "Stop": {"matcher": None, "script": "require_da_review.py", "timeout": 20},
    "SessionStart": {"matcher": "startup|resume|clear|compact", "script": "context_inject.py", "timeout": 10},
}
hooks = s.setdefault("hooks", {})
for ev, w in want.items():
    cmd = f"python3 {dst}/{w['script']}"
    groups = hooks.setdefault(ev, [])
    if any(h.get("command") == cmd for g in groups for h in g.get("hooks", [])):
        continue
    g = {"hooks": [{"type": "command", "command": cmd, "timeout": w["timeout"]}]}
    if w["matcher"]:
        g["matcher"] = w["matcher"]
    groups.append(g)
json.dump(s, open(sp, "w"), indent=2)
print("hooks in", sp)
EOF

MD="$HOME/.claude/CLAUDE.md"
if ! grep -q "research-harness:begin" "$MD" 2>/dev/null; then
  [ -f "$MD" ] && cp "$MD" "$MD.bak-$(date +%Y%m%d-%H%M%S)"
  { [ -s "$MD" ] && echo; cat "$SRC/research_CLAUDE.md"; } >> "$MD"
  echo "research rules appended to $MD"
else
  echo "research rules already in $MD (left unchanged)"
fi

# this project's old local hook entries -> drop them (the global ones replace them)
LOCAL="$(git -C "$SRC" rev-parse --show-toplevel)/.claude/settings.local.json"
if [ -f "$LOCAL" ]; then
  python3 - "$LOCAL" <<'EOF'
import json, shutil, sys, time
p = sys.argv[1]; s = json.load(open(p)); n = 0
for ev, groups in list(s.get("hooks", {}).items()):
    keep = []
    for g in groups:
        g["hooks"] = [h for h in g.get("hooks", []) if "hybrid/harness/hooks/" not in h.get("command", "") or not (n := n + 1)]
        if g["hooks"]:
            keep.append(g)
    s["hooks"][ev] = keep
    if not keep:
        del s["hooks"][ev]
if n:
    shutil.copy(p, p + time.strftime(".bak-%Y%m%d-%H%M%S"))
    json.dump(s, open(p, "w"), indent=2)
print(f"removed {n} project-local hook entries from {p}")
EOF
fi
echo "Done. Open /hooks once (or restart Claude Code) so it picks up the global hooks."
