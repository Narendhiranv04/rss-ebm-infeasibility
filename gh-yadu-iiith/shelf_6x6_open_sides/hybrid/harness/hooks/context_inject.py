#!/usr/bin/env python3
"""Claude Code SessionStart hook (matcher compact|resume): load hybrid/CONTEXT.md into the model's context so working state
survives compaction.  Prints nothing if the file is missing.

    python3 context_inject.py --selftest
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_config as HC  # noqa: E402


CAP = 9000  # hook context is truncated past ~10k chars; the full file stays on disk


def payload(root=None):
    root = root or HC.project_root()
    cfg = HC.load(root)
    if cfg is None:
        return None  # not a research project
    CTX = Path(root) / cfg["context_file"]
    if not CTX.exists():
        return None
    text = CTX.read_text()
    if len(text) > CAP:
        text = text[:CAP] + f"\n\n[... truncated at {CAP} chars; Read {CTX} for the rest]"
    return {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                   "additionalContext": f"Working-state file {CTX} (kept current by the harness; read it before continuing):\n\n" + text}}


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            assert payload(d) is None                                   # no config -> nothing
            os.makedirs(d + "/.claude"); open(d + "/.claude/research_harness.json", "w").write("{}"); open(d + "/CONTEXT.md", "w").write("state")
            p = payload(d); assert p and "state" in p["hookSpecificOutput"]["additionalContext"]
        print("selftest ok"); sys.exit(0)
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = {}
    p = payload(HC.project_root(data))
    if p:
        print(json.dumps(p))
