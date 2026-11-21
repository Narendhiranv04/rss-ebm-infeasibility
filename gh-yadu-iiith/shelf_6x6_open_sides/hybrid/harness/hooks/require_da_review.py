#!/usr/bin/env python3
"""Claude Code Stop hook (v2, after an ARS Devil's Advocate review): when files under hybrid/ changed since the user's last real
prompt, block stopping until (a) an Agent call that loads the ARS devils_advocate_reviewer_agent.md role ran this turn after the
first change, and (b) hybrid/CONTEXT.md was updated after the last change.

Changes are found by file mtimes under hybrid/ (so Bash, script and subagent writes count), not by parsing tool calls.
"Real prompt" excludes task notifications, agent hand-backs and other meta entries.  Blocks at most once per stop
(stop_hook_active), as Claude Code requires to avoid loops; fails open on any error.  Reason: user rules 2026-11-21
("always run the devils advocate reviewer"; keep a context file because of compaction).

    python3 require_da_review.py --selftest
"""
import json, os, sys, tempfile, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_config as HC  # noqa: E402

ROLE = "devils_advocate_reviewer_agent.md"
SKIP_DIRS = {"__pycache__", ".claude", ".git", "node_modules", ".venv", "venv"}
META = ("<task-notification", "[SYSTEM NOTIFICATION", "Another Claude session sent a message", "<system-reminder", "<command-",
        "<agent-message", "Stop hook feedback", "[Subagent hand-back]")  # not new user turns: agent reports and this hook's own feedback


def ts(e):
    try:
        return datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()
    except (KeyError, ValueError, AttributeError):
        return None


def text_of(e):
    c = (e.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return ""
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def is_prompt(e):
    if e.get("type") != "user" or e.get("isMeta") or e.get("isSidechain") or e.get("isCompactSummary"):  # compaction is not a new turn
        return False
    t = text_of(e).lstrip()
    return bool(t) and not t.startswith(META)


def da_times(es):
    out = []
    for e in es:
        if e.get("type") != "assistant":
            continue
        c = (e.get("message") or {}).get("content")
        for b in c if isinstance(c, list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Agent" and ROLE in json.dumps(b.get("input") or {}):
                out.append(ts(e))
    return [t for t in out if t is not None]


def mtime(p):
    try:
        return os.stat(p).st_mtime
    except OSError:  # broken symlink, file removed meanwhile
        return None


def changes_since(watch, since, areas, ctx_file):
    """mtimes of files changed after `since` under watch/<area> (all of watch if areas is empty), and the context file's mtime.
    user 2026-11-21: "don't do a DA review on everything": only these areas need one."""
    other = []
    for top in ([watch / a for a in areas] if areas else [watch]):
        walk = os.walk(top) if top.is_dir() else [(str(top.parent), [], [top.name])] if top.exists() else []
        for dp, dns, fns in walk:
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for f in fns:
                p = Path(dp) / f
                m = mtime(p)
                if m and m > since and p != ctx_file:
                    other.append(m)
    return other, mtime(ctx_file)


def decide(es, root, cfg):
    prompts = [ts(e) for e in es if is_prompt(e) and ts(e) is not None]
    if not prompts:
        return None
    t0 = prompts[-1]
    watch, ctx_file = Path(root) / cfg["watch_dir"], Path(root) / cfg["context_file"]
    changed, ctx = changes_since(watch, t0, cfg["review_areas"], ctx_file)
    if not changed:
        return None
    first, last = min(changed), max(changed)
    reasons = []
    if ctx is None or ctx < last:
        reasons.append(f"Update {cfg['context_file']} (compaction-safe working state: results, in-flight jobs, next steps) to reflect this turn's changes.")
    if not any(t >= first - 1 for t in da_times([e for e in es if (ts(e) or 0) >= t0])):
        reasons.append(f"Reviewed files ({', '.join(cfg['review_areas']) or 'all'} under {cfg['watch_dir']}) changed and no Devil's Advocate review ran "
                       f"this turn: dispatch ONE Agent (model: {cfg['review_model']}, narrow brief: just what changed and what to check) that "
                       f"reads the ARS role file academic-paper-reviewer/agents/{ROLE}, then adjudicate its CRITICAL issues visibly. "
                       f"Batch fixes; do not re-review fixes the reviewer itself proposed.")
    return " ".join(reasons) or None


def selftest():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); (root / "src").mkdir(); f = root / "src" / "a.py"; ctx = root / "CONTEXT.md"; cfg = dict(HC.DEFAULTS)
        d_ = lambda es, r: decide(es, r, cfg)  # noqa: E731
        now = time.time()
        iso = lambda t: datetime.fromtimestamp(t).astimezone().isoformat()  # noqa: E731
        prompt = {"type": "user", "timestamp": iso(now - 100), "message": {"content": "do it"}}
        notif = {"type": "user", "timestamp": iso(now - 10), "message": {"content": "<task-notification> done"}}
        da = {"type": "assistant", "timestamp": iso(now - 20), "message": {"content": [{"type": "tool_use", "name": "Agent", "input": {"prompt": "read .../" + ROLE}}]}}
        fake = {"type": "assistant", "timestamp": iso(now - 20), "message": {"content": [{"type": "tool_use", "name": "Agent", "input": {"prompt": "do NOT run devils_advocate"}}]}}
        f.write_text("x"); os.utime(f, (now - 50, now - 50)); ctx.write_text("c"); os.utime(ctx, (now - 200, now - 200))
        assert d_([prompt], root) and "Devil" in d_([prompt], root) and "CONTEXT" in d_([prompt], root)
        assert "Devil" in d_([prompt, fake], root)                  # a mere mention is not a review
        assert "Devil" not in (d_([prompt, da], root) or "")        # real DA after the first change
        assert "Devil" in d_([prompt, notif], root)                 # notification does not reset the turn
        summ = {"type": "user", "isCompactSummary": True, "timestamp": iso(now - 10), "message": {"content": "This session is being continued"}}
        assert "Devil" in d_([prompt, summ], root)                  # nor does a compaction summary (DA re-review C4)
        os.utime(ctx, (now - 5, now - 5))
        assert d_([prompt, da], root) is None                       # both done
        (root / "notes.txt").write_text("x")
        os.utime(f, (now - 300, now - 300))
        assert d_([prompt], root) is None                           # nothing changed in a reviewed area (notes.txt is not one)
        assert d_([], root) is None
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest(); sys.exit(0)
    try:
        data = json.load(sys.stdin)
        if data.get("stop_hook_active"):
            sys.exit(0)  # already blocked once for this stop; never loop
        es = []
        with open(data["transcript_path"]) as fh:
            for line in fh:
                try:
                    es.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        root = HC.project_root(data)
        cfg = HC.load(root)
        if cfg is None:
            sys.exit(0)  # not a research project: no review / context requirement
        reason = decide(es, root, cfg)
    except Exception:
        sys.exit(0)  # fail open: a broken hook must not trap the session
    if reason:
        print(json.dumps({"decision": "block", "reason": reason}))
