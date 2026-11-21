"""Shared settings for the research-harness hooks.

The safety guard runs in every project.  The research hooks (review-before-stop, context file) run only in projects that
opt in with <project>/.claude/research_harness.json (create one with init_project.py).  Keys (all optional):
  context_file      compaction-safe working-state file, relative to the project root           (default CONTEXT.md)
  watch_dir         folder whose changes need a review, relative to the project root           (default .)
  review_areas      top-level entries under watch_dir that need a Devil's Advocate review      (default src, docs, RESULTS.md, README.md)
  remote_ok         remote folders this project may write into (e.g. "Documents/RRC/rss_gen") (default none)
  reviewed_scripts  sha256 allowlist of scripts allowed to run on remote hosts                  (default .claude/reviewed_remote_scripts.txt)
  review_model      model for review subagents (cost: a long-running session plus Opus reviews burns usage) (default sonnet)
"""
import json, os, subprocess

DEFAULTS = {"context_file": "CONTEXT.md", "watch_dir": ".", "review_areas": ["src", "docs", "RESULTS.md", "README.md"],
            "remote_ok": [], "reviewed_scripts": ".claude/reviewed_remote_scripts.txt", "review_model": "sonnet"}


def project_root(data=None):
    r = os.environ.get("CLAUDE_PROJECT_DIR")
    if not r:
        r = (data or {}).get("cwd") or os.getcwd()
        try:
            r = subprocess.run(["git", "-C", r, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=3).stdout.strip() or r
        except (OSError, subprocess.SubprocessError):
            pass
    return os.path.realpath(r)


def load(root):
    """The project's research config merged over DEFAULTS, or None when the project has not opted in."""
    try:
        with open(os.path.join(root, ".claude", "research_harness.json")) as f:
            return {**DEFAULTS, **json.load(f)}
    except (OSError, ValueError):
        return None
