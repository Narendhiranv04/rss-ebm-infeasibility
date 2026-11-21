#!/usr/bin/env python3
"""Opt a project into the research harness: writes <project>/.claude/research_harness.json and a CONTEXT.md template
(only if missing; never overwrites), and adds both to .git/info/exclude so nothing is committed by accident.

    python3 ~/.claude/research-harness/init_project.py [PROJECT_DIR]      (default: git top level of the current folder)
"""
import json, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_config as HC  # noqa: E402

CONTEXT_TEMPLATE = """# CONTEXT: working state (read first after a compaction; keep under ~9,000 characters)

Last updated: <date/time>.

## Who / what
- User, collaborators, the research question, the ground truth / evaluation.

## Standing rules
- Never touch the held-out TEST split. Pre-register a confirmatory test before running it.

## Key results so far (say which split; exploratory vs confirmatory)

## In flight (check status first)
- Background jobs (task ids, machines, output paths), what each one is waiting on.

## Next steps
"""


def main(root):
    root = os.path.realpath(root)
    cfgf = os.path.join(root, ".claude", "research_harness.json")
    os.makedirs(os.path.dirname(cfgf), exist_ok=True)
    made = []
    if not os.path.exists(cfgf):
        with open(cfgf, "w") as f:
            json.dump(HC.DEFAULTS, f, indent=2)
        made.append(cfgf)
    ctx = os.path.join(root, HC.load(root)["context_file"])
    if not os.path.exists(ctx):
        with open(ctx, "w") as f:
            f.write(CONTEXT_TEMPLATE)
        made.append(ctx)
    ex = os.path.join(root, ".git", "info", "exclude")
    if os.path.isdir(os.path.dirname(ex)):
        have = open(ex).read() if os.path.exists(ex) else ""
        with open(ex, "a") as f:
            for line in (".claude/research_harness.json",):
                if line not in have.split("\n"):
                    f.write(line + "\n")
    print("created:" if made else "already set up:", *(made or [cfgf]), sep="\n  ")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HC.project_root({"cwd": os.getcwd()}))
