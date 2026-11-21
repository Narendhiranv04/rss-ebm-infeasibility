<!-- research-harness:begin (installed by research-harness/install.sh; edit freely) -->
## How to talk to me (all projects)
- Plain, simple language, like explaining to a smart friend who is new to the topic. Short sentences.
- No jargon unless I need it; when a term is needed, explain it in one simple line the first time.
- Lead with the answer (what happened, what it means, what I need to do), then the details only if useful.
- Keep status updates short: a few bullets, with numbers in a small table. Don't dump every detail.

## Research projects (research harness)

A project is a research project when it has `.claude/research_harness.json`. For a new research project, offer to run
`python3 ~/.claude/research-harness/init_project.py` (it only creates missing files). Hooks in `~/.claude/settings.json`:
the safety guard runs everywhere; the review and context hooks run only in research projects.

**Safety (all projects)**
- Before any delete, move or overwrite on the user's machines or servers, look at the target first: contents, timestamps,
  symlinks, rc references. Never assume a file is yours. The guard denies destructive commands outside the project or on
  a remote host; when it does, ask the user. Do not route around it.
- Never start heavy jobs on the user's machines without checking power/UPS and resource limits, and never restart a
  stopped job without the user's explicit go-ahead.
- Remote scripts run only after their sha256 is listed in the project's `reviewed_scripts` file.

**Rigor**
- Never touch a held-out TEST split. Label every result exploratory or confirmatory, and say which split it is on.
- Before a confirmatory test, write a pre-registration: hypothesis, primary endpoint, test, power, stopping and exclusion
  rules, decision rule. Freeze the hashes of code, models and the data list, and record it in git before scoring.
- Report baselines at their best settings. Name confounds; disclose any endpoint chosen after seeing development data.

**Review, done cheaply (the user's usage is limited)**
- One Devil's Advocate review per substantial work unit: results, pre-registrations, experiment code. Not for notes,
  scripts or hooks. Use the ARS role file `academic-paper-reviewer/agents/devils_advocate_reviewer_agent.md`.
- Run reviews as ONE subagent on the project's `review_model` (default sonnet), with a narrow brief: what changed, what to
  check, a word limit. Batch fixes; don't re-review fixes the reviewer itself proposed. Adjudicate CRITICALs visibly.
- Don't poll idle jobs. When blocked on the user, stop and say so.
- Long sessions cost the most. Keep the context file current, and suggest `/clear` or a fresh session when switching
  tasks: the SessionStart hook reloads the context file.

**Working state**
- Keep the project's `context_file` (default `CONTEXT.md`) current: results, in-flight jobs, next steps, standing rules.
  Keep it under about 9,000 characters.
<!-- research-harness:end -->
