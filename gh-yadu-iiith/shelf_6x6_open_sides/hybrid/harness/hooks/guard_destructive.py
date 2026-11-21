#!/usr/bin/env python3
"""Claude Code PreToolUse hook: deny destructive actions that reach outside the project or onto a remote host.

Bash (permissionDecision "deny"; the user is often away, so no prompt is left waiting):
  - local part: a destructive operation (rm/mv/cp/tee/ln -f/dd/sed -i/rmtree/uninstall/git push -f/...) whose paths leave the
    project root (absolute, ~, .., unresolved $VAR / $(...)); a command whose only write is `>` is judged on its targets alone.
  - remote part (v4): each ssh segment's remote command is checked on its own: any destructive word or overwrite redirect there
    is denied, whatever the path (relative = $HOME).  Running a script remotely (`ssh h 'bash x.sh'`, `bash -s < x`, `| ssh`)
    is allowed only if the local copy's sha256 is in REVIEWED_REMOTE_SCRIPTS.txt.  scp/rsync uploads may only target
    REMOTE_OK; rsync --del*/--remove-source-files with a remote side is denied.  Python that shells out to ssh with a
    destructive word is denied; heredocs fed to a shell or ssh are denied.
Write / Edit / MultiEdit / NotebookEdit: denies file_path outside the project (memory dir and session tmp exempt) and any
Claude Code settings file (it could switch this guard off).
Project root = CLAUDE_PROJECT_DIR (or the git top level); remote folders and the script allowlist come from the
project's .claude/research_harness.json (harness_config.py).  Fails open by design (a crashing hook must not lock the session).  A seatbelt against honest mistakes, not a sandbox: a regex
cannot classify all shell.  Reason: 2026-11-21 an unchecked cleanup deleted pre-existing user files on the home server.
v2-v4 after three ARS Devil's Advocate reviews (cases in selftest()).

    python3 guard_destructive.py --selftest
"""
import glob, hashlib, json, os, re, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness_config as HC  # noqa: E402

SAFE = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/stdin")
MEMORY = re.compile(r"^" + re.escape(os.path.realpath(os.path.expanduser("~/.claude/projects"))) + r"/[^/]+/memory(?:/|$)")
ROOT, ROOTS, REMOTE_OK, REVIEWED, WATCH = "", [], [], "", ""


def configure(root, cfg=None):
    """Per call: the project root (CLAUDE_PROJECT_DIR / git top level) plus the project's research config, if any."""
    global ROOT, ROOTS, REMOTE_OK, REVIEWED, WATCH
    cfg = cfg if cfg is not None else (HC.load(root) or HC.DEFAULTS)
    ROOT = root
    tmp = [f"/private/tmp/claude-{os.getuid()}"] + ([os.environ["TMPDIR"]] if os.environ.get("TMPDIR") else [])
    ROOTS = [os.path.realpath(r) for r in [root] + tmp]
    REMOTE_OK = list(cfg["remote_ok"])
    REVIEWED = os.path.join(root, cfg["reviewed_scripts"])
    WATCH = os.path.join(root, cfg["watch_dir"])


D = r"(?:^|[\s;&|(`'\"/=])"  # start of a command word (also after a path prefix like /bin/)
CORE = re.compile("|".join([
    D + r"(?:rm|rmdir|unlink|shred|truncate|mv|cp|install|ln\s+-\w*f\w*|tee|dd\s[^;&|]*of=)\b",
    r"\b(?:uv\s+(?:tool\s+)?(?:uninstall|remove)|pip3?\s+uninstall|brew\s+(?:uninstall|remove|rm)|(?:apt|apt-get|dnf|yum|snap)\s+(?:remove|purge|autoremove))\b",
    r"\bnpm\s+(?:uninstall|rm)\b|\bconda\s+(?:remove|env\s+remove)\b|\bdocker\s+(?:rm|rmi|system\s+prune)\b",
    r"rsync\b[^;&|]*--(?:del\b|delete|remove-source-files)",
    r"\bgit\b[^;&|]*\s(?:reset\s+--hard|clean\b|checkout\s+--|restore\b|push\b[^;&|]*(?:--force|\s-f\b|\s\+\S|\s:\S))",
    r"\bsed\s+(?:-\w*\s+)*-i|\bperl\s+-\w*i",
    r"\bfind\b[^;&|]*\s-(?:delete|exec\s+rm)\b", r"\bxargs\b[^;&|]*\brm\b",
    r"\b(?:chmod|chown)\s+(?:-R\b|0?00\b)", r"\bcrontab\s+-r\b", r"\bkill\s+-9\s+-1\b",
    r"(?:shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir|Path\([^)]*\)\.unlink|\bunlink\s*\()",
]))
REDIR = re.compile(r"(?:^|[^>-])(?:&|\d)?>\|?(?!>|&)\s*([^\s&|;)]+)")  # > or >| overwrite (not >>, 2>&1, ->); group = target
REMOTE = re.compile(r"(?:^|[\s;&|(])(?:ssh|scp|sftp)\s|rsync\b[^;&|]*\s(?:[\w.-]+@)?[A-Za-z][\w.-]*:(?!//)|"
                    r"\b[\w.-]+@\d+\.\d+\.\d+\.\d+")
REMOTE_HOST = re.compile(r"^(?:[\w.-]+@)?[A-Za-z0-9][\w.-]*:(?!//)(.*)$")
REMOTE_WORDS = re.compile(r"\b(?:rm|rmdir|unlink|mv|cp|ln|tee|truncate|shred|uninstall|remove|purge|rmtree|kill|pkill|killall|chmod|"
                          r"chown|crontab|prune|dd|eval|base64)\b|\bgit\b[^;&|]*\b(?:reset|push|clean|checkout|restore)\b|--del\b|--delete|"
                          r"--remove-source-files|-delete\b|os\.remove|shutil\.move|\btar\s+-?\w*x|\bsed\s+(?:-\w*\s+)*-i|\bperl\s+-\w*i|"
                          r"\bcurl\b[^|;&]*\s-\w*o\s*+(?!/dev/null)|\bwget\b[^|;&]*\s-\w*O\s*+(?!/dev/null|-)|(?:^|[;&|]\s*)install\b|\bunzip\s+-\w*o|\bcache\s+clean")
REMOTE_REDIR = re.compile(r">\|?\s*(?!/dev/(?:null|stdout|stderr)\b)([^\s>&]\S*)")  # overwrite on the remote side; REMOTE_OK targets are ok
CMDPOS = r"(?:^|[;&|(]\s*)"
SCRIPT_RUN = re.compile(CMDPOS + r"(?:\./|~/|/)(?!(?:usr|bin|sbin|opt)/)(?!\S*/python[\d.]*(?:\s|$))\S+|(?:^|[\s;&|(/])(?:ba|z|da|k)?sh\s+(?!-c\b)\S|"
                        r"(?:^|[\s;&|(/])python[\d.]*\s+(?!-c\b|-m\b|--version\b|-V\b)\S|\bsource\s+\S+\.sh\b|" + CMDPOS + r"\.\s+\S|\buv\s+run\b")
ORDINARY = re.compile(r"\s*(?:\S*/)?(?!(?:ba|z|da|k)?sh\b|python|perl\b|ruby\b|node\b|sudo\b|cd\b|env\b|exec\b|source\b)[\w.-]+")
ARG_OPTS = {"scp": {"-P", "-i", "-o", "-F", "-l", "-c", "-J", "-S", "-D", "-X"},  # options that take a separate value, per tool
            "rsync": {"-e", "-f", "-T", "-B", "-M", "--exclude", "--include", "--filter", "--rsh", "--chown", "--chmod", "--port",
                      "--password-file", "--exclude-from", "--include-from", "--files-from", "--log-file", "--partial-dir", "--temp-dir",
                      "--backup-dir", "--compare-dest", "--link-dest", "--copy-dest", "--rsync-path", "--timeout", "--bwlimit", "--suffix"}}
READ_ONLY = {"grep", "rg", "ag", "cat", "less", "head", "tail", "ls", "wc", "stat", "file", "jq", "diff"}
HEREDOC = re.compile(r"(<<-?\s*(['\"]?)(\w+)\2[^\n]*\n)(.*?\n)?[ \t]*\3[ \t]*(?=\n|$)", re.S)
SHELL_FED = re.compile(r"(?:^|[\s;&|(])(?:ba|z|da|k)?sh\b[^\n|;&]*<<|(?:^|[\s;&|(])ssh\b[^\n|;&]*<<")
UNRESOLVED = re.compile(r"\$\{?[A-Za-z_]\w*\}?|\$\(|`")
OPS = {"|", "||", "&&", ";", "&", "(", ")"}
OPLIKE = re.compile(r"[;&|()]+|\d*[<>][<>&|]*")
WRAPPERS = {"sudo", "nohup", "time", "env", "nice", "command", "exec"}
SSH_ARG_OPTS = set("bcDEeFIiJLlmOoPpQRSWw")


def strip_comments(cmd):
    """Remove shell comments: an unquoted # at the start of a word runs to end of line (apostrophes inside it are not quotes)."""
    out, q, i = [], None, 0
    while i < len(cmd):
        ch = cmd[i]
        if q:
            if ch == "\\" and q == '"' and i + 1 < len(cmd):
                out += cmd[i:i + 2]; i += 2; continue
            if ch == q:
                q = None
        elif ch == "\\" and i + 1 < len(cmd):
            out += cmd[i:i + 2]; i += 2; continue
        elif ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or (cmd[i - 1] in " \t\n;&|(" and not out[-2:-1] == ["\\"])):  # `my\ #x` is not a comment
            while i < len(cmd) and cmd[i] != "\n":
                i += 1
            continue
        out.append(ch); i += 1
    return "".join(out)


def inside(path, cwd):
    if path in SAFE or path.startswith("/dev/fd"):
        return True
    p = os.path.realpath(os.path.join(cwd, os.path.expanduser(path)))
    return bool(MEMORY.match(p)) or any(p == r or p.startswith(r + os.sep) for r in ROOTS)


def ok_remote(path):
    return any(re.search(r"(?:^|/)" + re.escape(ok) + r"(?:/|$)", path) for ok in REMOTE_OK) and ".." not in path.split("/")


def reviewed(cmd, cwd):
    """Every .sh/.py named in cmd has a local copy (cwd or under hybrid/) whose sha256 is listed in REVIEWED_REMOTE_SCRIPTS.txt.
    shortcut: checks the local copy, not the remote one; fine while scripts are always scp'd from here in the same command."""
    names = set(re.findall(r"[\w.~/-]+\.(?:sh|py)\b", cmd)) | set(re.findall(CMDPOS + r"((?:\./|~/)\S+)", cmd))
    try:
        ok = {ln.split()[0] for ln in open(REVIEWED) if ln.strip() and not ln.startswith("#")}
    except OSError:
        return False
    for n in names:
        cands = [p for p in [os.path.join(cwd, os.path.expanduser(n))] if os.path.isfile(p)]
        cands = cands or glob.glob(os.path.join(WATCH, "**", os.path.basename(n)), recursive=True)
        if not any(hashlib.sha256(open(c, "rb").read()).hexdigest() in ok for c in cands):
            return False
    return bool(names)


def split_remote(cmd, cwd):
    """Check every ssh / scp / rsync segment with the remote rules; return (reason, the remaining local command)."""
    lex = shlex.shlex(cmd.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lex.whitespace_split, lex.commenters = True, ""
    segs, cur = [], []
    for t in list(lex) + [";"]:  # comments were already stripped by strip_comments()
        if t in OPS:
            segs.append((cur, t)); cur = []
        else:
            cur.append(t)
    local, prev = [], None
    for seg, op in segs:
        i = next((n for n, t in enumerate(seg) if re.fullmatch(r"(?:/usr)?(?:/bin/)?(?:ssh|scp|rsync|sftp)", t)), len(seg))
        word = os.path.basename(seg[i]) if i < len(seg) else ""  # found anywhere: timeout/watch/then/do ... ssh
        hostish = [a for a in seg[i + 1:] if REMOTE_HOST.match(a)]
        if word == "ssh":
            j = i + 1
            while j < len(seg) and seg[j].startswith("-"):
                j += 2 if len(seg[j]) == 2 and seg[j][1] in SSH_ARG_OPTS else 1
            k = j + 1
            while k < len(seg) and not OPLIKE.fullmatch(seg[k]):
                k += 1
            remote, rest = " ".join(seg[j + 1:k]), seg[k:]
            if REMOTE_WORDS.search(remote) or any(not ok_remote(t) for t in REMOTE_REDIR.findall(remote)):
                return "destructive operation in a command that runs on a remote host", None
            fed = (prev == "|" or ("<" in rest and rest[rest.index("<") + 1:rest.index("<") + 2] != ["/dev/null"])) and not ORDINARY.match(remote)
            if (SCRIPT_RUN.search(remote) or fed) and not reviewed(cmd, cwd):
                return ("runs a script on a remote host whose local copy is not listed (sha256) in harness/hooks/REVIEWED_REMOTE_SCRIPTS.txt; "
                        "get it Devil's-Advocate reviewed first"), None
            local += seg[:i] + rest
        elif word in ("scp", "rsync", "sftp") and (hostish or word == "sftp"):
            args = seg[i + 1:]
            if word == "sftp" or re.search(r"--del\b|--delete|--remove-source-files", " ".join(args)):
                return "remote delete (rsync --delete / --remove-source-files) or interactive sftp", None
            pos, skip = [], False
            for a in args:
                if skip:
                    skip = False
                elif a.startswith("-"):
                    skip = a in ARG_OPTS[word]
                elif not a.isdigit():
                    pos.append(a)
            hosts = [n for n, a in enumerate(pos) if REMOTE_HOST.match(a)]
            locs = [n for n, a in enumerate(pos) if not REMOTE_HOST.match(a)]
            # both checks always (an unknown option value can look like a local source): host paths after the first local
            # positional must be in REMOTE_OK; local paths after the first host positional must be inside the project
            if locs and hosts:
                for n in hosts:
                    if n > locs[0] and not ok_remote(REMOTE_HOST.match(pos[n]).group(1)):
                        return f"upload to a remote path outside {REMOTE_OK or 'the allowed remote folders (none set)'} (could overwrite the user's files): {pos[n]}", None
            if True:
                for n in locs:
                    if hosts and n > hosts[0] and not inside(pos[n], cwd):
                        return f"download into a path outside the project: {pos[n]}", None
            local += seg[:i]
        else:
            local += seg
        local.append(op)
        prev = op
    while local and local[-1] in OPS:
        local.pop()
    return None, " ".join(t if OPLIKE.fullmatch(t) else shlex.quote(t) for t in local)


def local_reason(cmd, cwd):
    core = CORE.search(cmd)
    if not core and not REDIR.search(cmd):
        return None
    try:
        toks = shlex.split(cmd, comments=False, posix=True)
    except ValueError:
        return "destructive command that could not be parsed safely"
    if toks and os.path.basename(toks[0]) in READ_ONLY and not re.search(r"[|;&>`\n]|\$\(", cmd):
        return None  # a lone read-only command (grep/cat/ls...) whose arguments merely mention rm etc.
    if not core:  # the only write is `>`: judge its targets, not every path or $VAR in the command
        for t in (x.strip("'\"") for x in REDIR.findall(cmd)):
            if t in SAFE:
                continue
            if UNRESOLVED.search(t):
                return f"overwrite redirect to an unresolved path ({t})"
            if not inside(t, cwd):
                return f"overwrite redirect to a path outside the project: {t}"
        return None
    paths = []
    for t in toks:
        for part in re.split(r"[\s;&|]+", t):
            part = part.lstrip(">|").strip("'\"")
            if part and (part.startswith(("/", "~")) or ".." in part.split("/") or UNRESOLVED.search(part)):
                paths.append(part)
        paths += re.findall(r"(?<=['\"(=,])((?:~|/)[^\s'\"),;]+)", t)  # paths inside code strings, e.g. rmtree('/x')
    for p in paths:
        if UNRESOLVED.search(p):
            return f"destructive command with an unresolved path ({p}); cannot verify it stays inside the project"
        if not inside(p, cwd):
            return f"destructive command touches a path outside the project: {p}"
    if re.search(r"\b(?:uv\s+(?:tool\s+)?uninstall|pip3?\s+uninstall|brew\s+(?:uninstall|remove)|apt(?:-get)?\s+(?:remove|purge)|crontab\s+-r|kill\s+-9\s+-1|push\b[^;&|]*(?:--force|\s-f\b|\s\+\S|\s:\S))", cmd):
        return "uninstall / system-level destructive command"
    return None


def bash_reason(cmd, cwd):
    if re.search(r"(?:subprocess\.\w+|os\.system|os\.popen|Popen)\s*\(", cmd) and re.search(r"['\"\[\s]ssh['\"\s,]", cmd) \
            and REMOTE_WORDS.search(cmd):
        return "code shells out to ssh with a destructive operation"
    for body in (m.group(4) or "" for m in HEREDOC.finditer(cmd)):  # heredoc bodies are code: only explicit deletes count
        if re.search(r"subprocess\.\w+\(\s*\[\s*['\"]rm['\"]|os\.system\([^)]*\brm\b", body):
            return "heredoc code shells out to rm"
        for p in re.findall(r"(?:rmtree|os\.remove|os\.unlink|\.unlink|os\.rmdir)\s*\(\s*(?:os\.path\.expanduser\(\s*)?['\"]((?:~|/)[^'\"]+)", body):
            if not inside(p, cwd):
                return f"heredoc code deletes a path outside the project: {p}"
    cmd = strip_comments(HEREDOC.sub(lambda m: m.group(1), cmd))  # keep the opener line: `cat <<EOF > ~/.zshrc` must still be seen
    if SHELL_FED.search(cmd):
        return "heredoc script fed to a shell or ssh: cannot verify it; run it as a reviewed file instead"
    t = os.environ.get("TMPDIR")
    if t:
        cmd = cmd.replace("${TMPDIR}", t).replace("$TMPDIR", t)
    if REMOTE.search(cmd) and CORE.search(cmd) and re.search(r"\$\(|`", cmd):
        return "destructive command with $(...) or backticks next to a remote command; cannot verify its paths"
    if REMOTE.search(cmd):
        try:
            reason, cmd = split_remote(cmd, cwd)
        except ValueError:
            return "remote command that could not be parsed safely"
        if reason:
            return reason
    return local_reason(cmd, cwd)


def file_reason(fp, cwd):
    if not fp:
        return None
    if re.search(r"/\.claude/settings[^/]*\.json$", fp):
        return "edit of a Claude Code settings file (it could switch this guard off)"
    return None if inside(fp, cwd) else f"write to a file outside the project: {fp}"


def decide(data):
    tool, inp, cwd = data.get("tool_name", "Bash"), data.get("tool_input", {}), data.get("cwd") or os.getcwd()
    configure(HC.project_root(data))
    if tool == "Bash":
        return bash_reason(inp.get("command", ""), cwd)
    if tool in ("Write", "Edit", "NotebookEdit", "MultiEdit"):
        return file_reason(inp.get("file_path") or inp.get("notebook_path"), cwd)
    return None


def selftest():
    configure("/Users/nav/Documents/RRC_Work/RSS_2027", {**HC.DEFAULTS, "remote_ok": ["Documents/RRC/rss_gen"],
                                                         "watch_dir": "gh-yadu-iiith/shelf_6x6_open_sides/hybrid"})
    c = ROOT + "/gh-yadu-iiith"
    b = lambda cmd: bash_reason(cmd, c)  # noqa: E731
    ask = ["rm -rf ~/.local/share/uv", "mv ~/.local/bin/uv /tmp/x", "ssh home@1.2.3.4 'rm -rf ~/rss_gen'",
           "rsync -a --delete a/ home@1.2.3.4:b/", "echo hi > /Users/nav/.zshrc", "rm ../../../../../etc/hosts",
           # DA review 1 (v1 missed these):
           "rsync -a --delete a/ gvlab:b/", "rsync --remove-source-files gvlab:x/ .", "ssh gvlab 'uv tool uninstall ruff'",
           "uv tool uninstall ruff", "pip uninstall numpy", "sudo apt remove foo", "brew uninstall jq",
           "python3 -c \"import shutil; shutil.rmtree('/Users/nav/x')\"", "cp x ~/.zshrc", "echo a >| ~/.zshrc",
           "echo a | tee ~/.zshrc", "ln -sf /x ~/.zshrc", "/bin/rm -rf ~/x", "bash -c 'rm -rf ~/x'", "ls ~/.local/bin | xargs rm",
           "D=$HOME/.local; rm -rf $D", "rm -rf ${HOME}/.local", "rm -rf \"$(echo ~/.local)\"", "git push --force",
           "chmod 000 ~/.ssh", "crontab -r", "rm -rf /private/tmp/claude-other",
           "python3 - <<'EOF'\nimport shutil; shutil.rmtree('/Users/nav/.local')\nEOF",
           # DA review 2:
           "cat <<EOF > ~/.zshrc\nx\nEOF", "bash <<'EOF'\nrm -rf ~/x\nEOF", "ssh gvlab bash <<'EOF'\nrm -rf ~/rss_gen\nEOF",
           "ssh home@1.2.3.4 <<'EOF'\nrm -rf rss_gen\nEOF", "ls\nrm -rf ~/x", "cat notes.md\nrm -rf ~/.local", "ssh gvlab rm -rf rss_gen",
           "ssh gvlab 'find . -delete'", "git push -f", "git push origin +main", "git push origin :main", "sed -i '' s/a/b/ ~/.zshrc",
           "python3 - <<'EOF'\nimport subprocess; subprocess.run(['rm','-rf','x'])\nEOF",
           "python3 - <<'EOF'\nimport os; os.remove(os.path.expanduser('~/.zshrc'))\nEOF",
           # DA review 3:
           "scp cleanup.sh gvlab:~/ && ssh gvlab 'bash ~/cleanup.sh'", "ssh gvlab 'bash ~/cleanup.sh'", "ssh gvlab bash -s < cleanup.sh",
           "cat cleanup.sh | ssh gvlab bash", "ssh gvlab 'python3 -' < clean.py",
           "python3 - <<'EOF'\nimport subprocess\nsubprocess.run(['ssh','gvlab','rm -rf ~/rss_gen'])\nEOF",
           "python3 -c \"import os; os.system('ssh gvlab mv a b')\"", "rsync -a --del code/ gvlab:~/x/code/",
           "scp evil.txt gvlab:~/.bashrc", "rsync -a x gvlab:~/Documents/", "ssh gvlab 'pkill -f python'", "ssh gvlab 'killall python'",
           "ssh gvlab 'git -C ~/proj checkout .'", "ssh gvlab 'tar -xf a.tar -C ~/Documents'", "ssh gvlab 'docker system prune -af'",
           "ssh gvlab 'echo eA==|base64 -d|sh'", "ssh gvlab 'echo x > ~/.bashrc'", "scp gvlab:a.txt ~/.zshrc"]
    allow = ["ls -la", "rm -rf runs/PHASE9", "python3 x.py > out.txt", "cat a 2>/dev/null", "git status", "cmd 2>&1 | tail",
             "ssh gvlab 'squeue' > q.txt", "ssh gvlab 'nvidia-smi > /dev/null'", "awk '$1>3' f", "mv a b",
             "python3 - <<'EOF'\nprint(1>0)\nEOF", "grep -n 'rm -rf ~/' notes.md", "ssh home@1.2.3.4 'ls ~/Documents; df -h'",
             "cd x && python3 - <<'EOF'\ns = s.replace('a', '[|;&>`]|$(')\nopen('f','w').write(s)\nEOF\npython3 f.py --selftest",
             "python3 x.py > $TMPDIR/out.txt", "sed -i '' s/a/b/ src/x.py",
             # DA review 3 false positives:
             "awk '{print $1}' a.csv > col.txt", "python3 run.py --seed $SEED > out.txt", "echo \"$(date) tick\" > STATUS.txt",
             "for f in runs/*; do echo $f; done > list.txt", "ssh gvlab 'cat run.log' | tee runs/hx1.log",
             "ssh gvlab 'tail run.log' > $TMPDIR/x", "rsync -a gvlab:d/ local/ && cp local/a.json results/",
             "ssh gvlab 'ls' > f.txt  # what to remove later", "rsync -aL code home@1.2.3.4:Documents/RRC/rss_gen/code/",
             "ssh gvlab 'tail -n 5 ~/Documents/RRC/rss_gen/runs/hx1/run.log'", "ssh h 'echo 8 > ~/Documents/RRC/rss_gen/runs/hx1/MAX_RUNNING'", "scp gvlab:~/x/out.tgz runs/",
             "ssh h 'cd ~/x && .venv/bin/python -c \"import json; print(1)\"'", "python3 - <<'EOF'\nx = 'sh y <<Z'\nEOF"]
    ask += ["ssh gvlab ls  # peek\nrm -rf ~/.local/share/uv", "# look\nssh gvlab ls\nmv ~/a ~/b", "ssh h ls # c\ncrontab -r",
            "timeout 10 ssh gvlab 'rm -rf rss_gen'", "nice -n 5 ssh gvlab 'rm -rf x'", "if true; then ssh gvlab 'rm -rf rss_gen'; fi",
            "for i in 1 2; do ssh gvlab 'rm -rf rss_gen'; done", "ssh h ls; rm -rf $(cat list)", "ssh gvlab './cleanup.sh'",
            "ssh gvlab 'cd x && ./cleanup.sh'", "ssh gvlab '~/Documents/RRC/rss_gen/cleanup.sh'", "ssh gvlab 'bash -x cleanup.sh'",
            "ssh gvlab 'python3 -u gen.py'", "ssh gvlab 'uv run gen.py'", "ssh gvlab 'bash ~/cleanup'",
            "rsync -a src/ gvlab:~/ --exclude '*.pyc'", "scp x gvlab:Documents/RRC/rss_gen_old/", "echo x &> ~/.zshrc",
            "python3 x.py 2> ~/err.log"]
    allow += ["ssh gvlab 'tail run.log | grep -i reset'", "ssh gvlab 'ls Documents/RRC/rss_gen/runs/clean'",
              "ssh gvlab 'grep -c checkout run.log'", "ssh h 'uptime'  # check load\nls runs", "python3 x.py 2> err.log",
              "rsync -a gvlab:Documents/RRC/rss_gen/runs/hx1 runs/ --exclude '*.tmp'"]
    ask += ["ssh h grep -c '#' log && rm -rf ~/x", "ssh h ls | grep -v '#' && rm -rf ~/x", "ssh h ls && echo '#' && ssh h 'rm -rf rss_gen'",
            "ssh h echo '#' rm -rf x", "scp -P 2222 h:Documents/RRC/rss_gen/x.json ~/.zshrc", "rsync -a -e ssh h:Documents/RRC/rss_gen/runs/ ~/Desktop/runs/",
            "rsync -avz --exclude '*.tmp' h:Documents/RRC/rss_gen/runs/hx1 ~/", "ssh h \"sed -i 's/a/b/' ~/.bashrc\"",
            "ssh h ls; rm -rf ~/old/ssh", "ssh h 'bash launch_hx1.sh; ./cleanup'", "ssh h 'curl -o ~/.bashrc http://x'"]
    allow += ["ssh h '~/x/.venv/bin/python -c \"import json\"'", "ssh h 'du -sh ~/Documents'", "ssh h 'python3 --version'",
              "ssh h 'python3 -m json.tool f'", "ssh h 'ls ~/Documents/RRC/rss_gen/code/shelf66_v3/run_shelf66_full.py'", "ssh h 'tail run.py.log'",
              "git log | ssh h 'wc -l'", "ssh h 'uptime' < /dev/null", "scp -P 22 h:/var/log/syslog runs/",
              "out=$(git rev-parse HEAD 2>/dev/null)", "(cd runs && ls > /dev/null)", "ssh h 'ps aux | grep python'",
              "ssh h 'cat ~/Documents/RRC/rss_gen/runs/hx1/PROGRESS.json'", "ssh h 'cd ~/x && ls'"]
    ask += ["ssh gvlab ls  # it's fine\nrm -rf ~/x  # that's all", "ssh h find runs -name '#*' -delete  # tidy",
            "rsync -avz code/ gvlab:~/ --bwlimit 5000", "rsync -a h:Documents/RRC/rss_gen/runs/ ~/Desktop/runs --timeout 30",
            "ssh gvlab < cleanup.sh", "cat cleanup.sh | ssh gvlab", "cat x | ssh gvlab sudo bash", "ssh gvlab 'cd ~/x && bash' < cleanup.sh",
            "ssh h 'curl -sLo ~/.bashrc URL'", "ssh h 'unzip -qo a.zip'"]
    allow += ["ssh h 'source .venv/bin/activate && python -V'", "ssh h '/usr/bin/nvidia-smi'", "ssh h 'curl -s -o /dev/null -w x URL'",
              "ssh gvlab squeue  # user's jobs", "echo $# ${#x} > n.txt", "cat data.json | ssh h 'wc -c'"]
    ask += ["rsync -avz -P code/ gvlab:~/", "rsync -a -c code/ gvlab:~/.config/", "rsync -P gvlab:Documents/RRC/rss_gen/x ~/.zshrc",
            "scp -T gvlab:x ~/.zshrc", "ls my\\ #x; rm -rf ~/x", "rsync -avze ssh h:Documents/RRC/rss_gen/runs/ ~/Desktop/runs/",
            "rsync -avze 'ssh -p 2222' h:Documents/RRC/rss_gen/x.json ~/.zshrc", "rsync -a --info progress2 h:Documents/RRC/rss_gen/runs ~/Desktop/"]
    allow += ["scp h:a h:b runs/", "rsync -avz -P code/ h:Documents/RRC/rss_gen/code/", "rsync -a --rsync-path /usr/bin/rsync h:Documents/RRC/rss_gen/x runs/"]
    for cmd in ask:
        assert b(cmd), f"should deny: {cmd}"
    for cmd in allow:
        assert b(cmd) is None, f"should allow: {cmd} -> {b(cmd)}"
    assert file_reason("/Users/nav/.zshrc", c) and file_reason(ROOT + "/x.md", c) is None
    assert file_reason(os.path.expanduser("~/.claude/projects/-x-y/memory/m.md"), c) is None and file_reason(ROOT + "/.claude/settings.local.json", c)
    print(f"selftest ok ({len(ask)} deny, {len(allow)} allow)")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest(); sys.exit(0)
    reason = decide(json.load(sys.stdin))
    if reason:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                                 "permissionDecisionReason": f"Safety guard (deny): {reason}. Do not retry or work around this. Check what the target contains and whether it predates this session, then ask the user in chat to run it themselves or to confirm it explicitly."}}))
