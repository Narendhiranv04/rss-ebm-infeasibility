"""Union of two append-only ledgers (the laptop and server runs of one dataset): writes the complete, valid lines each side lacks.

    python3 merge_ledgers.py LOCAL_LEDGER REMOTE_COPY TO_LOCAL TO_REMOTE

A line still being written (no newline yet, or not valid JSON) is left for the next round."""
import json
import sys


def lines(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                if line.endswith("\n") and json.loads(line):
                    out.append(line)
            except ValueError:
                pass
    return out


local, remote = lines(sys.argv[1]), lines(sys.argv[2])
seen_local, seen_remote = set(local), set(remote)
with open(sys.argv[3], "w") as f:
    f.writelines(x for x in dict.fromkeys(remote) if x not in seen_local)
with open(sys.argv[4], "w") as f:
    f.writelines(x for x in dict.fromkeys(local) if x not in seen_remote)
