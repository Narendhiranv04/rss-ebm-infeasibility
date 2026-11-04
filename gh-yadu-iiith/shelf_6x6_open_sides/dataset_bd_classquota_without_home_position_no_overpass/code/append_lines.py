"""Appends a file's bytes to a ledger in ONE write (O_APPEND), so a runner appending at the same moment cannot interleave with it.

    python3 append_lines.py SOURCE LEDGER"""
import os
import sys

data = open(sys.argv[1], "rb").read()
if data:
    fd = os.open(sys.argv[2], os.O_WRONLY | os.O_APPEND)
    try:
        assert os.write(fd, data) == len(data)
    finally:
        os.close(fd)
