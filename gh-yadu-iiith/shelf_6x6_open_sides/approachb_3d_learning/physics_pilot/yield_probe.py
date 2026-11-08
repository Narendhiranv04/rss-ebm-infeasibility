"""Print the number of collaborator blockers on Blackwell (pending sajid queue entries + yadu generator processes), using the SAME rule as
bwlphys.guard (run as a file, so the pattern is never in this process's own command line).  0 = clear to resume after a yield stop."""
import bwlphys as B
f, bad = B.guard({"temp": 999, "mem_gb": 0, "swap_growth_mb": 1e9, "swap0": 0, "host": "blackwell"})
q = f["queue_pending"]
print(f["yield_processes"] + sum(v for v in q.values() if isinstance(v, int)) + (1 if "ERROR" in q else 0))
