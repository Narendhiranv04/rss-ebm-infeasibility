"""Run approachb_3d_learning's own GPU-server scripts unchanged on a laptop CPU.

Only the server-specific calls are stubbed: the resource guard / watchdog (nvidia-smi, /proc, CPU 7 pinning), CUDA memory stats and the
Linux-only os.sched_getaffinity. Training logic, configs and outputs are theirs, byte for byte.

    python ../hybrid/src/cpu_run.py src/train.py fit --config configs/sampler_v1.json --kind cvae --seed 0      (run from approachb_3d_learning/)
    python ../hybrid/src/cpu_run.py src/xy3d_train.py smoke --sidecar <sidecar> --out runs/XY3D_REAL/smoke_cpu
"""
import os, runpy, sys
from pathlib import Path

os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))  # Linux-only

import torch  # noqa: E402

torch.cuda.max_memory_allocated = lambda *a, **k: 0
torch.cuda.get_device_name = lambda *a, **k: "cpu (laptop, via hybrid/cpu_run.py)"

script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parent))
import train as TR  # noqa: E402  (their module; patched before their script imports it)


class _NoWatchdog:
    def terminate(self):
        pass


TR.preflight = lambda cfg, gcfg, dry: ({"device": "cpu"}, torch.device("cpu"))
TR.start_watchdog = lambda run, gcfg_path: (Path(run).mkdir(parents=True, exist_ok=True), _NoWatchdog())[1]

sys.argv = [str(script)] + sys.argv[2:]
if script.name in ("train.py", "xy3d_train.py"):
    # re-running the file as a script would redefine the stubbed functions; run only its CLI block against the imported module
    mod = TR if script.name == "train.py" else __import__("xy3d_train")
    if script.name == "xy3d_train.py" and os.environ.get("XY_SEED"):
        mod.TRAIN["seed"] = int(os.environ["XY_SEED"])  # their smoke fit hard-codes seed 0; extra seeds for the hybrid study only
    src = script.read_text()
    exec(compile(src[src.index('if __name__ == "__main__":'):], str(script), "exec"), {**vars(mod), "__name__": "__main__"})
else:
    runpy.run_path(str(script), run_name="__main__")
