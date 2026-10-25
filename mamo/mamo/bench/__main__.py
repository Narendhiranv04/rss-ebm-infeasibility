"""Command line for the MAMO benchmark tool.

    python -m mamo.bench app [--port 8501]
    python -m mamo.bench list
    python -m mamo.bench label reference-10 [--workers 6]
    python -m mamo.bench new --name demo --methods m4m em4m --datasets reference-10 \
        --difficulties easy medium hard --per-difficulty 5 --budget 300 --workers 6 [--run]
    python -m mamo.bench run <experiment folder>
    python -m mamo.bench generate my-dataset --envs open_tabletop access19 --tasks retrieve relocate         --difficulties easy medium hard --per 10 [--start 0] [--no-check] [--workers 4]

``--workspace DIR`` may be given before or after the command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mamo.bench.runner import DEFAULT_WORKSPACE, ROOT


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m mamo.bench")
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", default=argparse.SUPPRESS)  # also accepted after the command
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("app", help="start the web app", parents=[common])
    a.add_argument("--port", type=int, default=8501)
    sub.add_parser("list", help="list planners, datasets and experiments", parents=[common])
    lb = sub.add_parser("label", help="label a dataset's scenes by difficulty", parents=[common])
    lb.add_argument("dataset")
    lb.add_argument("--workers", type=int, default=6)
    nw = sub.add_parser("new", help="create an experiment", parents=[common])
    nw.add_argument("--name", required=True)
    nw.add_argument("--methods", nargs="+", required=True)
    nw.add_argument("--datasets", nargs="+", required=True)
    nw.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    nw.add_argument("--per-difficulty", type=int, default=None)
    nw.add_argument("--seeds", type=int, nargs="+", default=[0])
    nw.add_argument("--budget", type=float, default=None, help="override every method's default budget")
    nw.add_argument("--workers", type=int, default=6)
    nw.add_argument("--no-execute", action="store_true")
    nw.add_argument("--envs", nargs="+", default=None, help="generated datasets: only these environments")
    nw.add_argument("--tasks", nargs="+", default=None, help="generated datasets: only these tasks")
    nw.add_argument("--run", action="store_true", help="run it now (blocking)")
    rn = sub.add_parser("run", help="run or resume an experiment (blocking)", parents=[common])
    rn.add_argument("experiment")
    gn = sub.add_parser("generate", help="generate a dataset in the REAP environments (mamo.envs)",
                        parents=[common])
    gn.add_argument("name")
    gn.add_argument("--envs", nargs="+", default=["open_tabletop", "confined_shelf", "multi_level_shelf",
                                                  "access19", "stacking"])
    gn.add_argument("--tasks", nargs="+", default=["retrieve", "relocate"])
    gn.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    gn.add_argument("--per", type=int, default=10, help="problems per (environment, task, difficulty)")
    gn.add_argument("--start", type=int, default=0, help="index of the first problem")
    gn.add_argument("--no-check", action="store_true", help="skip the target feasibility check")
    gn.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)

    if args.cmd == "app":
        # Load MuJoCo on the main thread before Streamlit starts: loading its DLL for the first
        # time inside Streamlit's script thread can fail on Windows (WinError 1114).
        import mujoco  # noqa: F401
        from streamlit.web import bootstrap
        app = Path(__file__).with_name("app") / "Home.py"
        flags = {"server.port": args.port, "server.address": "localhost"}  # this machine only
        bootstrap.load_config_options(flag_options=flags)
        bootstrap.run(str(app), False, ["--workspace", str(ws)], flags)
        return
    from mamo.bench.datasets import list_datasets
    from mamo.bench.planners import REGISTRY, load_plugins
    from mamo.bench import runner
    errors = load_plugins(runner.plugin_dirs(ws))
    if args.cmd == "list":
        print("planners:")
        for s in REGISTRY.values():
            print(f"  {s.name:16s} [{s.family}] {s.description}")
        for e in errors:
            print("  plugin error:", e)
        print("datasets:")
        for d in list_datasets(ws).values():
            print(f"  {d.name:16s} {d.root}")
        print("experiments:")
        for e in runner.list_experiments(ws):
            print(f"  {e.name}: {runner.read_status(e).get('state')}")
    elif args.cmd == "label":
        runner.label_dataset(ws, args.dataset, args.workers)
    elif args.cmd == "new":
        methods = [{"name": m, "budget": args.budget or REGISTRY[m].budget, "params": {}} for m in args.methods]
        exp = runner.new_experiment(ws, args.name, methods, args.datasets, args.difficulties, args.per_difficulty,
                                    args.seeds, args.workers, execute=not args.no_execute, envs=args.envs,
                                    tasks=args.tasks)
        print(exp)
        if args.run:
            runner.run_experiment(ws, exp)
    elif args.cmd == "run":
        runner.run_experiment(ws, Path(args.experiment))
    elif args.cmd == "generate":
        from mamo.bench.datasets import generate_dataset
        ds = generate_dataset(ws, args.name, args.envs, args.tasks, args.difficulties, args.per, args.start,
                              not args.no_check, args.workers)
        print(f"{ds.name}: {len(ds.scenes())} scenes in {ds.root}")


if __name__ == "__main__":
    main()
