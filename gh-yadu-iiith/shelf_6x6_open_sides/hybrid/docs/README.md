# docs: what each file is

| File | What it is |
|---|---|
| `EXPERIMENT_LOG.md` | Every experiment so far: what we expected, how we tested it, what happened. |
| `HZ_V34_PREREG.md` | The test plan for the main TEST result, written and frozen before TEST was scored. |
| `HZ_V34_FREEZE.sha256` | Fingerprints (hashes) of the code, models and scene lists used for that test, frozen at the same time. |
| `HZ_V34_env2d_pip.txt`, `HZ_V34_env3d_pip.txt` | Exact Python package versions used for that test (2D and 3D). |
| `E9v2_FREEZE.sha256` | Fingerprints for an earlier, smaller pre-registered experiment (see `EXPERIMENT_LOG.md`, E9v2). |
| `E9v2_pip_freeze.txt` | Package versions for that earlier experiment. |
| `literature/related_work_brief.md` | Related work: about 40 checked papers on learned plan proposal and success prediction. |
| `literature/three_way_scan.md` | Notes on the team's reading list: why each paper matters, how it works, what it shows. |
| `literature/factor_graphs_and_heuristics.md` | Short note on whether factor graphs or better search heuristics could raise plan success. |

The `HZ_V34_*` and `E9v2_*` names stay as they are: the frozen records and the evaluation code refer to these exact paths, so renaming them would break the proof that the plan was fixed before scoring.
