"""Experiment orchestration CLI — thin wrapper around ExperimentRunner.

Replaces the original ``run.py`` whose top-level code executed at import time.
All logic now lives in :class:`~confit.runners.experiment.ExperimentRunner`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from confit.runners.experiment import ExperimentRunner, RunMode, RunnerConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], required=True)
    p.add_argument(
        "--mode", type=str,
        choices=[m.value for m in RunMode],
        default=RunMode.MAIN.value,
    )
    p.add_argument("--shot",          type=int, default=96)
    p.add_argument("--check_shot",    type=int, default=96)
    p.add_argument("--run_suffix",    type=str, default="rerun_fixed")
    p.add_argument("--max_datasets",  type=int, default=None)
    p.add_argument("--last_datasets", type=int, default=None)
    p.add_argument(
        "--model_seeds", type=int, nargs="+", default=[1],
        help="Validation fold indices to run (1–5). Pass multiple for CV, e.g. --model_seeds 1 2 3 4 5",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = RunnerConfig(
        quarter=args.quarter,
        mode=RunMode(args.mode),
        shot=args.shot,
        check_shot=args.check_shot,
        run_suffix=args.run_suffix,
        max_datasets=args.max_datasets,
        last_datasets=args.last_datasets,
        model_seeds=args.model_seeds,
        base_dir=Path("."),
    )
    ExperimentRunner(cfg).run()