"""Inference CLI — thin wrapper around InferenceAggregator.

Replaces the original ``inference.py`` whose top-level code executed at import time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from confit.runners.inference import InferenceAggregator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConFit inference aggregation")
    p.add_argument("--dataset",     type=str,   required=True)
    p.add_argument("--shot",        type=int,   required=True)
    p.add_argument("--no_retrival", action="store_true")
    p.add_argument("--alpha",       type=float, default=0.8)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    agg = InferenceAggregator(alpha=args.alpha)
    sr = agg.aggregate(
        dataset=args.dataset,
        shot=args.shot,
        use_retrieval=not args.no_retrival,
    )
    if sr is not None:
        print(f"Spearman ρ for {args.dataset}: {sr:.4f}")
    else:
        print(f"No predictions found for {args.dataset}.")