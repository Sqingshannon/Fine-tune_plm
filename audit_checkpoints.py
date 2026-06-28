#!/usr/bin/env python3
"""Audit ConFit checkpoints for corruption from the multi-rank save race condition.

Recursively scans every folder under a root directory that looks like a checkpoint
(contains adapter_config.json, adapter_model.bin, adapter_model.safetensors, or
A.pth) and runs the following health checks on each:

  1. adapter_config.json  — parseable JSON with all required PEFT keys
  2. adapter_model.bin/.safetensors — exists and above minimum size (1 MB)
  3. A.pth                — if present, non-empty and torch-loadable as a state dict

Usage
-----
    python audit_checkpoints.py /path/to/experiments
    python audit_checkpoints.py /path/to/experiments --output report.csv
    python audit_checkpoints.py /path/to/experiments --output report.json
    python audit_checkpoints.py /path/to/experiments --progress-every 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants — tuned to this project's checkpoint structure
# ---------------------------------------------------------------------------

# A healthy adapter_model.bin in this project is ~5.3 MB.
# 1 MB is a conservative floor; anything below almost certainly means a
# partial / race-corrupted write.
_ADAPTER_MIN_BYTES = 1 * 1024 * 1024  # 1 MB

# Checked in this order; first match wins.
_ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")

# Any directory containing at least one of these is treated as a checkpoint.
_CHECKPOINT_MARKERS = frozenset(
    {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "A.pth"}
)

# PEFT keys that must be present for a valid LoRA adapter config.
_REQUIRED_CONFIG_KEYS = frozenset(
    {"peft_type", "base_model_name_or_path", "r", "lora_alpha", "target_modules"}
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CheckpointResult:
    path: str
    ok: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # File-presence metadata
    has_adapter_config: bool = False
    adapter_config_bytes: int = 0
    has_adapter_weights: bool = False
    adapter_weights_file: str = ""
    adapter_weights_bytes: int = 0
    has_a_pth: bool = False
    a_pth_bytes: int = 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_checkpoint_dirs(root: Path) -> List[Path]:
    """Return every directory under *root* that contains checkpoint marker files."""
    found: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if set(filenames) & _CHECKPOINT_MARKERS:
            found.append(Path(dirpath))
    found.sort()
    return found


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_adapter_config(ckpt_dir: Path, result: CheckpointResult) -> None:
    config_path = ckpt_dir / "adapter_config.json"
    if not config_path.exists():
        # Optional: only an error when the folder was expected to have one
        # (inferred from the presence of adapter weights).
        result.warn("adapter_config.json: file not found")
        return

    result.has_adapter_config = True
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        result.error(f"adapter_config.json: cannot read — {exc}")
        return

    result.adapter_config_bytes = len(raw)

    if len(raw) == 0:
        result.error(
            "adapter_config.json: empty file (0 bytes) — typical symptom of the "
            "multi-rank race condition: a non-rank-0 process truncated the file "
            "before rank 0 could write it"
        )
        return

    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        result.error(
            f"adapter_config.json: invalid JSON — {exc} "
            f"(file size={len(raw)} B, first 120 chars: {raw[:120]!r})"
        )
        return

    if not isinstance(cfg, dict):
        result.error(
            f"adapter_config.json: top-level value is {type(cfg).__name__!r}, expected dict"
        )
        return

    missing = _REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        result.error(
            f"adapter_config.json: missing required keys: {sorted(missing)}"
        )


def _check_adapter_weights(ckpt_dir: Path, result: CheckpointResult) -> None:
    for fname in _ADAPTER_WEIGHT_NAMES:
        wpath = ckpt_dir / fname
        if not wpath.exists():
            continue

        result.has_adapter_weights = True
        result.adapter_weights_file = fname

        try:
            size = wpath.stat().st_size
        except OSError as exc:
            result.error(f"{fname}: cannot stat — {exc}")
            return

        result.adapter_weights_bytes = size

        if size == 0:
            result.error(
                f"{fname}: empty file (0 bytes) — likely a truncated write from the race condition"
            )
        elif size < _ADAPTER_MIN_BYTES:
            result.error(
                f"{fname}: suspiciously small ({size:,} B < {_ADAPTER_MIN_BYTES:,} B minimum). "
                f"Healthy files in this project are ~5.5 MB."
            )
        return  # found one — done

    # Neither file was found.
    if result.has_adapter_config:
        result.error(
            "adapter_model.safetensors / adapter_model.bin: neither found, "
            "but adapter_config.json is present — weights file may have been "
            "deleted or never written"
        )
    else:
        result.warn(
            "No adapter weights file found (A.pth-only checkpoint?). "
            "This is unexpected — investigate manually."
        )


def _check_a_pth(ckpt_dir: Path, result: CheckpointResult) -> None:
    a_path = ckpt_dir / "A.pth"
    if not a_path.exists():
        return  # A.pth is optional; absence is not an error by itself

    result.has_a_pth = True

    try:
        size = a_path.stat().st_size
    except OSError as exc:
        result.error(f"A.pth: cannot stat — {exc}")
        return

    result.a_pth_bytes = size

    if size == 0:
        result.error(
            "A.pth: empty file (0 bytes) — likely a truncated write from the race condition"
        )
        return

    # Try to load with weights_only=True (safe; rejects arbitrary pickle code).
    # Fall back to weights_only=False for older PyTorch versions that don't
    # support the flag, but emit a warning.
    load_error: Optional[Exception] = None
    obj = None
    for weights_only in (True, False):
        try:
            obj = _torch_load(a_path, weights_only=weights_only)
            if not weights_only:
                result.warn(
                    "A.pth: loaded with weights_only=False (upgrade PyTorch >= 2.0 "
                    "to enable the safer weights_only=True path)"
                )
            break
        except Exception as exc:
            load_error = exc

    if obj is None:
        result.error(f"A.pth: torch.load failed — {load_error}")
        return

    if not isinstance(obj, dict):
        result.error(
            f"A.pth: loaded object is {type(obj).__name__!r}, expected a state_dict (dict)"
        )
        return

    if len(obj) == 0:
        result.warn("A.pth: state_dict has no keys — module may have had no parameters")


def _torch_load(path: Path, *, weights_only: bool):
    """Thin wrapper so we can import torch lazily and handle version differences."""
    import torch  # local import keeps startup fast if torch is slow to import

    try:
        return torch.load(str(path), map_location="cpu", weights_only=weights_only)
    except TypeError:
        # weights_only kwarg not supported in PyTorch < 1.13
        if weights_only:
            raise
        return torch.load(str(path), map_location="cpu")


# ---------------------------------------------------------------------------
# Per-checkpoint audit
# ---------------------------------------------------------------------------


def audit_checkpoint(ckpt_dir: Path) -> CheckpointResult:
    result = CheckpointResult(path=str(ckpt_dir))
    try:
        _check_adapter_config(ckpt_dir, result)
        _check_adapter_weights(ckpt_dir, result)
        _check_a_pth(ckpt_dir, result)
    except Exception as exc:
        result.error(f"Unexpected audit error: {type(exc).__name__}: {exc}")
    result.ok = len(result.errors) == 0
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: List[CheckpointResult]) -> None:
    total = len(results)
    good = sum(1 for r in results if r.ok)
    bad = total - good
    warn_only = sum(1 for r in results if r.ok and r.warnings)

    print()
    print("=" * 72)
    print("  CHECKPOINT AUDIT SUMMARY")
    print("=" * 72)
    print(f"  Total scanned    : {total:,}")
    print(f"  Healthy          : {good:,}")
    print(f"  Corrupted        : {bad:,}")
    if warn_only:
        print(f"  Healthy+warnings : {warn_only:,}")
    print("=" * 72)

    if bad:
        print(f"\nCORRUPTED CHECKPOINTS ({bad}):")
        print("-" * 72)
        for r in results:
            if not r.ok:
                print(f"\n  {r.path}")
                for err in r.errors:
                    print(f"    [ERROR]   {err}")
                for w in r.warnings:
                    print(f"    [WARN]    {w}")

    if warn_only:
        print(f"\nHEALTHY CHECKPOINTS WITH WARNINGS ({warn_only}):")
        print("-" * 72)
        for r in results:
            if r.ok and r.warnings:
                print(f"\n  {r.path}")
                for w in r.warnings:
                    print(f"    [WARN]    {w}")

    print("\n" + "=" * 72)


def _flatten_result(r: CheckpointResult) -> dict:
    d = asdict(r)
    d["errors"] = " | ".join(r.errors)
    d["warnings"] = " | ".join(r.warnings)
    return d


def write_report(results: List[CheckpointResult], output_path: Path) -> None:
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        fieldnames = [
            "path", "ok", "errors", "warnings",
            "has_adapter_config", "adapter_config_bytes",
            "has_adapter_weights", "adapter_weights_file", "adapter_weights_bytes",
            "has_a_pth", "a_pth_bytes",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(_flatten_result(r))

    elif suffix == ".json":
        payload = [asdict(r) for r in results]
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    else:
        # Default to JSON for unknown extensions
        out = output_path.with_suffix(".json")
        print(f"  Unknown extension {suffix!r}; writing JSON to {out}")
        write_report(results, out)
        return

    print(f"  Report written → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to scan recursively (e.g. /path/to/checkpoint_rerun_fixed).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("checkpoint_audit_report.json"),
        metavar="FILE",
        help="Output report file (.json or .csv).  Default: checkpoint_audit_report.json",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        metavar="N",
        help="Print a progress line every N checkpoints (default: 200).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Root          : {root}")
    print(f"Output report : {args.output}")
    print()
    print("Discovering checkpoint directories...")
    ckpt_dirs = find_checkpoint_dirs(root)
    total = len(ckpt_dirs)
    print(f"Found {total:,} checkpoint directories.\n")

    if total == 0:
        print("No checkpoints found. Check that the root path is correct.")
        sys.exit(0)

    results: List[CheckpointResult] = []
    n_bad = 0

    for i, ckpt_dir in enumerate(ckpt_dirs, 1):
        result = audit_checkpoint(ckpt_dir)
        results.append(result)

        if not result.ok:
            n_bad += 1
            # Always print corrupted ones immediately so you can monitor live.
            print(f"  [{i:>{len(str(total))},}/{total:,}]  CORRUPTED  {ckpt_dir}")
            for err in result.errors:
                print(f"               [ERROR] {err}")
        elif i % args.progress_every == 0:
            pct = 100 * i / total
            print(f"  [{i:>{len(str(total))},}/{total:,}] ({pct:5.1f}%)  {n_bad} corrupted so far ...")

    # Final progress line
    print(f"  [{total:>{len(str(total))},}/{total:,}] (100.0%)  Done. {n_bad} corrupted total.")

    print_summary(results)
    write_report(results, args.output)

    sys.exit(1 if n_bad > 0 else 0)


if __name__ == "__main__":
    main()
