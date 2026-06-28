"""SPURS-augmented DeepSequence Ridge regression baseline.

Refactors the original ``spurs_ridge.py`` procedural functions into a cohesive
class hierarchy.  The public surface is :class:`SpursRidgeBaseline` with a
``run()`` method; the original ``run()`` top-level function is preserved as a
module-level alias for backward compatibility.

Feature matrix per mutant (all row-aligned by index):
  - One-hot encoding of mutant sequence → L × 20
  - DeepSequence log pseudo-likelihood (``pll.csv``, col ``'pll'``) → 1
  - SPURS-predicted ΔΔG (``spurs.pkl``, list of floats) → 1
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AA_ORDER: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(_AA_ORDER)}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MutantRecord:
    """A single aligned row from the merged data sources.

    Attributes:
        seq: Amino-acid sequence string.
        target: Experimental DMS score (``log_fitness``).
        pll: DeepSequence log pseudo-likelihood.
        spurs_ddg: SPURS-predicted ΔΔG scalar.
    """

    seq: str
    target: float
    pll: float
    spurs_ddg: float


@dataclass
class EvalResult:
    """Results for a single training-set-size evaluation point.

    Attributes:
        n_train: Number of training examples.
        mean_spearman: Mean Spearman ρ across repeats.
        std_spearman: Standard deviation of Spearman ρ across repeats.
        raw: Raw per-repeat Spearman ρ values.
    """

    n_train: int
    mean_spearman: float
    std_spearman: float
    raw: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Sub-components
# ---------------------------------------------------------------------------


class _DataLoader:
    """Loads and merges the three row-aligned data sources.

    Args:
        target_col: Column name in ``data.csv`` to use as the regression target.
    """

    def __init__(self, target_col: str = "log_fitness") -> None:
        self.target_col = target_col

    def load(
        self,
        data_path: Path,
        pll_path: Path,
        spurs_path: Path,
    ) -> List[MutantRecord]:
        """Load and merge all three sources into a list of :class:`MutantRecord`.

        Args:
            data_path: Path to ``data/{dataset}/data.csv``.
            pll_path: Path to ``DeepSequence/pll.csv``.
            spurs_path: Path to ``spurs.pkl``.

        Returns:
            List of merged :class:`MutantRecord` instances.

        Raises:
            SystemExit: On missing columns, empty files, or row-count mismatches.
        """
        dms_rows = self._read_csv(data_path, required=["seq", self.target_col])
        pll_rows = self._read_csv(pll_path, required=["pll"])
        spurs_list = self._read_pickle(spurs_path)

        n = len(dms_rows)
        if len(pll_rows) != n or len(spurs_list) != n:
            sys.exit(
                f"Row count mismatch — data.csv:{n}, "
                f"pll.csv:{len(pll_rows)}, spurs.pkl:{len(spurs_list)}"
            )

        records: List[MutantRecord] = []
        for i, (dms, pll) in enumerate(zip(dms_rows, pll_rows)):
            try:
                records.append(
                    MutantRecord(
                        seq=dms["seq"],
                        target=float(dms[self.target_col]),
                        pll=float(pll["pll"]),
                        spurs_ddg=float(spurs_list[i]),
                    )
                )
            except (ValueError, KeyError):
                pass

        if len(records) < n:
            print(f"  [warn] dropped {n - len(records)} rows with missing values")

        return records

    @staticmethod
    def _read_csv(path: Path, required: List[str]) -> List[Dict[str, str]]:
        """Read a CSV file and validate required columns."""
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            sys.exit(f"ERROR: {path} is empty.")
        missing = [c for c in required if c not in rows[0]]
        if missing:
            sys.exit(f"ERROR: missing columns {missing} in {path}")
        return rows

    @staticmethod
    def _read_pickle(path: Path) -> list:
        """Load a pickle file and return its contents."""
        with open(path, "rb") as fh:
            return pickle.load(fh)


class _FeatureBuilder:
    """Builds the feature matrix from a list of :class:`MutantRecord` instances.

    Feature layout per row: ``[one_hot(seq) | pll | spurs_ddg]``
    Shape: ``(N, L*20 + 2)``
    """

    def build(self, records: List[MutantRecord]) -> Tuple[np.ndarray, np.ndarray]:
        """Construct feature matrix X and target vector y.

        Args:
            records: List of loaded :class:`MutantRecord` instances.

        Returns:
            Tuple ``(X, y)`` with shapes ``(N, L*20+2)`` and ``(N,)``.
        """
        seq_len = len(records[0].seq)
        n = len(records)

        X = np.zeros((n, seq_len * 20 + 2), dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)

        for i, rec in enumerate(records):
            for j, aa in enumerate(rec.seq):
                if aa in _AA_INDEX:
                    X[i, j * 20 + _AA_INDEX[aa]] = 1.0
            X[i, seq_len * 20] = rec.pll
            X[i, seq_len * 20 + 1] = rec.spurs_ddg
            y[i] = rec.target

        return X, y


class _LowNEvaluator:
    """Low-N evaluation protocol for Ridge regression.

    For each training-set size N in ``[48, 96, 144, 192, 240, int(0.8*total)]``,
    repeats random train/test splits *n_repeats* times and averages Spearman ρ.

    Args:
        n_repeats: Number of random repeats per training size.
        random_seed: Master RNG seed.
        alpha: Ridge regularisation strength.
    """

    def __init__(
        self,
        n_repeats: int = 20,
        random_seed: int = 42,
        alpha: float = 1e-8,
    ) -> None:
        self.n_repeats = n_repeats
        self.random_seed = random_seed
        self.alpha = alpha

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[EvalResult]:
        """Run the low-N evaluation grid.

        Args:
            X: Feature matrix, shape ``(N, D)``.
            y: Target vector, shape ``(N,)``.

        Returns:
            List of :class:`EvalResult` instances, one per training size.
        """
        total = len(X)
        ns = sorted(
            {n for n in [48, 96, 144, 192, 240, int(0.8 * total)] if n < total}
        )
        rng = np.random.default_rng(self.random_seed)
        results: List[EvalResult] = []

        for n_train in ns:
            raw_scores = self._repeat_eval(X, y, n_train, total, rng)
            result = EvalResult(
                n_train=n_train,
                mean_spearman=float(np.mean(raw_scores)),
                std_spearman=float(np.std(raw_scores)),
                raw=raw_scores,
            )
            results.append(result)
            print(
                f"  N={n_train:>4d}  Spearman = "
                f"{result.mean_spearman:.4f} ± {result.std_spearman:.4f}"
            )

        return results

    def _repeat_eval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_train: int,
        total: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Run *n_repeats* train/test splits for a given *n_train*.

        Args:
            X: Feature matrix.
            y: Targets.
            n_train: Training set size.
            total: Total number of samples.
            rng: Master random generator.

        Returns:
            Array of Spearman ρ values, shape ``(n_repeats,)``.
        """
        scores = []
        for _ in range(self.n_repeats):
            seed = int(rng.integers(0, 2**31))
            idx = np.random.default_rng(seed).permutation(total)
            train_idx, test_idx = idx[:n_train], idx[n_train:]
            model = Ridge(alpha=self.alpha)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            rho, _ = spearmanr(y[test_idx], preds)
            scores.append(rho)
        return np.array(scores)


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class SpursRidgeBaseline:
    """Full SPURS-augmented DeepSequence Ridge baseline pipeline.

    Composes :class:`_DataLoader`, :class:`_FeatureBuilder`, and
    :class:`_LowNEvaluator` into a single cohesive entry point.

    Args:
        target_col: Column in ``data.csv`` to use as the regression target.
        n_repeats: Repeats per training-size evaluation point.
        random_seed: Master RNG seed.
        alpha: Ridge regularisation strength.

    Example:
        >>> baseline = SpursRidgeBaseline()
        >>> results = baseline.run(
        ...     data_path=Path("data/PTEN_HUMAN/data.csv"),
        ...     pll_path=Path("...pll.csv"),
        ...     spurs_path=Path("...spurs.pkl"),
        ... )
    """

    def __init__(
        self,
        target_col: str = "log_fitness",
        n_repeats: int = 20,
        random_seed: int = 42,
        alpha: float = 1e-8,
    ) -> None:
        self._loader = _DataLoader(target_col=target_col)
        self._feature_builder = _FeatureBuilder()
        self._evaluator = _LowNEvaluator(
            n_repeats=n_repeats, random_seed=random_seed, alpha=alpha
        )

    def run_with_fixed_splits(
        self,
        data_dir: Path,
        fitness_dir: Path,
        model_seed: int = 1,
    ) -> EvalResult:
        """Run Ridge on the same pre-built train/test splits used by PsiFit.

        Loads ``train_{1..5}.csv`` and ``test.csv`` from *data_dir*, looks up
        each row's PLL and SPURS score by its original integer index (the
        unnamed first column in the split CSVs), fits Ridge on the training
        folds, and evaluates Spearman ρ on the test set.

        Args:
            data_dir: ``data_rerun_fixed/{dataset}/`` containing
                ``train_*.csv``, ``test.csv``.
            fitness_dir: ``fitness/proteingym_deepsequence/{dataset}/``
                containing ``DeepSequence/pll.csv`` and ``spurs.pkl``.
            model_seed: Which ``train_{i}.csv`` to hold out as validation
                (matches the ``--model_seed`` passed to ``train_entry.py``).

        Returns:
            A single :class:`EvalResult` with ``std_spearman=0`` (no repeats
            — the split is fixed).
        """
        import pandas as pd

        pll_df = pd.read_csv(fitness_dir / "DeepSequence" / "pll.csv", index_col=0)
        pll_series = pll_df["pll"]
        pll_series.index = [int(s.replace("id_", "")) for s in pll_series.index]

        with open(fitness_dir / "spurs.pkl", "rb") as fh:
            spurs_list = pickle.load(fh)

        train_parts = [
            pd.read_csv(data_dir / f"train_{i}.csv", index_col=0)
            for i in range(1, 6)
            if i != model_seed
        ]
        train_df = pd.concat(train_parts)
        test_df = pd.read_csv(data_dir / "test.csv", index_col=0)

        def _to_records(df: "pd.DataFrame") -> List[MutantRecord]:
            records = []
            for idx, row in df.iterrows():
                try:
                    pid = int(row["PID"]) if "PID" in row else int(idx)
                    seq = row["seq"] if "seq" in row.index else idx
                    records.append(MutantRecord(
                        seq=seq,
                        target=float(row[self._loader.target_col]),
                        pll=float(pll_series[pid]),
                        spurs_ddg=float(spurs_list[pid]),
                    ))
                except (KeyError, IndexError):
                    pass
            return records

        train_records = _to_records(train_df)
        test_records = _to_records(test_df)
        print(f"  train: {len(train_records)}  test: {len(test_records)}")

        X_train, y_train = self._feature_builder.build(train_records)
        X_test, y_test = self._feature_builder.build(test_records)
        print(f"  X shape: {X_train.shape}  (L*20={X_train.shape[1]-2} + pll + spurs_ddg)")

        model = Ridge(alpha=self._evaluator.alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rho, _ = spearmanr(y_test, preds)

        print(f"  Spearman ρ on test: {rho:.4f}")
        return EvalResult(
            n_train=len(train_records),
            mean_spearman=float(rho),
            std_spearman=0.0,
            raw=np.array([rho]),
        )

    def run(
        self,
        data_path: Path,
        pll_path: Path,
        spurs_path: Path,
    ) -> List[EvalResult]:
        """Execute the full data-load → feature-build → evaluate pipeline.

        Args:
            data_path: Path to ``data/{dataset}/data.csv``.
            pll_path: Path to DeepSequence ``pll.csv``.
            spurs_path: Path to SPURS ``spurs.pkl``.

        Returns:
            List of :class:`EvalResult` instances, one per training size.
        """
        print("Loading data …")
        records = self._loader.load(data_path, pll_path, spurs_path)
        print(f"  {len(records)} samples loaded")

        print("Building feature matrix …")
        X, y = self._feature_builder.build(records)
        print(f"  X shape: {X.shape}  (L*20={X.shape[1]-2} + pll + spurs_ddg)")

        print(
            f"\nRunning low-N evaluation "
            f"({self._evaluator.n_repeats} repeats, alpha={self._evaluator.alpha}) …"
        )
        return self._evaluator.evaluate(X, y)

    def run_all(
        self,
        data_dir: Path = Path("data"),
        fitness_dir: Path = Path("fitness/proteingym_deepsequence"),
        verbose: bool = True,
    ) -> "pd.DataFrame":
        """Run the baseline on every dataset that has all three required files.

        Iterates over subdirectories of *fitness_dir* and skips any dataset
        missing ``data.csv``, ``pll.csv``, or ``spurs.pkl``.  Returns a tidy
        DataFrame with one row per (dataset, n_train) combination, ready for
        plotting or further analysis.

        Args:
            data_dir: Root directory containing ``{dataset}/data.csv``.
            fitness_dir: Root directory containing
                ``{dataset}/DeepSequence/pll.csv`` and ``{dataset}/spurs.pkl``.
            verbose: Print per-dataset progress when ``True``.

        Returns:
            DataFrame with columns:
            ``dataset``, ``n_train``, ``mean_spearman``, ``std_spearman``.

        Example:
            >>> baseline = SpursRidgeBaseline()
            >>> df = baseline.run_all()
            >>> df.groupby("n_train")["mean_spearman"].mean()
        """
        import pandas as pd  # lazy import

        fitness_root = Path(fitness_dir)
        data_root = Path(data_dir)
        rows = []
        datasets = sorted(d.name for d in fitness_root.iterdir() if d.is_dir())

        for i, dataset in enumerate(datasets):
            data_path  = data_root  / dataset / "data.csv"
            pll_path   = fitness_root / dataset / "DeepSequence" / "pll.csv"
            spurs_path = fitness_root / dataset / "spurs.pkl"

            if not (data_path.exists() and pll_path.exists() and spurs_path.exists()):
                if verbose:
                    print(f"[{i+1}/{len(datasets)}] SKIP {dataset} — missing files")
                continue

            if verbose:
                print(f"[{i+1}/{len(datasets)}] {dataset}")

            try:
                results = self.run(data_path, pll_path, spurs_path)
                for r in results:
                    rows.append({
                        "dataset":        dataset,
                        "n_train":        r.n_train,
                        "mean_spearman":  r.mean_spearman,
                        "std_spearman":   r.std_spearman,
                    })
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  ERROR: {exc}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_table(results: List[EvalResult]) -> None:
        """Print a formatted results table to stdout.

        Args:
            results: List of :class:`EvalResult` instances.
        """
        print("\n" + "=" * 46)
        print(f"{'N_train':>8}  {'Mean Spearman':>14}  {'Std':>8}")
        print("-" * 46)
        for r in results:
            print(f"{r.n_train:>8d}  {r.mean_spearman:>14.4f}  {r.std_spearman:>8.4f}")
        print("=" * 46)

    @staticmethod
    def save_plot(results: List[EvalResult], path: Path) -> None:
        """Save a mean Spearman vs. N training-size plot.

        Args:
            results: List of :class:`EvalResult` instances.
            path: Output file path for the figure.
        """
        import matplotlib.pyplot as plt  # lazy import — optional dependency

        ns = [r.n_train for r in results]
        means = [r.mean_spearman for r in results]
        stds = [r.std_spearman for r in results]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(ns, means, yerr=stds, marker="o", capsize=4)
        ax.set_xlabel("Training set size N")
        ax.set_ylabel("Spearman ρ (mean ± std, 20 repeats)")
        ax.set_title("SPURS-augmented DeepSequence – low-N evaluation")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"Plot saved to {path}")


# ---------------------------------------------------------------------------
# Module-level backward-compatible alias
# ---------------------------------------------------------------------------


def run(
    data_path: Path,
    pll_path: Path,
    spurs_path: Path,
    target_col: str = "log_fitness",
    n_repeats: int = 20,
    random_seed: int = 42,
    alpha: float = 1e-8,
) -> List[EvalResult]:
    """Backward-compatible functional entry point (delegates to SpursRidgeBaseline).

    Args:
        data_path: Path to ``data/{dataset}/data.csv``.
        pll_path: Path to DeepSequence ``pll.csv``.
        spurs_path: Path to SPURS ``spurs.pkl``.
        target_col: Regression target column.
        n_repeats: Repeats per training-size point.
        random_seed: Master seed.
        alpha: Ridge regularisation.

    Returns:
        List of :class:`EvalResult` instances.
    """
    return SpursRidgeBaseline(
        target_col=target_col,
        n_repeats=n_repeats,
        random_seed=random_seed,
        alpha=alpha,
    ).run(data_path=data_path, pll_path=pll_path, spurs_path=spurs_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline CLI."""
    p = argparse.ArgumentParser(
        description="SPURS-augmented DeepSequence Ridge regression"
    )
    p.add_argument("--data", required=True, help="Path to data/{dataset}/data.csv")
    p.add_argument("--pll", required=True, help="Path to DeepSequence/pll.csv")
    p.add_argument("--spurs", required=True, help="Path to spurs.pkl")
    p.add_argument("--target-col", default="log_fitness",
                   help="Column in data.csv to predict")
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=1e-8)
    p.add_argument("--plot", metavar="PATH", default=None,
                   help="Save Spearman-vs-N plot to this path")
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    _baseline = SpursRidgeBaseline(
        target_col=_args.target_col,
        n_repeats=_args.n_repeats,
        random_seed=_args.seed,
        alpha=_args.alpha,
    )
    _results = _baseline.run(
        data_path=Path(_args.data),
        pll_path=Path(_args.pll),
        spurs_path=Path(_args.spurs),
    )
    SpursRidgeBaseline.print_table(_results)
    if _args.plot:
        SpursRidgeBaseline.save_plot(_results, Path(_args.plot))