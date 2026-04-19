"""Convert runner outputs into pandas DataFrames and CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _is_run_result(obj: Any) -> bool:
    """Return True when object looks like a single ``run`` result."""
    if not isinstance(obj, dict):
        return False
    if not obj:
        return False
    sample = next(iter(obj.values()))
    return isinstance(sample, dict) and "metrics" in sample


def _is_loop_result(obj: Any) -> bool:
    """Return True when object looks like a ``loop_run`` result."""
    if not isinstance(obj, dict):
        return False
    if not obj:
        return False
    sample = next(iter(obj.values()))
    return _is_run_result(sample)


def run_result_to_dataframe(run_result: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert a single ``run`` result into a flat DataFrame.

    Parameters
    ----------
    run_result : dict[str, dict[str, Any]]
        Output produced by ``runner.run``.

    Returns
    -------
    pd.DataFrame
        One row per state (for example GHZ and W) with metric columns.
    """
    rows: list[dict[str, Any]] = []

    for state, payload in run_result.items():
        row: dict[str, Any] = {"state": state}
        metrics = payload.get("metrics", {})
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)

    return pd.DataFrame(rows)


def loop_result_to_dataframe(loop_result: dict[float, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    """Convert a ``loop_run`` result into a merged wide DataFrame.

    Parameters
    ----------
    loop_result : dict[float, dict[str, dict[str, Any]]]
        Output produced by ``runner.loop_run``.

    Returns
    -------
    pd.DataFrame
        One row per ``sweep_value``. Metric columns are prefixed with state
        labels, for example ``ghz_entropy_pure`` and ``w_entropy_pure``.
    """
    rows: list[dict[str, Any]] = []

    for sweep_value, run_result in loop_result.items():
        row: dict[str, Any] = {"sweep_value": float(sweep_value)}
        for state, payload in run_result.items():
            metrics = payload.get("metrics", {})
            for key, value in metrics.items():
                row[f"{state}_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def result_to_dataframe(result: dict[Any, Any]) -> pd.DataFrame:
    """Auto-convert ``run`` or ``loop_run`` output into a DataFrame.

    Parameters
    ----------
    result : dict[Any, Any]
        Either output from ``runner.run`` or ``runner.loop_run``.

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame for downstream analysis.

    Raises
    ------
    ValueError
        If the input format does not match known runner outputs.
    """
    if _is_run_result(result):
        return run_result_to_dataframe(result)
    if _is_loop_result(result):
        return loop_result_to_dataframe(result)
    raise ValueError("Unsupported result format. Provide run(...) or loop_run(...) output.")


def dataframe_to_csv(df: pd.DataFrame, output_csv: str | Path, index: bool = False) -> Path:
    """Write a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    output_csv : str | Path
        Output CSV path.
    index : bool, optional
        If ``True``, include DataFrame index in the CSV.

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path.resolve()


def dump_result_to_csv(
    result: dict[Any, Any],
    output_csv: str | Path,
    index: bool = False,
) -> pd.DataFrame:
    """Convert runner output to DataFrame and dump it as CSV.

    Parameters
    ----------
    result : dict[Any, Any]
        Either ``run`` output or ``loop_run`` output.
    output_csv : str | Path
        Output CSV file path.
    index : bool, optional
        If ``True``, include DataFrame index in the CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame written to disk.
    """
    df = result_to_dataframe(result)
    dataframe_to_csv(df, output_csv=output_csv, index=index)
    return df
