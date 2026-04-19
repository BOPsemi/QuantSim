"""Entry point for running density-matrix simulations from the extracted modules."""

from __future__ import annotations

import argparse
from runner import run, loop_run
from datadump import dump_result_to_csv
from visualization import (
    plot_density_matrix_heatmap_plotly,
    plot_multi_scatter_from_csv_plotly,
    plot_results,
    plot_scatter_from_csv_plotly,
)
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed runtime options.
    """
    parser = argparse.ArgumentParser(description="Density-matrix GHZ/W simulation demo")
    parser.add_argument("--bits", type=int, default=3, help="Number of qubits for GHZ/W circuits")
    parser.add_argument("--p1", type=float, default=0.01, help="1-qubit noise parameter")
    parser.add_argument("--p2", type=float, default=0.1, help="2-qubit noise parameter")
    parser.add_argument(
        "--noise-model",
        type=str,
        default="depolarizing",
        choices=["depolarizing", "amplitude", "phase"],
        help="Noise model family for both Kraus and Aer pipelines",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="0-1,1-2",
        help="Comma-separated qubit pairs, e.g. '0-1,1-2'",
    )
    return parser.parse_args()


def parse_pairs(raw_pairs: str) -> list[tuple[int, int]]:
    """Convert CLI pair text into a list of qubit index tuples.

    Parameters
    ----------
    raw_pairs : str
        Comma-separated pair text formatted as ``a-b,c-d``.

    Returns
    -------
    list[tuple[int, int]]
        Parsed list of qubit index pairs.

    Raises
    ------
    ValueError
        If the input format is invalid.
    """
    pairs: list[tuple[int, int]] = []
    for item in raw_pairs.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split("-", maxsplit=1)
        pairs.append((int(left), int(right)))
    if not pairs:
        raise ValueError("At least one pair is required, e.g. '0-1,1-2'.")
    return pairs


def main() -> None:
    """Run the command-line simulation workflow.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    args = parse_args()
    pairs = parse_pairs(args.pairs)
    
    
    
    results = run(
        bits=args.bits,
        p1=args.p1,
        p2=args.p2,
        pairs=pairs,
        noise_model=args.noise_model,
    )
    #plot_results(results)
    
    plot_density_matrix_heatmap_plotly(results, width=800, height=1200)
    
    w_metrics = results["w"]["metrics"]
    ghz_metrics = results["ghz"]["metrics"]
    
    '''
    print("W-state metrics:")
    for key in sorted(w_metrics):
        print(f"  {key}: {w_metrics[key]:.6f}")
        
    
    print("GHZ-state metrics:")
    for key in sorted(ghz_metrics):
        print(f"  {key}: {ghz_metrics[key]:.6f}")    
    '''
    
    # paramter sweep
    p2 = np.arange(0.0, 0.62, 0.02, dtype=float)
    lr = loop_run(
        bits=3,
        pairs=[(0, 1), (1, 2)],
        sweep_param="p2",
        sweep_values=p2,
        p1=0.0,
        noise_model=args.noise_model,
    )
    df2 = dump_result_to_csv(lr, "out/loop_metrics.csv")
    
    print(df2.head())

    '''
    # Scatter plot example from CSV
    plot_scatter_from_csv_plotly(
        csv_path="out/loop_metrics.csv",
        x_axis="sweep_value",
        x_label="p2",
        y_label="Concurrence",
        data=["w_concurrence_ab_pure", "ghz_concurrence_ab_pure"],
        title="Concurrence vs Sweep Value",
    )
    '''
    # Multi-scatter subplot example from CSV
    # GHZ vs. W-state
    plot_multi_scatter_from_csv_plotly(
        csv_path="out/loop_metrics.csv",
        x_axes="sweep_value",
        data_sets=[
            ("w_concurrence_ab_pure", "ghz_concurrence_ab_pure"),
            ("w_entropy_pure", "ghz_entropy_pure"),
            ("w_monogamy_pure", "ghz_monogamy_pure"),
        ],
        x_label="p2",
        y_labels=["Concurrence", "von Neumann Entropy", "Monogamy"],
        titles=["Concurrence Comparison", "von Neumann Entropy Comparison" ,"Monogamy Comparison"],
        figure_title="W vs GHZ Metric Sweep",
        width=600,
        height=1000,
    )
    
    # Multi-scatter subplot example from CSV
    # W-state pure vs. circuit
    plot_multi_scatter_from_csv_plotly(
        csv_path="out/loop_metrics.csv",
        x_axes="sweep_value",
        data_sets=[
            ("w_concurrence_ab_pure", "w_concurrence_ab_circuit"),
            ("w_entropy_pure", "w_entropy_circuit"),
            ("w_monogamy_pure", "w_monogamy_circuit"),
        ],
        x_label="p2",
        y_labels=["Concurrence", "von Neumann Entropy", "Monogamy"],
        titles=["Concurrence Comparison", "von Neumann Entropy Comparison" ,"Monogamy Comparison"],
        figure_title="State vs Circuit Metric Sweep",
        width=600,
        height=1000,
    )
    
if __name__ == "__main__":
    main()
