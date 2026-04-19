"""Plotting helpers for visualizing density matrices."""

from __future__ import annotations

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import DensityMatrix


class StateRunResult(TypedDict):
    """Per-state outputs produced by runner.run."""

    pure_kraus: DensityMatrix
    circuit_aer: DensityMatrix
    metrics: dict[str, float]


RunResults = dict[str, StateRunResult]


def plot_density_matrix_heatmap(
    rho: DensityMatrix,
    basis_labels: list[str] | None = None,
    cmap_real: str = "viridis",
    cmap_imag: str = "magma",
    title: str | None = None,
):
    """Plot real and imaginary parts of a density matrix as heatmaps.

    Parameters
    ----------
    rho : DensityMatrix
        Density matrix to visualize.
    basis_labels : list[str] | None, optional
        Axis labels for computational basis states. If ``None``, labels are
        generated automatically.
    cmap_real : str, optional
        Colormap used for the real-part heatmap.
    cmap_imag : str, optional
        Colormap used for the imaginary-part heatmap.
    title : str | None, optional
        Figure-level title shown above both heatmaps.

    Returns
    -------
    tuple
        ``(fig, ax)`` where ``fig`` is a ``matplotlib.figure.Figure`` and
        ``ax`` is an array of two axes.
    """
    density = DensityMatrix(rho)
    matrix = np.asarray(density.data)
    real_part = np.real(matrix)
    imag_part = np.imag(matrix)
    dim = matrix.shape[0]

    if basis_labels is None:
        n = density.num_qubits
        if n is not None and 2**n == dim:
            basis_labels = [format(i, f"0{n}b") for i in range(dim)]
        else:
            basis_labels = [str(i) for i in range(dim)]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    im0 = ax[0].imshow(real_part, cmap=cmap_real)
    ax[0].set_title("Re(rho)")
    ax[0].set_xlabel("col")
    ax[0].set_ylabel("row")
    ax[0].set_xticks(range(dim))
    ax[0].set_yticks(range(dim))
    ax[0].set_xticklabels(basis_labels, rotation=45, ha="right")
    ax[0].set_yticklabels(basis_labels)
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(imag_part, cmap=cmap_imag)
    ax[1].set_title("Im(rho)")
    ax[1].set_xlabel("col")
    ax[1].set_ylabel("row")
    ax[1].set_xticks(range(dim))
    ax[1].set_yticks(range(dim))
    ax[1].set_xticklabels(basis_labels, rotation=45, ha="right")
    ax[1].set_yticklabels(basis_labels)
    plt.colorbar(im1, ax=ax[1])

    if title:
        fig.suptitle(title)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        plt.tight_layout()
    return fig, ax


def plot_results(results: RunResults) -> None:
    """Plot density matrices returned by ``runner.run``.

    Parameters
    ----------
    results : RunResults
        Output produced by the simulation runner.

    Returns
    -------
    None
    """
    for state in ("ghz", "w"):
        state_label = state.upper()
        plot_density_matrix_heatmap(
            results[state]["pure_kraus"],
            title=f"{state_label} Pure State + Kraus Noise",
        )
        plot_density_matrix_heatmap(
            results[state]["circuit_aer"],
            title=f"{state_label} Circuit + Aer Noise Model",
        )

    plt.show()


def plot_density_matrix_heatmap_plotly(
    rho: DensityMatrix | np.ndarray | list[DensityMatrix | np.ndarray] | RunResults,
    basis_labels: list[str] | None = None,
    colorscale_real: str = "Viridis",
    colorscale_imag: str = "Viridis",
    titles: list[str] | None = None,
    auto_color_range: bool = True,
    zmax: float | None = None,
    width: int = 1000,
    height: int | None = None,
    horizontal_spacing: float = 0.22,
    vertical_spacing: float | None = None,
    show: bool = True,
):
    """Plot density-matrix heatmaps with Plotly.

    The input can be either:
    1) a single density matrix,
    2) a list of density matrices, or
    3) ``RunResults`` from ``runner.run`` (same high-level interface as
       ``plot_results``).

    Parameters
    ----------
    rho : DensityMatrix | np.ndarray | list[DensityMatrix | np.ndarray] | RunResults
        Input density matrix data or simulation run results.
    basis_labels : list[str] | None, optional
        Axis labels for basis states. If ``None``, labels are auto-generated.
    colorscale_real : str, optional
        Plotly colorscale used for the real-part heatmap.
    colorscale_imag : str, optional
        Plotly colorscale used for the imaginary-part heatmap.
    titles : list[str] | None, optional
        Optional per-matrix titles when multiple density matrices are provided.
    auto_color_range : bool, optional
        If ``True``, color range is scaled from each heatmap's data magnitude.
    zmax : float | None, optional
        Fixed symmetric color max (uses ``[-zmax, zmax]``) when provided.
        Ignored when ``auto_color_range`` is ``True``.
    width : int, optional
        Figure width in pixels.
    height : int | None, optional
        Figure height in pixels. If ``None``, height is set based on row count.
    horizontal_spacing : float, optional
        Horizontal gap between real/imag subplots (normalized figure coords).
    vertical_spacing : float | None, optional
        Vertical gap between rows. If ``None``, an automatic default is used.
    show : bool, optional
        If ``True``, immediately display the Plotly figure.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure containing heatmaps for real and imaginary parts.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    densities: list[DensityMatrix]
    resolved_titles: list[str] | None = titles

    if isinstance(rho, dict) and "ghz" in rho and "w" in rho:
        if titles is not None:
            raise ValueError("Do not pass titles when rho is RunResults.")
        densities = [
            DensityMatrix(rho["ghz"]["pure_kraus"]),
            DensityMatrix(rho["ghz"]["circuit_aer"]),
            DensityMatrix(rho["w"]["pure_kraus"]),
            DensityMatrix(rho["w"]["circuit_aer"]),
        ]
        resolved_titles = [
            "GHZ Pure State + Kraus Noise",
            "GHZ Circuit + Aer Noise Model",
            "W Pure State + Kraus Noise",
            "W Circuit + Aer Noise Model",
        ]
    else:
        items = rho if isinstance(rho, list) else [rho]
        densities = [DensityMatrix(item) for item in items] # type: ignore

    if resolved_titles is not None and len(resolved_titles) != len(densities):
        raise ValueError("titles length must match number of density matrices.")

    subplot_titles: list[str] = []
    for idx in range(len(densities)):
        header = (
            resolved_titles[idx]
            if resolved_titles is not None
            else f"Density Matrix {idx + 1}"
        )
        subplot_titles.append(f"{header} Re(rho)")
        subplot_titles.append(f"{header} Im(rho)")

    fig = make_subplots(
        rows=len(densities),
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=(
            vertical_spacing
            if vertical_spacing is not None
            else (0.12 if len(densities) > 1 else 0.08)
        ),
    )

    for row_idx, density in enumerate(densities, start=1):
        matrix = np.asarray(density.data)
        real_part = np.real(matrix)
        imag_part = np.imag(matrix)
        dim = matrix.shape[0]

        labels = basis_labels
        if labels is None:
            n = density.num_qubits
            if n is not None and 2**n == dim:
                labels = [format(i, f"0{n}b") for i in range(dim)]
            else:
                labels = [str(i) for i in range(dim)]

        fig.add_trace(
            go.Heatmap(
                z=real_part,
                x=labels,
                y=labels,
                colorscale=colorscale_real,
                zmin=0.0,
                zmax=(
                    max(float(np.max(real_part)), 1e-12)
                    if auto_color_range
                    else (zmax if zmax is not None else 1.0)
                ),
                colorbar=dict(title="", len=0.35),
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=imag_part,
                x=labels,
                y=labels,
                colorscale=colorscale_imag,
                zmin=0.0,
                zmax=(
                    max(float(np.max(real_part)), 1e-12)
                    if auto_color_range
                    else (zmax if zmax is not None else 1.0)
                ),
                colorbar=dict(title="", len=0.35),
            ),
            row=row_idx,
            col=2,
        )

        fig.update_xaxes(title_text="col", row=row_idx, col=1)
        fig.update_yaxes(title_text="row", autorange="reversed", row=row_idx, col=1)
        fig.update_xaxes(title_text="col", row=row_idx, col=2)
        fig.update_yaxes(
            title_text="row",
            autorange="reversed",
            side="left",
            automargin=True,
            title_standoff=8,
            row=row_idx,
            col=2,
        )

    # Match each colorbar to the vertical size of its subplot row.
    for row_idx in range(1, len(densities) + 1):
        y_axis_index = (row_idx - 1) * 2 + 1
        y_axis_name = "yaxis" if y_axis_index == 1 else f"yaxis{y_axis_index}"
        y_domain = fig.layout[y_axis_name].domain # type: ignore
        bar_len = float(y_domain[1] - y_domain[0])
        bar_y = float((y_domain[0] + y_domain[1]) / 2.0)

        x_axis_index_left = (row_idx - 1) * 2 + 1
        x_axis_index_right = x_axis_index_left + 1
        x_axis_name_left = "xaxis" if x_axis_index_left == 1 else f"xaxis{x_axis_index_left}"
        x_axis_name_right = (
            "xaxis" if x_axis_index_right == 1 else f"xaxis{x_axis_index_right}"
        )
        x_domain_left = fig.layout[x_axis_name_left].domain # type: ignore
        x_domain_right = fig.layout[x_axis_name_right].domain # type: ignore
        # Keep left colorbar near left subplot to avoid overlapping with
        # right subplot y-axis labels in the center gap.
        bar_x_left = float(x_domain_left[1] + 0.004)
        bar_x_right = float(x_domain_right[1] + 0.01)

        re_trace_index = (row_idx - 1) * 2
        im_trace_index = re_trace_index + 1

        fig.data[re_trace_index].update(
            colorbar=dict(
                title="",
                len=bar_len,
                y=bar_y,
                x=bar_x_left,
                xanchor="left",
                thickness=10,
            ),
        )
        fig.data[im_trace_index].update(
            colorbar=dict(
                title="",
                len=bar_len,
                y=bar_y,
                x=bar_x_right,
                xanchor="left",
                thickness=10,
            ),
        )

    fig.update_layout(
        height=height if height is not None else 350 * len(densities),
        width=width,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig


def plot_scatter_from_csv_plotly(
    csv_path: str,
    x_axis: str,
    data: list[str],
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int = 1000,
    height: int = 500,
    show: bool = True,
):
    """Load CSV data and plot selected columns against an x-axis column.

    Column lookup is keyword-based:
    1) exact column name match (case-insensitive),
    2) fallback to first column that contains the keyword (case-insensitive).

    Parameters
    ----------
    csv_path : str
        Path to input CSV file.
    x_axis : str
        Keyword used to find the x-axis column.
    data : list[str]
        List of keywords used to find y-axis columns.
    title : str | None, optional
        Figure title.
    x_label : str | None, optional
        Display label for x-axis. If ``None``, resolved x column name is used.
    y_label : str | None, optional
        Display label for y-axis. If ``None``, ``"value"`` is used.
    width : int, optional
        Figure width in pixels.
    height : int, optional
        Figure height in pixels.
    show : bool, optional
        If ``True``, immediately display the figure.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly scatter figure.

    Raises
    ------
    ValueError
        If x-axis or any data keyword does not match a CSV column.
    """
    import pandas as pd
    import plotly.graph_objects as go

    def resolve_column(columns: list[str], keyword: str) -> str:
        target = keyword.strip().lower()
        exact_matches = [c for c in columns if c.lower() == target]
        if exact_matches:
            return exact_matches[0]

        contains_matches = [c for c in columns if target in c.lower()]
        if contains_matches:
            return contains_matches[0]

        raise ValueError(f"Column keyword '{keyword}' not found in CSV.")

    df = pd.read_csv(csv_path)
    columns = list(df.columns)

    x_col = resolve_column(columns, x_axis)
    y_cols = [resolve_column(columns, key) for key in data]

    fig = go.Figure()
    for y_col in y_cols:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="lines+markers",
                name=y_col,
            )
        )

    fig.update_layout(
        title=title if title is not None else f"{', '.join(y_cols)} vs {x_col}",
        xaxis_title=x_label if x_label is not None else x_col,
        yaxis_title=y_label if y_label is not None else "value",
        width=width,
        height=height,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig


def plot_multi_scatter_from_csv_plotly(
    csv_path: str,
    x_axes: str | list[str],
    data_sets: list[tuple[str, str]],
    x_label: str | None = None,
    y_labels: list[str] | None = None,
    titles: list[str] | None = None,
    figure_title: str | None = None,
    width: int = 1000,
    height: int | None = None,
    height_per_plot: int = 350,
    show: bool = True,
):
    """Plot multiple scatter subplots from CSV by reusing single-scatter helper.

    Each entry in ``data_sets`` is a pair of y-series keywords ``(y1, y2)``.
    Both series are plotted in the same subplot. The next dataset is plotted in
    the next subplot row.

    Parameters
    ----------
    csv_path : str
        Path to input CSV file.
    x_axes : str | list[str]
        X-axis keyword(s). If a single string is provided, it is reused for all
        subplots. If a list is provided, its length must match ``data_sets``.
    data_sets : list[tuple[str, str]]
        List of y-series pairs, for example
        ``[(\"w_concurrence_ab_pure\", \"ghz_concurrence_ab_pure\"), ...]``.
    x_label : str | None, optional
        Optional x-axis display label used for all subplots.
    y_labels : list[str] | None, optional
        Optional y-axis display label per subplot.
    titles : list[str] | None, optional
        Optional subplot title per dataset.
    figure_title : str | None, optional
        Figure-level title.
    width : int, optional
        Figure width in pixels.
    height : int | None, optional
        Total figure height in pixels. If ``None``, computed from
        ``height_per_plot * number_of_subplots``.
    height_per_plot : int, optional
        Height per subplot row in pixels.
    show : bool, optional
        If ``True``, immediately display the figure.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure containing one subplot per dataset.

    Raises
    ------
    ValueError
        If list lengths for per-subplot options do not match ``data_sets``.
    """
    from plotly.subplots import make_subplots

    n_plots = len(data_sets)
    if n_plots == 0:
        raise ValueError("data_sets must not be empty.")

    if isinstance(x_axes, str):
        resolved_x_axes = [x_axes] * n_plots
    else:
        if len(x_axes) != n_plots:
            raise ValueError("x_axes length must match data_sets length.")
        resolved_x_axes = x_axes

    if y_labels is not None and len(y_labels) != n_plots:
        raise ValueError("y_labels length must match data_sets length.")
    if titles is not None and len(titles) != n_plots:
        raise ValueError("titles length must match data_sets length.")

    subplot_titles = (
        titles
        if titles is not None
        else [f"{pair[0]} vs {pair[1]}" for pair in data_sets]
    )
    fig = make_subplots(rows=n_plots, cols=1, subplot_titles=subplot_titles)
    pair_colors = ["#1f77b4", "#ff7f0e"]

    for idx, data_pair in enumerate(data_sets):
        x_axis_key = resolved_x_axes[idx]
        subplot_fig = plot_scatter_from_csv_plotly(
            csv_path=csv_path,
            x_axis=x_axis_key,
            data=[data_pair[0], data_pair[1]],
            title=None,
            x_label=x_label,
            y_label=(y_labels[idx] if y_labels is not None else None),
            width=width,
            height=height_per_plot,
            show=False,
        )

        row = idx + 1
        for trace_idx, trace in enumerate(subplot_fig.data):
            color = pair_colors[trace_idx % len(pair_colors)]
            trace.update(
                line=dict(color=color),
                marker=dict(color=color),
            )
            fig.add_trace(trace, row=row, col=1)

        fig.update_xaxes(
            title_text=(x_label if x_label is not None else x_axis_key),
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text=(y_labels[idx] if y_labels is not None else "value"),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=figure_title if figure_title is not None else "Multi Scatter Plot",
        width=width,
        height=height if height is not None else height_per_plot * n_plots,
        template="plotly_white",
        showlegend=True,
    )

    if show:
        fig.show()

    return fig
