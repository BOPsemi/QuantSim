"""Reusable simulation runner for GHZ/W density-matrix demos."""

from __future__ import annotations

from typing import Iterable, TypedDict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace

from metrics import concurrence, fidelity, monogamy, von_neumann_entropy

from noise import (
    apply_noise_all_qubits,
    apply_noise_pairs,
    build_noise_model,
    depolarizing_kraus_1q,
    depolarizing_kraus_2q,
)
from simulation import run_circuit_simulation
from states import (
    circuit_to_density_matrix,
    ghz_state_circuit,
    w_state_circuit,
)


class StateRunResult(TypedDict):
    """Per-state outputs from the simulation runner."""

    pure_kraus: DensityMatrix
    circuit_aer: DensityMatrix
    metrics: dict[str, float]


RunResults = dict[str, StateRunResult]
LoopRunResults = dict[float, RunResults]


def build_state_circuit(state: str, bits: int) -> QuantumCircuit:
    """Build a GHZ or W state-preparation circuit.

    Parameters
    ----------
    state : str
        Target state type. Supported values are ``"ghz"`` and ``"w"``.
    bits : int
        Number of qubits for the state-preparation circuit.

    Returns
    -------
    QuantumCircuit
        Circuit that prepares the requested state.

    Raises
    ------
    ValueError
        If ``state`` is not one of the supported values.
    """
    normalized = state.strip().lower()
    if normalized == "ghz":
        return ghz_state_circuit(bits)
    if normalized == "w":
        return w_state_circuit(bits)
    raise ValueError("state must be 'ghz' or 'w'.")


def build_pure_state(state: str, bits: int) -> DensityMatrix:
    """Build a pure-state density matrix for GHZ or W.

    Parameters
    ----------
    state : str
        Target state type. Supported values are ``"ghz"`` and ``"w"``.
    bits : int
        Number of qubits used to construct the pure state.

    Returns
    -------
    DensityMatrix
        Pure density matrix of the requested state.
    """
    circuit = build_state_circuit(state, bits)
    return circuit_to_density_matrix(circuit)


def _concurrence_ab(rho: DensityMatrix) -> float:
    """Compute concurrence on qubits (0, 1) when possible, else NaN."""
    n = rho.num_qubits
    if n is None or n < 2:
        return float(np.nan)

    if n == 2:
        return concurrence(rho)

    trace_out = list(range(2, n))
    reduced_ab = partial_trace(rho, trace_out)
    return concurrence(reduced_ab)


def _compute_metrics(
    reference: DensityMatrix,
    pure_kraus: DensityMatrix,
    circuit_aer: DensityMatrix,
) -> dict[str, float]:
    """Compute requested metrics for one state family."""
    def reduced_entropy_qubit0(rho: DensityMatrix) -> float:
        """Compute S(rho_0) where rho_0 is reduced state of qubit 0."""
        n = rho.num_qubits
        if n is None:
            raise ValueError("Unable to determine number of qubits for entropy.")
        if n == 1:
            return von_neumann_entropy(rho)

        trace_out = list(range(1, n))
        rho_0 = partial_trace(rho, trace_out)
        return von_neumann_entropy(rho_0)

    return {
        "entropy_global_pure": von_neumann_entropy(pure_kraus),
        "entropy_global_circuit": von_neumann_entropy(circuit_aer),
        "entropy_pure": reduced_entropy_qubit0(pure_kraus),
        "entropy_circuit": reduced_entropy_qubit0(circuit_aer),
        "fidelity_reference_pure": fidelity(reference, pure_kraus),
        "fidelity_reference_circuit": fidelity(reference, circuit_aer),
        "concurrence_ab_pure": _concurrence_ab(pure_kraus),
        "concurrence_ab_circuit": _concurrence_ab(circuit_aer),
        "monogamy_pure": monogamy(pure_kraus),
        "monogamy_circuit": monogamy(circuit_aer),
    }


def run(bits: int, p1: float, p2: float, pairs: list[tuple[int, int]]) -> RunResults:
    """Run GHZ and W density-matrix demos and return noisy outputs.

    Parameters
    ----------
    bits : int
        Number of qubits for circuit-based GHZ/W state preparation.
    p1 : float
        1-qubit depolarizing probability.
    p2 : float
        2-qubit depolarizing probability.
    pairs : list[tuple[int, int]]
        Qubit pairs for 2-qubit Kraus noise application.

    Returns
    -------
    RunResults
        Nested dictionary with per-state density matrices:
        ``results[state]["pure_kraus"]`` and ``results[state]["circuit_aer"]``.
        Calculated metrics are available at ``results[state]["metrics"]``.
    """
    noise_model = build_noise_model(mode="all", p1=p1, p2=p2)
    results: RunResults = {}

    for state in ("ghz", "w"):
        # Pure state + Kraus noise
        pure_state = build_pure_state(state, bits)
        pure_state_noisy = apply_noise_all_qubits(pure_state, depolarizing_kraus_1q, p1)
        pure_state_noisy = apply_noise_pairs(pure_state_noisy, depolarizing_kraus_2q, p2, pairs=pairs)

        # Circuit + Aer noise model
        circuit = build_state_circuit(state, bits)
        circuit_noisy = run_circuit_simulation(circuit, noise_model=noise_model, shots=10000)

        results[state] = {
            "pure_kraus": pure_state_noisy,
            "circuit_aer": circuit_noisy,
            "metrics": _compute_metrics(
                reference=pure_state,
                pure_kraus=pure_state_noisy,
                circuit_aer=circuit_noisy,
            ),
        }

    return results


def loop_run(
    bits: int,
    pairs: list[tuple[int, int]],
    sweep_param: str,
    sweep_values: Iterable[float],
    p1: float = 0.01,
    p2: float = 0.10,
) -> LoopRunResults:
    """Run GHZ/W simulations by sweeping one noise parameter.

    This function reuses ``run`` for each sweep point and collects outputs.

    Parameters
    ----------
    bits : int
        Number of qubits for circuit-based GHZ/W state preparation.
    pairs : list[tuple[int, int]]
        Qubit pairs for 2-qubit Kraus noise application.
    sweep_param : str
        Parameter to sweep. Must be ``"p1"`` or ``"p2"``.
    sweep_values : Iterable[float]
        Values used for the selected sweep parameter.
    p1 : float, optional
        Baseline 1-qubit depolarizing probability. Used when sweeping ``p2``.
    p2 : float, optional
        Baseline 2-qubit depolarizing probability. Used when sweeping ``p1``.

    Returns
    -------
    LoopRunResults
        Mapping ``{sweep_value: run_result}``, where each run result includes
        both GHZ and W outputs plus metrics.

    Raises
    ------
    ValueError
        If ``sweep_param`` is not ``"p1"`` or ``"p2"``.
    """
    if sweep_param not in ("p1", "p2"):
        raise ValueError("sweep_param must be 'p1' or 'p2'.")

    all_results: LoopRunResults = {}
    for value in sweep_values:
        sweep_value = float(value)
        if sweep_param == "p1":
            all_results[sweep_value] = run(bits=bits, p1=sweep_value, p2=p2, pairs=pairs)
        else:
            all_results[sweep_value] = run(bits=bits, p1=p1, p2=sweep_value, pairs=pairs)

    return all_results
