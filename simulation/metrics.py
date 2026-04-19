"""Quantum-state metrics for density-matrix simulations."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import (
    DensityMatrix,
    concurrence as qiskit_concurrence,
    entropy,
    partial_trace,
    state_fidelity as qiskit_state_fidelity,
)


def von_neumann_entropy(rho: DensityMatrix | np.ndarray, base: int = 2) -> float:
    """Compute the von Neumann entropy of a quantum state.

    Parameters
    ----------
    rho : DensityMatrix | np.ndarray
        Input density matrix.
    base : int, optional
        Logarithm base used in entropy calculation.

    Returns
    -------
    float
        Von Neumann entropy of ``rho``.
    """
    density = DensityMatrix(rho)
    return float(entropy(density, base=base))


def von_neumann_entoropy(rho: DensityMatrix | np.ndarray, base: int = 2) -> float:
    """Backward-compatible alias for ``von_neumann_entropy``.

    Parameters
    ----------
    rho : DensityMatrix | np.ndarray
        Input density matrix.
    base : int, optional
        Logarithm base used in entropy calculation.

    Returns
    -------
    float
        Von Neumann entropy of ``rho``.
    """
    return von_neumann_entropy(rho, base=base)


def concurrence(rho: DensityMatrix | np.ndarray) -> float:
    """Compute concurrence for a 2-qubit state.

    Parameters
    ----------
    rho : DensityMatrix | np.ndarray
        Two-qubit density matrix.

    Returns
    -------
    float
        Concurrence value in ``[0, 1]`` for valid 2-qubit states.

    Raises
    ------
    ValueError
        If the state is not a 2-qubit state.
    """
    density = DensityMatrix(rho)
    if density.num_qubits != 2:
        raise ValueError("concurrence is defined here for 2-qubit states only.")
    return float(qiskit_concurrence(density))


def fidelity(
    rho1: DensityMatrix | np.ndarray,
    rho2: DensityMatrix | np.ndarray,
    validate: bool = True,
) -> float:
    """Compute state fidelity between two quantum states.

    Parameters
    ----------
    rho1 : DensityMatrix | np.ndarray
        First state.
    rho2 : DensityMatrix | np.ndarray
        Second state.
    validate : bool, optional
        If ``True``, Qiskit validates that inputs are physical states.

    Returns
    -------
    float
        State fidelity between ``rho1`` and ``rho2``.
    """
    density1 = DensityMatrix(rho1)
    density2 = DensityMatrix(rho2)
    return float(qiskit_state_fidelity(density1, density2, validate=validate))


def monogamy(rho: DensityMatrix | np.ndarray, focus_qubit: int = 0) -> float:
    """Compute a 3-qubit monogamy score around one focus qubit.

    The score is:
    ``C^2_{A|BC} - C^2_{AB} - C^2_{AC}``
    where ``A`` is ``focus_qubit``.

    Parameters
    ----------
    rho : DensityMatrix | np.ndarray
        Input 3-qubit density matrix.
    focus_qubit : int, optional
        Qubit index used as subsystem ``A`` in the CKW relation.

    Returns
    -------
    float
        Monogamy score with respect to ``focus_qubit``.
        For mixed states this uses ``2 * (1 - Tr(rho_A^2))`` for the
        ``A|BC`` contribution.

    Raises
    ------
    ValueError
        If input is not a 3-qubit state, or focus index is invalid.
    """
    density = DensityMatrix(rho)
    if density.num_qubits != 3:
        raise ValueError("monogamy is implemented for 3-qubit states only.")
    if focus_qubit not in (0, 1, 2):
        raise ValueError("focus_qubit must be 0, 1, or 2.")

    other_qubits = [q for q in (0, 1, 2) if q != focus_qubit]

    # For pure states this equals C^2_{A|BC}; for mixed states this is a
    # linear-entropy based extension used as a practical indicator.
    rho_a = partial_trace(density, other_qubits)
    purity_a = float(np.real(np.trace(rho_a.data @ rho_a.data)))
    c_a_rest_sq = max(0.0, 2.0 * (1.0 - purity_a))

    trace_out_for_ab = [other_qubits[1]]
    trace_out_for_ac = [other_qubits[0]]
    rho_ab = partial_trace(density, trace_out_for_ab)
    rho_ac = partial_trace(density, trace_out_for_ac)

    c_ab_sq = float(qiskit_concurrence(rho_ab) ** 2)
    c_ac_sq = float(qiskit_concurrence(rho_ac) ** 2)

    score = c_a_rest_sq - c_ab_sq - c_ac_sq
    if score < 0.0 and abs(score) < 1e-10:
        return 0.0
    return float(score)
