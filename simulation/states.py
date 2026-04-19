"""State-preparation and state-conversion helpers for density-matrix simulations."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector


def ghz_state_circuit(bits: int) -> QuantumCircuit:
    """Build a circuit that prepares an n-qubit GHZ state.

    Parameters
    ----------
    bits : int
        Number of qubits in the circuit.

    Returns
    -------
    QuantumCircuit
        Circuit that prepares ``(|00...0> + |11...1>) / sqrt(2)``.

    Raises
    ------
    ValueError
        If ``bits`` is less than 1.
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")

    circuit = QuantumCircuit(bits)
    circuit.h(0)
    for i in range(1, bits):
        circuit.cx(0, i)
    return circuit


def w_state_circuit(bits: int) -> QuantumCircuit:
    """Build a circuit that prepares an n-qubit W state.

    Parameters
    ----------
    bits : int
        Number of qubits in the circuit.

    Returns
    -------
    QuantumCircuit
        Circuit that prepares an n-qubit W state.

    Raises
    ------
    ValueError
        If ``bits`` is less than 2.
    """
    if bits < 2:
        raise ValueError("bits must be >= 2")

    circuit = QuantumCircuit(bits)

    for i in range(bits - 1):
        theta = 2 * np.arccos(np.sqrt(1 / (bits - i)))
        if i == 0:
            circuit.ry(theta, 0)
        else:
            circuit.cry(theta, i - 1, i)

    for i in range(bits - 1, 0, -1):
        circuit.cx(i - 1, i)

    circuit.x(0)
    return circuit


def circuit_to_density_matrix(circuit: QuantumCircuit) -> DensityMatrix:
    """Convert a quantum circuit instruction into a density matrix.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit that defines the state-preparation instruction.

    Returns
    -------
    DensityMatrix
        Density matrix obtained by simulating the circuit as an instruction.
    """
    return DensityMatrix.from_instruction(circuit)


def ghz_state_pure_3q() -> DensityMatrix:
    """Return the ideal 3-qubit GHZ state as a density matrix.

    Parameters
    ----------
    None

    Returns
    -------
    DensityMatrix
        Density matrix representation of ``(|000> + |111>) / sqrt(2)``.
    """
    ghz_vec = np.array(
        [1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2)],
        dtype=complex,
    )
    return DensityMatrix(Statevector(ghz_vec))


def w_state_pure_3q() -> DensityMatrix:
    """Return the ideal 3-qubit W state as a density matrix.

    Parameters
    ----------
    None

    Returns
    -------
    DensityMatrix
        Density matrix representation of ``(|001> + |010> + |100>) / sqrt(3)``.
    """
    w_vec = np.array(
        [0, 1 / np.sqrt(3), 1 / np.sqrt(3), 0, 1 / np.sqrt(3), 0, 0, 0],
        dtype=complex,
    )
    return DensityMatrix(Statevector(w_vec))
