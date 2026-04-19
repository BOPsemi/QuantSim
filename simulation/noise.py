"""Kraus-channel based noise utilities for density matrices."""

from __future__ import annotations

from itertools import product
from typing import Any, cast

import numpy as np
from qiskit.quantum_info import DensityMatrix, Kraus
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)


def apply_kraus_noise_dm(
    rho: DensityMatrix | np.ndarray,
    kraus_ops: list[np.ndarray],
    qargs: list[int] | None = None,
) -> DensityMatrix:
    """Apply a Kraus channel to a density matrix.

    Parameters
    ----------
    rho : DensityMatrix | np.ndarray
        Input density matrix.
    kraus_ops : list[np.ndarray]
        Kraus operators ``[K0, K1, ...]`` describing the channel.
    qargs : list[int] | None, optional
        Target qubit indices. If ``None``, the channel is applied to the full system.

    Returns
    -------
    DensityMatrix
        Output density matrix after channel application.
    """
    density = DensityMatrix(rho)
    channel = Kraus(cast(Any, kraus_ops))
    return density.evolve(channel, qargs=qargs)


def build_noise_model(
    mode: str,
    p1: float = 0.01,
    p2: float = 0.02,
    t1: float = 100e3,
    t2: float = 80e3,
    t_1q: float = 50,
    t_2q: float = 300,
) -> NoiseModel:
    """Construct a Qiskit Aer noise model based on a mode string.

    Parameters
    ----------
    mode : str
        Noise configuration mode. Supported values are
        ``"cx_only"``, ``"single_only"``, ``"readout_only"``,
        ``"all"``, ``"all_thermal"``, and ``"ideal"``.
    p1 : float, optional
        1-qubit depolarizing probability.
    p2 : float, optional
        2-qubit depolarizing probability.
    t1 : float, optional
        T1 relaxation time for thermal noise.
    t2 : float, optional
        T2 dephasing time for thermal noise.
    t_1q : float, optional
        Duration of 1-qubit gates for thermal noise.
    t_2q : float, optional
        Duration of 2-qubit gates for thermal noise.

    Returns
    -------
    NoiseModel
        Configured Aer noise model.

    Raises
    ------
    ValueError
        If ``mode`` is not supported.
    """
    noise_model = NoiseModel()

    if mode == "cx_only":
        error_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
    elif mode == "single_only":
        error_1q = depolarizing_error(p1, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ["h", "x", "ry", "rz"])
    elif mode == "readout_only":
        ro = ReadoutError([[0.98, 0.02], [0.03, 0.97]])
        noise_model.add_all_qubit_readout_error(ro)
    elif mode == "all":
        error_1q = depolarizing_error(p1, 1)
        error_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ["h", "x", "ry", "rz"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cry"])
    elif mode == "all_thermal":
        error_1q = thermal_relaxation_error(t1, t2, t_1q)
        error_2q = thermal_relaxation_error(t1, t2, t_2q).tensor(
            thermal_relaxation_error(t1, t2, t_2q)
        )
        noise_model.add_all_qubit_quantum_error(error_1q, ["h", "x", "ry", "rz"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cry"])
    elif mode == "ideal":
        return noise_model
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return noise_model


def depolarizing_kraus_1q(p: float) -> list[np.ndarray]:
    """Build Kraus operators for a 1-qubit depolarizing channel.

    Parameters
    ----------
    p : float
        Depolarizing probability in ``[0, 1]``.

    Returns
    -------
    list[np.ndarray]
        List of four ``2x2`` Kraus operators.

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]``.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")

    i = np.array([[1, 0], [0, 1]], dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)

    return [
        np.sqrt(1 - p) * i,
        np.sqrt(p / 3) * x,
        np.sqrt(p / 3) * y,
        np.sqrt(p / 3) * z,
    ]


def depolarizing_kraus_2q(p: float) -> list[np.ndarray]:
    """Build Kraus operators for a 2-qubit depolarizing channel.

    This models ``E(rho) = (1 - p) * rho + p * I / 4`` using 16 tensor-product
    Pauli operators.

    Parameters
    ----------
    p : float
        Depolarizing probability in ``[0, 1]``.

    Returns
    -------
    list[np.ndarray]
        List of sixteen ``4x4`` Kraus operators.

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]``.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1].")

    i = np.array([[1, 0], [0, 1]], dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [i, x, y, z]

    ops = [np.kron(a, b) for a, b in product(paulis, paulis)]
    kraus_ops = [np.sqrt(1 - p) * ops[0]]
    kraus_ops += [np.sqrt(p / 15) * op for op in ops[1:]]
    return kraus_ops


def apply_noise_all_qubits(
    rho: DensityMatrix | np.ndarray,
    kraus_fn,
    param: float,
) -> DensityMatrix:
    """Apply a 1-qubit Kraus channel sequentially to all qubits.

    Parameters
    ----------
    rho : DensityMatrix | np.ndarray
        Input n-qubit density matrix.
    kraus_fn : callable
        Callable with signature ``kraus_fn(param) -> list[np.ndarray]``.
    param : float
        Parameter passed to ``kraus_fn``.

    Returns
    -------
    DensityMatrix
        Density matrix after channel application to every qubit.
    """
    density = DensityMatrix(rho)
    n = density.num_qubits
    if n is None:
        raise ValueError("Unable to determine the number of qubits from rho.")
    ops = kraus_fn(param)

    for q in range(n):
        density = apply_kraus_noise_dm(density, ops, qargs=[q])

    return density


def apply_noise_pairs(
    rho: DensityMatrix | np.ndarray,
    kraus_fn,
    param: float,
    pairs: list[tuple[int, int]],
) -> DensityMatrix:
    """Apply a 2-qubit Kraus channel to user-specified qubit pairs.

    Parameters
    ----------
    rho : DensityMatrix | np.ndarray
        Input n-qubit density matrix.
    kraus_fn : callable
        Callable with signature ``kraus_fn(param) -> list[np.ndarray]``.
    param : float
        Parameter passed to ``kraus_fn``.
    pairs : list[tuple[int, int]]
        Qubit pairs to which noise is applied, for example ``[(0, 1), (1, 2)]``.

    Returns
    -------
    DensityMatrix
        Density matrix after applying the 2-qubit channel to all pairs.

    Raises
    ------
    ValueError
        If any pair index is out of range or contains identical qubits.
    """
    density = DensityMatrix(rho)
    n = density.num_qubits
    if n is None:
        raise ValueError("Unable to determine the number of qubits from rho.")
    ops = kraus_fn(param)

    for a, b in pairs:
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Invalid pair ({a}, {b}) for {n} qubits.")
        if a == b:
            raise ValueError(f"Pair ({a}, {b}) must contain two different qubits.")
        density = apply_kraus_noise_dm(density, ops, qargs=[a, b])

    return density
