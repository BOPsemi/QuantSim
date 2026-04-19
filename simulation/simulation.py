"""Aer simulation helpers for noisy density-matrix workflows."""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveDensityMatrix
from qiskit_aer.noise import NoiseModel


def run_circuit_simulation(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    shots: int = 5024,
) -> DensityMatrix:
    """Simulate a circuit with Aer and return the final density matrix.

    Parameters
    ----------
    circuit : QuantumCircuit
        Input circuit to simulate.
    noise_model : NoiseModel
        Noise model to apply during simulation.
    shots : int, optional
        Number of shots passed to the simulator run.

    Returns
    -------
    DensityMatrix
        Final simulated density matrix.
    """
    simulator = AerSimulator(noise_model=noise_model)

    sim_circuit = circuit.copy()
    sim_circuit.append(
        SaveDensityMatrix(num_qubits=sim_circuit.num_qubits),
        sim_circuit.qubits,
    )

    transpiled = transpile(sim_circuit, simulator)
    result = simulator.run(transpiled, shots=shots).result()
    return DensityMatrix(result.data(0)["density_matrix"])
