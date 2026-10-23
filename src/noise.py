"""Noise models and error channel builders for Q-GUARD bit-flip experiments.

Provides deterministic Pauli X error injection, stochastic Pauli error models,
and execution helpers compatible with Qiskit 2.x and Qiskit Aer.
"""

from typing import Dict, Optional, Tuple, Union
import math
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


def validate_probability(p: float) -> float:
    """Validate that a probability value lies within [0.0, 1.0].

    Args:
        p: Float probability value.

    Returns:
        The validated probability as a float.

    Raises:
        TypeError: If p is not numeric.
        ValueError: If p is outside [0.0, 1.0] or is NaN/inf.
    """
    if not isinstance(p, (int, float, np.floating)):
        raise TypeError(f"Probability must be a float, got {type(p).__name__}")
    if math.isnan(p) or math.isinf(p):
        raise ValueError(f"Probability must be a finite number, got {p}")
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Bit-flip probability p must be in [0.0, 1.0], got {p}")
    return float(p)


def validate_qubit_index(qubit: int, num_qubits: int) -> int:
    """Validate that a qubit index is valid for a given circuit register size.

    Args:
        qubit: Integer index of the target qubit.
        num_qubits: Total number of qubits in the circuit.

    Returns:
        The validated qubit index.

    Raises:
        TypeError: If qubit is not an integer.
        ValueError: If qubit index is out of bounds [0, num_qubits - 1].
    """
    if not isinstance(qubit, (int, np.integer)):
        raise TypeError(f"Qubit index must be an integer, got {type(qubit).__name__}")
    if num_qubits <= 0:
        raise ValueError(f"num_qubits must be positive, got {num_qubits}")
    if qubit < 0 or qubit >= num_qubits:
        raise ValueError(
            f"Qubit index {qubit} out of bounds for circuit with {num_qubits} qubits (0 to {num_qubits - 1})"
        )
    return int(qubit)


def apply_deterministic_bitflip(circuit: QuantumCircuit, qubit: int = 0) -> QuantumCircuit:
    """Apply a deterministic bit-flip (Pauli X gate) to a specified qubit.

    Args:
        circuit: Target QuantumCircuit to modify.
        qubit: Zero-based index of the target qubit.

    Returns:
        The modified QuantumCircuit with the X gate appended.
    """
    validate_qubit_index(qubit, circuit.num_qubits)
    circuit.x(qubit)
    return circuit


def create_bitflip_error(p: float) -> pauli_error:
    """Construct a Qiskit Aer Pauli quantum error representing a bit-flip channel.

    The single-qubit bit-flip channel applies:
        rho -> (1 - p) rho + p X rho X

    Args:
        p: Physical bit-flip probability in [0.0, 1.0].

    Returns:
        Qiskit Aer pauli_error instance.
    """
    p_val = validate_probability(p)
    return pauli_error([("X", p_val), ("I", 1.0 - p_val)])


def apply_probabilistic_bitflip(
    circuit: QuantumCircuit, qubit: int = 0, p: float = 0.0
) -> QuantumCircuit:
    """Insert a probabilistic bit-flip channel directly into the circuit at the target qubit.

    Appending the Pauli error instruction directly to the circuit guarantees that the
    error channel is executed by AerSimulator without risking transpiler elimination of
    identity gates.

    Args:
        circuit: Target QuantumCircuit.
        qubit: Zero-based index of the target qubit.
        p: Physical bit-flip probability in [0.0, 1.0].

    Returns:
        The modified QuantumCircuit.
    """
    validate_qubit_index(qubit, circuit.num_qubits)
    p_val = validate_probability(p)
    if p_val == 0.0:
        # Perfectly noiseless channel: no error operation needed
        return circuit
    elif p_val == 1.0:
        # Full deterministic inversion
        circuit.x(qubit)
        return circuit
    else:
        error = create_bitflip_error(p_val)
        circuit.append(error.to_instruction(), [qubit])
        return circuit


def build_bitflip_noise_model(p: float, target_gate: str = "id") -> NoiseModel:
    """Build a Qiskit Aer NoiseModel with bit-flip Pauli error attached to a specific gate.

    Args:
        p: Bit-flip probability in [0.0, 1.0].
        target_gate: Gate name to attach the error to (default: 'id').

    Returns:
        Configured NoiseModel instance.
    """
    p_val = validate_probability(p)
    noise_model = NoiseModel()
    if p_val > 0.0:
        error = create_bitflip_error(p_val)
        noise_model.add_all_qubit_quantum_error(error, [target_gate])
    return noise_model


def run_circuit_simulation(
    circuit: QuantumCircuit,
    shots: int = 1024,
    seed: Optional[int] = None,
    noise_model: Optional[NoiseModel] = None,
) -> Dict[str, int]:
    """Execute a quantum circuit on AerSimulator with reproducible seeding and optional noise.

    Uses optimization_level=0 when a noise_model is provided so attached identity gates
    or custom gate instructions are not removed by the transpiler.

    Args:
        circuit: The QuantumCircuit to execute (must include measurements).
        shots: Number of measurement repetitions (must be >= 1).
        seed: Optional integer seed for simulator reproducibility.
        noise_model: Optional Qiskit Aer NoiseModel.

    Returns:
        Dictionary mapping bitstring outcomes to count frequencies, e.g. {'0': 820, '1': 180}.

    Raises:
        ValueError: If shots < 1.
    """
    if shots < 1:
        raise ValueError(f"Shots must be at least 1, got {shots}")

    backend = AerSimulator()
    opt_level = 0 if noise_model is not None else 1
    transpiled = transpile(circuit, backend, optimization_level=opt_level)
    job = backend.run(
        transpiled,
        shots=shots,
        seed_simulator=seed,
        noise_model=noise_model,
    )
    result = job.result()
    return result.get_counts()
