"""Quantum state preparation circuits, statevectors, and Bloch coordinates.

Supports standard computational states (|0>, |1>), superposition states (|+>, |->),
and reproducible Haar-random pure single-qubit states.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector
from src.noise import validate_qubit_index

SUPPORTED_STATES = ["|0>", "|1>", "|+>", "|->", "random"]

# Canonical statevector definitions for standard single-qubit states
CANONICAL_STATEVECTORS: Dict[str, np.ndarray] = {
    "|0>": np.array([1.0, 0.0], dtype=complex),
    "|1>": np.array([0.0, 1.0], dtype=complex),
    "|+>": np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=complex),
    "|->": np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)], dtype=complex),
}

# Analytical Bloch coordinates (x, y, z) for standard states
CANONICAL_BLOCH_COORDS: Dict[str, Tuple[float, float, float]] = {
    "|0>": (0.0, 0.0, 1.0),
    "|1>": (0.0, 0.0, -1.0),
    "|+>": (1.0, 0.0, 0.0),
    "|->": (-1.0, 0.0, 0.0),
}


def normalize_state_label(state_name: str) -> str:
    """Standardize state name string representation."""
    cleaned = state_name.strip()
    mapping = {
        "0": "|0>",
        "|0>": "|0>",
        "|0⟩": "|0>",
        "1": "|1>",
        "|1>": "|1>",
        "|1⟩": "|1>",
        "+": "|+>",
        "|+>": "|+>",
        "|+⟩": "|+>",
        "-": "|->",
        "|->": "|->",
        "|−⟩": "|->",
        "random": "random",
        "haar": "random",
        "random_pure": "random",
    }
    key = cleaned.lower() if cleaned.lower() in ("random", "haar", "random_pure") else cleaned
    if key not in mapping:
        raise ValueError(
            f"Unsupported state '{state_name}'. Expected one of {SUPPORTED_STATES}."
        )
    return mapping[key]


def generate_random_pure_state(seed: Optional[int] = None) -> Statevector:
    """Generate a reproducible Haar-random single-qubit pure statevector.

    Args:
        seed: Optional integer seed for reproducibility.

    Returns:
        Qiskit Statevector instance with norm 1.0.
    """
    sv = random_statevector(2, seed=seed)
    # Explicit numerical validation of normalization
    norm = np.linalg.norm(sv.data)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(f"Generated state is not normalized: norm = {norm}")
    return sv


def get_statevector(state_name: str, seed: Optional[int] = None) -> Statevector:
    """Return the ideal pure statevector for a named single-qubit state.

    Args:
        state_name: State label ('|0>', '|1>', '|+>', '|->', or 'random').
        seed: Optional seed for random state generation.

    Returns:
        Qiskit Statevector object.
    """
    canonical_name = normalize_state_label(state_name)
    if canonical_name == "random":
        return generate_random_pure_state(seed=seed)
    return Statevector(CANONICAL_STATEVECTORS[canonical_name])


def compute_bloch_coordinates(sv: Statevector) -> Tuple[float, float, float]:
    """Compute Bloch sphere Cartesian coordinates (x, y, z) for an arbitrary pure statevector.

    x = <psi| X |psi> = 2 * Re(alpha* * beta)
    y = <psi| Y |psi> = 2 * Im(alpha* * beta)
    z = <psi| Z |psi> = |alpha|^2 - |beta|^2

    Args:
        sv: Qiskit Statevector for a single qubit (dim=2).

    Returns:
        Tuple of (x, y, z) floats.
    """
    if sv.dim != 2:
        raise ValueError(f"Bloch sphere coordinates only defined for 1 qubit (dim=2), got dim={sv.dim}")
    data = sv.data
    alpha, beta = data[0], data[1]

    x = float(2.0 * np.real(np.conj(alpha) * beta))
    y = float(2.0 * np.imag(np.conj(alpha) * beta))
    z = float(np.abs(alpha) ** 2 - np.abs(beta) ** 2)

    # Clamp small floating point inaccuracies to [-1.0, 1.0]
    x = max(-1.0, min(1.0, x))
    y = max(-1.0, min(1.0, y))
    z = max(-1.0, min(1.0, z))
    return (x, y, z)


def get_bloch_coordinates(state_name: str, seed: Optional[int] = None) -> Tuple[float, float, float]:
    """Get the Bloch sphere coordinates for a named state.

    Args:
        state_name: State label ('|0>', '|1>', '|+>', '|->', or 'random').
        seed: Optional seed if state is random.

    Returns:
        Tuple of (x, y, z) coordinates.
    """
    canonical_name = normalize_state_label(state_name)
    if canonical_name in CANONICAL_BLOCH_COORDS:
        return CANONICAL_BLOCH_COORDS[canonical_name]
    sv = get_statevector(canonical_name, seed=seed)
    return compute_bloch_coordinates(sv)


def prepare_state(
    state_name: str,
    qubit: int = 0,
    num_qubits: int = 1,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """Generate a quantum circuit that prepares the requested pure test state.

    Uses standard unitary gates where possible:
      - |0>: No-op (qubit already initialized in |0>)
      - |1>: Pauli X gate
      - |+>: Hadamard H gate
      - |->: Pauli X followed by Hadamard H gate
      - random: Qiskit prepare_state with Haar-random statevector

    Args:
        state_name: Name of the state ('|0>', '|1>', '|+>', '|->', 'random').
        qubit: Index of the target qubit (default: 0).
        num_qubits: Total number of qubits in the circuit (default: 1).
        seed: Optional RNG seed for random state generation.

    Returns:
        QuantumCircuit of size num_qubits preparing the target state on `qubit`.
    """
    validate_qubit_index(qubit, num_qubits)
    canonical = normalize_state_label(state_name)
    qc = QuantumCircuit(num_qubits)

    if canonical == "|0>":
        pass  # Qubit is natively initialized in |0>
    elif canonical == "|1>":
        qc.x(qubit)
    elif canonical == "|+>":
        qc.h(qubit)
    elif canonical == "|->":
        qc.x(qubit)
        qc.h(qubit)
    elif canonical == "random":
        sv = generate_random_pure_state(seed=seed)
        qc.prepare_state(sv, qubit)

    return qc
