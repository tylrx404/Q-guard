"""Three-qubit quantum bit-flip error-correcting repetition code [[3, 1, 1]].

Provides typed functions for:
- Encoding an arbitrary single-qubit state into logical subspace alpha|000> + beta|111>.
- Physical Pauli X error injection across the 3 physical data qubits.
- Coherent / deferred syndrome extraction and conditional CCX recovery.
- Unrolling / decoding back to the recovered logical data qubit.
- Subsystem reduced density matrix partial trace and quantum state fidelity evaluation.
- Exhaustive evaluation of all 8 error masks demonstrating single-flip correction and double-flip miscorrection.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator

from src.noise import (
    apply_deterministic_bitflip,
    apply_probabilistic_bitflip,
    run_circuit_simulation,
    validate_probability,
    validate_qubit_index,
)
from src.states import compute_bloch_coordinates, get_statevector, normalize_state_label, prepare_state

# Fixed canonical syndrome mapping throughout Q-GUARD:
# s01 = b0 ^ b1, s12 = b1 ^ b2
SYNDROME_MAP: Dict[str, Dict[str, Any]] = {
    "00": {"error_qubit": None, "action": "Identity (No correction)", "description": "No bit-flip error detected"},
    "10": {"error_qubit": 0, "action": "Apply X on data qubit 0", "description": "Single bit-flip detected on data qubit 0"},
    "11": {"error_qubit": 1, "action": "Apply X on data qubit 1", "description": "Single bit-flip detected on data qubit 1"},
    "01": {"error_qubit": 2, "action": "Apply X on data qubit 2", "description": "Single bit-flip detected on data qubit 2"},
}


def normalize_syndrome_string(raw_syndrome: str) -> str:
    """Normalize raw syndrome string to standard two-character 's01 s12' representation.

    Args:
        raw_syndrome: Binary string of length 2.

    Returns:
        Standardized syndrome string in {'00', '10', '11', '01'}.
    """
    cleaned = raw_syndrome.strip().replace(" ", "")
    if len(cleaned) != 2 or any(c not in ("0", "1") for c in cleaned):
        raise ValueError(f"Invalid syndrome string '{raw_syndrome}'. Expected 2 binary bits.")
    return cleaned


def encode_repetition_code(
    circuit: QuantumCircuit,
    data_qubit: int = 0,
    physical_qubits: Tuple[int, int, int] = (0, 1, 2),
) -> QuantumCircuit:
    """Encode an arbitrary input state on data_qubit into the 3-qubit repetition code.

    Mapping:
        alpha|0> + beta|1> -> alpha|000> + beta|111>

    Circuit:
        CNOT(data_qubit, physical_qubits[1])
        CNOT(data_qubit, physical_qubits[2])

    Args:
        circuit: QuantumCircuit containing at least the physical qubits.
        data_qubit: Qubit holding the initial state (typically physical_qubits[0]).
        physical_qubits: 3-tuple of qubit indices for the code block.

    Returns:
        Modified QuantumCircuit.
    """
    q0, q1, q2 = physical_qubits
    validate_qubit_index(q0, circuit.num_qubits)
    validate_qubit_index(q1, circuit.num_qubits)
    validate_qubit_index(q2, circuit.num_qubits)

    circuit.cx(data_qubit, q1)
    circuit.cx(data_qubit, q2)
    return circuit


def extract_syndromes_coherent(
    circuit: QuantumCircuit,
    physical_qubits: Tuple[int, int, int] = (0, 1, 2),
    ancilla_qubits: Tuple[int, int] = (3, 4),
) -> QuantumCircuit:
    """Extract parity syndromes into ancilla qubits coherently using CNOT gates.

    Syndrome definitions:
        a0 = q0 ^ q1 (s01 parity check)
        a1 = q1 ^ q2 (s12 parity check)

    Args:
        circuit: QuantumCircuit containing data and ancilla registers.
        physical_qubits: 3-tuple of data qubit indices.
        ancilla_qubits: 2-tuple of ancilla qubit indices (a0, a1).

    Returns:
        Modified QuantumCircuit.
    """
    q0, q1, q2 = physical_qubits
    a0, a1 = ancilla_qubits

    # Parity check 1: s01 = q0 ^ q1
    circuit.cx(q0, a0)
    circuit.cx(q1, a0)

    # Parity check 2: s12 = q1 ^ q2
    circuit.cx(q1, a1)
    circuit.cx(q2, a1)
    return circuit


def apply_coherent_correction(
    circuit: QuantumCircuit,
    physical_qubits: Tuple[int, int, int] = (0, 1, 2),
    ancilla_qubits: Tuple[int, int] = (3, 4),
) -> QuantumCircuit:
    """Apply syndrome-conditioned Pauli X corrections to data qubits using CCX gates.

    Decoding rules:
        - Syndrome 10 (a0=1, a1=0) -> X on q0
        - Syndrome 11 (a0=1, a1=1) -> X on q1
        - Syndrome 01 (a0=0, a1=1) -> X on q2
        - Syndrome 00 (a0=0, a1=0) -> Identity (no correction)

    Ancilla states are preserved after correction so they can still be measured or analyzed.

    Args:
        circuit: QuantumCircuit containing data and ancilla qubits.
        physical_qubits: 3-tuple of data qubit indices.
        ancilla_qubits: 2-tuple of ancilla indices (a0, a1).

    Returns:
        Modified QuantumCircuit.
    """
    q0, q1, q2 = physical_qubits
    a0, a1 = ancilla_qubits

    # 1. Condition for s01=1, s12=0 -> Flip q0
    circuit.x(a1)
    circuit.ccx(a0, a1, q0)
    circuit.x(a1)

    # 2. Condition for s01=1, s12=1 -> Flip q1
    circuit.ccx(a0, a1, q1)

    # 3. Condition for s01=0, s12=1 -> Flip q2
    circuit.x(a0)
    circuit.ccx(a0, a1, q2)
    circuit.x(a0)

    return circuit


def decode_repetition_code(
    circuit: QuantumCircuit,
    data_qubit: int = 0,
    physical_qubits: Tuple[int, int, int] = (0, 1, 2),
) -> QuantumCircuit:
    """Decode the 3-qubit repetition code back to a single logical qubit.

    Reverses the encoding CNOTs:
        CNOT(data_qubit, physical_qubits[2])
        CNOT(data_qubit, physical_qubits[1])

    After decoding, the recovered logical state resides on data_qubit,
    while physical_qubits[1] and [2] return to |00>.

    Args:
        circuit: QuantumCircuit.
        data_qubit: Output qubit holding the recovered state.
        physical_qubits: 3-tuple of data qubit indices.

    Returns:
        Modified QuantumCircuit.
    """
    q0, q1, q2 = physical_qubits
    circuit.cx(data_qubit, q2)
    circuit.cx(data_qubit, q1)
    return circuit


def build_repetition_code_circuit(
    state_name: str = "|0>",
    p: float = 0.0,
    deterministic_mask: Optional[str] = None,
    seed: Optional[int] = None,
    measure_logical: bool = False,
    measure_syndrome: bool = True,
) -> QuantumCircuit:
    """Build a complete 5-qubit repetition code circuit with encoding, noise, syndrome, correction, and decode.

    Qubit Allocation:
        q[0]: Data qubit 0 (and primary input/output logical qubit)
        q[1]: Data qubit 1
        q[2]: Data qubit 2
        q[3]: Ancilla 0 (s01 parity check: q0 ^ q1)
        q[4]: Ancilla 1 (s12 parity check: q1 ^ q2)

    Classical Registers:
        - If measure_syndrome: 'syndrome' (2 bits: c_syn[0] <- a0, c_syn[1] <- a1)
        - If measure_logical: 'logical' (1 bit: c_log[0] <- q0)

    Args:
        state_name: Input state ('|0>', '|1>', '|+>', '|->', 'random').
        p: Physical bit-flip probability for stochastic noise (used if deterministic_mask is None).
        deterministic_mask: Optional 3-char binary string (e.g. '100' for flip on q0).
        seed: RNG seed for random state preparation.
        measure_logical: Whether to append measurement on decoded data qubit q0.
        measure_syndrome: Whether to append measurement on ancillas a0, a1.

    Returns:
        Configured QuantumCircuit.
    """
    qr_data = QuantumRegister(3, name="data")
    qr_anc = QuantumRegister(2, name="ancilla")

    regs = [qr_data, qr_anc]
    cr_syn = ClassicalRegister(2, name="syn") if measure_syndrome else None
    cr_log = ClassicalRegister(1, name="log") if measure_logical else None

    if cr_syn:
        regs.append(cr_syn)
    if cr_log:
        regs.append(cr_log)

    qc = QuantumCircuit(*regs)

    # 1. State Preparation on data[0]
    canonical_state = normalize_state_label(state_name)
    if canonical_state == "|1>":
        qc.x(qr_data[0])
    elif canonical_state == "|+>":
        qc.h(qr_data[0])
    elif canonical_state == "|->":
        qc.x(qr_data[0])
        qc.h(qr_data[0])
    elif canonical_state == "random":
        prep_qc = prepare_state("random", qubit=0, num_qubits=1, seed=seed)
        qc.compose(prep_qc, qubits=[qr_data[0]], inplace=True)

    qc.barrier(label="Encode")

    # 2. Encode into logical 3-qubit block: alpha|000> + beta|111>
    encode_repetition_code(qc, data_qubit=0, physical_qubits=(0, 1, 2))

    qc.barrier(label="Noise Channel")

    # 3. Noise Injection
    if deterministic_mask is not None:
        mask = deterministic_mask.strip()
        if len(mask) != 3 or any(b not in ("0", "1") for b in mask):
            raise ValueError(f"deterministic_mask must be a 3-character binary string, got '{deterministic_mask}'")
        for idx in range(3):
            if mask[idx] == "1":
                qc.x(qr_data[idx])
    else:
        p_val = validate_probability(p)
        for idx in range(3):
            apply_probabilistic_bitflip(qc, qubit=idx, p=p_val)

    qc.barrier(label="Syndrome Extraction")

    # 4. Coherent Syndrome Extraction onto ancilla[0] and ancilla[1]
    extract_syndromes_coherent(qc, physical_qubits=(0, 1, 2), ancilla_qubits=(3, 4))

    qc.barrier(label="Correction")

    # 5. Coherent CCX Recovery
    apply_coherent_correction(qc, physical_qubits=(0, 1, 2), ancilla_qubits=(3, 4))

    qc.barrier(label="Decode")

    # 6. Decode back to single logical qubit
    decode_repetition_code(qc, data_qubit=0, physical_qubits=(0, 1, 2))

    # 7. Optional Measurements
    # Measure ancillas to syndrome register:
    # We assign: cr_syn[0] <- ancilla 0 (s01), cr_syn[1] <- ancilla 1 (s12)
    if measure_syndrome and cr_syn is not None:
        qc.measure(qr_anc[0], cr_syn[0])
        qc.measure(qr_anc[1], cr_syn[1])

    if measure_logical and cr_log is not None:
        qc.measure(qr_data[0], cr_log[0])

    return qc


def evaluate_repetition_code_statevector(
    state_name: str = "|0>",
    deterministic_mask: Optional[str] = None,
    p: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate pure statevector fidelity of the repetition code without measurement collapse.

    Runs the unmeasured circuit to compute the exact 5-qubit pure Statevector,
    traces out qubits 1, 2, 3, 4 to obtain the single-qubit density matrix of the recovered
    data qubit q0, and calculates the exact quantum state fidelity with the ideal target state.

    Args:
        state_name: Ideal input state label ('|0>', '|1>', '|+>', '|->', 'random').
        deterministic_mask: Optional 3-bit binary error mask ('000' to '111').
        p: Physical error probability (if mask is None).
        seed: RNG seed for random state generation.

    Returns:
        Dictionary with state fidelity, reduced density matrix, and ideal state.
    """
    qc = build_repetition_code_circuit(
        state_name=state_name,
        p=p,
        deterministic_mask=deterministic_mask,
        seed=seed,
        measure_logical=False,
        measure_syndrome=False,
    )

    # Compute 5-qubit density matrix (handles both pure unitary circuits and stochastic Kraus noise channels)
    dm_full = DensityMatrix.from_instruction(qc)

    # Trace out qubits 1, 2 (data 1, 2) and 3, 4 (ancilla 0, 1) to isolate recovered qubit 0
    reduced_q0 = partial_trace(dm_full, [1, 2, 3, 4])

    # Ideal target pure state
    ideal_sv = get_statevector(state_name, seed=seed)
    ideal_dm = DensityMatrix(ideal_sv)

    fidelity = float(state_fidelity(reduced_q0, ideal_dm))

    # Also extract syndrome state from ancillas
    reduced_anc = partial_trace(dm_full, [0, 1, 2])
    anc_probs = reduced_anc.probabilities_dict()
    # In Qiskit, key is 'anc1 anc0' -> s12 s01
    syndrome_key = max(anc_probs, key=anc_probs.get) if anc_probs else "00"
    if len(syndrome_key) == 2:
        s01, s12 = syndrome_key[1], syndrome_key[0]
        syndrome_normalized = f"{s01}{s12}"
    else:
        syndrome_normalized = "00"

    return {
        "state_name": state_name,
        "deterministic_mask": deterministic_mask,
        "fidelity": fidelity,
        "reduced_density_matrix": reduced_q0,
        "ideal_density_matrix": ideal_dm,
        "syndrome": syndrome_normalized,
        "syndrome_info": SYNDROME_MAP.get(syndrome_normalized, SYNDROME_MAP["00"]),
    }


def analyze_eight_error_masks(
    input_state: str = "|0>",
) -> List[Dict[str, Any]]:
    """Exhaustively evaluate all 8 physical bit-flip combinations on the 3 data qubits.

    Demonstrates:
      - Weight 0 ('000'): Syndrome '00', 100% fidelity.
      - Weight 1 ('100', '010', '001'): Unique non-zero syndromes, 100% fidelity after correction.
      - Weight 2 ('110', '101', '011'): Miscorrection! Syndrome matches the remaining un-flipped qubit,
        causing the decoder to apply a third X gate, resulting in a logical X_L flip and 0.0 fidelity.
      - Weight 3 ('111'): Syndrome '00' (undetected logical flip), 0.0 fidelity.

    Args:
        input_state: Test state (|0>, |1>, |+>, |->).

    Returns:
        List of 8 dictionaries detailing mask, syndrome, diagnosed error, action,
        logical fidelity, and outcome explanation.
    """
    masks = ["000", "100", "010", "001", "110", "101", "011", "111"]
    records = []

    for mask in masks:
        eval_res = evaluate_repetition_code_statevector(
            state_name=input_state,
            deterministic_mask=mask,
        )
        syn = eval_res["syndrome"]
        syn_info = eval_res["syndrome_info"]
        fid = eval_res["fidelity"]
        weight = mask.count("1")

        if weight == 0:
            status = "No Error (Preserved)"
            outcome = "Ideal Logical State"
        elif weight == 1:
            status = "Correctable Single Error"
            outcome = "Successfully Corrected"
        elif weight == 2:
            status = "Uncorrectable Double Error (Miscorrected)"
            outcome = "Logical Bit-Flip Inversion"
        else:
            status = "Uncorrectable Triple Error (Undetected)"
            outcome = "Logical Bit-Flip Inversion"

        records.append(
            {
                "mask": mask,
                "weight": weight,
                "syndrome": syn,
                "diagnosed_error": syn_info["description"],
                "decoder_action": syn_info["action"],
                "logical_fidelity": fid,
                "classification": status,
                "outcome": outcome,
            }
        )

    return records


def run_repetition_code_simulation(
    state_name: str = "|0>",
    p: float = 0.0,
    deterministic_mask: Optional[str] = None,
    shots: int = 1024,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Execute repetition code circuit on AerSimulator with measurements on syndrome and logical output.

    Args:
        state_name: Target input state.
        p: Physical bit-flip probability.
        deterministic_mask: Optional 3-character error mask.
        shots: Number of measurement repetitions.
        seed: RNG seed for reproducible Aer simulation.

    Returns:
        Dictionary with syndrome counts, logical measurement counts, error rates, and circuit diagram.
    """
    qc = build_repetition_code_circuit(
        state_name=state_name,
        p=p,
        deterministic_mask=deterministic_mask,
        seed=seed,
        measure_logical=True,
        measure_syndrome=True,
    )

    counts = run_circuit_simulation(qc, shots=shots, seed=seed)

    # Parse counts:
    # Qiskit measurement string format with multiple classical registers:
    # "c_log c_syn" -> e.g. "0 00" or concatenated "0 10"
    syndrome_counts: Dict[str, int] = {"00": 0, "10": 0, "11": 0, "01": 0}
    logical_counts: Dict[str, int] = {"0": 0, "1": 0}

    for bitstr, freq in counts.items():
        parts = bitstr.split()
        if len(parts) == 2:
            log_bit, syn_raw = parts[0], parts[1]
        else:
            # Concatenated: 1 bit logical, 2 bits syndrome
            log_bit = bitstr[0]
            syn_raw = bitstr[1:]

        # Map syndrome bits:
        # cr_syn[0] is s01, cr_syn[1] is s12
        # In Qiskit, syn_raw is "cr_syn[1] cr_syn[0]"
        if len(syn_raw) == 2:
            s01 = syn_raw[1]
            s12 = syn_raw[0]
            syn_norm = f"{s01}{s12}"
        else:
            syn_norm = "00"

        syndrome_counts[syn_norm] = syndrome_counts.get(syn_norm, 0) + freq
        logical_counts[log_bit] = logical_counts.get(log_bit, 0) + freq

    # Calculate analytical theoretical formula
    p_phys = 1.0 if deterministic_mask is not None else p
    analytical_logical_error = 3.0 * (p_phys ** 2) - 2.0 * (p_phys ** 3)

    # Statevector pure fidelity evaluation
    sv_eval = evaluate_repetition_code_statevector(
        state_name=state_name,
        deterministic_mask=deterministic_mask,
        p=p,
        seed=seed,
    )

    # Empirical logical error rate:
    # If state_name is |0>, errors are measured '1' counts
    # If state_name is |1>, errors are measured '0' counts
    canonical = normalize_state_label(state_name)
    if canonical == "|0>":
        logical_errors = logical_counts.get("1", 0)
    elif canonical == "|1>":
        logical_errors = logical_counts.get("0", 0)
    else:
        logical_errors = logical_counts.get("1", 0)

    empirical_logical_error = logical_errors / shots if shots > 0 else 0.0

    return {
        "state_name": state_name,
        "physical_p": p_phys,
        "deterministic_mask": deterministic_mask,
        "shots": shots,
        "seed": seed,
        "circuit": qc,
        "raw_counts": counts,
        "syndrome_counts": syndrome_counts,
        "logical_counts": logical_counts,
        "logical_errors": logical_errors,
        "empirical_logical_error": empirical_logical_error,
        "analytical_logical_error": analytical_logical_error,
        "state_fidelity": sv_eval["fidelity"],
        "dominant_syndrome": max(syndrome_counts, key=syndrome_counts.get),
    }


def compute_repetition_code_theory_curve(
    p_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Calculate the analytical logical failure probability for the 3-qubit repetition code.

    Formula:
        P_logical(p) = 3*p^2 * (1-p) + p^3 = 3*p^2 - 2*p^3

    Breakeven crossover occurs at p = 0.5:
        P_logical(0.5) = 3*(0.25) - 2*(0.125) = 0.75 - 0.25 = 0.5

    For p < 0.5, P_logical(p) < p (Error suppression active).
    For p > 0.5, P_logical(p) > p (Error amplification).

    Args:
        p_grid: Sequence of physical error probabilities in [0.0, 1.0].

    Returns:
        Dictionary containing p_values, logical_error_values, and breakeven point.
    """
    if p_grid is None:
        p_grid = np.linspace(0.0, 1.0, 101)

    p_vals = [float(p) for p in p_grid]
    p_log = [float(3.0 * (p ** 2) - 2.0 * (p ** 3)) for p in p_vals]

    return {
        "p_physical": p_vals,
        "p_logical": p_log,
        "breakeven_threshold": 0.5,
    }


def evaluate_haar_random_benchmark(
    num_states: int = 200,
    p: float = 0.10,
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Evaluate unprotected vs [[3, 1, 1]] protected logical fidelity on an ensemble of Haar-random pure states.

    For each Haar-random pure state |ψ⟩:
      - Prepares |ψ⟩ on qubit 0.
      - Calculates unprotected single-qubit channel fidelity under Pauli X noise p:
            ρ_unprot = (1 - p)|ψ⟩⟨ψ| + p X|ψ⟩⟨ψ|X
            F_unprot = ⟨ψ|ρ_unprot|ψ⟩
      - Calculates [[3, 1, 1]] repetition code protected logical fidelity after encoding,
        independent physical noise p on each data qubit, syndrome extraction, coherent CCX correction,
        and decoding back to data qubit 0 (via partial trace of DensityMatrix).
      - Extracts Bloch sphere coordinates (x, y, z).

    Args:
        num_states: Number of Haar-random states to generate (default: 200).
        p: Physical bit-flip probability (default: 0.10).
        base_seed: Base RNG seed for reproducibility.

    Returns:
        List of dictionaries with evaluation metrics for each state.
    """
    p_val = validate_probability(p)
    x_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    records: List[Dict[str, Any]] = []

    for i in range(num_states):
        state_seed = base_seed + i
        sv_ideal = get_statevector("random", seed=state_seed)
        dm_ideal = DensityMatrix(sv_ideal)
        x_coord, y_coord, z_coord = compute_bloch_coordinates(sv_ideal)

        # 1. Unprotected single-qubit bit-flip channel
        rho_ideal = dm_ideal.data
        rho_unprot = (1.0 - p_val) * rho_ideal + p_val * (x_matrix @ rho_ideal @ x_matrix)
        f_unprotected = float(state_fidelity(DensityMatrix(rho_unprot), dm_ideal))

        # 2. Protected [[3, 1, 1]] repetition code
        prot_res = evaluate_repetition_code_statevector(
            state_name="random",
            p=p_val,
            deterministic_mask=None,
            seed=state_seed,
        )
        f_protected = float(prot_res["fidelity"])

        records.append(
            {
                "state_index": i,
                "seed": state_seed,
                "bloch_x": float(x_coord),
                "bloch_y": float(y_coord),
                "bloch_z": float(z_coord),
                "physical_p": p_val,
                "unprotected_fidelity": f_unprotected,
                "protected_fidelity": f_protected,
                "fidelity_gain": f_protected - f_unprotected,
                "is_improved": bool(f_protected >= f_unprotected),
            }
        )

    return records
