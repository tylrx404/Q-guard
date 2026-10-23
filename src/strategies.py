"""Error detection strategies for Q-GUARD bit-flip experiments.

Includes:
- Strategy A: Direct measurement of known computational-basis probe states.
- Strategy B: Repeated-trial statistical estimation, convergence analysis,
  and threshold-based channel fault detection.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from qiskit import QuantumCircuit
from src.noise import (
    apply_deterministic_bitflip,
    apply_probabilistic_bitflip,
    run_circuit_simulation,
    validate_probability,
)
from src.states import normalize_state_label, prepare_state
from src.analysis import (
    compute_wilson_confidence_interval,
    compute_confusion_matrix,
    compute_classification_metrics,
)


def run_strategy_a_direct_measurement(
    state_name: str = "|0>",
    error_mode: str = "probabilistic",
    p: float = 0.0,
    deterministic_qubit: int = 0,
    shots: int = 1024,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Strategy A: Direct measurement of a known basis state through a bit-flip channel.

    Steps:
      1. Prepares a known quantum state (e.g. |0> or |1>).
      2. Transmits the qubit through a deterministic or probabilistic bit-flip channel.
      3. Measures the output in the computational Z basis.
      4. Compares the observed measurement bit against the expected input.
      5. Computes the estimated error rate and 95% Wilson confidence interval.

    Args:
        state_name: Label of state to prepare ('|0>', '|1>', '|+>', '|->', 'random').
        error_mode: 'deterministic' (applies X gate) or 'probabilistic' (Pauli X noise).
        p: Physical bit-flip probability in [0.0, 1.0] (for probabilistic mode).
        deterministic_qubit: Target qubit index for deterministic injection (default: 0).
        shots: Number of measurement shots (n >= 1).
        seed: Optional RNG seed for simulator reproducibility.

    Returns:
        Structured dictionary containing counts, error metrics, Wilson interval, and explanatory metadata.

    Raises:
        ValueError: If error_mode is unrecognized, shots < 1, or p not in [0.0, 1.0].
    """
    if shots < 1:
        raise ValueError(f"Shots must be at least 1, got {shots}")

    canonical_state = normalize_state_label(state_name)
    mode = error_mode.strip().lower()

    if mode not in ("deterministic", "probabilistic"):
        raise ValueError(f"error_mode must be 'deterministic' or 'probabilistic', got '{error_mode}'")

    # Build quantum circuit
    qc = QuantumCircuit(1, 1)

    # 1. State preparation
    if canonical_state == "|1>":
        qc.x(0)
    elif canonical_state == "|+>":
        qc.h(0)
    elif canonical_state == "|->":
        qc.x(0)
        qc.h(0)
    elif canonical_state == "random":
        prep_qc = prepare_state("random", qubit=0, num_qubits=1, seed=seed)
        qc.compose(prep_qc, inplace=True)
    # (|0> requires no gates)

    # 2. Error Channel Transmission
    if mode == "deterministic":
        apply_deterministic_bitflip(qc, qubit=deterministic_qubit)
        effective_p = 1.0
    else:
        effective_p = validate_probability(p)
        apply_probabilistic_bitflip(qc, qubit=0, p=effective_p)

    # 3. Z-basis measurement
    qc.measure(0, 0)

    # 4. Simulation
    counts = run_circuit_simulation(qc, shots=shots, seed=seed)

    # Ensure binary keys are present
    shots_0 = counts.get("0", 0)
    shots_1 = counts.get("1", 0)
    total_valid = shots_0 + shots_1

    # 5. Error diagnosis
    is_computational = canonical_state in ("|0>", "|1>")

    if canonical_state == "|0>":
        expected_bit = "0"
        observed_errors = shots_1
        correct_shots = shots_0
        basis_explanation = (
            "Ideal outcome is |0>. Measured '1' counts are diagnosed as bit-flip errors."
        )
    elif canonical_state == "|1>":
        expected_bit = "1"
        observed_errors = shots_0
        correct_shots = shots_1
        basis_explanation = (
            "Ideal outcome is |1>. Measured '0' counts are diagnosed as bit-flip errors."
        )
    else:
        expected_bit = "N/A"
        observed_errors = shots_1  # Conventional metric for reporting
        correct_shots = shots_0
        basis_explanation = (
            f"Note: State {canonical_state} is in a superposition basis. Z-basis measurements "
            f"collapse probabilistically into |0> and |1> independent of bit flips (X|+> = |+>). "
            f"Observed counts reflect quantum state projection, not solely channel noise."
        )

    # Compute Wilson 95% Confidence Interval
    # For computational states, evaluate on observed_errors; for others, compute interval on '1' frequency
    wilson_ci = compute_wilson_confidence_interval(
        k=observed_errors,
        n=total_valid,
        confidence_level=0.95,
    )

    return {
        "strategy": "Strategy A: Direct Measurement",
        "state_name": canonical_state,
        "error_mode": mode,
        "configured_p": effective_p,
        "shots": total_valid,
        "seed": seed,
        "circuit": qc,
        "counts": counts,
        "shots_0": shots_0,
        "shots_1": shots_1,
        "expected_bit": expected_bit,
        "observed_errors": observed_errors,
        "correct_shots": correct_shots,
        "estimated_p": wilson_ci["point_estimate"],
        "lower_bound": wilson_ci["lower_bound"],
        "upper_bound": wilson_ci["upper_bound"],
        "wilson_ci": wilson_ci,
        "is_computational_basis": is_computational,
        "explanation": basis_explanation,
    }


def run_strategy_b_sample_size_sweep(
    true_p: float,
    sample_sizes: Sequence[int] = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
    input_state: str = "|0>",
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """Simulate repeated transmissions across increasing sample sizes to demonstrate estimation convergence.

    As sample size N increases, the empirical error estimate p_hat converges to true_p,
    and the 95% Wilson confidence interval contracts proportionally to 1 / sqrt(N).

    Args:
        true_p: True channel bit-flip probability.
        sample_sizes: Sequence of shot counts to simulate.
        input_state: Known computational probe state ('|0>' or '|1>').
        seed: Base RNG seed for reproducibility.

    Returns:
        List of dictionaries with fields: sample_size, true_p, estimated_p, lower_bound,
        upper_bound, error_margin, observed_errors.
    """
    validate_probability(true_p)
    results = []

    for idx, n in enumerate(sample_sizes):
        trial_seed = seed + idx if seed is not None else None
        res = run_strategy_a_direct_measurement(
            state_name=input_state,
            error_mode="probabilistic",
            p=true_p,
            shots=n,
            seed=trial_seed,
        )
        ci = res["wilson_ci"]
        margin = (ci["upper_bound"] - ci["lower_bound"]) / 2.0

        results.append(
            {
                "sample_size": n,
                "true_p": true_p,
                "estimated_p": ci["point_estimate"],
                "lower_bound": ci["lower_bound"],
                "upper_bound": ci["upper_bound"],
                "error_margin": margin,
                "observed_errors": res["observed_errors"],
            }
        )

    return results


def run_strategy_b_p_sweep(
    p_values: Sequence[float],
    shots_per_point: int = 1000,
    input_state: str = "|0>",
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """Simulate Strategy B across a range of physical error probabilities p in [0.0, 1.0].

    Args:
        p_values: Sequence of true error rates to test.
        shots_per_point: Number of probe shots for each error rate.
        input_state: Known probe state (|0> or |1>).
        seed: Base seed for reproducibility.

    Returns:
        List of dictionaries recording true_p vs estimated_p and Wilson bounds.
    """
    results = []
    for idx, p in enumerate(p_values):
        trial_seed = seed + idx * 7 if seed is not None else None
        res = run_strategy_a_direct_measurement(
            state_name=input_state,
            error_mode="probabilistic",
            p=p,
            shots=shots_per_point,
            seed=trial_seed,
        )
        ci = res["wilson_ci"]
        results.append(
            {
                "true_p": p,
                "estimated_p": ci["point_estimate"],
                "lower_bound": ci["lower_bound"],
                "upper_bound": ci["upper_bound"],
                "shots": shots_per_point,
                "observed_errors": res["observed_errors"],
            }
        )
    return results


def evaluate_strategy_b_fault_detector(
    channel_p_list: Sequence[float],
    faulty_threshold: float = 0.10,
    decision_threshold: float = 0.12,
    sample_size: int = 200,
    input_state: str = "|0>",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Evaluate a threshold-based statistical fault detector across an ensemble of quantum channels.

    Definitions:
      - Ground Truth: A channel is Truly Faulty (Positive class = 1) if true_p > faulty_threshold.
                      A channel is Truly Healthy (Negative class = 0) if true_p <= faulty_threshold.
      - Decision Rule: The detector transmits N probe states, observes estimated_p = k / N.
                       If estimated_p > decision_threshold, the channel is flagged as FAULTY (Predicted = 1).
                       Otherwise, it is passed as HEALTHY (Predicted = 0).

    Important Conceptual Note:
      Statistical estimation diagnoses channel reliability from repeated transmissions of known states.
      A narrow Wilson confidence interval means high certainty about the *average error rate of the channel*;
      it does NOT guarantee detecting an error on any individual unknown quantum state. Real-time protection
      of unknown states requires quantum error-correcting codes (such as repetition or surface codes).

    Args:
        channel_p_list: Array of true error probabilities for channels in the test fleet.
        faulty_threshold: True physical noise rate above which a channel is considered faulty.
        decision_threshold: Estimator cutoff threshold tau above which the detector flags a fault.
        sample_size: Number of probe qubits transmitted through each channel.
        input_state: Probe state used for calibration (|0> or |1>).
        seed: Base random seed for reproducibility.

    Returns:
        Dictionary containing confusion matrix, classification metrics, and detailed channel evaluations.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    evaluations: List[Dict[str, Any]] = []

    for idx, true_p in enumerate(channel_p_list):
        trial_seed = seed + idx * 13 if seed is not None else None
        res = run_strategy_a_direct_measurement(
            state_name=input_state,
            error_mode="probabilistic",
            p=true_p,
            shots=sample_size,
            seed=trial_seed,
        )
        est_p = res["estimated_p"]

        is_truly_faulty = 1 if true_p > faulty_threshold else 0
        is_flagged = 1 if est_p > decision_threshold else 0

        y_true.append(is_truly_faulty)
        y_pred.append(is_flagged)

        evaluations.append(
            {
                "channel_id": idx,
                "true_p": true_p,
                "is_truly_faulty": bool(is_truly_faulty),
                "estimated_p": est_p,
                "is_flagged": bool(is_flagged),
                "lower_bound": res["lower_bound"],
                "upper_bound": res["upper_bound"],
            }
        )

    cm = compute_confusion_matrix(y_true, y_pred, positive_label=1)
    metrics = compute_classification_metrics(cm["tp"], cm["fp"], cm["tn"], cm["fn"])

    return {
        "faulty_threshold": faulty_threshold,
        "decision_threshold": decision_threshold,
        "sample_size": sample_size,
        "confusion_matrix": cm,
        "metrics": metrics,
        "evaluations": evaluations,
    }


def compute_strategy_b_roc_curve(
    channel_p_list: Sequence[float],
    faulty_threshold: float = 0.10,
    sample_size: int = 200,
    threshold_steps: int = 50,
    input_state: str = "|0>",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Compute Receiver Operating Characteristic (ROC) curve data for the statistical fault detector.

    Sweeps the decision threshold tau over [0.0, 1.0] to map out the trade-off
    between True Positive Rate (Recall) and False Positive Rate.

    Args:
        channel_p_list: Sequence of true error rates across test channels.
        faulty_threshold: Ground-truth cutoff defining faulty channels.
        sample_size: Shots per channel evaluation.
        threshold_steps: Number of decision thresholds to evaluate between 0.0 and max(p) + margin.
        input_state: Probe state (|0> or |1>).
        seed: Base RNG seed.

    Returns:
        Dictionary with 'thresholds', 'fpr', 'tpr', 'auc', and 'sample_size'.
    """
    # 1. Pre-simulate estimated_p for each channel to keep evaluations consistent across thresholds
    estimates: List[float] = []
    y_true: List[int] = []

    for idx, true_p in enumerate(channel_p_list):
        trial_seed = seed + idx * 19 if seed is not None else None
        res = run_strategy_a_direct_measurement(
            state_name=input_state,
            error_mode="probabilistic",
            p=true_p,
            shots=sample_size,
            seed=trial_seed,
        )
        estimates.append(res["estimated_p"])
        y_true.append(1 if true_p > faulty_threshold else 0)

    # 2. Sweep decision thresholds
    max_thresh = min(1.0, max(channel_p_list) * 1.5 + 0.05)
    thresholds = np.linspace(0.0, max_thresh, threshold_steps)

    fpr_list: List[float] = []
    tpr_list: List[float] = []

    for tau in thresholds:
        y_pred = [1 if est > tau else 0 for est in estimates]
        cm = compute_confusion_matrix(y_true, y_pred, positive_label=1)
        metrics = compute_classification_metrics(cm["tp"], cm["fp"], cm["tn"], cm["fn"])
        fpr_list.append(metrics["false_positive_rate"])
        tpr_list.append(metrics["recall"])

    # 3. Compute AUC using trapezoidal rule
    # Sort points by FPR ascending
    sorted_pairs = sorted(zip(fpr_list, tpr_list), key=lambda x: (x[0], x[1]))
    # Add boundary anchor (0, 0) and (1, 1) if not present
    fpr_sorted = [0.0] + [p[0] for p in sorted_pairs] + [1.0]
    tpr_sorted = [0.0] + [p[1] for p in sorted_pairs] + [1.0]
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(trapz_fn(tpr_sorted, fpr_sorted))
    auc = max(0.0, min(1.0, auc))

    return {
        "thresholds": [float(t) for t in thresholds],
        "fpr": fpr_list,
        "tpr": tpr_list,
        "auc": auc,
        "sample_size": sample_size,
        "faulty_threshold": faulty_threshold,
    }


# ==============================================================================
# STRATEGY C: ANCILLA-ASSISTED PARITY DETECTION
# ==============================================================================
def build_strategy_c_parity_circuit(
    state_name: str = "|0>",
    error_mode: str = "probabilistic",
    p: float = 0.0,
    deterministic_mask: Optional[str] = None,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """Build a 5-qubit circuit for Strategy C (3 data qubits, 2 parity ancillas).

    Syndrome definitions:
      s01 = q0 ^ q1 (Ancilla a0)
      s12 = q1 ^ q2 (Ancilla a1)

    Crucial: Only the ancillas are measured. The 3 data qubits remain unmeasured,
    preserving the encoded logical superposition alpha|000> + beta|111>.

    Args:
        state_name: Input state to encode ('|0>', '|1>', '|+>', '|->', 'random').
        error_mode: 'deterministic' or 'probabilistic'.
        p: Physical error probability (if probabilistic).
        deterministic_mask: 3-bit binary string (e.g. '100' for flip on q0).
        seed: RNG seed for random state generation.

    Returns:
        Configured QuantumCircuit with 2 classical measurement bits.
    """
    from qiskit import ClassicalRegister, QuantumRegister

    qr_data = QuantumRegister(3, name="data")
    qr_anc = QuantumRegister(2, name="ancilla")
    cr_syn = ClassicalRegister(2, name="syn")
    qc = QuantumCircuit(qr_data, qr_anc, cr_syn)

    # 1. State preparation on data[0]
    canonical = normalize_state_label(state_name)
    if canonical == "|1>":
        qc.x(qr_data[0])
    elif canonical == "|+>":
        qc.h(qr_data[0])
    elif canonical == "|->":
        qc.x(qr_data[0])
        qc.h(qr_data[0])
    elif canonical == "random":
        prep_qc = prepare_state("random", qubit=0, num_qubits=1, seed=seed)
        qc.compose(prep_qc, qubits=[qr_data[0]], inplace=True)

    # 2. Encode repetition code: alpha|000> + beta|111>
    qc.cx(qr_data[0], qr_data[1])
    qc.cx(qr_data[0], qr_data[2])

    # 3. Error injection
    if error_mode == "deterministic" and deterministic_mask is not None:
        mask = deterministic_mask.strip()
        for idx in range(3):
            if mask[idx] == "1":
                qc.x(qr_data[idx])
    elif error_mode == "probabilistic":
        p_val = validate_probability(p)
        for idx in range(3):
            apply_probabilistic_bitflip(qc, qubit=idx, p=p_val)

    # 4. Parity extraction onto ancillas
    # s01 = q0 ^ q1
    qc.cx(qr_data[0], qr_anc[0])
    qc.cx(qr_data[1], qr_anc[0])

    # s12 = q1 ^ q2
    qc.cx(qr_data[1], qr_anc[1])
    qc.cx(qr_data[2], qr_anc[1])

    # 5. Measure ancillas
    # cr_syn[0] <- ancilla 0 (s01)
    # cr_syn[1] <- ancilla 1 (s12)
    qc.measure(qr_anc[0], cr_syn[0])
    qc.measure(qr_anc[1], cr_syn[1])

    return qc


def run_strategy_c_ancilla_parity(
    state_name: str = "|0>",
    error_mode: str = "probabilistic",
    p: float = 0.0,
    deterministic_mask: Optional[str] = None,
    shots: int = 1024,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Execute Strategy C: Ancilla-assisted parity extraction without collapsing data qubits.

    Fixed Syndrome Interpretation:
      '00': No single bit-flip detected
      '10': Single bit-flip detected on data qubit 0
      '11': Single bit-flip detected on data qubit 1
      '01': Single bit-flip detected on data qubit 2

    Scientific Limitation Notice:
      Parity measurements preserve the encoded repetition code subspace alpha|000> + beta|111>
      because both basis states have identical parity eigenvalues (even parity: s01=0, s12=0).
      However, parity checks do NOT protect an arbitrary unencoded 3-qubit state, as measuring
      parity projectors collapses superpositions across different parity sectors.

    Args:
        state_name: Input logical state.
        error_mode: 'deterministic' or 'probabilistic'.
        p: Physical bit-flip probability.
        deterministic_mask: Optional 3-bit string for deterministic error.
        shots: Number of measurement shots.
        seed: RNG seed for reproducible execution.

    Returns:
        Dictionary containing normalized syndrome counts, dominant syndrome,
        subspace preservation fidelity, and detailed diagnostic metadata.
    """
    if shots < 1:
        raise ValueError(f"Shots must be at least 1, got {shots}")

    qc = build_strategy_c_parity_circuit(
        state_name=state_name,
        error_mode=error_mode,
        p=p,
        deterministic_mask=deterministic_mask,
        seed=seed,
    )

    counts = run_circuit_simulation(qc, shots=shots, seed=seed)

    # Normalize Qiskit syndrome bitstrings:
    # We measured cr_syn[0] <- a0 (s01), cr_syn[1] <- a1 (s12).
    # Qiskit outputs "cr_syn[1] cr_syn[0]" -> "s12 s01".
    # We normalize to standardized "s01 s12".
    syndrome_counts: Dict[str, int] = {"00": 0, "10": 0, "11": 0, "01": 0}
    for raw_str, freq in counts.items():
        cleaned = raw_str.strip().replace(" ", "")
        if len(cleaned) == 2:
            s01 = cleaned[1]
            s12 = cleaned[0]
            syn_norm = f"{s01}{s12}"
        else:
            syn_norm = "00"
        syndrome_counts[syn_norm] = syndrome_counts.get(syn_norm, 0) + freq

    dominant_syn = max(syndrome_counts, key=syndrome_counts.get)

    # Evaluate preservation of encoded superposition using Statevector (unmeasured circuit)
    qc_unmeasured = build_strategy_c_parity_circuit(
        state_name=state_name,
        error_mode="deterministic",
        deterministic_mask="000",  # Ideal channel
        seed=seed,
    )
    # Remove measurement operations to test coherent statevector
    qc_unmeasured.remove_final_measurements()

    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

    sv = Statevector.from_instruction(qc_unmeasured)
    dm = DensityMatrix(sv)
    # Trace out ancillas (qubits 3, 4) to check 3-qubit data subspace
    dm_data = partial_trace(dm, [3, 4])

    # Target encoded 3-qubit state: alpha|000> + beta|111>
    qc_target = QuantumCircuit(3)
    canonical = normalize_state_label(state_name)
    if canonical == "|1>":
        qc_target.x(0)
    elif canonical == "|+>":
        qc_target.h(0)
    elif canonical == "|->":
        qc_target.x(0)
        qc_target.h(0)
    elif canonical == "random":
        prep_qc = prepare_state("random", qubit=0, num_qubits=1, seed=seed)
        qc_target.compose(prep_qc, qubits=[0], inplace=True)
    qc_target.cx(0, 1)
    qc_target.cx(0, 2)
    target_sv = Statevector.from_instruction(qc_target)
    target_dm = DensityMatrix(target_sv)

    subspace_fidelity = float(state_fidelity(dm_data, target_dm))

    explanation = (
        "Strategy C measures parity stabilizers (Z ⊗ Z ⊗ I and I ⊗ Z ⊗ Z) onto entangled ancillas. "
        f"For the encoded subspace alpha|000> + beta|111>, both basis components produce even parity (s01=0, s12=0), "
        "allowing syndrome diagnosis without collapsing the stored logical quantum information."
    )

    return {
        "strategy": "Strategy C: Ancilla-Assisted Parity Detection",
        "state_name": state_name,
        "error_mode": error_mode,
        "configured_p": p,
        "deterministic_mask": deterministic_mask,
        "shots": shots,
        "seed": seed,
        "circuit": qc,
        "raw_counts": counts,
        "syndrome_counts": syndrome_counts,
        "dominant_syndrome": dominant_syn,
        "subspace_preservation_fidelity": subspace_fidelity,
        "is_encoded_subspace_preserved": bool(subspace_fidelity > 0.999),
        "explanation": explanation,
    }


def get_strategy_comparison_table() -> List[Dict[str, Any]]:
    """Return comprehensive comparison of Strategies A, B, and C."""
    return [
        {
            "Strategy": "Strategy A: Direct Measurement",
            "What It Detects": "Direct computational-basis flip on known probe states",
            "State Compatibility": "Known basis states only (|0>, |1>)",
            "Preserves Superposition": "No (Destructive measurement collapses state)",
            "Qubit Overhead": "1 physical qubit, 0 ancillas",
            "Active Correction": "None (Diagnostic calibration only)",
            "Primary Role": "Quick single-qubit calibration and baseline error benchmarking",
            "Key Limitation": "Cannot protect unknown states; Z-measurement is destructive",
        },
        {
            "Strategy": "Strategy B: Statistical Estimation",
            "What It Detects": "Average channel noise probability p and fault status",
            "State Compatibility": "Known calibration probe states (|0>, |1>)",
            "Preserves Superposition": "No (Repeated destructive sampling)",
            "Qubit Overhead": "1 physical qubit per sample, 0 ancillas",
            "Active Correction": "None (Flags channel as faulty if p̂ > τ)",
            "Primary Role": "Systematic channel characterization, convergence, and ROC testing",
            "Key Limitation": "Measures average noise over time; does not protect a live state",
        },
        {
            "Strategy": "Strategy C: Ancilla Parity Detection",
            "What It Detects": "Specific location of single physical bit-flips via syndromes",
            "State Compatibility": "Encoded logical subspace alpha|000> + beta|111>",
            "Preserves Superposition": "Yes (Ancilla-mediated non-destructive readout)",
            "Qubit Overhead": "3 data qubits + 2 parity ancillas (5 total)",
            "Active Correction": "Provides syndrome input for active CCX recovery (Repetition Code)",
            "Primary Role": "Fault-tolerant error diagnosis without user state destruction",
            "Key Limitation": "Protects only the encoded subspace; cannot correct phase-flips",
        },
    ]
