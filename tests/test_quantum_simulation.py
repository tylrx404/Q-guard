"""Automated tests for Q-GUARD quantum simulation core (Milestone 2).

Verifies state preparations, deterministic and probabilistic bit-flip channels,
Wilson score intervals, statistical estimation convergence, and parameter validation.
"""

import pytest
import numpy as np

from qiskit import QuantumCircuit
from src.noise import (
    apply_deterministic_bitflip,
    apply_probabilistic_bitflip,
    build_bitflip_noise_model,
    run_circuit_simulation,
    validate_probability,
    validate_qubit_index,
)
from src.states import (
    prepare_state,
    get_statevector,
    get_bloch_coordinates,
    generate_random_pure_state,
    normalize_state_label,
)
from src.analysis import (
    compute_wilson_confidence_interval,
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_fidelity,
)
from src.strategies import (
    run_strategy_a_direct_measurement,
    run_strategy_b_sample_size_sweep,
    evaluate_strategy_b_fault_detector,
    compute_strategy_b_roc_curve,
)


# ==============================================================================
# 1. State Preparation Tests
# ==============================================================================
def test_preparation_of_all_four_named_states():
    """Verify preparation circuits and analytical properties for |0>, |1>, |+>, |->."""
    for state_label in ["|0>", "|1>", "|+>", "|->"]:
        qc = prepare_state(state_label)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 1

        sv = get_statevector(state_label)
        norm = np.linalg.norm(sv.data)
        assert np.isclose(norm, 1.0, atol=1e-9)

    # Validate specific Bloch coordinates
    assert get_bloch_coordinates("|0>") == (0.0, 0.0, 1.0)
    assert get_bloch_coordinates("|1>") == (0.0, 0.0, -1.0)
    coords_plus = get_bloch_coordinates("|+>")
    assert np.isclose(coords_plus[0], 1.0) and np.isclose(coords_plus[1], 0.0) and np.isclose(coords_plus[2], 0.0)
    coords_minus = get_bloch_coordinates("|->")
    assert np.isclose(coords_minus[0], -1.0) and np.isclose(coords_minus[1], 0.0) and np.isclose(coords_minus[2], 0.0)


# ==============================================================================
# 2. Deterministic X Injection Tests
# ==============================================================================
def test_deterministic_x_injection_flips_computational_states():
    """Verify deterministic X gate flips |0> -> |1> and |1> -> |0> with certainty."""
    # |0> + X -> |1>
    res_0 = run_strategy_a_direct_measurement(
        state_name="|0>",
        error_mode="deterministic",
        deterministic_qubit=0,
        shots=500,
        seed=42,
    )
    assert res_0["counts"] == {"1": 500}
    assert res_0["observed_errors"] == 500
    assert res_0["estimated_p"] == 1.0

    # |1> + X -> |0>
    res_1 = run_strategy_a_direct_measurement(
        state_name="|1>",
        error_mode="deterministic",
        deterministic_qubit=0,
        shots=500,
        seed=42,
    )
    assert res_1["counts"] == {"0": 500}
    assert res_1["observed_errors"] == 500
    assert res_1["estimated_p"] == 1.0


# ==============================================================================
# 3 & 4. Noiseless / p=0 Channel Tests
# ==============================================================================
def test_noiseless_channel_has_zero_injected_errors():
    """Verify that p=0 probabilistic channel introduces zero errors on |0> and |1>."""
    # |0> with p=0 -> always measured as '0'
    res_0 = run_strategy_a_direct_measurement(
        state_name="|0>",
        error_mode="probabilistic",
        p=0.0,
        shots=1000,
        seed=101,
    )
    assert res_0["counts"] == {"0": 1000}
    assert res_0["observed_errors"] == 0
    assert res_0["estimated_p"] == 0.0
    assert res_0["lower_bound"] == 0.0

    # |1> with p=0 -> always measured as '1'
    res_1 = run_strategy_a_direct_measurement(
        state_name="|1>",
        error_mode="probabilistic",
        p=0.0,
        shots=1000,
        seed=102,
    )
    assert res_1["counts"] == {"1": 1000}
    assert res_1["observed_errors"] == 0
    assert res_1["estimated_p"] == 0.0
    assert res_1["lower_bound"] == 0.0


# ==============================================================================
# 5. Probabilistic p=1 Channel Tests
# ==============================================================================
def test_probabilistic_p_one_flips_every_transmitted_bit():
    """Verify that p=1 probabilistic channel inverts every bit in the ideal model."""
    res_0 = run_strategy_a_direct_measurement(
        state_name="|0>",
        error_mode="probabilistic",
        p=1.0,
        shots=1000,
        seed=201,
    )
    assert res_0["counts"] == {"1": 1000}
    assert res_0["observed_errors"] == 1000
    assert res_0["estimated_p"] == 1.0
    assert res_0["upper_bound"] == 1.0

    res_1 = run_strategy_a_direct_measurement(
        state_name="|1>",
        error_mode="probabilistic",
        p=1.0,
        shots=1000,
        seed=202,
    )
    assert res_1["counts"] == {"0": 1000}
    assert res_1["observed_errors"] == 1000
    assert res_1["estimated_p"] == 1.0
    assert res_1["upper_bound"] == 1.0


# ==============================================================================
# 6. Statistical Consistency of Repeated Noisy Trials
# ==============================================================================
def test_repeated_noisy_trials_consistent_with_configured_p():
    """Verify empirical estimate matches configured p within reasonable statistical tolerance."""
    configured_p = 0.25
    shots = 2500
    res = run_strategy_a_direct_measurement(
        state_name="|0>",
        error_mode="probabilistic",
        p=configured_p,
        shots=shots,
        seed=303,
    )
    est_p = res["estimated_p"]
    # 3 standard deviations for binomial proportion: 3 * sqrt(0.25 * 0.75 / 2500) ≈ 0.026
    assert abs(est_p - configured_p) < 0.04
    # The true value must be within the 95% Wilson confidence interval
    assert res["lower_bound"] <= configured_p <= res["upper_bound"]


# ==============================================================================
# 7. Wilson Confidence Interval Boundary Behavior
# ==============================================================================
def test_wilson_confidence_interval_boundaries():
    """Verify Wilson interval behavior at boundary outcomes (k=0 and k=n) and interior values."""
    # Boundary k=0: lower must be 0.0, upper must be strictly positive
    ci_zero = compute_wilson_confidence_interval(k=0, n=100, confidence_level=0.95)
    assert ci_zero["point_estimate"] == 0.0
    assert ci_zero["lower_bound"] == 0.0
    assert 0.0 < ci_zero["upper_bound"] < 0.10

    # Boundary k=n: upper must be 1.0, lower must be strictly < 1.0
    ci_full = compute_wilson_confidence_interval(k=100, n=100, confidence_level=0.95)
    assert ci_full["point_estimate"] == 1.0
    assert ci_full["upper_bound"] == 1.0
    assert 0.90 < ci_full["lower_bound"] < 1.0

    # Intermediate k=50, n=100
    ci_mid = compute_wilson_confidence_interval(k=50, n=100, confidence_level=0.95)
    assert ci_mid["point_estimate"] == 0.50
    assert 0.35 < ci_mid["lower_bound"] < 0.45
    assert 0.55 < ci_mid["upper_bound"] < 0.65


# ==============================================================================
# 8. Haar-Random Pure State Generation
# ==============================================================================
def test_random_state_generation_normalized_and_reproducible():
    """Verify Haar-random state generation is normalized and deterministic with fixed seed."""
    sv1 = generate_random_pure_state(seed=42)
    sv2 = generate_random_pure_state(seed=42)
    sv3 = generate_random_pure_state(seed=999)

    # Norm must be exactly 1.0 within numerical precision
    assert np.isclose(np.linalg.norm(sv1.data), 1.0, atol=1e-10)
    assert np.isclose(np.linalg.norm(sv3.data), 1.0, atol=1e-10)

    # Same seed must yield identical statevector
    assert np.allclose(sv1.data, sv2.data)
    # Different seeds should yield different states
    assert not np.allclose(sv1.data, sv3.data)


# ==============================================================================
# 9. Results Structure for UI Integration
# ==============================================================================
def test_results_structured_correctly_for_ui():
    """Verify result dictionary contains all necessary fields for Streamlit rendering."""
    res = run_strategy_a_direct_measurement(
        state_name="|0>",
        error_mode="probabilistic",
        p=0.10,
        shots=500,
        seed=505,
    )
    expected_keys = {
        "strategy",
        "state_name",
        "error_mode",
        "configured_p",
        "shots",
        "seed",
        "circuit",
        "counts",
        "shots_0",
        "shots_1",
        "expected_bit",
        "observed_errors",
        "correct_shots",
        "estimated_p",
        "lower_bound",
        "upper_bound",
        "wilson_ci",
        "is_computational_basis",
        "explanation",
    }
    assert expected_keys.issubset(set(res.keys()))

    # Verify Strategy B sweep structure
    sweep = run_strategy_b_sample_size_sweep(true_p=0.10, sample_sizes=[20, 100], seed=42)
    assert len(sweep) == 2
    assert "error_margin" in sweep[0]
    assert "estimated_p" in sweep[0]


# ==============================================================================
# 10. Parameter Validation & Exception Handling
# ==============================================================================
def test_invalid_parameters_raise_exceptions():
    """Verify validation logic catches invalid probabilities, qubit indices, and shot counts."""
    # Invalid probability
    with pytest.raises(ValueError, match="must be in"):
        validate_probability(-0.01)
    with pytest.raises(ValueError, match="must be in"):
        validate_probability(1.01)
    with pytest.raises(TypeError, match="must be a float"):
        validate_probability("high")

    # Invalid qubit index
    with pytest.raises(ValueError, match="out of bounds"):
        validate_qubit_index(1, num_qubits=1)
    with pytest.raises(ValueError, match="out of bounds"):
        validate_qubit_index(-1, num_qubits=1)

    # Invalid shots
    with pytest.raises(ValueError, match="at least 1"):
        run_circuit_simulation(QuantumCircuit(1, 1), shots=0)

    # Invalid state
    with pytest.raises(ValueError, match="Unsupported state"):
        prepare_state("unknown_state")

    # Invalid Wilson CI params
    with pytest.raises(ValueError, match="at least 1"):
        compute_wilson_confidence_interval(k=0, n=0)
    with pytest.raises(ValueError, match="between 0 and n"):
        compute_wilson_confidence_interval(k=10, n=5)


# ==============================================================================
# 11. Superposition Basis Handling
# ==============================================================================
def test_superposition_state_basis_explanation():
    """Verify |+> and |-> include appropriate basis mismatch explanations."""
    res_plus = run_strategy_a_direct_measurement(
        state_name="|+>",
        error_mode="probabilistic",
        p=0.0,
        shots=1000,
        seed=606,
    )
    assert not res_plus["is_computational_basis"]
    assert "superposition" in res_plus["explanation"].lower()
    # In Z basis, |+> projects 50/50 even with p=0
    assert 400 < res_plus["shots_0"] < 600
    assert 400 < res_plus["shots_1"] < 600
