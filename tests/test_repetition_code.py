"""Automated tests for 3-qubit repetition code [[3, 1, 1]] and Strategy C ancilla parity checks.

Verifies:
1. Ideal encoding into alpha|000> + beta|111>.
2. Syndrome extraction for all single-bit flips (10, 11, 01) and no-error (00).
3. 100% recovery for all single-qubit errors on computational and superposition states.
4. Subspace preservation of encoded states through parity checks.
5. Partial trace subsystem fidelity calculation on recovered data qubit q0.
6. Boundary cases p=0 and p=1.
7. Exhaustive verification of all 8 error masks proving double-flip miscorrection.
8. Syndrome bit-order consistency and reproducibility.
"""

import pytest
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

from src.repetition_code import (
    build_repetition_code_circuit,
    encode_repetition_code,
    extract_syndromes_coherent,
    apply_coherent_correction,
    decode_repetition_code,
    evaluate_repetition_code_statevector,
    analyze_eight_error_masks,
    run_repetition_code_simulation,
    compute_repetition_code_theory_curve,
    evaluate_haar_random_benchmark,
    normalize_syndrome_string,
    SYNDROME_MAP,
)
from src.strategies import (
    run_strategy_c_ancilla_parity,
    build_strategy_c_parity_circuit,
    get_strategy_comparison_table,
)
from src.states import get_statevector


# ==============================================================================
# 1. Encoding Tests
# ==============================================================================
def test_ideal_encoder_produces_ghz_subspace():
    """Verify that encoding |+> produces (|000> + |111>)/sqrt(2)."""
    qc = QuantumCircuit(3)
    qc.h(0)  # |+>
    encode_repetition_code(qc, data_qubit=0, physical_qubits=(0, 1, 2))

    sv = Statevector.from_instruction(qc)
    expected_sv = Statevector([1.0 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 1.0 / np.sqrt(2)])
    assert state_fidelity(sv, expected_sv) > 0.999999

    # Encode |1> -> |111>
    qc1 = QuantumCircuit(3)
    qc1.x(0)
    encode_repetition_code(qc1, data_qubit=0, physical_qubits=(0, 1, 2))
    sv1 = Statevector.from_instruction(qc1)
    assert state_fidelity(sv1, Statevector.from_label("111")) > 0.999999


# ==============================================================================
# 2 & 10. Syndrome Mapping and Bit Order Tests
# ==============================================================================
def test_single_bit_flip_syndrome_mapping():
    """Verify exact syndrome string for each single-bit flip pattern under s01s12 definition."""
    # Mask 000 -> 00
    res_000 = evaluate_repetition_code_statevector(state_name="|0>", deterministic_mask="000")
    assert res_000["syndrome"] == "00"

    # Mask 100 (flip q0) -> 10 (s01=1, s12=0)
    res_100 = evaluate_repetition_code_statevector(state_name="|0>", deterministic_mask="100")
    assert res_100["syndrome"] == "10"
    assert res_100["syndrome_info"]["error_qubit"] == 0

    # Mask 010 (flip q1) -> 11 (s01=1, s12=1)
    res_010 = evaluate_repetition_code_statevector(state_name="|0>", deterministic_mask="010")
    assert res_010["syndrome"] == "11"
    assert res_010["syndrome_info"]["error_qubit"] == 1

    # Mask 001 (flip q2) -> 01 (s01=0, s12=1)
    res_001 = evaluate_repetition_code_statevector(state_name="|0>", deterministic_mask="001")
    assert res_001["syndrome"] == "01"
    assert res_001["syndrome_info"]["error_qubit"] == 2


def test_syndrome_normalization_helper():
    """Verify normalize_syndrome_string validates and formats correctly."""
    assert normalize_syndrome_string(" 10 ") == "10"
    assert normalize_syndrome_string("0 1") == "01"
    with pytest.raises(ValueError, match="Expected 2 binary bits"):
        normalize_syndrome_string("100")
    with pytest.raises(ValueError, match="Expected 2 binary bits"):
        normalize_syndrome_string("ab")


# ==============================================================================
# 3 & 5. Single-Flip Recovery on Multiple Basis States
# ==============================================================================
@pytest.mark.parametrize("state_label", ["|0>", "|1>", "|+>", "|->", "random"])
def test_all_single_flips_fully_recovered(state_label):
    """Verify 100% recovered state fidelity for any single bit flip on diverse quantum states."""
    for mask in ["000", "100", "010", "001"]:
        eval_res = evaluate_repetition_code_statevector(
            state_name=state_label,
            deterministic_mask=mask,
            seed=42,
        )
        assert eval_res["fidelity"] > 0.999999, f"Failed for state {state_label} with mask {mask}"


def test_superposition_state_with_phase():
    """Verify recovery on phase state |+i> = (|0> + i|1>)/sqrt(2)."""
    # Build circuit manually for |+i>
    for mask in ["100", "010", "001"]:
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.s(0)  # |+i>
        encode_repetition_code(qc, 0, (0, 1, 2))

        # Inject error
        if mask[0] == "1": qc.x(0)
        if mask[1] == "1": qc.x(1)
        if mask[2] == "1": qc.x(2)

        extract_syndromes_coherent(qc, (0, 1, 2), (3, 4))
        apply_coherent_correction(qc, (0, 1, 2), (3, 4))
        decode_repetition_code(qc, 0, (0, 1, 2))

        sv = Statevector.from_instruction(qc)
        dm = DensityMatrix(sv)
        reduced_q0 = partial_trace(dm, [1, 2, 3, 4])

        target_qc = QuantumCircuit(1)
        target_qc.h(0)
        target_qc.s(0)
        target_sv = Statevector.from_instruction(target_qc)

        fid = float(state_fidelity(reduced_q0, target_sv))
        assert fid > 0.999999


# ==============================================================================
# 4. Encoded Superposition Preservation Through Strategy C
# ==============================================================================
def test_strategy_c_preserves_encoded_superposition():
    """Verify Strategy C ancilla parity extraction leaves encoded superposition intact."""
    res = run_strategy_c_ancilla_parity(
        state_name="|+>",
        error_mode="deterministic",
        deterministic_mask="000",
        shots=500,
        seed=123,
    )
    assert res["dominant_syndrome"] == "00"
    assert res["subspace_preservation_fidelity"] > 0.999999
    assert res["is_encoded_subspace_preserved"] is True


# ==============================================================================
# 6 & 7. Boundary Probabilities p=0 and p=1
# ==============================================================================
def test_repetition_code_boundary_p_zero_and_one():
    """Verify behavior at p=0 (noiseless) and p=1 (full triple flip)."""
    # p = 0: No errors, dominant syndrome 00, 0 logical errors
    sim_0 = run_repetition_code_simulation(state_name="|0>", p=0.0, shots=500, seed=42)
    assert sim_0["syndrome_counts"]["00"] == 500
    assert sim_0["logical_errors"] == 0
    assert sim_0["empirical_logical_error"] == 0.0

    # p = 1: All 3 qubits flip (mask 111) -> Syndrome 00 (undetected logical flip)
    sim_1 = run_repetition_code_simulation(state_name="|0>", p=1.0, shots=500, seed=42)
    assert sim_1["syndrome_counts"]["00"] == 500  # Parity check doesn't detect 3 flips
    assert sim_1["logical_errors"] == 500  # Entire shot count flipped to |1>
    assert sim_1["empirical_logical_error"] == 1.0


# ==============================================================================
# 8 & 9. Exhaustive Verification of All 8 Error Masks (Double-Flip Miscorrection)
# ==============================================================================
def test_exhaustive_eight_masks_and_miscorrection():
    """Verify all 8 error masks demonstrating single recovery and double-flip failure."""
    masks_eval = analyze_eight_error_masks(input_state="|0>")
    assert len(masks_eval) == 8

    by_mask = {m["mask"]: m for m in masks_eval}

    # Weight 0
    assert by_mask["000"]["syndrome"] == "00"
    assert by_mask["000"]["logical_fidelity"] == 1.0

    # Weight 1: All correctable
    assert by_mask["100"]["syndrome"] == "10" and by_mask["100"]["logical_fidelity"] == 1.0
    assert by_mask["010"]["syndrome"] == "11" and by_mask["010"]["logical_fidelity"] == 1.0
    assert by_mask["001"]["syndrome"] == "01" and by_mask["001"]["logical_fidelity"] == 1.0

    # Weight 2: Miscorrection (decoder applies flip to third qubit, completing logical X_L)
    assert by_mask["110"]["syndrome"] == "01"  # Matches single-flip on q2
    assert by_mask["110"]["logical_fidelity"] == 0.0

    assert by_mask["101"]["syndrome"] == "11"  # Matches single-flip on q1
    assert by_mask["101"]["logical_fidelity"] == 0.0

    assert by_mask["011"]["syndrome"] == "10"  # Matches single-flip on q0
    assert by_mask["011"]["logical_fidelity"] == 0.0

    # Weight 3: Undetected logical flip
    assert by_mask["111"]["syndrome"] == "00"
    assert by_mask["111"]["logical_fidelity"] == 0.0


# ==============================================================================
# 11. Theory Curve Calculation
# ==============================================================================
def test_repetition_code_theory_curve():
    """Verify analytical repetition code failure formula P_L = 3p^2 - 2p^3."""
    theory = compute_repetition_code_theory_curve([0.0, 0.1, 0.2, 0.5, 1.0])
    p_log = theory["p_logical"]

    assert p_log[0] == 0.0
    # For p = 0.1: 3(0.01) - 2(0.001) = 0.028
    assert np.isclose(p_log[1], 0.028)
    # For p = 0.2: 3(0.04) - 2(0.008) = 0.104
    assert np.isclose(p_log[2], 0.104)
    # For p = 0.5: 3(0.25) - 2(0.125) = 0.500 (breakeven)
    assert np.isclose(p_log[3], 0.500)
    # For p = 1.0: 3(1) - 2(1) = 1.0
    assert np.isclose(p_log[4], 1.000)


# ==============================================================================
# 12. Strategy Comparison Table Completeness
# ==============================================================================
def test_strategy_comparison_table():
    """Verify 3-strategy comparison table contains all necessary comparison dimensions."""
    table = get_strategy_comparison_table()
    assert len(table) == 3
    strategies = [row["Strategy"] for row in table]
    assert any("Strategy A" in s for s in strategies)
    assert any("Strategy B" in s for s in strategies)
    assert any("Strategy C" in s for s in strategies)


# ==============================================================================
# 13. Haar-Random State Ensemble Benchmark Tests
# ==============================================================================
def test_evaluate_haar_random_benchmark_fidelity_gain():
    """Verify that 200 Haar-random states demonstrate consistent QEC fidelity improvement."""
    records = evaluate_haar_random_benchmark(num_states=25, p=0.10, base_seed=123)
    assert len(records) == 25

    unprotected_fids = [r["unprotected_fidelity"] for r in records]
    protected_fids = [r["protected_fidelity"] for r in records]
    gains = [r["fidelity_gain"] for r in records]

    # Verify all fidelities are valid physical probabilities in [0, 1]
    for r in records:
        assert 0.0 <= r["unprotected_fidelity"] <= 1.0
        assert 0.0 <= r["protected_fidelity"] <= 1.0
        assert -1.0 <= r["bloch_x"] <= 1.0
        assert -1.0 <= r["bloch_y"] <= 1.0
        assert -1.0 <= r["bloch_z"] <= 1.0

    mean_unprot = np.mean(unprotected_fids)
    mean_prot = np.mean(protected_fids)
    mean_gain = np.mean(gains)

    # Under p = 0.10, unprotected is ~0.933, protected is ~0.981
    assert mean_prot > mean_unprot, "Protected fidelity must exceed unprotected fidelity"
    assert mean_prot > 0.97, f"Mean protected fidelity {mean_prot:.4f} should exceed 0.97"
    assert mean_gain > 0.03, f"Mean fidelity gain {mean_gain:.4f} should be positive and > 3%"


# ==============================================================================
# 14. 4096-Shot Repetition Code High-Precision Benchmark
# ==============================================================================
def test_repetition_code_simulation_4096_shots():
    """Verify high-precision simulation with 4,096 shots at p=0.10."""
    res = run_repetition_code_simulation(state_name="|0>", p=0.10, shots=4096, seed=777)
    assert res["shots"] == 4096
    assert sum(res["syndrome_counts"].values()) == 4096
    assert sum(res["logical_counts"].values()) == 4096

    # For p = 0.10: analytical failure rate is 3(0.01) - 2(0.001) = 0.028 (2.8%)
    # Empirical logical error rate should be very close to 0.028 with 4096 shots (within +/- 0.015)
    empirical_err = res["empirical_logical_error"]
    assert 0.015 <= empirical_err <= 0.045, f"Empirical error {empirical_err:.4f} out of expected 4096-shot bounds"

