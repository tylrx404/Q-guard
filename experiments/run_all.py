"""Batch experiment runner for Q-GUARD Strategies A, B, C and 3-Qubit Repetition Code.

Executes reproducible quantum circuit simulations, saves CSV datasets under results/,
and outputs publication-quality figures under figures/.
"""

import os
import sys
from typing import Dict, List
import pandas as pd
import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.strategies import (
    run_strategy_a_direct_measurement,
    run_strategy_b_sample_size_sweep,
    run_strategy_b_p_sweep,
    evaluate_strategy_b_fault_detector,
    compute_strategy_b_roc_curve,
    run_strategy_c_ancilla_parity,
)
from src.repetition_code import (
    run_repetition_code_simulation,
    analyze_eight_error_masks,
    evaluate_repetition_code_statevector,
    evaluate_haar_random_benchmark,
)
from src.analysis import compute_wilson_confidence_interval
from src.plots import (
    save_sample_size_convergence_plot,
    save_true_vs_estimated_p_plot,
    save_roc_curve_plot,
    save_theory_vs_empirical_plot,
    save_double_flip_miscorrection_plot,
    save_haar_random_fidelity_plot,
)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")


def run_all_experiments() -> int:
    """Execute reproducible batch experiments for Milestones 2 and 3."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=================================================================")
    print("        Q-GUARD: FULL QUANTUM SIMULATION BATCH SUITE             ")
    print("=================================================================")

    # --------------------------------------------------------------------------
    # 1. Strategy A: Direct Measurement Benchmarks
    # --------------------------------------------------------------------------
    print("\n[1/6] Running Strategy A: Known-basis state transmissions...")
    strategy_a_records = []
    test_configs = [
        ("|0>", "probabilistic", 0.0, 1000, 101),
        ("|0>", "probabilistic", 0.10, 1000, 102),
        ("|0>", "probabilistic", 0.25, 1000, 103),
        ("|0>", "probabilistic", 0.50, 1000, 104),
        ("|0>", "probabilistic", 1.0, 1000, 105),
        ("|0>", "deterministic", 1.0, 1000, 106),
        ("|1>", "probabilistic", 0.0, 1000, 201),
        ("|1>", "probabilistic", 0.15, 1000, 202),
        ("|1>", "probabilistic", 0.50, 1000, 203),
        ("|1>", "probabilistic", 1.0, 1000, 204),
        ("|1>", "deterministic", 1.0, 1000, 205),
    ]

    for state, mode, p_val, shots, seed in test_configs:
        res = run_strategy_a_direct_measurement(
            state_name=state,
            error_mode=mode,
            p=p_val,
            shots=shots,
            seed=seed,
        )
        strategy_a_records.append(
            {
                "input_state": state,
                "error_mode": mode,
                "configured_p": p_val,
                "shots": shots,
                "seed": seed,
                "shots_0": res["shots_0"],
                "shots_1": res["shots_1"],
                "observed_errors": res["observed_errors"],
                "estimated_p": res["estimated_p"],
                "lower_bound_95": res["lower_bound"],
                "upper_bound_95": res["upper_bound"],
            }
        )

    df_a = pd.DataFrame(strategy_a_records)
    path_a_csv = os.path.join(RESULTS_DIR, "strategy_a_results.csv")
    df_a.to_csv(path_a_csv, index=False)
    print(f"      Saved: {path_a_csv} ({len(df_a)} configurations)")

    # --------------------------------------------------------------------------
    # 2. Strategy B: Sample Size Convergence Sweep
    # --------------------------------------------------------------------------
    print("\n[2/6] Running Strategy B: Sample size convergence sweep...")
    sample_sizes = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    sweep_b_samples = run_strategy_b_sample_size_sweep(
        true_p=0.15,
        sample_sizes=sample_sizes,
        input_state="|0>",
        seed=303,
    )
    df_b_samples = pd.DataFrame(sweep_b_samples)
    path_b_samples_csv = os.path.join(RESULTS_DIR, "strategy_b_sample_size_sweep.csv")
    df_b_samples.to_csv(path_b_samples_csv, index=False)
    print(f"      Saved: {path_b_samples_csv} ({len(df_b_samples)} sample sizes)")

    path_fig_conv = os.path.join(FIGURES_DIR, "strategy_b_sample_size_convergence.png")
    save_sample_size_convergence_plot(sweep_b_samples, path_fig_conv)
    print(f"      Saved: {path_fig_conv}")

    # --------------------------------------------------------------------------
    # 3. Strategy B: Physical Noise Calibration Sweep (p_true vs p_hat)
    # --------------------------------------------------------------------------
    print("\n[3/6] Running Strategy B: Physical noise calibration sweep (p in [0, 0.5])...")
    p_grid = np.linspace(0.0, 0.50, 21)
    sweep_b_p = run_strategy_b_p_sweep(
        p_values=p_grid,
        shots_per_point=1000,
        input_state="|0>",
        seed=404,
    )
    df_b_p = pd.DataFrame(sweep_b_p)
    path_b_p_csv = os.path.join(RESULTS_DIR, "strategy_b_p_sweep.csv")
    df_b_p.to_csv(path_b_p_csv, index=False)
    print(f"      Saved: {path_b_p_csv} ({len(df_b_p)} p points)")

    path_fig_p = os.path.join(FIGURES_DIR, "strategy_b_true_vs_estimated_p.png")
    save_true_vs_estimated_p_plot(sweep_b_p, path_fig_p)
    print(f"      Saved: {path_fig_p}")

    # --------------------------------------------------------------------------
    # 4. Strategy B: Fault Detector & ROC Curve
    # --------------------------------------------------------------------------
    print("\n[4/6] Running Strategy B: Channel fault detector & ROC analysis...")
    np.random.seed(505)
    healthy_p = list(np.random.uniform(0.01, 0.09, 50))
    faulty_p = list(np.random.uniform(0.12, 0.35, 50))
    fleet_p = healthy_p + faulty_p

    detector_eval = evaluate_strategy_b_fault_detector(
        channel_p_list=fleet_p,
        faulty_threshold=0.10,
        decision_threshold=0.11,
        sample_size=300,
        input_state="|0>",
        seed=606,
    )

    metrics = detector_eval["metrics"]
    cm = detector_eval["confusion_matrix"]
    metrics_summary = [
        {
            "faulty_ground_truth_threshold": detector_eval["faulty_threshold"],
            "decision_threshold_tau": detector_eval["decision_threshold"],
            "sample_size_N": detector_eval["sample_size"],
            "true_positives": cm["tp"],
            "false_positives": cm["fp"],
            "true_negatives": cm["tn"],
            "false_negatives": cm["fn"],
            "precision": metrics["precision"],
            "recall_sensitivity": metrics["recall"],
            "specificity": metrics["specificity"],
            "false_positive_rate": metrics["false_positive_rate"],
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
        }
    ]
    df_metrics = pd.DataFrame(metrics_summary)
    path_metrics_csv = os.path.join(RESULTS_DIR, "strategy_b_classification_metrics.csv")
    df_metrics.to_csv(path_metrics_csv, index=False)
    print(f"      Saved: {path_metrics_csv}")

    roc_data = compute_strategy_b_roc_curve(
        channel_p_list=fleet_p,
        faulty_threshold=0.10,
        sample_size=300,
        threshold_steps=40,
        input_state="|0>",
        seed=707,
    )
    path_fig_roc = os.path.join(FIGURES_DIR, "strategy_b_roc_curve.png")
    save_roc_curve_plot(roc_data, path_fig_roc)
    print(f"      Saved: {path_fig_roc} (AUC = {roc_data['auc']:.4f})")

    # --------------------------------------------------------------------------
    # 5. Phase E: 3-Qubit Repetition Code Analytical Benchmark Sweep (4096 shots)
    # --------------------------------------------------------------------------
    print("\n[5/7] Running Phase E: 3-Qubit Repetition Code parametric sweep (p in [0.0, 0.50], 4,096 shots/pt)...")
    qec_p_grid = np.linspace(0.0, 0.50, 21)
    shots_qec = 4096
    qec_sweep_records = []

    for idx, p_val in enumerate(qec_p_grid):
        trial_seed = 800 + idx * 11

        # 1. Unprotected baseline (single qubit)
        unprotected_res = run_strategy_a_direct_measurement(
            state_name="|0>",
            error_mode="probabilistic",
            p=p_val,
            shots=shots_qec,
            seed=trial_seed,
        )
        unprotected_err = unprotected_res["observed_errors"]
        unprotected_rate = unprotected_res["estimated_p"]
        unprotected_ci = unprotected_res["wilson_ci"]

        # 2. Protected repetition code [[3, 1, 1]]
        protected_res = run_repetition_code_simulation(
            state_name="|0>",
            p=p_val,
            shots=shots_qec,
            seed=trial_seed,
        )
        protected_err = protected_res["logical_errors"]
        protected_rate = protected_res["empirical_logical_error"]
        protected_ci = compute_wilson_confidence_interval(protected_err, shots_qec, 0.95)

        # 3. Analytical theoretical formula: 3p^2 - 2p^3
        analytical_rate = 3.0 * (p_val ** 2) - 2.0 * (p_val ** 3)

        qec_sweep_records.append(
            {
                "p_physical": p_val,
                "shots": shots_qec,
                "unprotected_errors": unprotected_err,
                "unprotected_error_rate": unprotected_rate,
                "unprotected_ci_lower": unprotected_ci["lower_bound"],
                "unprotected_ci_upper": unprotected_ci["upper_bound"],
                "protected_logical_errors": protected_err,
                "protected_logical_error_rate": protected_rate,
                "protected_ci_lower": protected_ci["lower_bound"],
                "protected_ci_upper": protected_ci["upper_bound"],
                "analytical_logical_error_rate": analytical_rate,
                "error_suppression_gain": unprotected_rate - protected_rate,
            }
        )

    df_qec = pd.DataFrame(qec_sweep_records)
    path_qec_csv = os.path.join(RESULTS_DIR, "repetition_code_p_sweep.csv")
    df_qec.to_csv(path_qec_csv, index=False)
    print(f"      Saved: {path_qec_csv} ({len(df_qec)} p points, {shots_qec} shots each)")

    path_fig_qec = os.path.join(FIGURES_DIR, "repetition_code_theory_vs_experiment.png")
    save_theory_vs_empirical_plot(qec_sweep_records, path_fig_qec)
    print(f"      Saved: {path_fig_qec}")

    # --------------------------------------------------------------------------
    # 6. Phase F: Exhaustive 8-Error-Mask Evaluation (Double-Flip Miscorrection)
    # --------------------------------------------------------------------------
    print("\n[6/7] Running Phase F: Exhaustive 8-Error-Mask Evaluation...")
    mask_records = analyze_eight_error_masks(input_state="|0>")
    df_masks = pd.DataFrame(mask_records)
    path_masks_csv = os.path.join(RESULTS_DIR, "eight_error_masks_evaluation.csv")
    df_masks.to_csv(path_masks_csv, index=False)
    print(f"      Saved: {path_masks_csv} (8 masks evaluated)")

    path_fig_masks = os.path.join(FIGURES_DIR, "double_flip_miscorrection.png")
    save_double_flip_miscorrection_plot(mask_records, path_fig_masks)
    print(f"      Saved: {path_fig_masks}")

    # --------------------------------------------------------------------------
    # 7. Phase G: 200 Haar-Random Input State Fidelity Evaluation
    # --------------------------------------------------------------------------
    print("\n[7/7] Running Phase G: 200 Haar-Random Pure State Fidelity Benchmark (p=0.10)...")
    haar_records = evaluate_haar_random_benchmark(num_states=200, p=0.10, base_seed=999)
    df_haar = pd.DataFrame(haar_records)
    path_haar_csv = os.path.join(RESULTS_DIR, "haar_random_fidelity_evaluation.csv")
    df_haar.to_csv(path_haar_csv, index=False)
    print(f"      Saved: {path_haar_csv} ({len(df_haar)} random states)")

    path_fig_haar = os.path.join(FIGURES_DIR, "haar_random_fidelity_comparison.png")
    save_haar_random_fidelity_plot(haar_records, path_fig_haar)
    print(f"      Saved: {path_fig_haar}")

    mean_unprot = float(df_haar["unprotected_fidelity"].mean())
    mean_prot = float(df_haar["protected_fidelity"].mean())
    mean_gain = float(df_haar["fidelity_gain"].mean())
    print(f"      Mean Unprotected Fidelity: {mean_unprot:.4f}")
    print(f"      Mean Protected Fidelity:   {mean_prot:.4f}")
    print(f"      Mean Fidelity Gain:        +{mean_gain * 100:.2f}%")

    print("\n=================================================================")
    print("    ALL EXPERIMENTS & BENCHMARKS COMPLETED SUCCESSFULLY!        ")
    print(f"    Numerical results: {RESULTS_DIR}")
    print(f"    Figures:           {FIGURES_DIR}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_experiments())
