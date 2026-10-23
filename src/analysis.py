"""Statistical and quantum metrics analysis for Q-GUARD.

Provides Wilson score confidence intervals for binomial error proportions,
confusion matrix evaluation, classification metrics for channel fault detectors,
and distribution overlap analysis.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
from scipy import stats


COMMON_Z_SCORES: Dict[float, float] = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.98: 2.3263478740408408,
    0.99: 2.5758293035489004,
}


def get_z_score(confidence_level: float) -> float:
    """Return standard normal quantile for a given two-tailed confidence level."""
    rounded = round(confidence_level, 4)
    if rounded in COMMON_Z_SCORES:
        return COMMON_Z_SCORES[rounded]
    alpha = 1.0 - confidence_level
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def compute_wilson_confidence_interval(
    k: int,
    n: int,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    The Wilson score interval has superior coverage properties for small sample sizes
    and extreme proportions (p near 0 or 1) compared to the standard Wald normal
    approximation interval.

    Args:
        k: Observed number of successes / errors (integer, 0 <= k <= n).
        n: Total number of trials / shots (integer, n >= 1).
        confidence_level: Desired coverage probability (default: 0.95).

    Returns:
        Dictionary containing:
            - 'point_estimate': Observed proportion k / n
            - 'lower_bound': Lower confidence bound in [0.0, 1.0]
            - 'upper_bound': Upper confidence bound in [0.0, 1.0]
            - 'confidence_level': Nominal confidence level
            - 'k': Observed errors
            - 'n': Total trials

    Raises:
        TypeError: If k or n are not integers.
        ValueError: If n < 1, k < 0, k > n, or confidence_level not in (0, 1).
    """
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"Sample size n must be at least 1, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must be between 0 and n ({n}), got {k}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    z = get_z_score(confidence_level)
    p_hat = float(k) / float(n)
    z_sq = z * z

    # Numerically stable Wilson formulation
    denominator = 1.0 + z_sq / float(n)
    center = (p_hat + z_sq / (2.0 * float(n))) / denominator
    spread = (z / denominator) * math.sqrt(
        (p_hat * (1.0 - p_hat)) / float(n) + z_sq / (4.0 * float(n) * float(n))
    )

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    # Clean boundary clamp for exact zero or full errors
    if k == 0:
        lower = 0.0
    if k == n:
        upper = 1.0

    return {
        "point_estimate": p_hat,
        "lower_bound": lower,
        "upper_bound": upper,
        "confidence_level": confidence_level,
        "k": int(k),
        "n": int(n),
    }


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive_label: int = 1,
) -> Dict[str, int]:
    """Compute binary confusion matrix counts from ground truth and predictions.

    Positive Class Definition:
        Instances with value == positive_label are considered POSITIVE (e.g. faulty channel).
        Instances with value != positive_label are considered NEGATIVE (e.g. healthy channel).

    Args:
        y_true: Sequence of true binary labels (0 or 1).
        y_pred: Sequence of predicted binary labels (0 or 1).
        positive_label: Label designating the positive condition (default: 1).

    Returns:
        Dictionary with keys 'tp', 'fp', 'tn', 'fn', 'total'.

    Raises:
        ValueError: If sequences have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("Cannot compute confusion matrix for empty sequences.")

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for actual, predicted in zip(y_true, y_pred):
        is_actual_pos = (actual == positive_label)
        is_pred_pos = (predicted == positive_label)

        if is_actual_pos and is_pred_pos:
            tp += 1
        elif not is_actual_pos and is_pred_pos:
            fp += 1
        elif not is_actual_pos and not is_pred_pos:
            tn += 1
        elif is_actual_pos and not is_pred_pos:
            fn += 1

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": len(y_true)}


def compute_classification_metrics(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> Dict[str, float]:
    """Calculate standard binary diagnostic metrics avoiding division by zero.

    Args:
        tp: True positives count.
        fp: False positives count.
        tn: True negatives count.
        fn: False negatives count.

    Returns:
        Dictionary of metrics: precision, recall (TPR), false_positive_rate (FPR),
        specificity (TNR), accuracy, f1_score.
    """
    total = tp + fp + tn + fn
    if total == 0:
        raise ValueError("Total population count cannot be zero.")

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = float(tp + tn) / float(total)
    f1_score = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def compute_fidelity(
    ideal_counts: Dict[str, int],
    noisy_counts: Dict[str, int],
) -> float:
    """Compute classical statistical fidelity (Bhattacharyya coefficient) between measurement distributions.

    Note on Error Metrics:
    In classical detection strategies (Strategy A & B), we measure in the computational basis,
    and fidelity here represents classical probability overlap:
        BC(P_ideal, P_noisy) = sum_{x} sqrt(P_ideal(x) * P_noisy(x))
    This is distinct from true quantum state fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    or logical fidelity in quantum error correction (repetition code), which will be modeled
    in future milestones.

    Args:
        ideal_counts: Measurement counts from ideal/noiseless circuit.
        noisy_counts: Measurement counts from noisy circuit.

    Returns:
        Classical fidelity in [0.0, 1.0].
    """
    total_ideal = sum(ideal_counts.values())
    total_noisy = sum(noisy_counts.values())

    if total_ideal == 0 or total_noisy == 0:
        return 0.0

    all_keys = set(ideal_counts.keys()).union(set(noisy_counts.keys()))
    bc = 0.0
    for k in all_keys:
        p_ideal = float(ideal_counts.get(k, 0)) / float(total_ideal)
        p_noisy = float(noisy_counts.get(k, 0)) / float(total_noisy)
        bc += math.sqrt(p_ideal * p_noisy)

    return max(0.0, min(1.0, float(bc)))
