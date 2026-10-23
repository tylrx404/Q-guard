"""Plotting and visualization routines for Q-GUARD.

Provides both Plotly interactive figures for the Streamlit research dashboard
and Matplotlib static figure exporters for reproducible batch experiment artifacts.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Q-GUARD Design System Color Tokens
COLOR_BG_BASE = "#0A0F1D"
COLOR_BG_SURFACE = "#111827"
COLOR_BORDER = "#334155"
COLOR_TEXT_PRIMARY = "#F8FAFC"
COLOR_TEXT_SECONDARY = "#94A3B8"
COLOR_ACCENT_CYAN = "#06B6D4"
COLOR_ACCENT_PURPLE = "#8B5CF6"
COLOR_STATUS_SUCCESS = "#10B981"
COLOR_STATUS_WARNING = "#F59E0B"
COLOR_STATUS_DANGER = "#EF4444"


# ==============================================================================
# PLOTLY INTERACTIVE BUILDERS (FOR STREAMLIT)
# ==============================================================================
def build_counts_histogram_plotly(
    counts: Dict[str, int],
    expected_bit: str = "0",
    title: str = "Measurement Shot Distribution (Z Basis)",
) -> go.Figure:
    """Build high-contrast Plotly bar chart of measurement counts."""
    keys = sorted(counts.keys())
    values = [counts[k] for k in keys]
    total = sum(values)

    colors = []
    for k in keys:
        if k == expected_bit:
            colors.append(COLOR_ACCENT_CYAN)
        else:
            colors.append(COLOR_STATUS_WARNING if expected_bit != "N/A" else COLOR_ACCENT_PURPLE)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"|{k}⟩" for k in keys],
            y=values,
            text=[f"{v} ({v/total*100:.1f}%)" if total > 0 else "0" for v in values],
            textposition="auto",
            marker_color=colors,
            marker_line=dict(color=COLOR_BORDER, width=1),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=COLOR_TEXT_PRIMARY, size=15)),
        paper_bgcolor=COLOR_BG_SURFACE,
        plot_bgcolor=COLOR_BG_SURFACE,
        font=dict(color=COLOR_TEXT_SECONDARY, family="Inter, sans-serif"),
        xaxis=dict(
            title="Computational Basis State",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            title="Recorded Shot Frequency",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def build_sample_size_convergence_plotly(
    sweep_data: List[Dict[str, Any]],
) -> go.Figure:
    """Build Plotly chart showing error estimation convergence across sample sizes."""
    df = pd.DataFrame(sweep_data)
    true_p = df["true_p"].iloc[0] if not df.empty else 0.0

    fig = go.Figure()

    # 95% Confidence Interval Band
    fig.add_trace(
        go.Scatter(
            x=list(df["sample_size"]) + list(df["sample_size"][::-1]),
            y=list(df["upper_bound"]) + list(df["lower_bound"][::-1]),
            fill="toself",
            fillcolor="rgba(6, 182, 212, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Wilson CI",
        )
    )

    # Theoretical True Value Line
    fig.add_trace(
        go.Scatter(
            x=df["sample_size"],
            y=[true_p] * len(df),
            mode="lines",
            line=dict(color=COLOR_ACCENT_PURPLE, width=2, dash="dash"),
            name=f"True Error Rate p = {true_p:.3f} (Theoretical)",
        )
    )

    # Empirical Estimates
    fig.add_trace(
        go.Scatter(
            x=df["sample_size"],
            y=df["estimated_p"],
            mode="lines+markers",
            marker=dict(size=7, color=COLOR_ACCENT_CYAN),
            line=dict(color=COLOR_ACCENT_CYAN, width=2),
            name="Empirical Estimate p̂",
        )
    )

    fig.update_layout(
        title=dict(
            text="Strategy B: Estimation Convergence vs Sample Size N",
            font=dict(color=COLOR_TEXT_PRIMARY, size=15),
        ),
        paper_bgcolor=COLOR_BG_SURFACE,
        plot_bgcolor=COLOR_BG_SURFACE,
        font=dict(color=COLOR_TEXT_SECONDARY, family="Inter, sans-serif"),
        xaxis=dict(
            title="Sample Size N (Shots)",
            type="log",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            title="Estimated Error Probability p̂",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            range=[0.0, min(1.0, max(0.5, true_p * 2.0 + 0.1))],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return fig


def build_true_vs_estimated_p_plotly(
    sweep_data: List[Dict[str, Any]],
) -> go.Figure:
    """Build Plotly chart showing true p vs estimated p with error bars."""
    df = pd.DataFrame(sweep_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color=COLOR_TEXT_SECONDARY, width=1.5, dash="dot"),
            name="Ideal Identity p̂ = p_true (Theoretical)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(df["true_p"]) + list(df["true_p"][::-1]),
            y=list(df["upper_bound"]) + list(df["lower_bound"][::-1]),
            fill="toself",
            fillcolor="rgba(6, 182, 212, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Wilson CI",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["true_p"],
            y=df["estimated_p"],
            mode="lines+markers",
            marker=dict(size=7, color=COLOR_ACCENT_CYAN),
            line=dict(color=COLOR_ACCENT_CYAN, width=2),
            name="Empirical Estimator p̂",
        )
    )

    fig.update_layout(
        title=dict(
            text="Strategy B: Physical Noise Calibration (True p vs Estimated p̂)",
            font=dict(color=COLOR_TEXT_PRIMARY, size=15),
        ),
        paper_bgcolor=COLOR_BG_SURFACE,
        plot_bgcolor=COLOR_BG_SURFACE,
        font=dict(color=COLOR_TEXT_SECONDARY, family="Inter, sans-serif"),
        xaxis=dict(
            title="Configured True Physical Probability p",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            range=[-0.02, 1.02],
        ),
        yaxis=dict(
            title="Estimated Probability p̂",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            range=[-0.02, 1.02],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return fig


def build_roc_curve_plotly(roc_data: Dict[str, Any]) -> go.Figure:
    """Build Plotly chart showing ROC curve for the statistical fault detector."""
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc = roc_data["auc"]
    n = roc_data.get("sample_size", "N")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color=COLOR_BORDER, width=1.5, dash="dash"),
            name="Random Classifier (AUC = 0.50)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines+markers",
            marker=dict(size=5, color=COLOR_ACCENT_CYAN),
            line=dict(color=COLOR_ACCENT_CYAN, width=2.5),
            name=f"Detector ROC (N={n}, AUC = {auc:.3f})",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Strategy B: Fault Detector ROC Curve (AUC = {auc:.3f})",
            font=dict(color=COLOR_TEXT_PRIMARY, size=15),
        ),
        paper_bgcolor=COLOR_BG_SURFACE,
        plot_bgcolor=COLOR_BG_SURFACE,
        font=dict(color=COLOR_TEXT_SECONDARY, family="Inter, sans-serif"),
        xaxis=dict(
            title="False Positive Rate (1 - Specificity)",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            range=[-0.02, 1.02],
        ),
        yaxis=dict(
            title="True Positive Rate (Recall / Sensitivity)",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            range=[-0.02, 1.02],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return fig


def build_repetition_code_syndrome_plotly(
    syndrome_counts: Dict[str, int],
    title: str = "Syndrome Extraction Histogram (s01 s12)",
) -> go.Figure:
    """Build Plotly bar chart displaying 4 syndrome outcomes for the 3-qubit repetition code."""
    syndromes = ["00", "10", "11", "01"]
    labels = [
        "00 (No Error)",
        "10 (Flip Q0)",
        "11 (Flip Q1)",
        "01 (Flip Q2)",
    ]
    counts = [syndrome_counts.get(s, 0) for s in syndromes]
    total = sum(counts)

    colors = [
        COLOR_STATUS_SUCCESS,
        COLOR_ACCENT_CYAN,
        COLOR_ACCENT_CYAN,
        COLOR_ACCENT_CYAN,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=counts,
            text=[f"{c} ({c/total*100:.1f}%)" if total > 0 else "0" for c in counts],
            textposition="auto",
            marker_color=colors,
            marker_line=dict(color=COLOR_BORDER, width=1),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=COLOR_TEXT_PRIMARY, size=15)),
        paper_bgcolor=COLOR_BG_SURFACE,
        plot_bgcolor=COLOR_BG_SURFACE,
        font=dict(color=COLOR_TEXT_SECONDARY, family="Inter, sans-serif"),
        xaxis=dict(
            title="Diagnosed Error Syndrome (s01 s12)",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            title="Recorded Syndrome Counts",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def build_theory_vs_empirical_plotly(
    sweep_data: List[Dict[str, Any]],
) -> go.Figure:
    """Build Plotly comparison chart between unprotected and 3-qubit repetition code."""
    df = pd.DataFrame(sweep_data)

    fig = go.Figure()

    # 1. Unprotected Baseline
    fig.add_trace(
        go.Scatter(
            x=df["p_physical"],
            y=df["unprotected_error_rate"],
            mode="lines+markers",
            marker=dict(size=6, color=COLOR_STATUS_DANGER),
            line=dict(color=COLOR_STATUS_DANGER, width=2, dash="dash"),
            name="Unprotected Physical Qubit (Baseline)",
        )
    )

    # 2. Analytical Theory Curve
    fig.add_trace(
        go.Scatter(
            x=df["p_physical"],
            y=df["analytical_logical_error_rate"],
            mode="lines",
            line=dict(color=COLOR_ACCENT_PURPLE, width=2.5),
            name="Theoretical [[3,1,1]] Scaling (3p² - 2p³)",
        )
    )

    # 3. Confidence Band
    fig.add_trace(
        go.Scatter(
            x=list(df["p_physical"]) + list(df["p_physical"][::-1]),
            y=list(df["protected_ci_upper"]) + list(df["protected_ci_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(6, 182, 212, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Wilson CI (Empirical QEC)",
        )
    )

    # 4. Empirical Repetition Code
    fig.add_trace(
        go.Scatter(
            x=df["p_physical"],
            y=df["protected_logical_error_rate"],
            mode="lines+markers",
            marker=dict(size=7, color=COLOR_ACCENT_CYAN),
            line=dict(color=COLOR_ACCENT_CYAN, width=2),
            name="Empirical [[3,1,1]] Code (Aer Simulator)",
        )
    )

    fig.add_vline(
        x=0.5,
        line_width=1.5,
        line_dash="dot",
        line_color=COLOR_STATUS_WARNING,
        annotation_text="Breakeven Threshold (p = 0.50)",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_STATUS_WARNING, size=11),
    )

    fig.update_layout(
        title=dict(
            text="3-Qubit Repetition Code: Logical Failure Probability vs Physical Noise p",
            font=dict(color=COLOR_TEXT_PRIMARY, size=15),
        ),
        paper_bgcolor=COLOR_BG_SURFACE,
        plot_bgcolor=COLOR_BG_SURFACE,
        font=dict(color=COLOR_TEXT_SECONDARY, family="Inter, sans-serif"),
        xaxis=dict(
            title="Physical Bit-Flip Error Probability p",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            range=[-0.02, 0.52],
        ),
        yaxis=dict(
            title="Logical Error Probability P_L",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            range=[-0.02, 0.55],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=400,
    )
    return fig


def build_haar_random_fidelity_plotly(
    haar_data: List[Dict[str, Any]],
    title: str = "Haar-Random State Ensemble: Unprotected vs [[3, 1, 1]] Protected Fidelity",
) -> go.Figure:
    """Build interactive Plotly chart comparing unprotected vs repetition-code protected fidelity."""
    df = pd.DataFrame(haar_data).sort_values("unprotected_fidelity").reset_index(drop=True)
    mean_unprot = float(df["unprotected_fidelity"].mean())
    mean_prot = float(df["protected_fidelity"].mean())
    mean_gain = float(df["fidelity_gain"].mean())

    fig = go.Figure()

    # Gain ribbon between unprotected and protected
    fig.add_trace(
        go.Scatter(
            x=list(df.index) + list(df.index[::-1]),
            y=list(df["protected_fidelity"]) + list(df["unprotected_fidelity"][::-1]),
            fill="toself",
            fillcolor="rgba(6, 182, 212, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=f"Fidelity Gain Band (Mean Δ = +{mean_gain:.3f})",
        )
    )

    # Unprotected curve
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["unprotected_fidelity"],
            mode="lines+markers",
            marker=dict(size=4, color=COLOR_STATUS_DANGER),
            line=dict(color=COLOR_STATUS_DANGER, width=1.5, dash="dash"),
            name=f"Unprotected Channel (Mean = {mean_unprot:.3f})",
            hovertemplate="State #%{x}<br>Unprotected F: %{y:.4f}<extra></extra>",
        )
    )

    # Protected curve
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["protected_fidelity"],
            mode="lines+markers",
            marker=dict(size=5, color=COLOR_ACCENT_CYAN),
            line=dict(color=COLOR_ACCENT_CYAN, width=2),
            name=f"Protected [[3,1,1]] (Mean = {mean_prot:.3f})",
            hovertemplate="State #%{x}<br>Protected F: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=COLOR_TEXT_PRIMARY, size=15)),
        paper_bgcolor=COLOR_BG_SURFACE,
        plot_bgcolor=COLOR_BG_SURFACE,
        font=dict(color=COLOR_TEXT_SECONDARY, family="Inter, sans-serif"),
        xaxis=dict(
            title="Random Pure States (Sorted by Baseline Susceptibility)",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            title="State Fidelity F(ρ, |ψ⟩)",
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            range=[max(0.0, float(df["unprotected_fidelity"].min()) - 0.05), 1.01],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=400,
    )
    return fig


# ==============================================================================
# MATPLOTLIB DISK EXPORTERS (FOR BATCH RUNS)
# ==============================================================================
def save_sample_size_convergence_plot(
    sweep_data: List[Dict[str, Any]],
    output_path: str = "figures/strategy_b_sample_size_convergence.png",
) -> str:
    """Generate and save publication-quality convergence figure to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(sweep_data)
    true_p = df["true_p"].iloc[0]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG_SURFACE)
    ax.set_facecolor(COLOR_BG_SURFACE)

    ax.fill_between(
        df["sample_size"],
        df["lower_bound"],
        df["upper_bound"],
        color=COLOR_ACCENT_CYAN,
        alpha=0.2,
        label="95% Wilson Confidence Interval",
    )

    ax.axhline(
        y=true_p,
        color=COLOR_ACCENT_PURPLE,
        linestyle="--",
        linewidth=1.8,
        label=f"True Error Rate p = {true_p:.3f} (Theoretical)",
    )

    ax.plot(
        df["sample_size"],
        df["estimated_p"],
        color=COLOR_ACCENT_CYAN,
        marker="o",
        markersize=5,
        linewidth=1.8,
        label="Empirical Estimate p̂",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Sample Size N (Shots)", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylabel("Estimated Error Probability p̂", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_title("Strategy B: Estimation Convergence vs Sample Size N", color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.grid(True, linestyle=":", alpha=0.4, color=COLOR_BORDER)
    ax.legend(facecolor=COLOR_BG_BASE, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_PRIMARY)

    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def save_true_vs_estimated_p_plot(
    sweep_data: List[Dict[str, Any]],
    output_path: str = "figures/strategy_b_true_vs_estimated_p.png",
) -> str:
    """Generate and save true p vs estimated p calibration figure to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(sweep_data)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG_SURFACE)
    ax.set_facecolor(COLOR_BG_SURFACE)

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=COLOR_TEXT_SECONDARY,
        linestyle=":",
        linewidth=1.5,
        label="Identity Reference p̂ = p_true (Theoretical)",
    )

    ax.fill_between(
        df["true_p"],
        df["lower_bound"],
        df["upper_bound"],
        color=COLOR_ACCENT_CYAN,
        alpha=0.2,
        label="95% Wilson CI",
    )

    ax.plot(
        df["true_p"],
        df["estimated_p"],
        color=COLOR_ACCENT_CYAN,
        marker="s",
        markersize=5,
        linewidth=1.8,
        label="Empirical Calibration p̂",
    )

    ax.set_xlabel("Configured Physical Probability p", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylabel("Empirically Estimated Probability p̂", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_title("Strategy B: True Error Probability vs Estimated Rate", color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.grid(True, linestyle=":", alpha=0.4, color=COLOR_BORDER)
    ax.legend(facecolor=COLOR_BG_BASE, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_PRIMARY)

    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def save_roc_curve_plot(
    roc_data: Dict[str, Any],
    output_path: str = "figures/strategy_b_roc_curve.png",
) -> str:
    """Generate and save ROC curve plot to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc = roc_data["auc"]
    n = roc_data.get("sample_size", 200)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor(COLOR_BG_SURFACE)
    ax.set_facecolor(COLOR_BG_SURFACE)

    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_BORDER, label="Random Guess (AUC = 0.50)")
    ax.plot(fpr, tpr, color=COLOR_ACCENT_CYAN, marker="o", markersize=4, linewidth=2, label=f"Fault Detector (N={n}, AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_title(f"Strategy B: Fault Detector ROC Curve (AUC = {auc:.3f})", color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.grid(True, linestyle=":", alpha=0.4, color=COLOR_BORDER)
    ax.legend(facecolor=COLOR_BG_BASE, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_PRIMARY, loc="lower right")

    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def save_theory_vs_empirical_plot(
    sweep_data: List[Dict[str, Any]],
    output_path: str = "figures/repetition_code_theory_vs_experiment.png",
) -> str:
    """Generate and save publication figure comparing theoretical vs empirical QEC scaling."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(sweep_data)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG_SURFACE)
    ax.set_facecolor(COLOR_BG_SURFACE)

    ax.plot(
        df["p_physical"],
        df["unprotected_error_rate"],
        color=COLOR_STATUS_DANGER,
        linestyle="--",
        linewidth=1.8,
        label="Unprotected Single Qubit (p_L = p)",
    )

    ax.plot(
        df["p_physical"],
        df["analytical_logical_error_rate"],
        color=COLOR_ACCENT_PURPLE,
        linestyle="-",
        linewidth=2.2,
        label="Theoretical [[3,1,1]] Scaling (3p² - 2p³)",
    )

    ax.fill_between(
        df["p_physical"],
        df["protected_ci_lower"],
        df["protected_ci_upper"],
        color=COLOR_ACCENT_CYAN,
        alpha=0.2,
        label="95% Wilson Confidence Band",
    )

    ax.plot(
        df["p_physical"],
        df["protected_logical_error_rate"],
        color=COLOR_ACCENT_CYAN,
        marker="o",
        markersize=5,
        linewidth=1.8,
        label="Empirical [[3,1,1]] Repetition Code",
    )

    ax.axvline(x=0.5, color=COLOR_STATUS_WARNING, linestyle=":", linewidth=1.5, label="Breakeven Threshold (p = 0.50)")

    ax.set_xlabel("Physical Bit-Flip Probability p", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylabel("Logical Error Probability P_L", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_title("Q-GUARD: 3-Qubit Repetition Code Suppression vs Physical Noise", color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.grid(True, linestyle=":", alpha=0.4, color=COLOR_BORDER)
    ax.legend(facecolor=COLOR_BG_BASE, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_PRIMARY, loc="upper left")

    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def save_double_flip_miscorrection_plot(
    mask_data: List[Dict[str, Any]],
    output_path: str = "figures/double_flip_miscorrection.png",
) -> str:
    """Generate and save visual bar chart of all 8 error masks demonstrating miscorrection."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(mask_data)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG_SURFACE)
    ax.set_facecolor(COLOR_BG_SURFACE)

    colors = []
    for w in df["weight"]:
        if w == 0:
            colors.append(COLOR_STATUS_SUCCESS)
        elif w == 1:
            colors.append(COLOR_ACCENT_CYAN)
        elif w == 2:
            colors.append(COLOR_STATUS_WARNING)
        else:
            colors.append(COLOR_STATUS_DANGER)

    bars = ax.bar(
        [f"Mask {m}\n(w={w})" for m, w in zip(df["mask"], df["weight"])],
        df["logical_fidelity"],
        color=colors,
        edgecolor=COLOR_BORDER,
        width=0.6,
    )

    ax.set_ylabel("Recovered Logical Fidelity F", color=COLOR_TEXT_PRIMARY, fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Exhaustive Evaluation of 8 Error Masks: Single-Correction vs Double-Flip Miscorrection", color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, color=COLOR_BORDER)

    for bar, syn in zip(bars, df["syndrome"]):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.04,
            f"Syn: {syn}",
            ha="center",
            va="bottom",
            color=COLOR_TEXT_PRIMARY,
            fontsize=9,
            fontfamily="monospace",
        )

    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def save_haar_random_fidelity_plot(
    haar_data: List[Dict[str, Any]],
    output_path: str = "figures/haar_random_fidelity_comparison.png",
) -> str:
    """Generate and save publication figure comparing unprotected vs protected fidelities on Haar-random states."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(haar_data).sort_values("unprotected_fidelity").reset_index(drop=True)
    mean_unprot = float(df["unprotected_fidelity"].mean())
    mean_prot = float(df["protected_fidelity"].mean())
    mean_gain = float(df["fidelity_gain"].mean())
    n_states = len(df)
    p_phys = float(df["physical_p"].iloc[0]) if not df.empty else 0.10

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300, gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(COLOR_BG_SURFACE)

    # Panel 1: State-by-State Fidelity Comparison
    ax1.set_facecolor(COLOR_BG_SURFACE)
    ax1.fill_between(
        df.index,
        df["unprotected_fidelity"],
        df["protected_fidelity"],
        color=COLOR_ACCENT_CYAN,
        alpha=0.18,
        label=f"Fidelity Recovery Gap (Mean +{mean_gain*100:.1f}%)",
    )
    ax1.plot(
        df.index,
        df["unprotected_fidelity"],
        color=COLOR_STATUS_DANGER,
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        label=f"Unprotected Single Qubit (Mean = {mean_unprot:.3f})",
    )
    ax1.plot(
        df.index,
        df["protected_fidelity"],
        color=COLOR_ACCENT_CYAN,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label=f"[[3,1,1]] QEC Protected (Mean = {mean_prot:.3f})",
    )

    ax1.axhline(y=mean_unprot, color=COLOR_STATUS_DANGER, linestyle=":", alpha=0.6)
    ax1.axhline(y=mean_prot, color=COLOR_ACCENT_CYAN, linestyle=":", alpha=0.6)

    ax1.set_xlabel(f"{n_states} Haar-Random States (Sorted by Baseline Susceptibility)", color=COLOR_TEXT_PRIMARY, fontsize=10)
    ax1.set_ylabel("Pure State Fidelity F(ρ, |ψ⟩)", color=COLOR_TEXT_PRIMARY, fontsize=10)
    ax1.set_title(f"A. Logical Fidelity Across {n_states} Random States (p = {p_phys:.2f})", color=COLOR_TEXT_PRIMARY, fontsize=11, pad=10)
    ax1.grid(True, linestyle=":", alpha=0.4, color=COLOR_BORDER)
    ax1.legend(facecolor=COLOR_BG_BASE, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_PRIMARY, loc="lower right", fontsize=9)
    for spine in ax1.spines.values():
        spine.set_color(COLOR_BORDER)
    ax1.tick_params(colors=COLOR_TEXT_SECONDARY)

    # Panel 2: Distribution Comparison (Histograms)
    ax2.set_facecolor(COLOR_BG_SURFACE)
    bins = np.linspace(float(df["unprotected_fidelity"].min()) - 0.01, 1.005, 25)
    ax2.hist(
        df["unprotected_fidelity"],
        bins=bins,
        color=COLOR_STATUS_DANGER,
        alpha=0.55,
        edgecolor=COLOR_BORDER,
        label=f"Unprotected (μ={mean_unprot:.3f}, σ={df['unprotected_fidelity'].std():.3f})",
    )
    ax2.hist(
        df["protected_fidelity"],
        bins=bins,
        color=COLOR_ACCENT_CYAN,
        alpha=0.7,
        edgecolor=COLOR_BORDER,
        label=f"Protected (μ={mean_prot:.3f}, σ={df['protected_fidelity'].std():.3f})",
    )
    ax2.set_xlabel("Fidelity Metric F", color=COLOR_TEXT_PRIMARY, fontsize=10)
    ax2.set_ylabel("State Count", color=COLOR_TEXT_PRIMARY, fontsize=10)
    ax2.set_title("B. Empirical Fidelity Distributions", color=COLOR_TEXT_PRIMARY, fontsize=11, pad=10)
    ax2.grid(True, linestyle=":", alpha=0.4, color=COLOR_BORDER)
    ax2.legend(facecolor=COLOR_BG_BASE, edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT_PRIMARY, loc="upper left", fontsize=9)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_BORDER)
    ax2.tick_params(colors=COLOR_TEXT_SECONDARY)

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path
