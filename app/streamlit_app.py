"""Q-GUARD: Quantum Bit-Flip Detection and Error-Correction Lab.

Main Streamlit Application Shell with Integrated Quantum Simulation Core
and 3-Qubit Quantum Error Correction Suite (Milestone 3).
"""

import os
import sys
from typing import Any, Dict, List, Optional
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is on sys.path for robust absolute imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.strategies import (
    run_strategy_a_direct_measurement,
    run_strategy_b_sample_size_sweep,
    evaluate_strategy_b_fault_detector,
    compute_strategy_b_roc_curve,
    run_strategy_c_ancilla_parity,
    get_strategy_comparison_table,
)
from src.repetition_code import (
    run_repetition_code_simulation,
    evaluate_repetition_code_statevector,
    analyze_eight_error_masks,
    compute_repetition_code_theory_curve,
    evaluate_haar_random_benchmark,
    SYNDROME_MAP,
)
from src.analysis import compute_fidelity, compute_wilson_confidence_interval
from src.plots import (
    build_counts_histogram_plotly,
    build_sample_size_convergence_plotly,
    build_roc_curve_plotly,
    build_repetition_code_syndrome_plotly,
    build_theory_vs_empirical_plotly,
    build_haar_random_fidelity_plotly,
)


# ==============================================================================
# PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Q-GUARD | Quantum Error Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS (Aligned with docs/UI_SPEC.md)
CUSTOM_CSS = """
<style>
/* Base typography & palette */
:root {
    --bg-base: #0A0F1D;
    --bg-surface: #111827;
    --bg-surface-elevated: #1E293B;
    --border-subtle: #334155;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --accent-cyan: #06B6D4;
    --accent-purple: #8B5CF6;
    --status-success: #10B981;
    --status-warning: #F59E0B;
    --status-danger: #EF4444;
}

/* App Background override */
.stApp {
    background-color: var(--bg-base);
    color: var(--text-primary);
}

/* Clean Brand Header */
.brand-header {
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
}

.brand-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.brand-title .brand-accent {
    color: var(--accent-cyan);
}

.brand-subtitle {
    font-size: 0.92rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
}

.brand-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 4px;
    background-color: rgba(6, 182, 212, 0.15);
    color: var(--accent-cyan);
    border: 1px solid rgba(6, 182, 212, 0.3);
}

/* Q-Card Container */
.q-card {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}

.q-card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Metric Cards */
.q-metric-card {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    text-align: left;
}

.q-metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}

.q-metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: var(--text-primary);
    line-height: 1.1;
}

.q-metric-subtext {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 0.4rem;
}

/* Honest Empty State Container */
.q-empty-state {
    background-color: rgba(17, 24, 39, 0.6);
    border: 1px dashed var(--border-subtle);
    border-radius: 8px;
    padding: 2.25rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.q-empty-icon {
    font-size: 2rem;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
}

.q-empty-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
}

.q-empty-desc {
    font-size: 0.88rem;
    color: var(--text-secondary);
    max-width: 520px;
    margin: 0 auto;
}

/* Status Pill */
.q-status-idle {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.8rem;
    background-color: rgba(148, 163, 184, 0.1);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.q-status-active {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.8rem;
    background-color: rgba(16, 185, 129, 0.15);
    color: var(--status-success);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.q-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: currentColor;
}

/* Code & Formula blocks */
.q-formula {
    background-color: var(--bg-surface-elevated);
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-family: monospace;
    border-left: 3px solid var(--accent-cyan);
    margin: 0.75rem 0;
    color: var(--text-primary);
}

.q-circuit-block {
    background-color: #0D1322;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 1rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    overflow-x: auto;
    white-space: pre;
    color: #38BDF8;
}

.q-syndrome-badge {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.1em;
}

.q-badge-00 { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
.q-badge-err { background: rgba(6, 182, 212, 0.2); color: #06B6D4; border: 1px solid #06B6D4; }
.q-badge-danger { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "experiment_status" not in st.session_state:
    st.session_state["experiment_status"] = "No experiment run yet"
if "shots" not in st.session_state:
    st.session_state["shots"] = 1024
if "seed" not in st.session_state:
    st.session_state["seed"] = 42
if "error_injection_result" not in st.session_state:
    st.session_state["error_injection_result"] = None
if "strategy_b_sweep" not in st.session_state:
    st.session_state["strategy_b_sweep"] = None
if "strategy_b_detector" not in st.session_state:
    st.session_state["strategy_b_detector"] = None
if "strategy_c_result" not in st.session_state:
    st.session_state["strategy_c_result"] = None
if "repetition_code_result" not in st.session_state:
    st.session_state["repetition_code_result"] = None
if "theory_sweep_results" not in st.session_state:
    st.session_state["theory_sweep_results"] = None


# ==============================================================================
# UI COMPONENTS & HELPERS
# ==============================================================================
def render_header(title: str, subtitle: str, category: str = "LAB MODULE") -> None:
    """Render standardized section header."""
    is_idle = st.session_state["experiment_status"] == "No experiment run yet"
    status_class = "q-status-idle" if is_idle else "q-status-active"

    st.markdown(
        f"""
        <div class="brand-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <span class="brand-badge">{category}</span>
                    <h1 class="brand-title" style="margin-top: 0.4rem;">{title}</h1>
                    <div class="brand-subtitle">{subtitle}</div>
                </div>
                <div class="{status_class}">
                    <div class="q-status-dot"></div>
                    <span>{st.session_state['experiment_status']}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_empty_state(label: str, subtext: str = "Awaiting simulation") -> None:
    """Render honest empty state metric card without placeholder numbers."""
    st.markdown(
        f"""
        <div class="q-metric-card">
            <div class="q-metric-label">{label}</div>
            <div class="q-metric-value">—</div>
            <div class="q-metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str, is_highlight: bool = False, color: Optional[str] = None) -> None:
    """Render populated real metric card."""
    if color:
        val_color = color
    elif is_highlight:
        val_color = "var(--accent-cyan)"
    else:
        val_color = "var(--text-primary)"

    st.markdown(
        f"""
        <div class="q-metric-card">
            <div class="q-metric-label">{label}</div>
            <div class="q-metric-value" style="color: {val_color};">{value}</div>
            <div class="q-metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_empty_state(title: str, description: str) -> None:
    """Render clean, honest empty state placeholder for plots and visualizations."""
    st.markdown(
        f"""
        <div class="q-empty-state">
            <div class="q-empty-icon">📊</div>
            <div class="q-empty-title">{title}</div>
            <div class="q-empty-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION RENDERERS
# ==============================================================================
def render_overview() -> None:
    """Render the Overview section introducing Q-GUARD architecture."""
    render_header(
        title="Overview & Architecture",
        subtitle="Quantum bit-flip error detection, syndrome diagnosis, and 3-qubit repetition code testbed.",
        category="SYSTEM OVERVIEW",
    )

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">What is Q-GUARD?</div>
                <p style="color: var(--text-secondary); line-height: 1.6;">
                    Physical quantum bits are susceptible to environmental coupling and imperfect control pulses.
                    Among the most fundamental error mechanisms is the <b>bit-flip channel</b>, represented by
                    the Pauli operator $\\sigma_x = X$. An uncontrolled bit-flip inverts computational basis states:
                </p>
                <div class="q-formula">
                    X |0⟩ = |1⟩ &nbsp;&nbsp;&nbsp;&nbsp; X |1⟩ = |0⟩
                </div>
                <p style="color: var(--text-secondary); line-height: 1.6;">
                    <b>Q-GUARD</b> provides a laboratory to study how bit-flip errors corrupt arbitrary quantum states,
                    how ancilla-assisted syndrome measurements isolate errors without collapsing user information,
                    and how the canonical <b>[[3, 1, 1]] 3-qubit repetition code</b> actively restores state fidelity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Complete Experimental Pipeline</div>
                <ol style="color: var(--text-secondary); line-height: 1.8; margin-left: 1.25rem;">
                    <li><b>State Preparation:</b> Initialize target states (|0⟩, |1⟩, |+⟩, |−⟩, or Haar-random pure states).</li>
                    <li><b>Encoding:</b> Entangle data qubit into 3 physical qubits: $\\alpha|000\\rangle + \\beta|111\\rangle$.</li>
                    <li><b>Noise Channel:</b> Transmit through stochastic Pauli X noise ($p$) or deterministic error masks.</li>
                    <li><b>Ancilla Parity Extraction:</b> Non-destructively measure stabilizers $Z \\otimes Z \\otimes I$ and $I \\otimes Z \\otimes Z$.</li>
                    <li><b>Active CCX Recovery:</b> Apply conditioned correction unitaries to restore data qubits.</li>
                    <li><b>Decoding:</b> Unroll the logical block back to recovered single-qubit data state for fidelity evaluation.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Milestone Status</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.8;">
                    <div>• <b>Step 1 — UI Foundation:</b> <span style="color: var(--status-success);">Completed</span></div>
                    <div>• <b>Step 2 — Quantum Simulation Core:</b> <span style="color: var(--status-success);">Completed (Strategies A & B)</span></div>
                    <div>• <b>Step 3 — 3-Qubit Repetition Code:</b> <span style="color: var(--status-success);">Active & Verified (Milestone 3)</span></div>
                    <div>• <b>Ancilla Parity Extraction:</b> <span style="color: var(--accent-cyan);">Operational ($s_{01}, s_{12}$)</span></div>
                    <div>• <b>Double-Flip Miscorrection:</b> <span style="color: var(--status-warning);">Analyzed ($O(p^2)$ breakdown)</span></div>
                    <div>• <b>Analytical Threshold:</b> <span style="color: var(--accent-cyan);">$P_L = 3p^2 - 2p^3$ (Crossover $p=0.50$)</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Execution Environment</div>
                <div style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.7;">
                    <div><b>Simulator:</b> Qiskit Aer 0.18.0 (C++ backend)</div>
                    <div><b>Qiskit Core:</b> Version 2.6.1</div>
                    <div><b>Subsystem Tracing:</b> Exact partial trace on recovered logical qubit</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_error_injection_lab() -> None:
    """Render the Error Injection Lab with real quantum simulation core integration."""
    render_header(
        title="Error Injection Lab",
        subtitle="Simulate and analyze stochastic Pauli X bit-flip noise on arbitrary quantum states.",
        category="MODULE 01",
    )

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 2, 2])
    with ctrl_col1:
        state_option = st.selectbox(
            "Target State",
            ["|0>", "|1>", "|+>", "|->", "Haar-Random Pure State"],
            index=0,
            help="Select initial quantum state before channel transmission.",
        )
    with ctrl_col2:
        mode_option = st.selectbox(
            "Noise Mode",
            ["Probabilistic Pauli X Channel", "Deterministic X-Gate Injection"],
            index=0,
            help="Probabilistic applies stochastic noise p; Deterministic inverts the state with certainty.",
        )
    with ctrl_col3:
        is_prob = mode_option.startswith("Probabilistic")
        p_val = st.slider(
            "Bit-Flip Probability (p)",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            disabled=not is_prob,
            help="Physical probability of bit-flip Pauli X error during transmission.",
        )
    with ctrl_col4:
        shots_val = st.selectbox(
            "Shot Count",
            [100, 256, 512, 1024, 2048, 4096, 8192],
            index=3,
            help="Number of circuit execution repetitions.",
        )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("Run Error Injection Simulation", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Executing quantum circuit on AerSimulator..."):
            norm_state = "random" if state_option.startswith("Haar") else state_option
            err_mode = "probabilistic" if is_prob else "deterministic"

            res = run_strategy_a_direct_measurement(
                state_name=norm_state,
                error_mode=err_mode,
                p=p_val,
                deterministic_qubit=0,
                shots=shots_val,
                seed=st.session_state["seed"],
            )
            st.session_state["error_injection_result"] = res
            st.session_state["experiment_status"] = f"Executed Strategy A ({norm_state}, p={res['configured_p']:.2f})"

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    res = st.session_state["error_injection_result"]
    m1, m2, m3, m4 = st.columns(4)

    if res is None:
        with m1:
            render_metric_empty_state("Observed Errors (k)", "Awaiting simulation")
        with m2:
            render_metric_empty_state("Estimated Error Rate p̂", "Awaiting simulation")
        with m3:
            render_metric_empty_state("95% Wilson CI", "Confidence interval")
        with m4:
            render_metric_empty_state("Classical Fidelity", "Overlap with ideal")

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        render_chart_empty_state(
            title="No Error Injection Simulation Run",
            description="Configure the initial state and noise parameters above, then click 'Run Error Injection Simulation' to simulate the circuit on Qiskit Aer.",
        )
    else:
        k = res["observed_errors"]
        n = res["shots"]
        est_p = res["estimated_p"]
        ci = res["wilson_ci"]

        ideal_counts = {res["expected_bit"]: n} if res["expected_bit"] in ("0", "1") else {"0": n // 2, "1": n // 2}
        fidelity = compute_fidelity(ideal_counts, res["counts"])

        with m1:
            render_metric_card("Observed Errors (k)", f"{k} / {n}", f"{k / n * 100:.1f}% flipped", is_highlight=True)
        with m2:
            render_metric_card("Estimated Error Rate p̂", f"{est_p:.4f}", f"Configured p = {res['configured_p']:.4f}")
        with m3:
            render_metric_card("95% Wilson CI", f"[{ci['lower_bound']:.3f}, {ci['upper_bound']:.3f}]", f"Margin: ±{(ci['upper_bound']-ci['lower_bound'])/2:.3f}")
        with m4:
            render_metric_card("Distribution Overlap", f"{fidelity:.4f}", "Bhattacharyya coefficient")

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        col_chart, col_circ = st.columns([3, 2], gap="medium")
        with col_chart:
            fig_hist = build_counts_histogram_plotly(
                counts=res["counts"],
                expected_bit=res["expected_bit"],
                title=f"Measurement Distribution ({res['shots']} shots)",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_circ:
            st.markdown(
                f"""
                <div class="q-card" style="height: 100%;">
                    <div class="q-card-title">Qiskit Circuit Representation</div>
                    <div class="q-circuit-block">{res['circuit'].draw(output='text')}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.75rem;">
                        <b>Stage 1:</b> State Prep ({res['state_name']})<br>
                        <b>Stage 2:</b> {res['error_mode'].capitalize()} Channel ({res['configured_p']:.2f})<br>
                        <b>Stage 3:</b> Computational Z-Basis Measurement
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="q-card">
                <div class="q-card-title">Quantum Measurement Interpretation</div>
                <p style="color: var(--text-secondary); font-size: 0.92rem; line-height: 1.6;">
                    {res['explanation']}
                </p>
                <div style="font-size: 0.85rem; color: var(--accent-cyan); margin-top: 0.5rem;">
                    💡 <b>Key Scientific Insight:</b> The Wilson 95% interval [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}] 
                    demonstrates the statistical precision of our estimate from {n} independent shots. 
                    Unlike the naive Wald interval, the Wilson formulation remains strictly bounded in [0, 1] even at extreme probabilities.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detection_strategies() -> None:
    """Render the Detection Strategies section including Strategy C and the 3-Strategy Comparison."""
    render_header(
        title="Detection Strategies",
        subtitle="Empirical error detection: Strategy A (Direct Probe), Strategy B (Statistical Estimation), and Strategy C (Ancilla Parity Extraction).",
        category="MODULE 02",
    )

    tab_c, tab_b1, tab_b2, tab_comp = st.tabs([
        "🛡️ Strategy C: Ancilla Parity Extraction",
        "📈 Strategy B: Estimation Convergence",
        "🎯 Strategy B: Channel Fault Detector",
        "⚖️ Strategy A vs B vs C Comparison",
    ])

    # Tab 1: Strategy C
    with tab_c:
        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Ancilla-Assisted Parity Extraction (Non-Destructive)</div>
                <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                    Strategy C couples two ancilla qubits to the 3-qubit repetition code block to measure parity stabilizers 
                    <b>without measuring data qubits directly</b>:
                </p>
                <div class="q-formula">
                    s₀₁ = b₀ ⊕ b₁ (Ancilla a₀) &nbsp;&nbsp;&nbsp;&nbsp; s₁₂ = b₁ ⊕ b₂ (Ancilla a₁)
                </div>
                <p style="color: var(--text-secondary); font-size: 0.85rem;">
                    Because both logical basis states |000⟩ and |111⟩ share identical even parity eigenvalues (s₀₁=0, s₁₂=0), 
                    the parity check extracts the error syndrome while <b>completely preserving the logical superposition</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            c_state = st.selectbox("Logical State", ["|0>", "|1>", "|+>", "|->", "Haar-Random Pure State"], key="c_state_sel")
        with sc2:
            c_mode = st.selectbox("Error Injection", ["Deterministic Single Flip", "Probabilistic Pauli Noise"], key="c_mode_sel")
        with sc3:
            if c_mode.startswith("Deterministic"):
                c_mask_choice = st.selectbox("Injected Error Mask", ["100 (Flip Q0)", "010 (Flip Q1)", "001 (Flip Q2)", "000 (No Error)"], index=0)
                c_mask = c_mask_choice[:3]
                c_p = 0.0
            else:
                c_mask = None
                c_p = st.slider("Bit-Flip Probability (p)", 0.0, 0.50, 0.15, 0.01, key="c_p_slider")
        with sc4:
            c_shots = st.select_slider("Ancilla Shots", [100, 256, 512, 1024, 2048], value=1024, key="c_shots_slider")

        if st.button("Run Strategy C Parity Extraction", type="primary", use_container_width=True):
            with st.spinner("Extracting parity syndromes on AerSimulator..."):
                norm_s = "random" if c_state.startswith("Haar") else c_state
                err_m = "deterministic" if c_mask is not None else "probabilistic"
                res_c = run_strategy_c_ancilla_parity(
                    state_name=norm_s,
                    error_mode=err_m,
                    p=c_p,
                    deterministic_mask=c_mask,
                    shots=c_shots,
                    seed=st.session_state["seed"],
                )
                st.session_state["strategy_c_result"] = res_c
                st.session_state["experiment_status"] = f"Strategy C Run ({norm_s}, Syn={res_c['dominant_syndrome']})"

        if st.session_state["strategy_c_result"] is not None:
            res_c = st.session_state["strategy_c_result"]
            syn_info = SYNDROME_MAP.get(res_c["dominant_syndrome"], SYNDROME_MAP["00"])

            # Metric Row
            cm1, cm2, cm3, cm4 = st.columns(4)
            with cm1:
                badge_class = "q-badge-00" if res_c["dominant_syndrome"] == "00" else "q-badge-err"
                st.markdown(
                    f"""
                    <div class="q-metric-card">
                        <div class="q-metric-label">Dominant Syndrome (s₀₁s₁₂)</div>
                        <div style="margin-top: 0.3rem;"><span class="q-syndrome-badge {badge_class}">{res_c['dominant_syndrome']}</span></div>
                        <div class="q-metric-subtext">{syn_info['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cm2:
                render_metric_card("Subspace Preservation", f"{res_c['subspace_preservation_fidelity']:.4f}", "Overlap with encoded GHZ state", is_highlight=True)
            with cm3:
                render_metric_card("Diagnosed Error", f"Qubit {syn_info['error_qubit']}" if syn_info['error_qubit'] is not None else "None", syn_info['action'])
            with cm4:
                dominant_count = res_c["syndrome_counts"][res_c["dominant_syndrome"]]
                render_metric_card("Syndrome Purity", f"{dominant_count/res_c['shots']*100:.1f}%", f"{dominant_count}/{res_c['shots']} shots")

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            col_syn_hist, col_syn_circ = st.columns([3, 2], gap="medium")
            with col_syn_hist:
                fig_syn = build_repetition_code_syndrome_plotly(res_c["syndrome_counts"], title=f"Syndrome Extraction Histogram ({res_c['shots']} shots)")
                st.plotly_chart(fig_syn, use_container_width=True)

            with col_syn_circ:
                st.markdown(
                    f"""
                    <div class="q-card" style="height: 100%;">
                        <div class="q-card-title">Parity Extraction Circuit</div>
                        <div class="q-circuit-block">{res_c['circuit'].draw(output='text')}</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                            Data qubits 0, 1, 2 remain unmeasured. Only ancillas are read into classical bits.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            render_chart_empty_state(
                title="Strategy C Not Run",
                description="Click 'Run Strategy C Parity Extraction' above to simulate ancilla-assisted parity extraction.",
            )

    # Tab 2: Strategy B Convergence
    with tab_b1:
        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Repeated-Trial Statistical Estimator</div>
                <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                    Strategy B transmits batches of known probe qubits through a noisy channel to estimate the 
                    physical error probability <i>p</i>. As the sample size <i>N</i> grows, the empirical estimate 
                    p̂ converges toward the true value, and the Wilson 95% confidence interval contracts proportionally to 1/√N.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            b_true_p = st.slider("True Channel Error Probability (p_true)", 0.0, 0.50, 0.15, 0.01)
        with c2:
            b_probe_state = st.selectbox("Calibration Probe State", ["|0>", "|1>"], index=0)
        with c3:
            st.markdown("<div style='margin-top: 1.7rem;'></div>", unsafe_allow_html=True)
            run_sweep_btn = st.button("Run Sample Size Sweep", use_container_width=True)

        if run_sweep_btn:
            with st.spinner("Executing sample size convergence sweep..."):
                sweep_results = run_strategy_b_sample_size_sweep(
                    true_p=b_true_p,
                    sample_sizes=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
                    input_state=b_probe_state,
                    seed=st.session_state["seed"],
                )
                st.session_state["strategy_b_sweep"] = sweep_results
                st.session_state["experiment_status"] = f"Strategy B Convergence (p={b_true_p:.2f})"

        if st.session_state["strategy_b_sweep"] is not None:
            data = st.session_state["strategy_b_sweep"]
            df_sw = pd.DataFrame(data)
            fig_conv = build_sample_size_convergence_plotly(data)
            st.plotly_chart(fig_conv, use_container_width=True)

            st.markdown("<div class='q-card-title'>Empirical Convergence Table</div>", unsafe_allow_html=True)
            st.dataframe(
                df_sw[["sample_size", "true_p", "estimated_p", "lower_bound", "upper_bound", "error_margin"]].rename(
                    columns={
                        "sample_size": "Shots N",
                        "true_p": "True p",
                        "estimated_p": "Estimate p̂",
                        "lower_bound": "Wilson Lower",
                        "upper_bound": "Wilson Upper",
                        "error_margin": "Half-Width (±)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_chart_empty_state(
                title="Convergence Sweep Not Executed",
                description="Click 'Run Sample Size Sweep' above to simulate quantum transmissions across sample sizes from N=10 to N=5000.",
            )

    # Tab 3: Strategy B Fault Detector
    with tab_b2:
        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Channel Fault Detector & ROC Analysis</div>
                <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                    The detector classifies a channel as <b>Faulty</b> if the estimated error rate p̂ exceeds a decision 
                    threshold τ. By evaluating a test fleet across varying thresholds, we construct the Receiver Operating 
                    Characteristic (ROC) curve and measure the trade-off between True Positives and False Positives.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            fault_cut = st.slider("Ground Truth Fault Cutoff (p*)", 0.05, 0.25, 0.10, 0.01)
        with d2:
            tau_cut = st.slider("Decision Threshold (τ)", 0.05, 0.30, 0.12, 0.01)
        with d3:
            eval_shots = st.select_slider("Probe Shots (N)", [50, 100, 200, 500, 1000], value=200)
        with d4:
            st.markdown("<div style='margin-top: 1.7rem;'></div>", unsafe_allow_html=True)
            run_detector_btn = st.button("Evaluate Fault Detector", use_container_width=True)

        if run_detector_btn:
            with st.spinner("Evaluating test fleet across Aer simulator..."):
                np.random.seed(st.session_state["seed"] + 101)
                healthy = list(np.random.uniform(0.01, fault_cut - 0.01, 30))
                faulty = list(np.random.uniform(fault_cut + 0.02, 0.35, 30))
                fleet = healthy + faulty

                eval_res = evaluate_strategy_b_fault_detector(
                    channel_p_list=fleet,
                    faulty_threshold=fault_cut,
                    decision_threshold=tau_cut,
                    sample_size=eval_shots,
                    seed=st.session_state["seed"],
                )
                roc_res = compute_strategy_b_roc_curve(
                    channel_p_list=fleet,
                    faulty_threshold=fault_cut,
                    sample_size=eval_shots,
                    threshold_steps=35,
                    seed=st.session_state["seed"],
                )
                st.session_state["strategy_b_detector"] = {
                    "eval": eval_res,
                    "roc": roc_res,
                }
                st.session_state["experiment_status"] = f"Strategy B Detector Evaluated (N={eval_shots})"

        if st.session_state["strategy_b_detector"] is not None:
            det_data = st.session_state["strategy_b_detector"]
            metrics = det_data["eval"]["metrics"]
            cm = det_data["eval"]["confusion_matrix"]
            roc = det_data["roc"]

            dm1, dm2, dm3, dm4 = st.columns(4)
            with dm1:
                render_metric_card("Precision", f"{metrics['precision']:.3f}", f"TP / (TP + FP) = {cm['tp']}/{cm['tp']+cm['fp']}")
            with dm2:
                render_metric_card("Recall (Sensitivity)", f"{metrics['recall']:.3f}", f"TP / (TP + FN) = {cm['tp']}/{cm['tp']+cm['fn']}", is_highlight=True)
            with dm3:
                render_metric_card("False Positive Rate", f"{metrics['false_positive_rate']:.3f}", f"FP / (FP + TN) = {cm['fp']}/{cm['fp']+cm['tn']}")
            with dm4:
                render_metric_card("ROC AUC", f"{roc['auc']:.3f}", f"Fleet Size = {cm['total']} channels")

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            col_roc_plot, col_cm_table = st.columns([3, 2], gap="medium")
            with col_roc_plot:
                fig_roc = build_roc_curve_plotly(roc)
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_cm_table:
                st.markdown(
                    f"""
                    <div class="q-card">
                        <div class="q-card-title">Diagnostic Confusion Matrix</div>
                        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; color: var(--text-secondary);">
                            <tr style="border-bottom: 1px solid var(--border-subtle); text-align: left;">
                                <th style="padding: 0.5rem;">Condition</th>
                                <th style="padding: 0.5rem;">Flagged (Faulty)</th>
                                <th style="padding: 0.5rem;">Passed (Healthy)</th>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(51, 65, 85, 0.4);">
                                <td style="padding: 0.5rem; color: var(--text-primary);"><b>Truly Faulty</b></td>
                                <td style="padding: 0.5rem; color: var(--status-success); font-family: monospace;">TP = {cm['tp']}</td>
                                <td style="padding: 0.5rem; color: var(--status-danger); font-family: monospace;">FN = {cm['fn']}</td>
                            </tr>
                            <tr>
                                <td style="padding: 0.5rem; color: var(--text-primary);"><b>Truly Healthy</b></td>
                                <td style="padding: 0.5rem; color: var(--status-warning); font-family: monospace;">FP = {cm['fp']}</td>
                                <td style="padding: 0.5rem; color: var(--status-success); font-family: monospace;">TN = {cm['tn']}</td>
                            </tr>
                        </table>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 1rem;">
                            Overall Classification Accuracy: <b>{metrics['accuracy']*100:.1f}%</b><br>
                            F1-Score: <b>{metrics['f1_score']:.3f}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            render_chart_empty_state(
                title="Detector Fleet Not Evaluated",
                description="Click 'Evaluate Fault Detector' to benchmark the threshold classifier on an ensemble of simulated quantum channels.",
            )

    # Tab 4: Strategy Comparison Table
    with tab_comp:
        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Architectural Comparison: Strategy A vs B vs C</div>
                <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                    Understanding the fundamental scientific differences between destructive state diagnostics and 
                    fault-tolerant quantum error detection:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        comp_data = get_strategy_comparison_table()
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)


def render_repetition_code() -> None:
    """Render the 3-Qubit Repetition Code active quantum error correction suite."""
    render_header(
        title="3-Qubit Repetition Code",
        subtitle="Complete [[3, 1, 1]] quantum error correction: Encoding, Stochastic/Deterministic Noise, Syndrome Extraction, and Coherent Recovery.",
        category="MODULE 03",
    )

    r_col1, r_col2, r_col3, r_col4 = st.columns([2, 2, 2, 2])
    with r_col1:
        rep_state = st.selectbox("Logical State |ψ⟩", ["|0>", "|1>", "|+>", "|->", "Haar-Random Pure State"], index=0)
    with r_col2:
        rep_mode = st.selectbox("Error Injection Mode", ["Deterministic Error Mask", "Probabilistic Pauli X Noise"], index=0)
    with r_col3:
        if rep_mode.startswith("Deterministic"):
            mask_select = st.selectbox(
                "3-Qubit Error Mask (q0 q1 q2)",
                [
                    "000 (No Error)",
                    "100 (Flip Q0)",
                    "010 (Flip Q1)",
                    "001 (Flip Q2)",
                    "110 (Double Flip Q0+Q1 - Miscorrected!)",
                    "101 (Double Flip Q0+Q2 - Miscorrected!)",
                    "011 (Double Flip Q1+Q2 - Miscorrected!)",
                    "111 (Triple Flip - Undetected!)",
                ],
                index=1,
            )
            rep_mask = mask_select[:3]
            rep_p = 0.0
        else:
            rep_mask = None
            rep_p = st.slider("Physical Error Rate p", 0.0, 0.50, 0.15, 0.01)
    with r_col4:
        rep_shots = st.select_slider("Simulation Shots", [100, 256, 512, 1024, 2048, 4096], value=1024)

    if st.button("Run [[3, 1, 1]] Repetition Code Experiment", type="primary", use_container_width=True):
        with st.spinner("Simulating encoding, noise, syndrome extraction, and CCX recovery..."):
            norm_s = "random" if rep_state.startswith("Haar") else rep_state
            res_rep = run_repetition_code_simulation(
                state_name=norm_s,
                p=rep_p,
                deterministic_mask=rep_mask,
                shots=rep_shots,
                seed=st.session_state["seed"],
            )
            st.session_state["repetition_code_result"] = res_rep
            st.session_state["experiment_status"] = f"[[3,1,1]] QEC Run ({norm_s}, Syn={res_rep['dominant_syndrome']})"

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    if st.session_state["repetition_code_result"] is not None:
        res_rep = st.session_state["repetition_code_result"]
        syn = res_rep["dominant_syndrome"]
        syn_info = SYNDROME_MAP.get(syn, SYNDROME_MAP["00"])
        fid = res_rep["state_fidelity"]

        # Metric Cards
        rm1, rm2, rm3, rm4 = st.columns(4)
        with rm1:
            badge_class = "q-badge-00" if syn == "00" else "q-badge-err"
            st.markdown(
                f"""
                <div class="q-metric-card">
                    <div class="q-metric-label">Observed Syndrome</div>
                    <div style="margin-top: 0.3rem;"><span class="q-syndrome-badge {badge_class}">{syn}</span></div>
                    <div class="q-metric-subtext">{syn_info['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rm2:
            fid_color = "var(--status-success)" if fid > 0.90 else ("var(--status-warning)" if fid > 0.40 else "var(--status-danger)")
            render_metric_card("Logical State Fidelity", f"{fid:.4f}", "Recovered qubit overlap", color=fid_color)
        with rm3:
            unprotected_ref = res_rep["physical_p"] if rep_mask is None else (1.0 if rep_mask.count("1") > 0 else 0.0)
            render_metric_card("Logical Error Rate", f"{res_rep['empirical_logical_error']:.4f}", f"Unprotected: {unprotected_ref:.4f}")
        with rm4:
            render_metric_card("Decoder Recovery Action", syn_info['action'], f"Target: Qubit {syn_info['error_qubit']}" if syn_info['error_qubit'] is not None else "None")

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        col_qec_hist, col_qec_circ = st.columns([3, 2], gap="medium")
        with col_qec_hist:
            fig_rep_syn = build_repetition_code_syndrome_plotly(
                res_rep["syndrome_counts"],
                title=f"Syndrome Extraction Histogram ({res_rep['shots']} shots)",
            )
            st.plotly_chart(fig_rep_syn, use_container_width=True)

        with col_qec_circ:
            st.markdown(
                f"""
                <div class="q-card" style="height: 100%;">
                    <div class="q-card-title">[[3, 1, 1]] QEC Circuit Diagram</div>
                    <div class="q-circuit-block">{res_rep['circuit'].draw(output='text')}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                        <b>Stages:</b> Encode (CNOTs) ⟶ Channel ⟶ Syndrome Extraction (Ancillas) ⟶ CCX Recovery ⟶ Decode
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Result explanation
        st.markdown(
            f"""
            <div class="q-card">
                <div class="q-card-title">Error Correction Mechanics & Subsystem Analysis</div>
                <p style="color: var(--text-secondary); font-size: 0.92rem; line-height: 1.6;">
                    The 3-qubit bit-flip code protects 1 logical qubit against arbitrary single physical bit-flips. 
                    The recovered state fidelity of <b>{fid:.4f}</b> was computed by performing an exact <b>partial trace</b> 
                    over the two auxiliary data qubits and the two syndrome ancillas, isolating the recovered logical subsystem q₀.
                </p>
                <div style="font-size: 0.85rem; color: var(--accent-cyan); margin-top: 0.4rem;">
                    💡 <b>Note on Error Regimes:</b> If a single qubit flipped, the syndrome uniquely diagnosed its position and CCX recovery 
                    restored the state with 100% fidelity. If a double-flip occurred, the syndrome misidentified the error, applying a third flip 
                    and inducing a logical X_L error (see Failure Analysis for complete breakdown).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_empty_state("Observed Syndrome", "Awaiting execution")
        with m2:
            render_metric_empty_state("Logical State Fidelity", "Awaiting execution")
        with m3:
            render_metric_empty_state("Logical Error Rate", "Awaiting execution")
        with m4:
            render_metric_empty_state("Decoder Action", "Awaiting execution")

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        render_chart_empty_state(
            title="Repetition Code Experiment Not Run",
            description="Configure parameters above and click 'Run [[3, 1, 1]] Repetition Code Experiment' to simulate full encoding, error injection, syndrome extraction, and CCX recovery.",
        )


def render_theory_experiments() -> None:
    """Render the Theory & Experiments section with analytical scaling vs empirical simulation and Haar-random evaluation."""
    render_header(
        title="Theory & Benchmark Experiments",
        subtitle="Rigorous parametric sweeps comparing Qiskit Aer simulation against theoretical QEC scaling (P_L = 3p² - 2p³) and 200 Haar-random states.",
        category="MODULE 04",
    )

    tab_sweep, tab_haar = st.tabs([
        "Parametric Scaling Sweep (4,096 Shots)",
        "200 Haar-Random Pure State Ensemble",
    ])

    with tab_sweep:
        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Mathematical Foundation of 3-Qubit Bit-Flip Correction</div>
                <p style="color: var(--text-secondary); line-height: 1.6;">
                    For an independent bit-flip channel with physical error probability <i>p</i>, the 3-qubit repetition code
                    successfully corrects all zero- and single-qubit errors. It fails only when two or three physical qubits flip simultaneously.
                </p>
                <div class="q-formula">
                    P_{logical\\ failure}(p) = \\binom{3}{2} p^2 (1-p) + \\binom{3}{3} p^3 = 3p^2 - 2p^3
                </div>
                <p style="color: var(--text-secondary); line-height: 1.6;">
                    • When <b>p < 0.50</b>: P_L < p (Quantum error correction actively suppresses noise).<br>
                    • At <b>p = 0.50</b>: P_L = p = 0.50 (Theoretical Breakeven Threshold).<br>
                    • When <b>p > 0.50</b>: P_L > p (Multi-qubit errors dominate, amplifying noise).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            run_theory_btn = st.button("Run Live 21-Point Parametric Sweep (p ∈ [0, 0.5], 4096 shots)", type="primary", use_container_width=True)
        with col_btn2:
            csv_path = os.path.join(ROOT_DIR, "results", "repetition_code_p_sweep.csv")
            load_cached = os.path.exists(csv_path)
            load_cached_btn = st.button("Load Pre-Computed 4,096-Shot Benchmark Dataset", disabled=not load_cached, use_container_width=True)

        if run_theory_btn:
            with st.spinner("Running 21-point live Aer simulation sweep (4,096 shots each)..."):
                p_grid = np.linspace(0.0, 0.50, 21)
                records = []
                for idx, p_val in enumerate(p_grid):
                    res_unprotected = run_strategy_a_direct_measurement("|0>", "probabilistic", p=p_val, shots=4096, seed=42 + idx)
                    res_protected = run_repetition_code_simulation("|0>", p=p_val, shots=4096, seed=42 + idx)
                    analytical_rate = 3.0 * (p_val ** 2) - 2.0 * (p_val ** 3)
                    p_ci = compute_wilson_confidence_interval(res_protected["logical_errors"], 4096, 0.95)

                    records.append(
                        {
                            "p_physical": p_val,
                            "shots": 4096,
                            "unprotected_error_rate": res_unprotected["estimated_p"],
                            "protected_logical_error_rate": res_protected["empirical_logical_error"],
                            "protected_ci_lower": p_ci["lower_bound"],
                            "protected_ci_upper": p_ci["upper_bound"],
                            "analytical_logical_error_rate": analytical_rate,
                            "error_suppression_gain": res_unprotected["estimated_p"] - res_protected["empirical_logical_error"],
                        }
                    )
                st.session_state["theory_sweep_results"] = records
                st.session_state["experiment_status"] = "Live Theory Sweep Completed (4096 shots)"

        elif load_cached_btn and load_cached:
            df_cached = pd.read_csv(csv_path)
            st.session_state["theory_sweep_results"] = df_cached.to_dict(orient="records")
            st.session_state["experiment_status"] = "Loaded 4,096-Shot Benchmark"

        if st.session_state["theory_sweep_results"] is not None:
            sweep_data = st.session_state["theory_sweep_results"]
            df_data = pd.DataFrame(sweep_data)

            # Plotly Comparison Chart
            fig_theory = build_theory_vs_empirical_plotly(sweep_data)
            st.plotly_chart(fig_theory, use_container_width=True)

            st.markdown("<div class='q-card-title'>Empirical vs Theoretical Benchmark Dataset</div>", unsafe_allow_html=True)
            st.dataframe(
                df_data[["p_physical", "shots", "unprotected_error_rate", "protected_logical_error_rate", "analytical_logical_error_rate", "protected_ci_lower", "protected_ci_upper", "error_suppression_gain"]].rename(
                    columns={
                        "p_physical": "Physical p",
                        "shots": "Shots/Pt",
                        "unprotected_error_rate": "Unprotected p_L",
                        "protected_logical_error_rate": "Empirical QEC P_L",
                        "analytical_logical_error_rate": "Theoretical 3p²-2p³",
                        "protected_ci_lower": "95% CI Lower",
                        "protected_ci_upper": "95% CI Upper",
                        "error_suppression_gain": "Suppression Gain Δp",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = df_data.to_csv(index=False).encode("utf-8")
            st.download_button("Download Benchmark CSV", data=csv_bytes, file_name="qguard_repetition_code_sweep.csv", mime="text/csv")
        else:
            render_chart_empty_state(
                title="Theory Sweep Not Executed",
                description="Click 'Run Live 21-Point Parametric Sweep' or load the pre-computed benchmark dataset to view theoretical vs empirical curves.",
            )

    with tab_haar:
        st.markdown(
            """
            <div class="q-card">
                <div class="q-card-title">Ensemble Evaluation of 200 Haar-Random Pure States</div>
                <p style="color: var(--text-secondary); line-height: 1.6;">
                    A common misconception is that the 3-qubit bit-flip code only protects computational basis states |0⟩ and |1⟩.
                    In fact, for any arbitrary superposition state <i>|ψ⟩ = α|0⟩ + β|1⟩</i> (including arbitrary complex phase angles),
                    the encoded logical state <i>|ψ_L⟩ = α|000⟩ + β|111⟩</i> preserves the quantum amplitudes intact after syndrome extraction 
                    and CCX recovery.
                </p>
                <div style="font-size: 0.88rem; color: var(--accent-cyan); margin-top: 0.4rem;">
                    We evaluate an ensemble of <b>200 Haar-random pure single-qubit states</b> uniformly distributed over the Bloch sphere,
                    subjecting each to physical Pauli X noise at p = 0.10, and computing exact reduced density matrix fidelities.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        h_col1, h_col2 = st.columns([1, 1])
        with h_col1:
            run_haar_btn = st.button("Run Live 200 Haar-Random Evaluation (p=0.10)", use_container_width=True)
        with h_col2:
            haar_csv_path = os.path.join(ROOT_DIR, "results", "haar_random_fidelity_evaluation.csv")
            load_haar_cached = os.path.exists(haar_csv_path)
            load_haar_btn = st.button("Load Pre-Computed 200-State Benchmark", disabled=not load_haar_cached, use_container_width=True)

        if run_haar_btn:
            with st.spinner("Generating 200 Haar-random states and computing density matrices..."):
                haar_data = evaluate_haar_random_benchmark(num_states=200, p=0.10, base_seed=999)
                st.session_state["haar_random_results"] = haar_data
                st.session_state["experiment_status"] = "Haar-Random Benchmark Evaluated"

        elif load_haar_btn and load_haar_cached:
            df_haar_loaded = pd.read_csv(haar_csv_path)
            st.session_state["haar_random_results"] = df_haar_loaded.to_dict(orient="records")
            st.session_state["experiment_status"] = "Loaded 200-State Benchmark"

        if "haar_random_results" in st.session_state and st.session_state["haar_random_results"] is not None:
            haar_list = st.session_state["haar_random_results"]
            df_h = pd.DataFrame(haar_list)
            m_unprot = df_h["unprotected_fidelity"].mean()
            m_prot = df_h["protected_fidelity"].mean()
            m_gain = df_h["fidelity_gain"].mean()

            hm1, hm2, hm3, hm4 = st.columns(4)
            with hm1:
                render_metric_card("Ensemble Size", f"{len(df_h)} States", "Haar-random pure states")
            with hm2:
                render_metric_card("Mean Unprotected F", f"{m_unprot:.4f}", "Single-qubit baseline", color="var(--status-danger)")
            with hm3:
                render_metric_card("Mean Protected F", f"{m_prot:.4f}", "[[3,1,1]] Repetition Code", color="var(--status-success)", is_highlight=True)
            with hm4:
                render_metric_card("Fidelity Gain ΔF", f"+{m_gain*100:.2f}%", "Average error suppression", color="var(--accent-cyan)")

            fig_haar_plotly = build_haar_random_fidelity_plotly(haar_list)
            st.plotly_chart(fig_haar_plotly, use_container_width=True)

            st.markdown("<div class='q-card-title'>200 Haar-Random State Evaluation Dataset</div>", unsafe_allow_html=True)
            st.dataframe(
                df_h[["state_index", "bloch_x", "bloch_y", "bloch_z", "unprotected_fidelity", "protected_fidelity", "fidelity_gain", "is_improved"]].rename(
                    columns={
                        "state_index": "State #",
                        "bloch_x": "Bloch X",
                        "bloch_y": "Bloch Y",
                        "bloch_z": "Bloch Z",
                        "unprotected_fidelity": "Unprotected Fidelity",
                        "protected_fidelity": "Protected Fidelity",
                        "fidelity_gain": "Fidelity Gain ΔF",
                        "is_improved": "Gain Positive?",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            haar_bytes = df_h.to_csv(index=False).encode("utf-8")
            st.download_button("Download 200 Haar-Random States CSV", data=haar_bytes, file_name="qguard_haar_random_fidelity.csv", mime="text/csv")
        else:
            render_chart_empty_state(
                title="Haar-Random Evaluation Not Run",
                description="Click 'Run Live 200 Haar-Random Evaluation' or load the pre-computed benchmark to view fidelity suppression across the Bloch sphere.",
            )


def render_failure_analysis() -> None:
    """Render the Failure Analysis section with exhaustive 8 error masks and miscorrection analysis."""
    render_header(
        title="Failure & Error Budget Analysis",
        subtitle="Exhaustive evaluation of all 8 physical error masks demonstrating single-flip correction and double-flip miscorrection.",
        category="MODULE 05",
    )

    st.markdown(
        """
        <div class="q-card">
            <div class="q-card-title">The Fundamental Double-Flip Miscorrection Dilemma</div>
            <p style="color: var(--text-secondary); line-height: 1.6;">
                The 3-qubit repetition code has distance <i>d = 3</i>, which allows it to correct any error of weight 
                <i>t = ⌊(d - 1) / 2⌋ = 1</i>. When a double bit-flip occurs (weight 2), the error syndrome precisely mimics 
                a single bit-flip on the remaining third qubit:
            </p>
            <div class="q-formula" style="border-left-color: var(--status-danger);">
                Physical Error X₀X₁ |000⟩ = |110⟩ ⟶ Syndrome s₀₁s₁₂ = 01 ⟶ Decoder applies X₂ ⟶ Final State: |111⟩ = |1_L⟩
            </div>
            <p style="color: var(--text-secondary); line-height: 1.6;">
                By applying an X gate to the undamaged qubit, the classical decoder inadvertently completes a logical operator 
                <i>X_L = X₀ X₁ X₂</i>, resulting in an undetected logical bit-flip inversion (Fidelity = 0.0).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Exhaustive 8 masks evaluation
    mask_records = analyze_eight_error_masks(input_state="|0>")
    df_masks = pd.DataFrame(mask_records)

    st.markdown("<div class='q-card-title'>Exhaustive 8-Mask Truth Table (Distance d=3)</div>", unsafe_allow_html=True)
    st.dataframe(
        df_masks[["mask", "weight", "syndrome", "diagnosed_error", "decoder_action", "logical_fidelity", "classification"]].rename(
            columns={
                "mask": "Physical Error Mask",
                "weight": "Weight w(e)",
                "syndrome": "Syndrome (s₀₁s₁₂)",
                "diagnosed_error": "Diagnosed Error",
                "decoder_action": "Decoder Action",
                "logical_fidelity": "Logical Fidelity",
                "classification": "Regime Classification",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    fig_masks_path = os.path.join(ROOT_DIR, "figures", "double_flip_miscorrection.png")
    if os.path.exists(fig_masks_path):
        st.image(fig_masks_path, caption="Exhaustive 8-Error-Mask Fidelity: Weight-1 Recovery vs Weight-2 Miscorrection")


# ==============================================================================
# MAIN APPLICATION ROUTER
# ==============================================================================
def main() -> None:
    """Main application entry point."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 1rem; border-bottom: 1px solid #334155;">
                <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">
                    🛡️ <span style="color: #06B6D4;">Q-GUARD</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.25rem;">
                    Quantum Error Detection & Correction Lab
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.caption("NAVIGATION")

        section = st.radio(
            label="Select Module",
            options=[
                "Overview",
                "Error Injection Lab",
                "Detection Strategies",
                "Repetition Code",
                "Theory & Experiments",
                "Failure Analysis",
            ],
            index=3,  # Default to Repetition Code for Step 3 testing
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.caption("BACKEND CONFIGURATION")

        st.selectbox(
            "Quantum Backend",
            ["Qiskit Aer (AerSimulator)"],
            index=0,
            help="High-performance C++ quantum circuit and noise simulator.",
        )

        st.session_state["shots"] = st.select_slider(
            "Default Simulation Shots",
            options=[100, 256, 512, 1024, 2048, 4096, 8192],
            value=1024,
            help="Global default shot count.",
        )

        st.session_state["seed"] = st.number_input(
            "RNG Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            help="Fixed seed for reproducible simulator results.",
        )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        if st.button("Reset Lab Session", use_container_width=True):
            st.session_state["experiment_status"] = "No experiment run yet"
            st.session_state["error_injection_result"] = None
            st.session_state["strategy_b_sweep"] = None
            st.session_state["strategy_b_detector"] = None
            st.session_state["strategy_c_result"] = None
            st.session_state["repetition_code_result"] = None
            st.session_state["theory_sweep_results"] = None
            st.rerun()

        st.markdown(
            """
            <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #334155; font-size: 0.75rem; color: #64748B;">
                Q-GUARD v0.3.0 • Qiskit Marathon<br>
                Step 3: [[3, 1, 1]] Repetition Code & Parity
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Content Routing
    if section == "Overview":
        render_overview()
    elif section == "Error Injection Lab":
        render_error_injection_lab()
    elif section == "Detection Strategies":
        render_detection_strategies()
    elif section == "Repetition Code":
        render_repetition_code()
    elif section == "Theory & Experiments":
        render_theory_experiments()
    elif section == "Failure Analysis":
        render_failure_analysis()


if __name__ == "__main__":
    main()
