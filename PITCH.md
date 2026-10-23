# Q-GUARD: Quantum Bit-Flip Detection & Error-Correction Lab
### High-Precision Research Platform for Qiskit Hackathons & Quantum Education

---

## 1. Executive Summary

Quantum processors are inherently vulnerable to physical decoherence and environmental noise. In superconducting, trapped-ion, and neutral-atom hardware, uncontrolled bit-flip ($X$) errors invert computational basis states and destroy fragile quantum superpositions.

**Q-GUARD** is an interactive, mathematically rigorous research laboratory built with **Qiskit** and **Qiskit Aer**. It allows researchers, quantum engineers, and students to:
- Simulate deterministic and stochastic bit-flip noise channels across diverse quantum states.
- Contrast destructive calibration strategies (direct readout, sample-size convergence, ROC fault classification) with non-destructive parity extraction.
- Deploy and analyze the canonical $[[3, 1, 1]]$ 3-qubit repetition code through all 6 operational stages (Encode, Channel, Parity, CCX Recovery, Decode, and Subsystem Partial Trace).
- Benchmark theoretical noise suppression ($P_L = 3p^2 - 2p^3$) against high-precision 4,096-shot empirical simulations.
- Rigorously prove logical preservation on an ensemble of **200 Haar-random pure states** uniformly distributed across the Bloch sphere.

---

## 2. The Four Pillars of Q-GUARD

### Pillar I: Realistic Noise Modeling & State Preparation
- **Transpiler-Safe Pauli $X$ Noise Channel:** Implements physical bit-flip noise $p \in [0, 1]$ and deterministic error masks using identity-anchored Kraus error models that survive Qiskit transpiler optimizations.
- **Universal Single-Qubit State Preparation:** Supports $|0\rangle, |1\rangle, |+\rangle, |-\rangle$, and seeded Haar-random pure states with exact Bloch vector conversions $(x, y, z)$.

### Pillar II: Destructive vs. Non-Destructive Detection
- **Strategy A (Direct Readout):** Known computational-basis transmission with Wilson 95% confidence intervals.
- **Strategy B (Statistical Convergence & ROC Detector):** Empirically proves $O(1/\sqrt{N})$ variance scaling across $N \in [10, 5000]$ shots and benchmarks an automated channel fault classifier achieving an ROC AUC of **0.9958**.
- **Strategy C (Ancilla-Assisted Parity Extraction):** Non-destructively measures stabilizers $Z \otimes Z \otimes I$ and $I \otimes Z \otimes Z$ onto auxiliary ancillas, preserving encoded logical superpositions ($\text{Fidelity} = 1.0$) without wave-function collapse.

### Pillar III: Active Quantum Error Correction ([[3, 1, 1]] Code)
- **Coherent Recovery:** Applies syndrome-conditioned Toffoli (CCX) unitaries to restore data qubits while preserving diagnostic ancilla registers.
- **Subsystem Partial Trace:** Executes exact reduced density matrix tracing $\rho_0 = \text{Tr}_{1,2,3,4}(\rho_{\text{full}})$ to evaluate state fidelity without projective measurement destruction.
- **Failure Analysis:** Exhaustively maps all $2^3 = 8$ physical error masks, proving 100% single-error correction ($w=1$) and explaining the double-flip miscorrection dilemma ($w=2 \implies X_L$ inversion).

### Pillar IV: Publication-Grade Empirical Benchmarking
- **4,096 Shots per Benchmark Point:** A 21-point parametric sweep ($p \in [0.0, 0.50]$) confirming empirical alignment with analytical scaling $P_L = 3p^2 - 2p^3$ and the theoretical breakeven threshold at $p = 0.50$.
- **200 Haar-Random Pure State Ensemble ($p = 0.10$):** Demonstrates that active error correction improves logical fidelity by **$+4.86\%$** on average across all 200 states ($100\%$ positive gain).

---

## 3. Product Experience & Visual Identity

Q-GUARD is packaged with a custom research UI built in Streamlit following a refined scientific design system:
- **Palette:** Dark navy/slate base (`#0A0F1D`), cyan primary accent (`#06B6D4`), subtle purple secondary (`#8B5CF6`), and semantic status indicators (`#10B981`, `#F59E0B`, `#EF4444`).
- **Interactive Modules:**
  1. *Overview & Architecture:* Visual system diagram and quantum circuit pipeline.
  2. *Error Injection Lab:* Real-time channel simulation with custom error masks.
  3. *Detection Strategies:* Direct readout, convergence sweeps, ROC curves, and Strategy A vs B vs C comparison.
  4. *Repetition Code Workbench:* Interactive 5-qubit circuit diagrams, syndrome badges, and CCX recovery.
  5. *Theory & Experiments:* Live 21-point parametric sweep (4,096 shots) and 200 Haar-random state explorer.
  6. *Failure Analysis:* Exhaustive 8-mask truth table and miscorrection diagnostics.

---

## 4. Key Performance Metrics

- **Automated Tests:** **36 / 36 passed** in pytest (8.83s execution time).
- **Repetition Code Shots:** **4,096 shots** per benchmark point across 21 points ($p \in [0.0, 0.50]$).
- **Haar-Random State Ensemble:** **200 states** evaluated under $p = 0.10$:
  - Mean Unprotected Fidelity: **0.9325**
  - Mean Protected Fidelity: **0.9811**
  - Mean Fidelity Gain: **+4.86%**
- **Strategy B Fault Classifier ROC AUC:** **0.9958**.
- **Executable Walkthrough Notebook:** `notebooks/walkthrough.ipynb` with 22 cells covering the entire curriculum.

---

## 5. Verification Commands

```bash
# 1. Run all 36 automated unit & integration tests
python -m pytest -q

# 2. Run the complete 7-phase batch experiment pipeline
python experiments/run_all.py

# 3. Launch the Streamlit research workbench
python -m streamlit run app/streamlit_app.py
```
