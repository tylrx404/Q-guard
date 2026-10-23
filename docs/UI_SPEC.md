# Q-GUARD — UI/UX & Design System Specification
**Product:** Q-GUARD (Quantum Error Detection & Correction Lab)  
**Document Version:** 1.0.0 (Milestone 1 — Foundation)  
**Target Environment:** Streamlit 1.x / Modern Web Browsers (Chrome, Firefox, Safari, Edge)

---

## 1. Brand Identity & Philosophy

### 1.1 Mission
Q-GUARD is a high-fidelity research and simulation workbench for exploring, benchmarking, and visualizing quantum bit-flip errors ($X$-noise), syndrome extraction, and fault-tolerant quantum error correction protocols using Qiskit.

### 1.2 Design Philosophy
- **Scientific Rigor over Flash:** Avoid fake terminal animations, decorative neon scans, sci-fi tropes, or cluttered gadgetry. Every visual element corresponds to physical quantum state properties, error rates, or syndrome outcomes.
- **Honest Feedback:** Zero synthetic/fictitious performance numbers. When an experiment has not been executed, the UI clearly displays descriptive empty states with actionable next steps rather than placeholder data.
- **Visual Ergonomics:** Long research sessions require subdued dark-mode backgrounds, deliberate color coding, generous component margins (16–24px), crisp typography, and high contrast compliant with WCAG 2.1 AA standards.

---

## 2. Design System & Tokens

### 2.1 Color Palette

| Token Name | Hex Code | Role & Usage |
| :--- | :--- | :--- |
| `--bg-base` | `#0A0F1D` | Deep navy base viewport background |
| `--bg-surface` | `#111827` | Primary surface: cards, sidebar, containers |
| `--bg-surface-elevated` | `#1E293B` | Elevated cards, input fields, active pill items |
| `--border-subtle` | `#334155` | Structural borders, card outlines, chart axes |
| `--text-primary` | `#F8FAFC` | Primary text: titles, metric values, headers |
| `--text-secondary` | `#94A3B8` | Secondary labels, descriptions, metadata |
| `--text-muted` | `#64748B` | Footers, disabled controls, empty state hints |
| `--accent-cyan` | `#06B6D4` | Primary brand accent: active tabs, primary actions, logical $|0_L\rangle$ |
| `--accent-purple` | `#8B5CF6` | Secondary accent: ancilla qubits, syndrome measurements, $|1_L\rangle$ |
| `--status-success` | `#10B981` | Positive fidelity threshold, successful correction ($s=00$) |
| `--status-warning` | `#F59E0B` | Single-qubit detected error ($s \ne 00$), marginal fidelity |
| `--status-danger` | `#EF4444` | Multi-qubit uncorrectable error, fidelity breakdown |

### 2.2 Typography
- **Primary Interface Font:** `Inter`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `Roboto`, sans-serif.
- **Monospace / Quantum Operators Font:** `JetBrains Mono`, `Fira Code`, `Consolas`, monospace.
- **Type Scale:**
  - Page Title: `28px` / Bold (`700`) / tracking `-0.02em`
  - Section Header: `20px` / Semi-bold (`600`) / tracking `-0.01em`
  - Subheader / Card Title: `15px` / Medium (`500`)
  - Body: `14px` / Regular (`400`) / line-height `1.6`
  - Metric Big Number: `32px` / Bold (`700`) / Monospace tabular
  - Captions / Microcopy: `12px` / Regular (`400`) / uppercase tracking `+0.05em`

### 2.3 Layout & Grid
- **Global Container Width:** Fluid layout max-width `1440px` with 24px horizontal padding.
- **Grid Layout:** 12-column responsive layout with 16px gutter spacing.
- **Card Padding:** `20px` inner padding with `8px` border radius and `1px solid var(--border-subtle)`.
- **Elevation:** Subtle shadow `0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.3)`.

---

## 3. Information Architecture & Navigation

Q-GUARD is organized into six core functional sections accessible via the persistent navigation rail / sidebar:

```
[Q-GUARD Lab]
├── 1. Overview                 (Executive summary, lab architecture, state pipeline)
├── 2. Error Injection Lab      (Single & multi-qubit bit-flip channels, p-sweeps)
├── 3. Detection Strategies     (Ancilla parity checks, non-destructive measurements)
├── 4. Repetition Code          (3-qubit [[3,1,1]] code: encoding, syndrome, recovery)
├── 5. Theory & Experiments     (Benchmark sweeps, threshold analysis vs. physical p)
└── 6. Failure Analysis         (Multi-error breakdown, lookup table, error budgets)
```

---

## 4. Component Specifications

### 4.1 Header & Brand Bar
- **Logo / Wordmark:** "Q-GUARD" styled in bold cyan (`#06B6D4`) with a subtle badge "QUANTUM ERROR LAB".
- **Environment Status Pill:** Indicates execution backend (`Qiskit Aer Simulator` or `Statevector`).
- **Run Status Indicator:** Live status badge (`IDLE`, `RUNNING`, `COMPLETED`, `ERROR`).

### 4.2 Sidebar & Global Controls
- **Section Selector:** Clean radio/pill navigation with section icons and distinct active highlights.
- **Backend Configuration:**
  - Simulator selection (Aer `qasm_simulator`, `aer_simulator_statevector`).
  - Shots slider ($100$ to $10,000$, step $100$, default $1,024$).
  - Seed control for deterministic reproducibility.
- **Reset Button:** "Reset Lab Session" action button returning all controls to defaults.

### 4.3 Metric Cards (KPI Display)
- High-visibility cards displaying key experimental values.
- **Structure:**
  - Label (uppercase, secondary text, e.g. `STATE FIDELITY F`)
  - Primary Value (32px bold, e.g. `0.9842` or `—` when idle)
  - Subtitle / Delta indicator (e.g. `+12.4% vs uncorrected` in green)
  - Baseline comparison badge.
- **Empty State Rule:** When no experiment has been executed, displays `—` with subtext `Awaiting simulation run`. Never displays placeholder numbers like `99.9%`.

### 4.4 Quantum Circuit Viewer
- Clean container with dark background (`#0D1322`), subtle 1px border.
- Supports interactive SVG / high-resolution Matplotlib circuit exports.
- Labeled circuit stages:
  1. `State Prep`
  2. `Encoding`
  3. `Noise Channel (Bit-Flip)`
  4. `Syndrome Extraction`
  5. `Correction / Recovery`
  6. `Measurement`

### 4.5 Chart & Plot Layouts
- **Engine:** Plotly with custom dark theme matching design tokens.
- **Chart Palette:**
  - Unprotected Baseline: Muted red (`#F87171`) / dashed.
  - Detected Parity: Amber (`#FBBF24`).
  - Repetition Code (Corrected): Vibrant cyan (`#06B6D4`) / solid.
  - Theoretical Threshold: Purple (`#A78BFA`) / dotted reference line.
- **Layout Standards:**
  - Margin: `{l: 48, r: 24, t: 36, b: 48}`.
  - Grid: subtle dotted `#1E293B` lines.
  - Tooltips: unified dark card with monospace coordinates and delta values.

### 4.6 Status Indicators & Badges
- **Semantic Color Coding:**
  - `SUCCESS` (`#10B981`): Syndromes $00$, error successfully eliminated, fidelity $> 0.95$.
  - `WARNING` (`#F59E0B`): Syndrome detected error ($01, 10, 11$), single bit corrected.
  - `DANGER` (`#EF4444`): Uncorrectable multi-bit flip occurrence ($w(e) \ge 2$), fidelity breakdown.
  - `INFO` (`#06B6D4`): General system announcements and configuration updates.

### 4.7 Form Controls & Sliders
- Custom styled sliders with cyan progress fill.
- Input labels clearly state units (e.g., `Error Probability p [0.00 – 0.50]`).
- Hover and focus states use subtle cyan halos (`box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.25)`).

### 4.8 Honest Empty States
- Whenever a module lacks experiment data:
  - **Visual:** Minimal outlined icon (shield/circuit glyph).
  - **Headline:** Clear state description (e.g., `"No Simulation Executed"`).
  - **Explanation:** Concise 1-sentence prompt explaining what parameters need configuration.
  - **Action Button:** Direct action button (`"Run Experiment"` or `"Initialize Circuit"`).

---

## 5. Responsive Behavior & Accessibility
- **Breakpoints:**
  - Desktop ($> 1200\text{px}$): Dual-column layout (parameters left, outputs right).
  - Tablet ($768\text{px} - 1200\text{px}$): Single-column stacked with sticky control summary.
  - Mobile ($< 768\text{px}$): Full-width stacked view with collapsible parameters.
- **Accessibility:**
  - All text meets 4.5:1 contrast against its background.
  - Color is never the sole conveyer of information; icons, text labels, and patterns accompany all status flags.
