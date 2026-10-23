"""Initial environment and package verification test."""

import importlib
import pytest


def test_qguard_package_imports():
    """Verify that the src/qguard submodules can be imported cleanly."""
    import src
    import src.noise
    import src.states
    import src.strategies
    import src.repetition_code
    import src.analysis
    import src.plots

    assert src.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "pkg_name",
    [
        "qiskit",
        "qiskit_aer",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "plotly",
        "streamlit",
        "pytest",
    ],
)
def test_dependency_availability(pkg_name):
    """Verify that required core dependencies are installed and importable."""
    module = importlib.import_module(pkg_name)
    assert module is not None
