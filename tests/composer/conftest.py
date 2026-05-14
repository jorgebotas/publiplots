"""Shared pytest fixtures for tests/composer/."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to keep memory clean."""
    yield
    plt.close("all")
