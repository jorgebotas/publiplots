"""Tests for the global edgecolor rcParam."""
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import pandas as pd
import numpy as np

import publiplots as pp


@pytest.fixture(autouse=True)
def _restore_edgecolor_rcparam():
    """Snapshot and restore pp.rcParams['edgecolor'] so tests don't leak state."""
    original = pp.rcParams["edgecolor"]
    yield
    pp.rcParams["edgecolor"] = original


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_edgecolor_rcparam_default_is_none():
    """The default edgecolor rcParam is None — 'auto' mode, preserves current behavior."""
    assert pp.rcParams["edgecolor"] is None