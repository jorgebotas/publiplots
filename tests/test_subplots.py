"""Tests for pp.subplots() and its supporting components."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

SUBPLOT_KEYS = [
    "subplots.title_space",
    "subplots.xlabel_space",
    "subplots.ylabel_space",
    "subplots.right",
    "subplots.hspace",
    "subplots.wspace",
    "subplots.outer_pad",
]


def test_subplots_rcparams_keys_exist():
    for key in SUBPLOT_KEYS:
        assert key in pp.rcParams, f"missing rcParam: {key}"


def test_subplots_rcparams_publication_defaults():
    pp.set_publication_style()
    try:
        assert pp.rcParams["subplots.title_space"] == 5
        assert pp.rcParams["subplots.xlabel_space"] == 8
        assert pp.rcParams["subplots.ylabel_space"] == 10
        assert pp.rcParams["subplots.right"] == 2
        assert pp.rcParams["subplots.hspace"] == 8
        assert pp.rcParams["subplots.wspace"] == 10
        assert pp.rcParams["subplots.outer_pad"] == 2
    finally:
        pp.reset_style()


def test_subplots_rcparams_notebook_defaults():
    pp.set_notebook_style()
    try:
        assert pp.rcParams["subplots.title_space"] == 8
        assert pp.rcParams["subplots.xlabel_space"] == 12
        assert pp.rcParams["subplots.ylabel_space"] == 14
        assert pp.rcParams["subplots.right"] == 2
        assert pp.rcParams["subplots.hspace"] == 12
        assert pp.rcParams["subplots.wspace"] == 14
        assert pp.rcParams["subplots.outer_pad"] == 3
    finally:
        pp.reset_style()