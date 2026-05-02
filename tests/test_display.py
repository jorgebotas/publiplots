"""Tests for pp.show and pp.suptitle."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_show_is_exported():
    assert callable(pp.show)


def test_suptitle_is_exported():
    assert callable(pp.suptitle)


def test_show_returns_none():
    """pp.show() is a pass-through; returns None on the Agg backend."""
    fig, ax = pp.subplots(axes_size=(40, 30))
    result = pp.show()
    assert result is None


def test_suptitle_attaches_to_current_figure():
    fig, ax = pp.subplots(axes_size=(50, 30))
    result = pp.suptitle("Overall Title")
    assert isinstance(result, Text)
    assert result.get_text() == "Overall Title"
    # It sits on the figure, not the axes
    assert result in fig.texts


def test_suptitle_forwards_kwargs():
    fig, ax = pp.subplots(axes_size=(50, 30))
    result = pp.suptitle("Title", fontsize=14, fontweight="bold")
    assert result.get_fontsize() == 14
    assert result.get_fontweight() == "bold"
