"""Tests for MultiAxesLegendGroup auto-collect (PR #81)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Claims + figure registration
# ---------------------------------------------------------------------------


def test_legend_group_registers_on_figure():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    group = pp.legend_group(anchor=axes[-1])
    assert fig._publiplots_legend_group is group


def test_legend_group_claims_collect_none_returns_true():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    group = pp.legend_group(anchor=axes[-1])
    assert group.claims("anything") is True
    assert group.claims("other") is True


def test_legend_group_claims_with_collect_list():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    group = pp.legend_group(anchor=axes[-1], collect=["treatment"])
    assert group.claims("treatment") is True
    assert group.claims("dose") is False


def test_legend_group_rejects_bare_string_collect():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    with pytest.raises(TypeError, match="list/tuple"):
        pp.legend_group(anchor=axes[-1], collect="treatment")
