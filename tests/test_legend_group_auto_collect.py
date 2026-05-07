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
    assert group in fig._publiplots_legend_groups


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


# ---------------------------------------------------------------------------
# _materialize — auto-collection
# ---------------------------------------------------------------------------

from matplotlib.patches import Rectangle

from publiplots.utils.legend_entries import LegendEntry, stash_entry


def _stub_handle(color="red", label="A"):
    """Real matplotlib Rectangle so it both hashes (get_facecolor) and renders (get_label)."""
    return Rectangle((0, 0), 1, 1, facecolor=color, label=label)


def _stash_on_axes(ax, name, kind, color="red", labels=("A",)):
    entry = LegendEntry.build(
        name=name, kind=kind,
        handles=[_stub_handle(color, label=lab) for lab in labels],
        labels=labels,
    )
    stash_entry(ax, entry)


def test_materialize_dedups_identical_entries_across_axes():
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    _stash_on_axes(axes[1], "treatment", "hue")
    _stash_on_axes(axes[2], "treatment", "hue")
    group = pp.legend_group(anchor=axes[-1])
    group._materialize()
    # Only one legend element added to the builder (not 3).
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    assert len(legend_elements) == 1


def test_materialize_collects_distinct_names_across_axes():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    _stash_on_axes(axes[1], "dose", "hue")
    group = pp.legend_group(anchor=axes[-1])
    group._materialize()
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    assert len(legend_elements) == 2


def test_materialize_with_collect_filter_drops_unlisted_names():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    _stash_on_axes(axes[1], "dose", "hue")
    group = pp.legend_group(anchor=axes[-1], collect=["treatment"])
    group._materialize()
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    assert len(legend_elements) == 1


def test_materialize_collect_list_preserves_order():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "dose", "hue")
    _stash_on_axes(axes[1], "treatment", "hue")
    # First-in-axes-order is dose; but collect says treatment first.
    group = pp.legend_group(anchor=axes[-1], collect=["treatment", "dose"])
    group._materialize()
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    assert len(legend_elements) == 2
    # Legend.get_title().get_text() gives the "label" of each legend.
    titles = [e[1].get_title().get_text() for e in legend_elements]
    assert titles == ["treatment", "dose"]


def test_materialize_is_idempotent():
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    group = pp.legend_group(anchor=axes[-1])
    group._materialize()
    count_after_first = len(group._builder.elements)
    group._materialize()
    count_after_second = len(group._builder.elements)
    assert count_after_first == count_after_second


def test_materialize_mismatched_signature_warns_once():
    fig, axes = pp.subplots(1, 3, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue", color="red")
    _stash_on_axes(axes[1], "treatment", "hue", color="blue")   # different palette
    _stash_on_axes(axes[2], "treatment", "hue", color="green")  # yet another
    group = pp.legend_group(anchor=axes[-1])
    with pytest.warns(UserWarning, match="inconsistent handles"):
        group._materialize()
    # Warn-once: calling _materialize again should not re-warn.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # fail if any warning fires
        # Reset the _materialized flag so it re-runs the collection
        group._materialized = False
        group._materialize()  # must not warn again


def test_materialize_no_publiplots_axes_is_noop():
    """Group on a plt.subplots figure (no _publiplots_axes) must not crash."""
    fig, ax = plt.subplots()
    group = pp.legend_group(anchor=ax)
    group._materialize()
    assert group._builder.elements == []


def test_materialize_manual_and_auto_coexist():
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    _stash_on_axes(axes[0], "treatment", "hue")
    group = pp.legend_group(anchor=axes[-1])
    # Manual add BEFORE materialize
    group.add_legend(
        handles=create_legend_handles(labels=["x"], colors=["#000"], alpha=0.5, linewidth=1.0),
        label="manual",
    )
    # Now materialize
    group._materialize()
    legend_elements = [e for e in group._builder.elements if e[0] == "legend"]
    # One manual + one auto-collected treatment entry.
    assert len(legend_elements) == 2
