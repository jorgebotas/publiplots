"""Tests for legend_entries.py — entries, stashing, flag resolution."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _FakeHandle:
    def __init__(self, fc="red"):
        self._fc = fc
    def get_facecolor(self):
        return self._fc


def test_legend_entry_build_is_deterministic():
    e1 = LegendEntry.build("g", "hue", [_FakeHandle("red")], ["A"])
    e2 = LegendEntry.build("g", "hue", [_FakeHandle("red")], ["A"])
    assert e1.signature == e2.signature


def test_legend_entry_different_handles_differ():
    e1 = LegendEntry.build("g", "hue", [_FakeHandle("red")], ["A"])
    e2 = LegendEntry.build("g", "hue", [_FakeHandle("blue")], ["A"])
    assert e1.signature != e2.signature


def test_legend_entry_name_and_kind_preserved():
    e = LegendEntry.build("treatment", "size", [], [])
    assert e.name == "treatment"
    assert e.kind == "size"
    assert e.handles == ()
    assert e.labels == ()


def test_stash_entry_creates_list_first_time():
    fig, ax = plt.subplots()
    assert get_entries(ax) == []
    stash_entry(ax, LegendEntry.build("g", "hue", [], []))
    assert len(get_entries(ax)) == 1


def test_stash_entry_appends_in_order():
    fig, ax = plt.subplots()
    stash_entry(ax, LegendEntry.build("g1", "hue", [], []))
    stash_entry(ax, LegendEntry.build("g2", "size", [], []))
    names = [e.name for e in get_entries(ax)]
    assert names == ["g1", "g2"]


def test_resolve_legend_flags_true():
    flags = resolve_legend_flags(True)
    assert flags == {"hue": True, "size": True, "style": True, "marker": True}


def test_resolve_legend_flags_false():
    flags = resolve_legend_flags(False)
    assert flags == {"hue": False, "size": False, "style": False, "marker": False}


def test_resolve_legend_flags_dict_partial_missing_defaults_to_true():
    flags = resolve_legend_flags({"hue": False})
    assert flags == {"hue": False, "size": True, "style": True, "marker": True}


def test_resolve_legend_flags_dict_full():
    flags = resolve_legend_flags({"hue": True, "size": False, "style": False, "marker": False})
    assert flags == {"hue": True, "size": False, "style": False, "marker": False}


def test_resolve_legend_flags_rejects_bad_type():
    with pytest.raises(TypeError, match="legend must be bool or dict"):
        resolve_legend_flags(1)


def test_entry_is_in_group_no_group():
    fig, ax = plt.subplots()
    entry = LegendEntry.build("g", "hue", [], [])
    assert entry_is_in_group(fig, entry) is False


def test_entry_is_in_group_with_claim():
    """entry_is_in_group delegates to group.claims(name) — mock one."""
    fig, ax = plt.subplots()
    entry = LegendEntry.build("treatment", "hue", [], [])

    class _FakeGroup:
        def claims(self, name):
            return name == "treatment"

    fig._publiplots_legend_group = _FakeGroup()
    assert entry_is_in_group(fig, entry) is True

    other = LegendEntry.build("other", "hue", [], [])
    assert entry_is_in_group(fig, other) is False


def test_is_continuous_hue_true_for_scalar_mappable():
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="viridis", norm=Normalize(0, 1))
    assert is_continuous_hue([sm]) is True


def test_is_continuous_hue_false_for_empty():
    assert is_continuous_hue([]) is False


def test_is_continuous_hue_false_for_regular_handles():
    assert is_continuous_hue([_FakeHandle()]) is False
