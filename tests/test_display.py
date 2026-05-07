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


def test_suptitle_grows_figure_height():
    """pp.suptitle reserves vertical space, so the figure grows."""
    fig, ax = pp.subplots(axes_size=(60, 40))
    # Settle so the first draw-event has passed.
    fig.canvas.draw()
    h_before_in = fig.get_size_inches()[1]
    pp.suptitle("Figure-level title", fontsize=16)
    h_after_in = fig.get_size_inches()[1]
    assert h_after_in > h_before_in + 0.1, (
        f"Figure height should grow to accommodate suptitle; "
        f"got before={h_before_in:.3f} in, after={h_after_in:.3f} in"
    )


def test_suptitle_second_call_replaces_first():
    """Calling pp.suptitle twice should leave exactly one publiplots
    suptitle attached; the new text is reflected in the stashed
    artist (matplotlib reuses ``fig._suptitle``) and the figure still
    has at most one suptitle-style Text attached."""
    fig, ax = pp.subplots(axes_size=(60, 40))
    first = pp.suptitle("First title")
    second = pp.suptitle("Second title")
    # The latest artist is stashed.
    assert fig._publiplots_suptitle is second
    # The title text reflects the latest call — prior "First title" is
    # gone regardless of whether matplotlib reused the Text instance.
    assert second.get_text() == "Second title"
    # Only one publiplots-managed suptitle on the figure: whatever
    # artist is in fig._publiplots_suptitle must be the one currently
    # attached. No stale "First title" Text should coexist.
    suptitle_like = [t for t in fig.texts if t.get_text() == "First title"]
    assert suptitle_like == [], (
        f"Stale first-call suptitle should be removed from fig.texts, "
        f"found: {suptitle_like}"
    )
    # The stashed artist is actually on the figure.
    assert fig._publiplots_suptitle in fig.texts


def test_suptitle_does_not_overlap_top_row_axes():
    """After settle(), the suptitle's y0 (pixel space) sits above the
    top-row axes' y1 — no overlap."""
    fig, ax = pp.subplots(axes_size=(60, 40))
    ax.set_title("axis title")  # a decoration that lives in title_space
    artist = pp.suptitle("Big figure title", fontsize=18)
    # settle has already been called inside pp.suptitle; one more draw
    # to ensure renderer is populated for window extent reads.
    fig.canvas.draw()
    ax_ext = ax.get_window_extent()
    su_ext = artist.get_window_extent()
    # Top-row axes' top edge (y1) must be strictly below suptitle's bottom (y0).
    assert ax_ext.y1 < su_ext.y0, (
        f"Top-row axes should sit entirely below suptitle; "
        f"got ax.y1={ax_ext.y1:.1f}, suptitle.y0={su_ext.y0:.1f}"
    )


def test_suptitle_lands_inside_reserved_band():
    """Suptitle y fraction lands inside the reserved band:
    between 1 - (outer_pad + suptitle_space)/H and 1 - outer_pad/H."""
    fig, ax = pp.subplots(axes_size=(60, 40))
    artist = pp.suptitle("Banded title", fontsize=16)
    layout = fig._publiplots_layout
    _, H = layout.figure_size()
    outer_pad = layout.outer_pad
    suptitle_space = layout.suptitle_space
    # y is in figure fractions.
    _, y = artist.get_position()
    lower = 1.0 - (outer_pad + suptitle_space) / H
    upper = 1.0 - outer_pad / H
    assert lower <= y <= upper, (
        f"suptitle y={y:.4f} should sit in the reserved band "
        f"[{lower:.4f}, {upper:.4f}]; "
        f"H={H:.2f}, outer_pad={outer_pad:.2f}, suptitle_space={suptitle_space:.2f}"
    )
