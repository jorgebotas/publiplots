"""Numeric ``loc`` codes for an inside colorbar (#223).

matplotlib's ``ax.legend(loc=...)`` accepts integer location codes as
well as position strings, and publiplots' categorical inside legend
passes ``loc`` straight through to it. The inside *colorbar* path
resolves ``loc`` itself, and used to reject every integer except ``0``:
switching a hue column from categorical to continuous broke a working
call with no other change.

These tests pin two things:

1. The integer -> corner mapping is matplotlib's own
   (``matplotlib.legend.Legend.codes``), so a code lands in the same
   corner the categorical legend would put it in.
2. Genuinely invalid ``loc`` values still fail with publiplots' readable
   message rather than a matplotlib internal error.

``loc=0``/``'best'`` is the one deliberate divergence: matplotlib's
'best' searches for the emptiest region using the legend's own handles,
and a colorbar strip has no equivalent search, so it resolves to a fixed
``'upper right'`` (the same fallback matplotlib uses for figure legends,
where 'best' is likewise unimplemented). That is asserted, not compared.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.legend import Legend

import publiplots as pp
from publiplots.utils.legend import LegendBuilder


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def df():
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "x": rng.normal(size=60),
        "y": rng.normal(size=60),
        "z": rng.normal(size=60),
        "cat": np.array(list("abc"))[rng.integers(0, 3, 60)],
    })


# Every code matplotlib accepts, minus 'best' (0), which has no
# strip equivalent and is covered separately.
POSITION_CODES = sorted(c for c in Legend.codes.values() if c != 0)
CODE_TO_NAME = {code: name for name, code in Legend.codes.items()}

AXES_SIZE = (60, 50)


def _corner(bbox, ax_bbox):
    """Classify a bbox's centre into (vertical, horizontal) thirds of ``ax``."""
    fx = ((bbox.x0 + bbox.x1) / 2 - ax_bbox.x0) / ax_bbox.width
    fy = ((bbox.y0 + bbox.y1) / 2 - ax_bbox.y0) / ax_bbox.height
    horizontal = "left" if fx < 1 / 3 else ("right" if fx > 2 / 3 else "center")
    vertical = "lower" if fy < 1 / 3 else ("upper" if fy > 2 / 3 else "center")
    return vertical, horizontal


def _colorbar_corner(df, loc):
    """Render an inside colorbar at ``loc`` and report the corner it landed in."""
    fig, ax = pp.subplots(axes_size=AXES_SIZE)
    pp.scatterplot(
        data=df, x="x", y="y", hue="z", ax=ax,
        legend_kws={"inside": True, "loc": loc},
    )
    fig.canvas.draw()
    # An inside colorbar is an ``ax.inset_axes``, so it is a child of the
    # parent axes and never appears in ``fig.get_axes()``.
    strips = list(ax.child_axes)
    assert len(strips) == 1, f"loc={loc!r}: expected one strip, got {len(strips)}"
    return _corner(strips[0].get_window_extent(), ax.get_window_extent())


def _legend_corner(df, loc):
    """Render an inside categorical legend at ``loc`` and report its corner."""
    fig, ax = pp.subplots(axes_size=AXES_SIZE)
    pp.scatterplot(
        data=df, x="x", y="y", hue="cat", ax=ax,
        legend_kws={"inside": True, "loc": loc},
    )
    fig.canvas.draw()
    legend = ax.get_legend()
    assert legend is not None, f"loc={loc!r}: no inside legend"
    return _corner(legend.get_window_extent(), ax.get_window_extent())


def test_code_table_is_matplotlibs_own():
    """The mapping is inverted from the installed matplotlib, not a copy.

    Also pins the table as of matplotlib 3.10 so a future matplotlib that
    renumbers or adds a code shows up here rather than silently moving a
    strip. Note 5 ('right') and 7 ('center right') are distinct codes
    that anchor identically (``offsetbox._get_anchored_bbox`` maps both
    to "E").
    """
    assert LegendBuilder._INSIDE_CBAR_LOC_NAMES == CODE_TO_NAME
    assert CODE_TO_NAME == {
        0: "best",
        1: "upper right",
        2: "upper left",
        3: "lower left",
        4: "lower right",
        5: "right",
        6: "center left",
        7: "center right",
        8: "lower center",
        9: "upper center",
        10: "center",
    }


@pytest.mark.parametrize("code", POSITION_CODES)
def test_anchor_resolves_every_code_like_its_string(code):
    """``_inside_cbar_anchor`` agrees with itself on int vs equivalent string."""
    assert (
        LegendBuilder._inside_cbar_anchor(code)
        == LegendBuilder._inside_cbar_anchor(CODE_TO_NAME[code])
    )


@pytest.mark.parametrize("code", POSITION_CODES)
def test_numeric_loc_lands_where_the_string_does(df, code):
    """A rendered strip lands in the same corner for the int and the string."""
    assert _colorbar_corner(df, code) == _colorbar_corner(df, CODE_TO_NAME[code])


@pytest.mark.parametrize("code", POSITION_CODES)
def test_numeric_loc_matches_the_categorical_legend(df, code):
    """The corner a code picks is the corner ``ax.legend(loc=code)`` picks.

    This is the actual #223 complaint: the same ``legend_kws`` should
    place the mark the same way whether the hue is categorical or
    continuous.
    """
    assert _colorbar_corner(df, code) == _legend_corner(df, code)


def test_best_and_zero_resolve_to_upper_right(df):
    """'best' has no strip equivalent, so both spellings pin to upper right.

    Deliberately NOT compared against the categorical legend: matplotlib's
    'best' search is data-dependent (it picks the emptiest region), so the
    two legitimately disagree here.
    """
    assert LegendBuilder._inside_cbar_anchor(0) == ("upper", "right")
    assert LegendBuilder._inside_cbar_anchor("best") == ("upper", "right")
    assert _colorbar_corner(df, 0) == ("upper", "right")
    assert _colorbar_corner(df, "best") == ("upper", "right")


@pytest.mark.parametrize("loc", [
    11,             # one past matplotlib's highest code
    -1,             # matplotlib rejects negatives too
    99,
    1.5,            # matplotlib requires an int, not a float
    0.0,            # ... including a float that equals a valid code
    None,
    "nowhere",
    "top right",    # plausible but not matplotlib's vocabulary
    "upper",        # only one of the two words
    "outside upper right",  # figure-legend-only spelling
    (0.5, 0.5),     # coordinate tuple: legend-only, no strip meaning
])
def test_invalid_loc_still_raises_readable_error(df, loc):
    with pytest.raises(ValueError, match="inside colorbar loc must be one of"):
        LegendBuilder._inside_cbar_anchor(loc)

    fig, ax = pp.subplots(axes_size=AXES_SIZE)
    with pytest.raises(ValueError, match="inside colorbar loc must be one of"):
        pp.scatterplot(
            data=df, x="x", y="y", hue="z", ax=ax,
            legend_kws={"inside": True, "loc": loc},
        )


def test_error_message_names_the_numeric_range():
    """The message tells the user integers are an option, and which ones."""
    with pytest.raises(ValueError) as excinfo:
        LegendBuilder._inside_cbar_anchor(42)
    message = str(excinfo.value)
    assert "0-10" in message
    assert "42" in message
