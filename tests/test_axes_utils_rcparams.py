"""pp.adjust_spines / add_grid / add_reference_line honour rcParams.

These three helpers used to hardcode every style default, so
pp.adjust_spines(ax) silently drew 1.5pt spines over a 0.75pt
axes.linewidth setting and pp.add_grid(ax) ignored all four grid rcParams.
"""
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_adjust_spines_uses_axes_linewidth():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.adjust_spines(ax, spines="left-bottom")
    for name in ("left", "bottom"):
        assert ax.spines[name].get_linewidth() == pytest.approx(
            plt.rcParams["axes.linewidth"]
        )


def test_adjust_spines_uses_axes_edgecolor():
    from matplotlib.colors import to_rgba

    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.adjust_spines(ax, spines="left-bottom")
    assert to_rgba(ax.spines["left"].get_edgecolor()) == to_rgba(
        plt.rcParams["axes.edgecolor"]
    )


def test_adjust_spines_explicit_value_still_wins():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.adjust_spines(ax, spines="left-bottom", linewidth=3.0)
    assert ax.spines["left"].get_linewidth() == pytest.approx(3.0)


def test_adjust_spines_explicit_color_still_wins():
    from matplotlib.colors import to_rgba

    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.adjust_spines(ax, spines="left-bottom", color="steelblue")
    assert to_rgba(ax.spines["left"].get_edgecolor()) == to_rgba("steelblue")


def test_adjust_spines_respects_a_changed_rcparam():
    saved = plt.rcParams["axes.linewidth"]
    try:
        plt.rcParams["axes.linewidth"] = 2.25
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.adjust_spines(ax, spines="left-bottom")
        assert ax.spines["left"].get_linewidth() == pytest.approx(2.25)
    finally:
        plt.rcParams["axes.linewidth"] = saved


def test_add_grid_uses_grid_rcparams():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_grid(ax, axis="y")
    gridlines = ax.yaxis.get_gridlines()
    assert gridlines
    line = gridlines[0]
    assert line.get_linewidth() == pytest.approx(plt.rcParams["grid.linewidth"])
    assert line.get_alpha() == pytest.approx(plt.rcParams["grid.alpha"])


def test_add_grid_uses_grid_color_and_linestyle_rcparams():
    from matplotlib.colors import to_rgba

    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_grid(ax, axis="y")
    line = ax.yaxis.get_gridlines()[0]
    assert to_rgba(line.get_color()) == to_rgba(plt.rcParams["grid.color"])
    assert line.get_linestyle() == plt.rcParams["grid.linestyle"]


def test_add_grid_explicit_value_still_wins():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_grid(ax, axis="y", linewidth=3.0, alpha=1.0)
    line = ax.yaxis.get_gridlines()[0]
    assert line.get_linewidth() == pytest.approx(3.0)
    assert line.get_alpha() == pytest.approx(1.0)


def test_add_grid_explicit_color_and_linestyle_still_win():
    from matplotlib.colors import to_rgba

    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_grid(ax, axis="y", color="steelblue", linestyle=":")
    line = ax.yaxis.get_gridlines()[0]
    assert to_rgba(line.get_color()) == to_rgba("steelblue")
    assert line.get_linestyle() == ":"


def test_add_grid_respects_a_changed_rcparam():
    saved = plt.rcParams["grid.linewidth"]
    try:
        plt.rcParams["grid.linewidth"] = 2.25
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.add_grid(ax, axis="y")
        assert ax.yaxis.get_gridlines()[0].get_linewidth() == pytest.approx(2.25)
    finally:
        plt.rcParams["grid.linewidth"] = saved


def test_add_reference_line_uses_lines_linewidth():
    """A reference line is a data line, so it takes lines.linewidth."""
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_reference_line(ax, value=0.0, axis="y")
    assert ax.lines
    assert ax.lines[-1].get_linewidth() == pytest.approx(
        plt.rcParams["lines.linewidth"]
    )


def test_add_reference_line_explicit_linewidth_still_wins():
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_reference_line(ax, value=0.0, axis="y", linewidth=3.0)
    assert ax.lines[-1].get_linewidth() == pytest.approx(3.0)


def test_add_reference_line_keeps_its_deliberate_red():
    """color='red' has no rcParam equivalent -- a reference line is meant to
    be conspicuous. This is a documented literal, not an oversight."""
    fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
    pp.add_reference_line(ax, value=0.0, axis="y")
    from matplotlib.colors import to_rgba
    assert to_rgba(ax.lines[-1].get_color()) == to_rgba("red")


# ---------------------------------------------------------------------------
# set_axis_labels: the per-role font rcParams must survive an unset kwarg.
#
# Forwarding fontsize=None / fontweight=None is not the same as omitting them:
# set_fontsize(None) / set_fontweight(None) reset the Text to the *generic*
# font.size / font.weight, discarding axes.labelsize / axes.labelweight and
# axes.titlesize / axes.titleweight that matplotlib applied at creation.
# ---------------------------------------------------------------------------

_ROLE_FONT_RCPARAMS = {
    "font.size": 7,
    "font.weight": "normal",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
}


def test_set_axis_labels_keeps_axes_labelsize():
    with plt.rc_context(_ROLE_FONT_RCPARAMS):
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.set_axis_labels(ax, xlabel="X", ylabel="Y")
        assert ax.xaxis.label.get_fontsize() == pytest.approx(11)
        assert ax.yaxis.label.get_fontsize() == pytest.approx(11)


def test_set_axis_labels_keeps_axes_labelweight():
    with plt.rc_context(_ROLE_FONT_RCPARAMS):
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.set_axis_labels(ax, xlabel="X", ylabel="Y")
        assert ax.xaxis.label.get_fontweight() == "bold"
        assert ax.yaxis.label.get_fontweight() == "bold"


def test_set_axis_labels_keeps_axes_titlesize():
    with plt.rc_context(_ROLE_FONT_RCPARAMS):
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.set_axis_labels(ax, title="T")
        assert ax.title.get_fontsize() == pytest.approx(13)


def test_set_axis_labels_keeps_axes_titleweight():
    with plt.rc_context(_ROLE_FONT_RCPARAMS):
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.set_axis_labels(ax, title="T")
        assert ax.title.get_fontweight() == "bold"


def test_set_axis_labels_matches_bare_matplotlib_when_nothing_is_passed():
    """The whole contract: omitting the font kwargs must be indistinguishable
    from never having gone through set_axis_labels at all."""
    with plt.rc_context(_ROLE_FONT_RCPARAMS):
        fig, ax = pp.subplots(1, 2, axes_size=(40, 30))
        pp.set_axis_labels(ax[0], xlabel="X", ylabel="Y", title="T")
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Y")
        ax[1].set_title("T")
        for a, b in (
            (ax[0].xaxis.label, ax[1].xaxis.label),
            (ax[0].yaxis.label, ax[1].yaxis.label),
            (ax[0].title, ax[1].title),
        ):
            assert a.get_fontsize() == pytest.approx(b.get_fontsize())
            assert a.get_fontweight() == b.get_fontweight()


def test_set_axis_labels_explicit_fontsize_still_wins():
    with plt.rc_context(_ROLE_FONT_RCPARAMS):
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.set_axis_labels(ax, xlabel="X", ylabel="Y", title="T", fontsize=20)
        assert ax.xaxis.label.get_fontsize() == pytest.approx(20)
        assert ax.yaxis.label.get_fontsize() == pytest.approx(20)
        assert ax.title.get_fontsize() == pytest.approx(20)


def test_set_axis_labels_explicit_fontweight_still_wins():
    with plt.rc_context(_ROLE_FONT_RCPARAMS):
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.set_axis_labels(
            ax, xlabel="X", ylabel="Y", title="T", fontweight="light"
        )
        assert ax.xaxis.label.get_fontweight() == "light"
        assert ax.yaxis.label.get_fontweight() == "light"
        assert ax.title.get_fontweight() == "light"


def test_set_axis_labels_explicit_normal_overrides_a_bold_rcparam():
    """'normal' is a real value, not a sentinel -- it must still override."""
    with plt.rc_context(_ROLE_FONT_RCPARAMS):
        fig, ax = pp.subplots(1, 1, axes_size=(40, 30))
        pp.set_axis_labels(ax, xlabel="X", title="T", fontweight="normal")
        assert ax.xaxis.label.get_fontweight() == "normal"
        assert ax.title.get_fontweight() == "normal"
