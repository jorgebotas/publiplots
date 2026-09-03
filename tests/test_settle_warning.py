"""``settle()`` must say so when it gives up (#230, #244).

``SubplotsAutoLayout.settle`` loops at most ``_MAX_CONVERGENCE_ITERS``
draws and used to ``return`` whether or not the layout had settled. Three
runaways shipped behind that silence: the #230 inner-anchor band, at
+67.24mm on every draw; the same feedback for a ``twinx`` scope, at
+115.56mm; and an ``ax.inset_axes`` scope (#244) that grows the figure by
tens of millimetres per draw until matplotlib raises on the image size.
The user-visible symptom in every case is a figure whose saved size is
not reproducible, with nothing in the output to explain it.

The warning is gated on the residual's *magnitude*, in the same
millimetres ``_needs_update`` compares against its 0.1mm tolerance, and
not on the residual growing: #230's residual is flat to four decimals
while the figure runs away underneath it. The floor
(``_NONCONVERGENCE_WARN_MM``) sits an order of magnitude above the
largest dpi-induced drift measured across a corpus of layouts saved at
72 / 100 / 150 / 300 / 600 dpi, and two orders below either real
divergence.
"""

import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.layout import auto_layout as al


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


_rng = np.random.default_rng(0)
DF = pd.DataFrame(
    {
        "x": _rng.normal(size=60),
        "y": _rng.normal(size=60),
        "g": _rng.choice(["alpha", "beta", "gamma"], 60),
    }
)


def _grid(nrows=1, ncols=2):
    fig, axes = pp.subplots(nrows, ncols, axes_size=(40, 32))
    flat = list(np.asarray(axes).flat)
    for ax in flat:
        pp.scatterplot(data=DF, x="x", y="y", hue="g", ax=ax)
    return fig, flat


def _force_runaway(fig, mm_per_draw):
    """Make every draw measure ``right`` ``mm_per_draw`` past what it holds.

    A synthetic runaway rather than a layout bug: the measurement is
    invariably one step ahead of the reservation, so ``_needs_update``
    can never come back False and the figure grows by ``mm_per_draw``
    forever. It reproduces the *shape* of #230 — a residual that stays
    exactly constant while the figure diverges — under a knob that lets
    a test sit either side of the warning's floor.
    """
    auto = fig._publiplots_auto_layout

    def _measure():
        return {"right": tuple(v + mm_per_draw for v in auto._layout.right)}

    auto._measure = _measure
    return auto


def _capture(fn):
    """Run ``fn`` and return the LayoutConvergenceWarnings it emitted."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
        return [
            w for w in rec
            if issubclass(w.category, pp.LayoutConvergenceWarning)
        ]


# --- it fires on a divergence -------------------------------------------

def test_warns_when_settle_exhausts_its_budget():
    fig, _ = _grid()
    auto = _force_runaway(fig, mm_per_draw=5.0)
    caught = _capture(auto.settle)
    assert len(caught) == 1, (
        f"a layout diverging by 5.00mm per draw produced {len(caught)} "
        f"warnings, expected 1"
    )


def test_warns_on_the_live_inset_axes_divergence():
    """#244, still open on main: an ``ax.inset_axes`` scope never settles.

    Kept as a test against the real thing and not only the synthetic
    runaway, because the synthetic one cannot prove the warning survives
    a genuine measurement path. When #244 is fixed this test should be
    removed along with it — it asserts a bug is still reported, not that
    the bug must remain.
    """
    fig, flat = _grid()
    inset = flat[0].inset_axes([0.55, 0.55, 0.4, 0.4])
    pp.scatterplot(data=DF, x="x", y="y", hue="g", ax=inset)
    pp.legend(anchor=inset, axes=[inset], side="right")
    auto = fig._publiplots_auto_layout

    caught = _capture(auto.settle)
    assert len(caught) == 1, (
        f"the #244 inset_axes runaway produced {len(caught)} warnings. "
        "If this is 0, the likely cause is that #244 was FIXED, not that the "
        "warning broke — check whether an inset_axes scope now converges, and "
        "if so delete this test rather than the warning."
    )
    assert "right[0]" in str(caught[0].message)


def test_the_warning_is_a_userwarning_subclass():
    """So a user's existing broad filter keeps catching it."""
    assert issubclass(pp.LayoutConvergenceWarning, UserWarning)


def test_the_warning_is_filterable_on_its_own_category():
    """The documented escape hatch must actually silence it."""
    fig, _ = _grid()
    auto = _force_runaway(fig, mm_per_draw=5.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=pp.LayoutConvergenceWarning)
        auto.settle()
    assert not [
        w for w in rec
        if issubclass(w.category, pp.LayoutConvergenceWarning)
    ]


# --- it does not fire on convergence ------------------------------------

@pytest.mark.parametrize("side", ["right", "left", "top", "bottom"])
def test_silent_on_a_converging_band(side):
    fig, flat = _grid(1, 3)
    pp.legend(anchor=flat[1], axes=flat, side=side)
    auto = fig._publiplots_auto_layout
    caught = _capture(auto.settle)
    assert not caught, f"{side} band warned: {caught[0].message}"


def test_silent_on_a_figure_with_no_legend():
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.lineplot(data=DF.sort_values("x"), x="x", y="y", ax=ax)
    caught = _capture(fig._publiplots_auto_layout.settle)
    assert not caught, str(caught[0].message) if caught else ""


@pytest.mark.parametrize("dpi", [72, 100, 150, 300, 600])
def test_silent_across_a_savefig_loop_at_any_dpi(tmp_path, dpi):
    """The plausible false positive: warning only on save, at a dpi the
    rest of the suite never exercises.

    ``settle()`` runs inside the ``print_figure`` wrapper *before*
    matplotlib swaps ``fig.dpi`` to the render's, so its check always
    evaluates at ``figure.dpi``; a render at another dpi must leave no
    hysteresis behind for the next one to trip over.
    """
    fig, flat = _grid(1, 3)
    pp.legend(anchor=flat[0], axes=flat, side="right")

    def _save():
        for i in range(4):
            fig.savefig(tmp_path / f"s{i}.png", dpi=dpi)

    caught = _capture(_save)
    assert not caught, f"dpi={dpi} warned: {caught[0].message}"


def test_silent_on_a_residual_below_the_floor():
    """A layout that never settles but only ever moves half a millimetre
    is inside the range dpi-dependent text metrics can produce, so it is
    reported as noise, not as a bug."""
    fig, _ = _grid()
    auto = _force_runaway(fig, mm_per_draw=0.5)
    caught = _capture(auto.settle)
    assert not caught, (
        f"a 0.50mm residual warned, below the "
        f"{al._NONCONVERGENCE_WARN_MM}mm floor: {caught[0].message}"
    )


# --- once per figure ----------------------------------------------------

def test_warns_once_per_figure_across_repeated_saves(tmp_path):
    """A user saving a broken figure in a loop gets one warning, not N."""
    fig, _ = _grid()
    _force_runaway(fig, mm_per_draw=5.0)

    def _save():
        for i in range(5):
            fig.savefig(tmp_path / f"s{i}.png")

    caught = _capture(_save)
    assert len(caught) == 1, f"5 saves produced {len(caught)} warnings"


def test_two_broken_figures_each_get_their_own_warning():
    """Once *per figure*, not once per process."""
    def _both():
        for _ in range(2):
            fig, _ = _grid()
            _force_runaway(fig, mm_per_draw=5.0).settle()

    assert len(_capture(_both)) == 2


# --- the message is actionable ------------------------------------------

def test_the_message_names_the_field_and_its_residual_in_mm():
    fig, _ = _grid()
    auto = _force_runaway(fig, mm_per_draw=5.0)
    caught = _capture(auto.settle)
    assert len(caught) == 1
    msg = str(caught[0].message)

    assert re.search(r"\bright\[\d+\]", msg), (
        f"message names no offending field: {msg}"
    )
    assert re.search(r"5\.00 mm per draw", msg), (
        f"message does not quote the residual in mm: {msg}"
    )
    assert str(al._UPDATE_THRESHOLD_MM) in msg, (
        f"message does not quote the tolerance the residual is measured "
        f"against: {msg}"
    )
    assert "did not converge" in msg
    assert "not reproducible" in msg
    assert "LayoutConvergenceWarning" in msg, (
        f"message does not say how to silence it: {msg}"
    )


def test_worst_residual_names_the_offending_cell():
    """The residual is reported in ``_needs_update``'s own terms: an
    absolute millimetre deviation, per position for the tuple sides."""
    fig, _ = _grid(1, 3)
    auto = fig._publiplots_auto_layout
    auto.settle()
    current = auto._layout.right
    bumped = (current[0], current[1], current[2] + 4.25)
    field, residual = auto._worst_residual({"right": bumped})
    assert field == "right[2]"
    assert residual == pytest.approx(4.25)


def test_worst_residual_reports_a_scalar_side_without_an_index():
    fig, _ = _grid(2, 2)
    auto = fig._publiplots_auto_layout
    auto.settle()
    field, residual = auto._worst_residual(
        {"legend_column": auto._layout.legend_column + 3.0}
    )
    assert field == "legend_column"
    assert residual == pytest.approx(3.0)
