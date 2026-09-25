"""A layout that leaks under the residual floor must still be reported (#251).

``_warn_not_converged``'s original trigger is the worst per-draw residual's
*magnitude*, against a 1.0 mm floor that was measured rather than guessed: it
clears the 0.76 mm of purely dpi-induced drift a corpus of healthy layouts
showed, and sits two orders of magnitude below either real divergence. Neither
end of that is negotiable, so the floor stays where it is.

What it cannot see is a leak *under* it. Measured on a 1x2 grid with a constant
per-measure drift injected on ``right``, one ``pp.savefig`` each, the figure
went from 95.10 mm to 336.70 mm across 20 saves at 0.99 mm/draw — one notch
below the floor — without a word. The per-draw magnitude is simply the wrong
discriminator for a slow leak, because at 0.99 mm it cannot tell a leak from
dpi noise. Cumulative growth can, since the two differ in kind: dpi jitter is
bounded and does not accumulate across saves, while a leak accumulates — that
is what makes it a leak.

So there is now a second, independent trigger on how far the figure has drifted
from the size it settled at. These tests pin both halves of it: that the leak
the floor misses is reported, and that the things which legitimately move a
figure — a one-time first settle, a pure oscillation, repeated saves at every
dpi and format — are not. The magnitude trigger is tested here too, but only
for the property this change could plausibly have broken: that it still fires
on the *first* exhausted settle, with no accumulated growth to help it.
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


MM = 25.4

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


def _force_leak(fig, mm_per_draw):
    """Measure ``right`` ``mm_per_draw`` past whatever the layout holds.

    Reproduces the *shape* of a real runaway — a residual that stays
    exactly constant while the figure diverges — under a knob that lets a
    test sit either side of the 1.0 mm floor. Same construction as the
    synthetic runaway in ``test_settle_warning.py``, and the same one the
    #251 measurements were taken with.
    """
    auto = fig._publiplots_auto_layout

    def _measure():
        return {"right": tuple(v + mm_per_draw for v in auto._layout.right)}

    auto._measure = _measure
    return auto


def _force_oscillation(fig, mm):
    """Never settle, but hold the figure at exactly one size.

    The discriminator's near miss: a residual that is permanently
    ``mm`` — so ``_needs_update`` can never come back False — while the
    reservation flips between two values the figure has already grown to
    accommodate, so nothing accumulates. Sub-floor and non-convergent, and
    yet not a leak; the growth trigger must not report it.
    """
    auto = fig._publiplots_auto_layout
    base = tuple(auto._layout.right)
    state = {"i": 0}

    def _measure():
        state["i"] += 1
        offset = mm if state["i"] % 2 else 0.0
        return {"right": tuple(v + offset for v in base)}

    auto._measure = _measure
    return auto


def _capture(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
        return [
            w for w in rec
            if issubclass(w.category, pp.LayoutConvergenceWarning)
        ]


def _save_loop(fig, tmp_path, n, dpis=(72, 100, 150, 300, 600),
               formats=("png", "pdf", "svg")):
    def _run():
        for i in range(n):
            dpi = dpis[i % len(dpis)]
            fmt = formats[i % len(formats)]
            fig.savefig(tmp_path / f"s{i}.{fmt}", dpi=dpi)
    return _run


# --- the leak the floor misses is now reported --------------------------

@pytest.mark.parametrize("mm_per_draw", [0.2, 0.5, 0.9, 0.99])
def test_a_sub_floor_leak_warns_once_the_growth_accumulates(
    tmp_path, mm_per_draw
):
    """Every drift the 1.0 mm floor lets through, at five saves.

    0.99 is the interesting one: one notch under the floor, and on main it
    took the figure to 3.5x its declared width in silence.
    """
    fig, _ = _grid()
    _force_leak(fig, mm_per_draw)
    caught = _capture(_save_loop(fig, tmp_path, 5))
    assert len(caught) == 1, (
        f"a {mm_per_draw:.2f} mm/draw leak — below the "
        f"{al._NONCONVERGENCE_WARN_MM} mm floor — produced {len(caught)} "
        f"warnings over 5 saves, expected 1"
    )


def test_the_sub_floor_leak_is_reported_before_the_figure_doubles(tmp_path):
    """It has to fire early enough to be worth having.

    The 0.5 mm/draw leak reaches 219.10 mm — a 2.3x runaway — by save 20.
    A warning that only arrived at the end of that would be no better than
    the silence it replaces, so require it within the first three saves.
    """
    fig, _ = _grid()
    _force_leak(fig, 0.5)
    fired_at = None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for i in range(20):
            fig.savefig(tmp_path / f"s{i}.png")
            if fired_at is None and any(
                issubclass(w.category, pp.LayoutConvergenceWarning) for w in rec
            ):
                fired_at = i + 1
    assert fired_at is not None, "the 0.5 mm/draw leak never warned"
    assert fired_at <= 3, f"the leak was only reported on save {fired_at}"


def test_growth_from_a_size_the_figure_settled_at_needs_no_repeat(tmp_path):
    """A layout that converged and then started leaking warns immediately.

    Growth measured from a size the figure actually reached is not
    ambiguous the way growth from a never-converged start is: there is
    nothing legitimate left for the layout to be doing, so one exhausted
    settle is evidence enough. Measured healthy value for this quantity is
    0.0000 mm across 3960 saves.
    """
    fig, _ = _grid()
    auto = fig._publiplots_auto_layout
    auto.settle()
    assert auto._settled_size is not None, "the healthy grid never settled"

    _force_leak(fig, 0.5)
    caught = _capture(auto.settle)
    assert len(caught) == 1, (
        f"a leak starting after a clean settle produced {len(caught)} "
        f"warnings on its first exhausted settle, expected 1"
    )


def test_the_growth_origin_is_the_first_settled_size_and_is_never_refreshed():
    """``_settled_size`` is pinned to the FIRST convergence, on purpose.

    Refreshing it on every convergent settle looks tidier and removes one
    false positive (a figure legitimately re-converging larger is then
    measured from its new size), but it masks a whole class of leak: a
    layout that alternates converging and exhausting across saves while
    growing monotonically would have its origin reset to the already-grown
    size on every convergence, and the growth ever measured would be one
    save's worth. Erring toward reporting is the right bias here — #249 and
    #251 both exist because silence let real bugs ship.

    Constructed so the two designs disagree and nothing else can account
    for the result: the figure settles, then legitimately re-converges
    40 mm larger, then leaks so slowly that the final settle's *own* growth
    is under the budget. A refreshed origin therefore has nothing to report
    and this test fails; the pinned origin reports the full 40 mm-plus.
    """
    fig, _ = _grid()
    auto = fig._publiplots_auto_layout
    auto.settle()
    first_settled = fig.get_size_inches()[0] * MM

    # A second, still-convergent settle at a genuinely larger size — the
    # constant offset is absorbed in one apply, so this converges.
    base = tuple(auto._layout.right)
    auto._measure = lambda: {"right": tuple(v + 20.0 for v in base)}
    auto.settle()
    re_settled = fig.get_size_inches()[0] * MM
    assert re_settled - first_settled > al._GROWTH_BUDGET_MM, (
        "the re-convergence did not move the figure, so this test cannot "
        "tell the two origins apart"
    )

    # A leak slow enough that this settle alone stays inside the budget.
    auto._measure = lambda: {
        "right": tuple(v + 0.15 for v in auto._layout.right)
    }
    caught = _capture(auto.settle)
    own_growth = fig.get_size_inches()[0] * MM - re_settled
    assert own_growth < al._GROWTH_BUDGET_MM, (
        f"the final settle grew {own_growth:.2f} mm on its own, at or past "
        f"the {al._GROWTH_BUDGET_MM} mm budget — it would warn under either "
        f"origin and this test proves nothing"
    )

    assert len(caught) == 1, (
        f"growth was measured from the re-converged size, not the first "
        f"settled one: a {own_growth:.2f} mm settle produced "
        f"{len(caught)} warnings"
    )
    assert auto._settled_size[0] * MM == pytest.approx(first_settled), (
        "the growth origin was refreshed by the second convergent settle"
    )
    m = re.search(r"has grown (\d+\.\d\d) mm", str(caught[0].message))
    assert m, f"no accumulated growth quoted: {caught[0].message}"
    assert float(m.group(1)) == pytest.approx(
        re_settled - first_settled + own_growth, abs=0.02
    ), (
        f"the warning quotes {m.group(1)} mm, not the drift from the first "
        f"settled size: {caught[0].message}"
    )
    assert "since it first settled" in str(caught[0].message)


# --- and the things that legitimately move a figure are not -------------

def test_a_converging_layout_saved_twenty_times_never_warns(tmp_path):
    """The false positive that would matter: 20 saves, five dpis, three
    formats, on a layout whose convergence ``test_band_convergence.py``
    already pins."""
    fig, flat = _grid(1, 3)
    pp.legend(anchor=flat[1], axes=flat, side="right")
    caught = _capture(_save_loop(fig, tmp_path, 20))
    assert not caught, (
        f"a converging band warned across 20 saves: {caught[0].message}"
    )


@pytest.mark.parametrize("side", ["right", "left", "top", "bottom"])
def test_bands_on_every_side_stay_silent_across_repeated_saves(tmp_path, side):
    fig, flat = _grid(2, 2)
    pp.legend(side=side, figure=fig)
    caught = _capture(_save_loop(fig, tmp_path, 10))
    assert not caught, f"the {side} band warned: {caught[0].message}"


def test_a_no_legend_figure_stays_silent_across_repeated_saves(tmp_path):
    fig, ax = pp.subplots(1, 1, axes_size=(50, 40))
    pp.lineplot(data=DF.sort_values("x"), x="x", y="y", ax=ax)
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    ax.set_title("a title")
    caught = _capture(_save_loop(fig, tmp_path, 10))
    assert not caught, str(caught[0].message) if caught else ""


def test_a_pure_oscillation_is_not_growth(tmp_path):
    """Sub-floor, never convergent, and yet not accumulating.

    This is the case that distinguishes "the figure grew" from "the layout
    is unhappy", and the reason the budget is a budget and not zero: the
    reservation flips forever between two values, so the residual never
    falls under the tolerance, but the figure holds exactly one size. Three
    exhausted settles, so the repeat requirement is satisfied and the
    budget is the only thing left standing between this and a warning.
    """
    fig, _ = _grid()
    auto = _force_oscillation(fig, 0.3)
    caught = _capture(lambda: [auto.settle() for _ in range(3)])
    assert not caught, (
        f"a non-convergent layout holding one figure size warned as growth: "
        f"{caught[0].message}"
    )


def test_one_exhausted_settle_alone_does_not_trip_the_growth_trigger(tmp_path):
    """Deliberate, and the limit of what this can detect.

    A single exhausted settle on a layout that has never converged shows
    growth that is indistinguishable from a legitimate first settle running
    long — worth up to 68.00 mm on the healthy corpus. Healthy layouts do
    not come close to the 5-draw cap (worst measured: 3), but the headroom
    is two draws, so a layout marginally more complex than anything
    measured could exhaust once while still terminating. The growth trigger
    therefore waits for the drift to repeat across a second save. The cost
    is that a sub-floor leak saved exactly once is still silent; a single
    save has no reproducibility symptom to report, and the compounding this
    guards against needs repeated saves by definition.
    """
    fig, _ = _grid()
    _force_leak(fig, 0.5)
    caught = _capture(_save_loop(fig, tmp_path, 1))
    assert not caught, (
        f"one exhausted settle warned on growth alone: {caught[0].message}"
    )


# --- the magnitude trigger keeps its fast path --------------------------

def test_a_supra_floor_residual_still_warns_on_the_first_settle():
    """#230 and #244 must not be reported one save later than before.

    A single ``settle()``, so there is no accumulated growth and no second
    exhausted settle to lean on: whatever fires here is the magnitude
    trigger, unassisted.
    """
    fig, _ = _grid()
    auto = _force_leak(fig, 5.0)
    caught = _capture(auto.settle)
    assert len(caught) == 1, (
        f"a 5.00 mm/draw runaway produced {len(caught)} warnings on its "
        f"first settle, expected 1"
    )
    assert auto._exhausted_settles == 1


def test_the_floor_itself_has_not_moved():
    """The growth trigger is an addition, not a relaxation.

    A residual in the ambiguous 0.762-1.0 mm band is still not reported on
    magnitude, and not reported at a lower severity either: a second
    severity would split one actionable signal in two for a band no layout
    has been observed in, and the growth trigger already covers that band's
    real risk — along with the rest of ``(tolerance, floor)``.
    """
    assert al._NONCONVERGENCE_WARN_MM == 1.0
    fig, _ = _grid()
    auto = _force_leak(fig, 0.9)
    caught = _capture(auto.settle)
    assert not caught, (
        f"a 0.90 mm residual warned on its first settle: {caught[0].message}"
    )


# --- the message says growth is what tripped it -------------------------

def _growth_message(tmp_path):
    fig, _ = _grid()
    _force_leak(fig, 0.5)
    caught = _capture(_save_loop(fig, tmp_path, 5))
    assert len(caught) == 1
    return str(caught[0].message)


def test_the_growth_message_names_growth_as_the_cause(tmp_path):
    msg = _growth_message(tmp_path)
    assert "cumulative growth" in msg, (
        f"a growth-triggered warning does not say growth tripped it: {msg}"
    )
    assert "not the per-draw residual" in msg, (
        f"the message leaves the sub-floor residual looking like the "
        f"reason: {msg}"
    )


def test_the_growth_message_quotes_the_accumulated_millimetres(tmp_path):
    """Without the number there is nothing to act on: a 0.50 mm residual
    reads as noise, and only the accumulated total shows it is not."""
    msg = _growth_message(tmp_path)
    m = re.search(r"has grown (\d+\.\d\d) mm", msg)
    assert m, f"the message quotes no accumulated growth: {msg}"
    grown = float(m.group(1))
    assert grown >= al._GROWTH_BUDGET_MM, (
        f"the message quotes {grown} mm, under the "
        f"{al._GROWTH_BUDGET_MM} mm budget it claims to have passed: {msg}"
    )
    assert re.search(r"\([-+]\d+\.\d\d x [-+]\d+\.\d\d mm\)", msg), (
        f"the message does not break the growth into width and height: {msg}"
    )
    assert f"{al._GROWTH_BUDGET_MM} mm budget" in msg, (
        f"the message does not say what budget was passed: {msg}"
    )


def test_the_growth_message_keeps_the_original_diagnostics(tmp_path):
    """A growth-triggered report is still a non-convergence report: the
    field, its residual and the tolerance are what a maintainer needs to
    find the leak, and dropping them would trade one blind spot for
    another."""
    msg = _growth_message(tmp_path)
    assert re.search(r"\bright\[\d+\]", msg), f"names no field: {msg}"
    assert re.search(r"0\.50 mm per draw", msg), f"no residual: {msg}"
    assert str(al._UPDATE_THRESHOLD_MM) in msg, f"no tolerance: {msg}"
    assert "not reproducible" in msg
    assert "LayoutConvergenceWarning" in msg, f"no escape hatch: {msg}"


# --- once per figure ----------------------------------------------------

def test_a_growth_warning_fires_once_per_figure(tmp_path):
    """Ten saves of a leaking figure, one warning."""
    fig, _ = _grid()
    _force_leak(fig, 0.5)
    caught = _capture(_save_loop(fig, tmp_path, 10))
    assert len(caught) == 1, f"10 saves produced {len(caught)} warnings"


def test_two_leaking_figures_each_get_their_own_growth_warning(tmp_path):
    """Once *per figure*, not once per process — the counter and the
    growth origin are per-figure state, and a shared one would report the
    first figure and hide the second."""
    def _both():
        for n in range(2):
            fig, _ = _grid()
            _force_leak(fig, 0.5)
            for i in range(5):
                fig.savefig(tmp_path / f"f{n}s{i}.png")

    assert len(_capture(_both)) == 2


def test_the_growth_warning_is_filterable_on_its_own_category(tmp_path):
    fig, _ = _grid()
    _force_leak(fig, 0.5)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=pp.LayoutConvergenceWarning)
        _save_loop(fig, tmp_path, 5)()
    assert not [
        w for w in rec
        if issubclass(w.category, pp.LayoutConvergenceWarning)
    ]
