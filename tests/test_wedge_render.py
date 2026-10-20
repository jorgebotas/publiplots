import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp
from publiplots.plot.wedge.render import render_wedge_chart


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


DF = pd.DataFrame({"kind": list("ABCD"), "n": [50.0, 25.0, 15.0, 10.0]})


def _render(ax, **overrides):
    kwargs = dict(
        owner="pieplot", source_frame=DF, y="n", hue=None, order_map=None,
        sort=False, hue_order=None, start_angle=90.0, direction="clockwise",
        donut=0.0, pull=None, gap=0.0, palette=None, hatch=None,
        hatch_map=None, hatch_order=None, color="#5d83c3", edgecolor=None,
        linewidth=0.75, alpha=0.1, center_label=None, aspect="cell",
        legend=False, legend_kws=None, title="",
    )
    kwargs.update(overrides)
    return render_wedge_chart(ax, DF, ["kind"], **kwargs)


def test_render_draws_what_pieplot_draws():
    _, a = pp.subplots(axes_size=(40, 40))
    pp.pieplot(data=DF, x="kind", y="n", legend=False, ax=a)
    _, b = pp.subplots(axes_size=(40, 40))
    meta = _render(b)
    assert b._publiplots_wedge_meta is meta
    pa = [p.get_path().vertices for p in a.patches]
    pb = [p.get_path().vertices for p in b.patches]
    assert len(pa) == len(pb) == 4
    for va, vb in zip(pa, pb):
        assert va.tolist() == vb.tolist()
    assert [tuple(p.get_facecolor()) for p in a.patches] == \
           [tuple(p.get_facecolor()) for p in b.patches]


def test_messages_name_the_owner():
    _, ax = pp.subplots(axes_size=(40, 40))
    with pytest.raises(TypeError, match=r"^sunburstplot: pull="):
        _render(ax, owner="sunburstplot", pull="wide")


@pytest.mark.parametrize("kws", [
    {}, {"donut": 0.5}, {"pull": 0.05}, {"gap": 1.0},
    {"pull": {"A": 0.1}}, {"pull": 0.05, "gap": 1.0, "donut": 0.4},
])
def test_meta_radii_and_angles_are_read_off_the_drawn_patch(kws):
    _, ax = pp.subplots(axes_size=(40, 40))
    meta = _render(ax, **kws)
    for rec in meta.wedges:
        assert rec.r_outer == rec.patch.r
        assert rec.r_inner == pytest.approx(rec.patch.r - rec.patch.width)
        assert rec.theta0 == rec.patch.theta1
        assert rec.theta1 == rec.patch.theta2


def test_meta_is_built_after_the_gap_is_applied(monkeypatch):
    """Plan 2's radial insets move radii in apply_gap; the meta must see them."""
    import publiplots.plot.wedge.render as render
    order = []
    real_gap, real_meta = render.apply_gap, render.build_wedge_meta
    monkeypatch.setattr(render, "apply_gap",
                        lambda *a, **k: (order.append("gap"), real_gap(*a, **k))[1])
    monkeypatch.setattr(render, "build_wedge_meta",
                        lambda *a, **k: (order.append("meta"), real_meta(*a, **k))[1])
    _, ax = pp.subplots(axes_size=(40, 40))
    _render(ax, gap=1.0)
    assert order == ["gap", "meta"]
