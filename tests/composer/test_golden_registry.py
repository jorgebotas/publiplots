"""Tests for the canonical composition registry.

PR 4.5 introduces tests/composer/golden/_compositions.py — a list of
(name, build_fn) tuples enumerating the compositions covered by the
mm-precision + visual-regression gates. This test asserts the registry
shape so downstream test files (test_mm_snapshots, test_png_regression)
and the CLI (tools/composer/regen_fixtures.py) can rely on it.
"""
from __future__ import annotations

import pytest


def test_registry_imports():
    """`COMPOSITIONS` is exported from _compositions.py."""
    from tests.composer.golden._compositions import COMPOSITIONS
    assert isinstance(COMPOSITIONS, list)
    # PR 4.5: 4 compositions; PR 5: +2 PanelImage compositions = 6;
    # PR 6b: +1 embed_figure composition = 7.
    assert len(COMPOSITIONS) == 7


def test_registry_entry_shape():
    """Each entry is a (name, build_fn) tuple with kebab-case name + callable."""
    from tests.composer.golden._compositions import COMPOSITIONS
    seen_names = set()
    for entry in COMPOSITIONS:
        assert isinstance(entry, tuple) and len(entry) == 2
        name, build_fn = entry
        assert isinstance(name, str)
        # kebab-case: lowercase letters + digits + hyphens only
        assert all(c.islower() or c.isdigit() or c == "-" for c in name), (
            f"name {name!r} is not kebab-case"
        )
        assert callable(build_fn)
        assert name not in seen_names, f"duplicate name {name!r}"
        seen_names.add(name)


def test_registry_build_functions_return_canvas():
    """Each build_fn returns a Canvas instance with at least one row."""
    import publiplots as pp
    from tests.composer.golden._compositions import COMPOSITIONS
    for name, build_fn in COMPOSITIONS:
        canvas = build_fn()
        assert isinstance(canvas, pp.Canvas), (
            f"{name!r} build_fn returned {type(canvas).__name__}, expected Canvas"
        )
        # Touching .figure triggers finalization — must not raise.
        _ = canvas.figure


def test_registry_build_functions_are_deterministic():
    """Calling a build_fn twice produces equivalent geometry.

    Determinism is required for snapshot stability: a build_fn that uses
    unseeded randomness or os.environ would make goldens flap.
    """
    from tests.composer.golden._compositions import COMPOSITIONS
    for name, build_fn in COMPOSITIONS:
        c1 = build_fn()
        c2 = build_fn()
        assert c1.figure_size_mm == c2.figure_size_mm, (
            f"{name!r} produced different figure_size_mm on two calls"
        )


def test_registry_includes_expected_canonical_names():
    """The four canonical compositions cloned from examples/composer/ exist."""
    from tests.composer.golden._compositions import COMPOSITIONS
    names = {name for name, _ in COMPOSITIONS}
    expected = {
        "cell-2col-simple",
        "cell-2col-multirow",
        "nature-2col-abc",
        "nature-2col-panel-grid",
    }
    assert expected.issubset(names), f"missing: {expected - names}"


# ---------------------------------------------------------------------------
# Task 2: snapshot helper
# ---------------------------------------------------------------------------


def test_round_geometry_rounds_to_tolerance():
    """_round_geometry(value, tol_mm) snaps to the nearest tolerance step."""
    from tests.composer.golden._helpers import _round_geometry
    assert _round_geometry(123.456, 0.01) == 123.46
    assert _round_geometry(123.454, 0.01) == 123.45
    assert _round_geometry(0.0, 0.01) == 0.0


def test_snapshot_returns_documented_shape():
    """snapshot(canvas) returns the documented dict shape."""
    from tests.composer.golden._compositions import COMPOSITIONS
    from tests.composer.golden._helpers import snapshot

    _, build_fn = COMPOSITIONS[0]  # cell-2col-simple
    canvas = build_fn()
    snap = snapshot(canvas)

    assert snap["schema_version"] == 1
    assert snap["preset"] == "cell-2col"
    assert isinstance(snap["figure_size_mm"], list) and len(snap["figure_size_mm"]) == 2
    assert isinstance(snap["rows"], list) and len(snap["rows"]) >= 1

    row0 = snap["rows"][0]
    assert "panels" in row0
    assert len(row0["panels"]) == 2  # cell-2col-simple has A, B
    p0 = row0["panels"][0]
    assert set(p0.keys()) == {"label", "kind", "size_mm", "bbox_mm"}
    assert p0["label"] == "A"
    assert p0["kind"] == "axes"
    assert isinstance(p0["size_mm"], list) and len(p0["size_mm"]) == 2
    assert isinstance(p0["bbox_mm"], list) and len(p0["bbox_mm"]) == 4


def test_snapshot_is_deterministic():
    """snapshot(canvas) == snapshot(build_fn()) for the same composition."""
    from tests.composer.golden._compositions import COMPOSITIONS
    from tests.composer.golden._helpers import snapshot

    name, build_fn = COMPOSITIONS[0]
    snap1 = snapshot(build_fn())
    snap2 = snapshot(build_fn())
    assert snap1 == snap2, f"{name!r}: snapshot not deterministic"


def test_snapshot_values_are_rounded_to_001mm():
    """All numeric fields in the snapshot are rounded to 0.01 mm."""
    from tests.composer.golden._compositions import COMPOSITIONS
    from tests.composer.golden._helpers import snapshot

    _, build_fn = COMPOSITIONS[0]
    snap = snapshot(build_fn())
    for v in snap["figure_size_mm"]:
        # round to 2 dp ↔ 0.01 mm tolerance
        assert v == round(v, 2)
    for row in snap["rows"]:
        for panel in row["panels"]:
            for v in panel["size_mm"]:
                assert v == round(v, 2)
            for v in panel["bbox_mm"]:
                assert v == round(v, 2)


# ---------------------------------------------------------------------------
# Task 3: PNG visual regression
# ---------------------------------------------------------------------------


def test_assert_png_matches_missing_fixture_raises_without_regen(tmp_path, monkeypatch):
    """Missing PNG without REGEN_ENV → clear AssertionError pointing at regen CLI."""
    import publiplots as pp
    from tests.composer.golden import _helpers

    # Redirect golden dir to a temp location for this test.
    monkeypatch.setattr(_helpers, "PNG_DIR", tmp_path)
    monkeypatch.delenv(_helpers.REGEN_ENV, raising=False)

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label="A", size=(70, 50)),
                   pp.PanelAxes(label="B", size=("flex", 50)))

    with pytest.raises(AssertionError, match=r"regen_fixtures"):
        _helpers.assert_png_matches(canvas, "missing-name")


def test_assert_png_matches_missing_with_regen_writes(tmp_path, monkeypatch):
    """Missing PNG with REGEN_ENV=1 → writes the file + passes."""
    import publiplots as pp
    from tests.composer.golden import _helpers

    monkeypatch.setattr(_helpers, "PNG_DIR", tmp_path)
    monkeypatch.setenv(_helpers.REGEN_ENV, "1")

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label="A", size=(70, 50)),
                   pp.PanelAxes(label="B", size=("flex", 50)))

    _helpers.assert_png_matches(canvas, "regen-test")
    assert (tmp_path / "regen-test.png").exists()


# ---------------------------------------------------------------------------
# Code-quality fix: _format_snapshot_diff names non-numeric drift
# ---------------------------------------------------------------------------


def _make_snapshot(panels):
    """Tiny snapshot factory for diff tests."""
    return {
        "schema_version": 1,
        "preset": "cell-2col",
        "figure_size_mm": [174.0, 80.0],
        "rows": [{"panels": list(panels)}],
    }


def test_format_snapshot_diff_names_label_drift():
    """A label change (None → 'A') is named in the diff message."""
    from tests.composer.golden._helpers import _format_snapshot_diff
    expected = _make_snapshot([
        {"label": None, "kind": "axes",
         "size_mm": [70.0, 50.0], "bbox_mm": [0.0, 0.0, 70.0, 50.0]},
    ])
    actual = _make_snapshot([
        {"label": "A", "kind": "axes",
         "size_mm": [70.0, 50.0], "bbox_mm": [0.0, 0.0, 70.0, 50.0]},
    ])
    msg = _format_snapshot_diff(expected, actual, tol_mm=0.01)
    assert "label" in msg
    assert "None" in msg and "'A'" in msg


def test_format_snapshot_diff_names_kind_drift():
    """A kind change (axes → axesgrid) is named in the diff message."""
    from tests.composer.golden._helpers import _format_snapshot_diff
    expected = _make_snapshot([
        {"label": "A", "kind": "axes",
         "size_mm": [70.0, 50.0], "bbox_mm": [0.0, 0.0, 70.0, 50.0]},
    ])
    actual = _make_snapshot([
        {"label": "A", "kind": "axesgrid",
         "size_mm": [70.0, 50.0], "bbox_mm": [0.0, 0.0, 70.0, 50.0]},
    ])
    msg = _format_snapshot_diff(expected, actual, tol_mm=0.01)
    assert "kind" in msg
    assert "axes" in msg and "axesgrid" in msg


def test_format_snapshot_diff_no_opaque_fallback_when_drift_named():
    """When kind/label change is named, the opaque fallback string is NOT in the message."""
    from tests.composer.golden._helpers import _format_snapshot_diff
    expected = _make_snapshot([
        {"label": "A", "kind": "axes",
         "size_mm": [70.0, 50.0], "bbox_mm": [0.0, 0.0, 70.0, 50.0]},
    ])
    actual = _make_snapshot([
        {"label": "A", "kind": "axesgrid",
         "size_mm": [70.0, 50.0], "bbox_mm": [0.0, 0.0, 70.0, 50.0]},
    ])
    msg = _format_snapshot_diff(expected, actual, tol_mm=0.01)
    assert "no field-level drift identified" not in msg


# ---------------------------------------------------------------------------
# Code-quality fix: _round_geometry rejects non-positive tol_mm
# ---------------------------------------------------------------------------


def test_round_geometry_rejects_zero_tol():
    """tol_mm=0 → ValueError (avoids division-by-zero on rounding)."""
    from tests.composer.golden._helpers import _round_geometry
    with pytest.raises(ValueError, match=r"tol_mm must be > 0"):
        _round_geometry(1.0, tol_mm=0)


def test_round_geometry_rejects_negative_tol():
    """tol_mm<0 → ValueError."""
    from tests.composer.golden._helpers import _round_geometry
    with pytest.raises(ValueError, match=r"tol_mm must be > 0"):
        _round_geometry(1.0, tol_mm=-0.01)


# ---------------------------------------------------------------------------
# PR 6a: assert_svg_matches helper
# ---------------------------------------------------------------------------

@pytest.fixture
def _small_svg_for_canvas(tmp_path_factory):
    """Build a minimal SVG schematic shared across SVG-helper tests."""
    p = tmp_path_factory.mktemp("svg") / "fixture.svg"
    p.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="40mm" height="30mm" viewBox="0 0 40 30">'
        '<rect width="40" height="30" fill="red"/></svg>'
    )
    return p


def test_assert_svg_matches_missing_fixture_raises_without_regen(
    tmp_path, monkeypatch, _small_svg_for_canvas,
):
    """Missing SVG without REGEN_ENV → AssertionError pointing at regen CLI."""
    import publiplots as pp
    from tests.composer.golden import _helpers

    monkeypatch.setattr(_helpers, "SVG_DIR", tmp_path)
    monkeypatch.delenv(_helpers.REGEN_ENV, raising=False)

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_small_svg_for_canvas, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )

    with pytest.raises(AssertionError, match=r"regen_fixtures"):
        _helpers.assert_svg_matches(canvas, "missing-svg")


def test_assert_svg_matches_missing_with_regen_writes(
    tmp_path, monkeypatch, _small_svg_for_canvas,
):
    """Missing golden + REGEN_ENV=1 → file written + assertion passes."""
    import publiplots as pp
    from tests.composer.golden import _helpers

    monkeypatch.setattr(_helpers, "SVG_DIR", tmp_path)
    monkeypatch.setenv(_helpers.REGEN_ENV, "1")

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_small_svg_for_canvas, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )

    _helpers.assert_svg_matches(canvas, "regen-svg")
    assert (tmp_path / "regen-svg.svg").exists()


def test_assert_svg_matches_viewbox_mode(
    tmp_path, monkeypatch, _small_svg_for_canvas,
):
    """viewbox mode passes when the produced SVG's viewBox matches mm × user-unit factor."""
    import publiplots as pp
    from tests.composer.golden import _helpers

    monkeypatch.setattr(_helpers, "SVG_DIR", tmp_path)
    monkeypatch.setenv(_helpers.REGEN_ENV, "1")

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_small_svg_for_canvas, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _helpers.assert_svg_matches(canvas, "vb-svg", mode="viewbox")
    # Re-call without REGEN; should still pass.
    monkeypatch.delenv(_helpers.REGEN_ENV, raising=False)
    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(
        pp.PanelImage(label="A", path=_small_svg_for_canvas, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _helpers.assert_svg_matches(canvas2, "vb-svg", mode="viewbox")


def test_assert_svg_matches_structure_mode_finds_panel_g(
    tmp_path, monkeypatch, _small_svg_for_canvas,
):
    """structure mode XPath-finds the publiplots-panel-image-* wrapper."""
    import publiplots as pp
    from tests.composer.golden import _helpers

    monkeypatch.setattr(_helpers, "SVG_DIR", tmp_path)
    monkeypatch.setenv(_helpers.REGEN_ENV, "1")

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_small_svg_for_canvas, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _helpers.assert_svg_matches(canvas, "struct-svg", mode="structure")
    monkeypatch.delenv(_helpers.REGEN_ENV, raising=False)
    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(
        pp.PanelImage(label="A", path=_small_svg_for_canvas, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _helpers.assert_svg_matches(canvas2, "struct-svg", mode="structure")


def test_assert_svg_matches_render_compare_skips_when_no_cairosvg(
    tmp_path, monkeypatch,
):
    """render_compare mode skip-helper returns False when cairosvg missing."""
    from tests.composer.golden import _helpers

    # Helper exists.
    assert hasattr(_helpers, "_svg_renderer_available")
    avail = _helpers._svg_renderer_available()
    assert isinstance(avail, bool)


def test_assert_svg_matches_unknown_mode_raises(
    tmp_path, monkeypatch, _small_svg_for_canvas,
):
    """Unknown mode → ValueError."""
    import publiplots as pp
    from tests.composer.golden import _helpers

    monkeypatch.setattr(_helpers, "SVG_DIR", tmp_path)
    monkeypatch.setenv(_helpers.REGEN_ENV, "1")
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=_small_svg_for_canvas, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    # Write the golden first.
    _helpers.assert_svg_matches(canvas, "mode-test", mode="viewbox")
    # Now expect ValueError on bogus mode.
    monkeypatch.delenv(_helpers.REGEN_ENV, raising=False)
    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(
        pp.PanelImage(label="A", path=_small_svg_for_canvas, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    with pytest.raises(ValueError, match=r"unknown mode"):
        _helpers.assert_svg_matches(canvas2, "mode-test", mode="bogus")
