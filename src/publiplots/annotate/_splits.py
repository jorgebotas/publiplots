"""Shared barplot dodge-spec resolution.

Single source of truth for: given (x, y, hue, hatch, categorical_axis),
which dimensions actually cause seaborn to dodge bars?

Used by:
- `publiplots.plot.bar.barplot` to decide the `sns_hue` argument and
  the flags driving `_apply_hatches_and_override_colors`.
- `publiplots.annotate._builders.build_from_barplot_call` to iterate
  data in seaborn's draw order and pair bars with aggregated rows.

Keeping these two call sites in sync by construction prevents the class
of bug where the plotter and the annotator disagree on how many bars
exist or in what order they were drawn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Tuple


def _categories_in_draw_order(series) -> list:
    """Ordered category labels seaborn would use.

    Honors pd.Categorical's explicit order; falls back to first-occurrence
    order for object/str dtypes.
    """
    if hasattr(series, "cat"):
        return list(series.cat.categories)
    seen = []
    for v in series:
        if v not in seen:
            seen.append(v)
    return seen


@dataclass
class BarSplitSpec:
    """Resolved dodge spec for a barplot call.

    Fields:
        categorical_axis: "x" or "y" column name; the one that carries
            category labels for bar positions.
        orient: "v" (vertical bars) or "h" (horizontal).
        split_hue: the `hue` column only if it genuinely splits bars —
            i.e. hue is given, distinct from the categorical axis. `None`
            otherwise.
        split_hatch: the `hatch` column only if it genuinely splits —
            distinct from the categorical axis AND distinct from
            `split_hue`. `None` otherwise.
    """
    categorical_axis: str
    orient: Literal["v", "h"]
    split_hue: Optional[str]
    split_hatch: Optional[str]

    @classmethod
    def resolve(
        cls,
        x: str,
        y: str,
        hue: Optional[str],
        hatch: Optional[str],
        categorical_axis: str,
    ) -> "BarSplitSpec":
        split_hue = hue if (hue is not None and hue != categorical_axis) else None
        split_hatch = hatch if (
            hatch is not None
            and hatch != categorical_axis
            and hatch != hue
        ) else None
        orient: Literal["v", "h"] = "v" if categorical_axis == x else "h"
        return cls(
            categorical_axis=categorical_axis,
            orient=orient,
            split_hue=split_hue,
            split_hatch=split_hatch,
        )

    @property
    def n_dodge_dims(self) -> int:
        return int(self.split_hue is not None) + int(self.split_hatch is not None)

    def iter_draw_order(
        self, data
    ) -> Iterator[Tuple[object, Optional[object], Optional[object]]]:
        """Yield `(cat, hue_value, hatch_value)` triples in seaborn's draw order.

        Seaborn's iteration is outer-to-inner: split_hue > split_hatch >
        categorical axis. `hue_value` is `None` when `split_hue` is `None`;
        same for `hatch_value`. Empty (mask-less) combinations are skipped.
        """
        cats = _categories_in_draw_order(data[self.categorical_axis])

        if self.split_hue is None and self.split_hatch is None:
            for cat in cats:
                yield cat, None, None
            return

        if self.split_hue is not None and self.split_hatch is None:
            for h in _categories_in_draw_order(data[self.split_hue]):
                for cat in cats:
                    mask = (data[self.categorical_axis] == cat) & (data[self.split_hue] == h)
                    if mask.any():
                        yield cat, h, None
            return

        if self.split_hue is None and self.split_hatch is not None:
            for ht in _categories_in_draw_order(data[self.split_hatch]):
                for cat in cats:
                    mask = (data[self.categorical_axis] == cat) & (data[self.split_hatch] == ht)
                    if mask.any():
                        yield cat, None, ht
            return

        # Both split dimensions active.
        for h in _categories_in_draw_order(data[self.split_hue]):
            for ht in _categories_in_draw_order(data[self.split_hatch]):
                for cat in cats:
                    mask = (
                        (data[self.categorical_axis] == cat)
                        & (data[self.split_hue] == h)
                        & (data[self.split_hatch] == ht)
                    )
                    if mask.any():
                        yield cat, h, ht
