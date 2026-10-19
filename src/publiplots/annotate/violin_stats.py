"""Value-label strategy for violinplots.

The same statistics and records as box_stats — violins and boxes both
produce BoxStatsMeta. The default anchor differs: "inside", on the forward
side of the median line inside the body, where box_stats keeps "right"
because a label fits inside a violin but not inside a ~3 mm box. On a
dodged vertical violin the label turns 90 degrees unless the caller passes
rotation=.
"""
from __future__ import annotations

from typing import List

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate.box_stats import _stats_strategy


def _violin_stats_strategy(ax: Axes, **kwargs) -> List[Text]:
    return _stats_strategy(ax, kind="violin_stats", default_anchor="inside", **kwargs)
