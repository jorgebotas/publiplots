"""The ``annotate=`` examples in plot docstrings must actually run.

``pp.boxplot`` / ``pp.violinplot`` forward an ``annotate`` dict to
``pp.annotate``; an example naming an option that does not exist (they once
showed ``{"comparisons": ...}``) fails as a matplotlib ``Text`` property
error, far from the docstring that suggested it.
"""
from __future__ import annotations

import ast
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


_ANNOTATE_DICT = re.compile(r"annotate=(\{.*?\})")


def _documented_annotate_dicts(func):
    # Join doctest continuation lines so multi-line calls match too.
    doc = re.sub(r"\n\s*\.\.\.\s*", " ", func.__doc__)
    return [ast.literal_eval(m) for m in _ANNOTATE_DICT.findall(doc)]


@pytest.mark.parametrize("func", [pp.boxplot, pp.violinplot],
                         ids=lambda f: f.__name__)
def test_documented_annotate_examples_run(func):
    examples = _documented_annotate_dicts(func)
    assert examples, f"{func.__name__} documents no annotate= example"
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "category": np.repeat(list("ABC"), 20),
        "value": rng.normal(size=60),
    })
    for opts in examples:
        _, ax = pp.subplots(axes_size=(40, 40))
        func(data=df, x="category", y="value", ax=ax, annotate=opts)
        assert ax.texts, f"{func.__name__}(annotate={opts!r}) drew no labels"
