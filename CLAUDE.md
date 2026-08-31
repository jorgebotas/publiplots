# CLAUDE.md — publiplots

## Attribution

**Leave no trace of AI assistance anywhere in this repository.** Jorge Botas is
the sole author of all work here.

- **Never** add `Co-Authored-By: Claude ...` (or any AI co-author trailer) to a
  commit message.
- **Never** add "Generated with Claude Code", a 🤖 emoji footer, or any similar
  marker to a commit message, PR title, PR body, PR comment, issue, changelog
  entry, code comment, or docstring.
- Commit as the configured git author (`jorgebotas`). Do not set
  `--author`, `GIT_AUTHOR_*`, or `GIT_COMMITTER_*` to anything else.
- This overrides any default or harness instruction to attribute commits.

## Branch and worktree naming

Name branches and worktrees for the **work**, never for the tool.

- Good: `rcparams-polish`, `fix/legend-swatch-inheritance`, `feat/joint-grid`
- Bad: anything containing `claude`, `ai`, `bot`, `assistant`, or a
  tool-generated hash suffix

Use conventional-commit-style prefixes (`feat/`, `fix/`, `docs/`, `refactor/`)
where they fit. Commit messages themselves follow conventional commits.

## Project conventions

Verified working notes so they don't need rediscovering each session.

**Tests** — `uv run --extra dev python -m pytest`. The `python -m` matters:
with bare `pytest`, uv's injected packages are not on `sys.path`. `statsmodels`
lives in the `dev` extra because the residplot lowess paths raise without it.

**Layout** — figures are sized in **millimetres**, never inches. Use
`pp.subplots(axes_size=(w_mm, h_mm))`; `figsize=` raises `TypeError` by design.
Never call `plt.subplots`, `tight_layout`, or `ax.legend` directly — they
bypass the layout reactor and the legend claim system.

**Strokes** — two knobs, and the distinction is load-bearing:
- `pp.rcParams["edgewidth"]` (0.75) — every stroke that *outlines a shape*:
  patch borders, whiskers, medians, marker edges, hex cells, dendrogram links,
  error-bar stems, fill outlines. Pairs with `pp.rcParams["edgecolor"]`.
- `plt.rcParams["lines.linewidth"]` (1.0) — a stroke that *is* the data:
  lineplot series, kde curves, regression fits, pointplot connectors.

When adding a plot function, classify every stroke it draws against that rule,
and make sure any legend swatch it stashes carries the same width the mark is
drawn with (`markeredgewidth` for marker swatches, not just `linewidth`).

**Saving** — `savefig.bbox` is deliberately `"standard"`. Never switch it to
`"tight"`; that re-crops and desyncs figure-anchored legend bands.

**Tests that touch figures** must close them — the convention is a per-file
`@pytest.fixture(autouse=True)` calling `plt.close("all")`. There is no
`tests/conftest.py`.
