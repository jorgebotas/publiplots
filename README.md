# PubliPlots

Publication-ready plots

## Overview

PubliPlots is a Python visualization library that provides beautiful, publication-ready plots with a seaborn-like API. It focuses on:

- **Beautiful defaults**: Carefully designed pastel color palettes and styles
- **Intuitive API**: Follows seaborn conventions for ease of use
- **Modular design**: Compose complex visualizations from simple building blocks
- **Highly configurable**: Extensive customization while maintaining sensible defaults
- **Publication-ready**: Optimized for scientific publications and presentations

> [!IMPORTANT]
> **Documentation**: Full documentation is available at [jorgebotas.github.io/publiplots](https://jorgebotas.github.io/publiplots/)

## Gallery

<p align="center" style="background-color: white">
  <img src="docs/images/barplot_hatch_hue.png" width="45%" alt="Barplot with Hatch and Hue">
  <img src="docs/images/raincloud_hue.png" width="45%" alt="Raincloud Plot">
</p>
<p align="center" style="background-color: white">
  <img src="docs/images/venn_4way.png" width="45%" alt="4-Way Venn Diagram" style="vertical-align: middle">
  <img src="docs/images/upsetplot.png" width="45%" alt="UpSet Plot" style="vertical-align: middle">
</p>

For interactive examples, check out the [examples.ipynb](examples/examples.ipynb) notebook.

## Installation

### From PyPI

```bash
pip install publiplots
```

Or if you are using [uv](https://github.com/astral-sh/uv) for Python environment management:

```bash
uv pip install publiplots
```

### From source (development)

```bash
git clone https://github.com/jorgebotas/publiplots.git
cd publiplots
pip install -e .
```

### Development with uv and Jupyter

If you're using [uv](https://github.com/astral-sh/uv) for Python environment management and want to use the package in Jupyter notebooks:

```bash
# Clone the repository
git clone https://github.com/jorgebotas/publiplots.git
cd publiplots

# Create a new uv environment with Python 3.11 (or your preferred version)
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Install the package in editable mode with all dependencies
uv pip install -e .

# Install ipykernel to make the environment available in Jupyter
uv pip install ipykernel

# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name=publiplots --display-name="Python (publiplots)"
```

Now you can select the "Python (publiplots)" kernel in Jupyter Lab or Jupyter Notebook and import publiplots:

```python
import publiplots as pp
```


## Quick Start

```python
import publiplots as pp
import pandas as pd

# Apply publication style globally
pp.set_publication_style()

# Create a scatter plot
ax = pp.scatterplot(
    data=df,
    x='measurement_a',
    y='measurement_b',
    hue='condition',
    palette=pp.color_palette('pastel', n_colors=3)
)

# Save with publication-ready settings
pp.savefig('figure.pdf')
```

## Matplotlib backends

publiplots is **backend-agnostic** — every plot works under any
matplotlib backend (PNG/JPG via Agg, PDF, SVG, PS, interactive
Jupyter `inline` / `widget`, desktop GUIs). The library never calls
`matplotlib.use(...)` implicitly, so it won't override a backend you've
already picked.

For headless rendering (scripts, CI, notebooks without displays) the
common pattern is to set Agg in your own code **before** importing
pyplot:

```python
import matplotlib
matplotlib.use("Agg")        # must come before pyplot touches the GUI
import matplotlib.pyplot as plt
import publiplots as pp
```

## Claude Code plugin

publiplots ships a [Claude Code](https://claude.com/claude-code) plugin with skills that teach Claude to write idiomatic publiplots code (mm-units layout, `pp.subplots` over `plt.subplots`, unified `pp.legend` scoping). Install via the built-in marketplace:

```
/plugin marketplace add jorgebotas/publiplots
/plugin install publiplots@publiplots
```

The plugin includes two skills:

- `/publiplots:publiplots-guide` — core library conventions, full `pp.*` API, canonical idioms and gotchas.
- `/publiplots:legend-placement` — `pp.legend` scoping decision tree (per-axes, row/column bands, figure-level bands) and the `pp.legend(ax)` vs `pp.legend(anchor=ax)` asymmetry.

Both skills also auto-activate when Claude detects publiplots-related work. The plugin version tracks the library version — pin a specific release with `@v0.10.1` in the marketplace source.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use PubliPlots in your research, please cite:

```
Botas, J. (2025). PubliPlots: Publication-ready plotting for Python.
GitHub: https://github.com/jorgebotas/publiplots
```

## License

MIT License - see LICENSE file for details.

## Author

Jorge Botas ([@jorgebotas](https://github.com/jorgebotas))

## Acknowledgments

PubliPlots builds upon excellent work from the Python visualization community:

- **[ggvenn](https://github.com/yanlinlin82/ggvenn)** by Yan Linlin - The Venn diagram implementation (2-5 sets) is based on the geometry from this R package
- **[UpSetPlot](https://github.com/jnothman/UpSetPlot)** by Joel Nothman - The UpSet plot implementation is inspired by concepts from this library (BSD-3-Clause license)
- **[matplotlib](https://matplotlib.org/)** - The foundational plotting library that powers PubliPlots
- **[seaborn](https://seaborn.pydata.org/)** - Inspiration for API design and color palettes
