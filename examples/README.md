# PubliPlots Examples

This directory contains example scripts for sphinx-gallery. These examples demonstrate
the various plotting capabilities of PubliPlots and are automatically converted into
a browsable gallery in the documentation.

## Examples

- **plot_bar_plots.py**: Comprehensive bar plot examples including simple bars, grouped bars, error bars, and hatch patterns
- **plot_scatter_plots.py**: Scatter plot examples with size and color encoding, including bubble plots
- **plot_venn_diagrams.py**: Venn diagram examples from 2-way to 5-way diagrams
- **plot_upset_plots.py**: UpSet plot examples for visualizing set intersections
- **plot_hatch_patterns.py**: Hatch pattern examples and density modes
- **plot_configuration.py**: Configuration and styling examples using rcParams

## Running Examples

You can run any example directly:

```bash
python plot_bar_plots.py
```

Or run them through sphinx-gallery when building the documentation:

```bash
cd docs
make html
```

## Adding New Examples

To add a new example:

1. Create a new Python file starting with `plot_`
2. Use docstring format for the title and description:
   ```python
   """
   Example Title
   =============

   Description of what this example demonstrates.
   """
   ```

3. Use `# %%` to separate code blocks (sphinx-gallery sections)
4. Add descriptive comments before each section:
   ```python
   # %%
   # Section Title
   # -------------
   # Description of this section.
   ```

5. Include `pp.show()` after creating figures
6. Run the example to ensure it works before building docs

## Notes

- Examples are executed during documentation build
- Generated output images are automatically included in the docs
- Keep examples concise and focused on specific features
- Use realistic sample data when possible
- Include clear comments and explanations
