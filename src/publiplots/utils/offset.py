from typing import List
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.transforms as transforms
from typing import Literal

def offset_lines(
    lines: List[Line2D],
    offset: float,
    orientation: Literal["vertical", "horizontal"],
) -> None:
    """
    Offset lines by a given amount.
    """
    for line in lines:
        if orientation == "vertical":
            line.set_xdata(line.get_xdata() + offset)
        else:
            line.set_ydata(line.get_ydata() + offset)

def offset_patches(
    patches: List[Patch],
    offset: float,
    orientation: Literal["vertical", "horizontal"],
) -> None:
    """
    Offset patches by a given amount.
    """
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import Rectangle, FancyArrowPatch

    for patch in patches:
        # Skip FancyArrowPatch - arrows are positioned by their coordinates, not paths
        # They should be positioned correctly when created
        if isinstance(patch, FancyArrowPatch):
            # Get current arrow positions
            x1, y1 = patch.get_positions()[0]
            x2, y2 = patch.get_positions()[1]

            # Offset both positions
            if orientation == "vertical":
                x1 += offset
                x2 += offset
            else:
                y1 += offset
                y2 += offset

            # Update arrow positions
            patch.set_positions((x1, y1), (x2, y2))
        # Handle Rectangle patches specially (barplot uses Rectangles)
        elif isinstance(patch, Rectangle):
            if orientation == "vertical":
                # Shift x position
                patch.set_x(patch.get_x() + offset)
            else:
                # Shift y position
                patch.set_y(patch.get_y() + offset)
        else:
            # For other patches, modify the path
            path = patch.get_path()
            # Create a copy of vertices to avoid read-only array errors
            vertices = path.vertices.copy()
            if orientation == "vertical":
                vertices[:, 0] += offset
            else:
                vertices[:, 1] += offset
            # Create new path with modified vertices
            new_path = MplPath(vertices, path.codes)
            patch.set_path(new_path)
        
def offset_collections(
    collections: List[PathCollection],
    offset: float,
    ax: Axes,
    orientation: Literal["vertical", "horizontal"],
) -> None:
    """
    Offset collections by a given amount.
    """
    fig = ax.figure
    
    if orientation == 'vertical':
        # Offset in x-direction
        display_points = ax.transData.transform([(0, 0), (1, 0)])
        pixels_per_unit = display_points[1, 0] - display_points[0, 0]
        display_offset = offset * pixels_per_unit
        offset = transforms.ScaledTranslation(display_offset/fig.dpi, 0, fig.dpi_scale_trans)
    else:  # horizontal
        # Offset in y-direction
        display_points = ax.transData.transform([(0, 0), (0, 1)])
        pixels_per_unit = display_points[1, 1] - display_points[0, 1]
        display_offset = offset * pixels_per_unit
        offset = transforms.ScaledTranslation(0, display_offset/fig.dpi, fig.dpi_scale_trans)
    
    for collection in collections:
        collection.set_transform(collection.get_transform() + offset)