"""
Dynamic geometry generation for Venn diagrams.

This module generates Venn diagram geometry dynamically using parametric equations
for circles and ellipses, similar to the ggvenn R package approach.

Based on the geometry approach from ggvenn by yanlinlin82:
https://github.com/yanlinlin82/ggvenn/blob/main/R/venn_geometry.R
"""

import numpy as np
from typing import Tuple, Dict, List, Literal
from dataclasses import dataclass


@dataclass
class Circle:
    """
    Represents a circle or ellipse in the Venn diagram.

    Attributes
    ----------
    x_offset : float
        X-coordinate of the center
    y_offset : float
        Y-coordinate of the center
    radius_a : float
        Horizontal radius (width)
    radius_b : float
        Vertical radius (height)
    theta_offset : float
        Rotation angle in radians
    """
    x_offset: float
    y_offset: float
    radius_a: float
    radius_b: float
    theta_offset: float

    def generate_points(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points along the circle/ellipse perimeter.

        Uses parametric equations with rotation:
        x = x_offset + radius_a * cos(theta) * cos(theta_offset) - radius_b * sin(theta) * sin(theta_offset)
        y = y_offset + radius_a * cos(theta) * sin(theta_offset) + radius_b * sin(theta) * cos(theta_offset)

        Parameters
        ----------
        n_points : int
            Number of points to generate

        Returns
        -------
        x, y : np.ndarray, np.ndarray
            Arrays of x and y coordinates
        """
        theta = np.linspace(0, 2 * np.pi, n_points)

        # Parametric circle/ellipse
        x_raw = self.radius_a * np.cos(theta)
        y_raw = self.radius_b * np.sin(theta)

        # Apply rotation
        cos_rot = np.cos(self.theta_offset)
        sin_rot = np.sin(self.theta_offset)

        x = self.x_offset + x_raw * cos_rot - y_raw * sin_rot
        y = self.y_offset + x_raw * sin_rot + y_raw * cos_rot

        return x, y


def compute_2way_geometry(
    overlap_size: float = 0.5,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> Tuple[List[Circle], Dict[str, Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute geometry for 2-way Venn diagram.

    Two circles with horizontal overlap.
    Uses the exact formula from ggvenn R package.

    Returns
    -------
    circles : List[Circle]
        List of Circle objects
    label_positions : Dict[str, Tuple[float, float]]
        Positions for intersection labels (binary logic keys)
    set_label_positions : List[Tuple[float, float]]
        Positions for set name labels
    """
    # Two circles with unit radius, horizontally aligned
    # Using R's formula: x_dist = (a_radius + b_radius - overlap_size * 2) / 2
    # Default overlap_size = 0.5 for larger, more visible intersections
    a_radius = 1.0
    b_radius = 1.0
    x_dist = (a_radius + b_radius - overlap_size * 2) / 2

    circles = [
        Circle(x_offset=-x_dist, y_offset=0, radius_a=a_radius, radius_b=a_radius, theta_offset=0),
        Circle(x_offset=x_dist, y_offset=0, radius_a=b_radius, radius_b=b_radius, theta_offset=0),
    ]

    # Intersection label positions (using binary logic)
    # "10" = only first set, "01" = only second set, "11" = both sets
    label_positions = {
        "10": (-x_dist - a_radius/2, 0),      # Left only
        "01": (x_dist + b_radius/2, 0),       # Right only
        "11": (0, 0),                          # Intersection
    }

    # Set name label positions (outside circles)
    set_label_positions = [
        (-x_dist, -a_radius - 0.3),   # Below left circle
        (x_dist, -b_radius - 0.3),    # Below right circle
    ]

    if orientation == "vertical":
        # Vertical = 90-deg clockwise rotation of the horizontal layout:
        # (x, y) -> (y, -x). Set A (index 0) lands on top, Set B on the bottom.
        circles = [
            Circle(
                x_offset=c.y_offset,
                y_offset=-c.x_offset,
                radius_a=c.radius_a,
                radius_b=c.radius_b,
                theta_offset=c.theta_offset,
            )
            for c in circles
        ]
        label_positions = {
            key: (y, -x) for key, (x, y) in label_positions.items()
        }
        # Set-name labels are NOT rotated: place them at the outer ends so they
        # don't collide with the overlap. Set A above the top circle, Set B
        # below the bottom circle.
        set_label_positions = [
            (0.0, x_dist + a_radius + 0.3),    # Set A (top)
            (0.0, -x_dist - b_radius - 0.3),   # Set B (bottom)
        ]
    elif orientation != "horizontal":
        raise ValueError(
            f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
        )

    return circles, label_positions, set_label_positions


def compute_3way_geometry(radius_scale: float = 1.3) -> Tuple[List[Circle], Dict[str, Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute geometry for 3-way Venn diagram.

    Three circles arranged in an equilateral triangle pattern.
    Based on ggvenn R package geometry with increased radius for better visibility.

    Returns
    -------
    circles : List[Circle]
        List of Circle objects
    label_positions : Dict[str, Tuple[float, float]]
        Positions for intersection labels (binary logic keys)
    set_label_positions : List[Tuple[float, float]]
        Positions for set name labels
    """
    # Circle arrangement from ggvenn
    # Equilateral triangle with scaled radius for larger, more visible intersections
    # Default radius_scale = 1.3
    sqrt3 = np.sqrt(3)
    base_radius = 1.0 * radius_scale

    circles = [
        # Top left circle
        Circle(x_offset=-2/3, y_offset=(sqrt3 + 2)/6, radius_a=base_radius, radius_b=base_radius, theta_offset=0),
        # Top right circle
        Circle(x_offset=2/3, y_offset=(sqrt3 + 2)/6, radius_a=base_radius, radius_b=base_radius, theta_offset=0),
        # Bottom circle
        Circle(x_offset=0, y_offset=-(sqrt3 + 2)/6, radius_a=base_radius, radius_b=base_radius, theta_offset=0),
    ]

    # Intersection label positions
    # Binary logic: "100" = only first, "010" = only second, "001" = only third
    # "110" = first+second, "101" = first+third, "011" = second+third, "111" = all three
    # Adjust bottom intersection positions based on radius scale
    bottom_intersection_y = -0.2 - (radius_scale - 1.0) * 0.5

    label_positions = {
        "100": (-1.2, (sqrt3 + 2)/6 + 0.2),           # Left only
        "010": (1.2, (sqrt3 + 2)/6 + 0.2),            # Right only
        "001": (0, -(sqrt3 + 2)/6 - 0.7),       # Bottom only
        "110": (0, (sqrt3 + 2)/6 + 0.4),        # Top intersection
        "101": (-0.8, bottom_intersection_y),   # Left-bottom intersection (adjusted for radius)
        "011": (0.8, bottom_intersection_y),    # Right-bottom intersection (adjusted for radius)
        "111": (0, 0.1),                         # Center (all three)
    }

    # Set name label positions (outside circles)
    # Place labels by pushing outward from each circle's center
    # The offset distance adjusts with radius_scale to maintain proper spacing
    label_offset = base_radius + 0.3

    # For each circle, push the label outward from the circle center
    # Direction is from origin toward the circle center, then extended
    set_label_positions = []
    for circle in circles:
        # Calculate direction from origin to circle center
        distance = np.sqrt(circle.x_offset**2 + circle.y_offset**2)
        if distance > 0:
            # Normalize and extend
            dir_x = circle.x_offset / distance
            dir_y = circle.y_offset / distance
            # Place label at circle center plus offset in that direction
            label_x = circle.x_offset + dir_x * label_offset
            label_y = circle.y_offset + dir_y * label_offset
        else:
            # Fallback if circle is at origin (shouldn't happen for 3-way)
            label_x = circle.x_offset
            label_y = circle.y_offset + label_offset
        set_label_positions.append((label_x, label_y))

    return circles, label_positions, set_label_positions


def compute_4way_geometry() -> Tuple[List[Circle], Dict[str, Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute geometry for 4-way Venn diagram.

    Four ellipses arranged with rotation to create all intersections.
    Uses exact coordinates from ggvenn R package.

    Returns
    -------
    circles : List[Circle]
        List of Circle (ellipse) objects
    label_positions : Dict[str, Tuple[float, float]]
        Positions for intersection labels (binary logic keys)
    set_label_positions : List[Tuple[float, float]]
        Positions for set name labels
    """
    # Four ellipses with rotation
    # Exact coordinates from ggvenn gen_circle_4:
    # gen_circle(1L, -.7, -1/2, .75, 1.5, pi/4),
    # gen_circle(2L, -.72+2/3, -1/6, .75, 1.5, pi/4),
    # gen_circle(3L, .72-2/3, -1/6, .75, 1.5, -pi/4),
    # gen_circle(4L, .7, -1/2, .75, 1.5, -pi/4)

    circles = [
        Circle(x_offset=-0.7, y_offset=-1/2, radius_a=0.75, radius_b=1.5, theta_offset=np.pi/4),
        Circle(x_offset=-0.72+2/3, y_offset=-1/6, radius_a=0.75, radius_b=1.5, theta_offset=np.pi/4),
        Circle(x_offset=0.72-2/3, y_offset=-1/6, radius_a=0.75, radius_b=1.5, theta_offset=-np.pi/4),
        Circle(x_offset=0.7, y_offset=-1/2, radius_a=0.75, radius_b=1.5, theta_offset=-np.pi/4),
    ]

    # Intersection label positions from gen_text_pos_4
    # Converting R labels (A, B, C, D) to binary logic (1000, 0100, 0010, 0001)
    label_positions = {
        # Individual sets
        "1000": (-1.5, 0.0),        # A
        "0100": (-0.6, 0.7),        # B
        "0010": (0.6, 0.7),         # C
        "0001": (1.5, 0.0),         # D
        # Two-way intersections
        "1100": (-0.9, 0.3),        # AB
        "0110": (0.0, 0.4),         # BC
        "0011": (0.9, 0.3),         # CD
        "1010": (-0.8, -0.9),       # AC
        "0101": (0.8, -0.9),        # BD
        "1001": (0.0, -1.4),        # AD
        # Three-way intersections
        "1110": (-0.5, -0.2),       # ABC
        "0111": (0.5, -0.2),        # BCD
        "1011": (-0.3, -1.1),       # ACD
        "1101": (0.3, -1.1),        # ABD
        # All four
        "1111": (0.0, -0.7),        # ABCD
    }

    # Set name label positions (outside ellipses)
    # Top petals (B and C): push labels up (y=1.2), outside the ellipse
    # Bottom petals (A and D): pull labels down (y=-0.9), just below the center
    # X-coordinates match original positions for consistency
    set_label_positions = [
        (-1.5, -0.9),    # A (left-bottom)
        (-0.6, 1.2),     # B (upper left)
        (0.6, 1.2),      # C (upper right)
        (1.5, -0.9),     # D (right-bottom)
    ]

    return circles, label_positions, set_label_positions


def _compute_radial_label_positions(binary_labels: List[str], radius: float, start_angle: float) -> Dict[str, Tuple[float, float]]:
    """
    Compute label positions in a radial/circular arrangement.

    Distributes labels evenly around a circle at the specified radius.

    Parameters
    ----------
    binary_labels : List[str]
        List of binary label strings (e.g., ["10000", "11000", "11100"])
    radius : float
        Radial distance from origin
    start_angle : float
        Starting angle in radians

    Returns
    -------
    positions : Dict[str, Tuple[float, float]]
        Dictionary mapping binary label to (x, y) position
    """
    n = len(binary_labels)
    positions = {}

    for i, binary in enumerate(binary_labels):
        theta = start_angle + i * 2 * np.pi / n
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        positions[binary] = (x, y)

    return positions


def compute_5way_geometry() -> Tuple[List[Circle], Dict[str, Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute geometry for 5-way Venn diagram.

    Five ellipses arranged in a pentagonal pattern with rotation.
    Uses exact coordinates from ggvenn R package.

    Returns
    -------
    circles : List[Circle]
        List of Circle (ellipse) objects
    label_positions : Dict[str, Tuple[float, float]]
        Positions for intersection labels (binary logic keys)
    set_label_positions : List[Tuple[float, float]]
        Positions for set name labels
    """
    # Five ellipses using gen_circle_list parameters:
    # gen_circle_list(5, 1, 3.1, 1.5, pi * 0.45, pi * 0.1)
    # center_radius = 1, ellipse_a = 3.1, ellipse_b = 1.5
    # start_angle = 0.45π, rotation_offset = 0.1π

    circles = []
    n = 5
    center_radius = 1.0
    ellipse_a = 3.1
    ellipse_b = 1.5
    start_angle = np.pi * 0.45
    rotation_offset = np.pi * 0.1

    for i in range(n):
        theta = start_angle + 2 * np.pi * i / n
        x = center_radius * np.cos(theta)
        y = center_radius * np.sin(theta)
        rotation = theta + rotation_offset
        circles.append(Circle(
            x_offset=x,
            y_offset=y,
            radius_a=ellipse_a,
            radius_b=ellipse_b,
            theta_offset=rotation
        ))

    # Generate label positions using gen_text_pos_5 logic
    label_positions = {}

    # Center: ABCDE
    label_positions["11111"] = (0.0, 0.0)

    # Four-way intersections (radius 1.42, start 1.12π)
    label_positions.update(_compute_radial_label_positions(
        ["01111", "10111", "11011", "11101", "11110"],  # BCDE, ACDE, ABDE, ABCE, ABCD
        radius=1.42, start_angle=np.pi * 1.12
    ))

    # Three-way intersections - first ring (radius 1.55, start 1.33π)
    label_positions.update(_compute_radial_label_positions(
        ["00111", "10011", "11001", "11100", "01110"],  # CDE, ADE, ABE, ABC, BCD
        radius=1.55, start_angle=np.pi * 1.33
    ))

    # Three-way intersections - second ring (radius 1.88, start 1.13π)
    label_positions.update(_compute_radial_label_positions(
        ["01101", "10110", "01011", "10101", "11010"],  # BCE, ACD, BDE, ACE, ABD
        radius=1.88, start_angle=np.pi * 1.13
    ))

    # Two-way intersections - first ring (radius 1.98, start 0.44π)
    label_positions.update(_compute_radial_label_positions(
        ["10100", "01010", "00101", "10010", "01001"],  # AC, BD, CE, AD, BE
        radius=1.98, start_angle=np.pi * 0.44
    ))

    # Two-way intersections - second ring (radius 2.05, start 0.68π)
    label_positions.update(_compute_radial_label_positions(
        ["11000", "01100", "00110", "00011", "10001"],  # AB, BC, CD, DE, AE
        radius=2.05, start_angle=np.pi * 0.68
    ))

    # Single sets (radius 3.0, start 0.52π)
    label_positions.update(_compute_radial_label_positions(
        ["10000", "01000", "00100", "00010", "00001"],  # A, B, C, D, E
        radius=3.0, start_angle=np.pi * 0.52
    ))

    # Set name label positions (pushed outward from ellipses)
    # Generate positions at a radius just outside the ellipse extent
    # Max ellipse extent ≈ center_radius + ellipse_a = 1.0 + 3.1 = 4.1
    # Use radius = 4.3 to place labels clearly outside
    label_radius = 4.3
    label_start_angle = np.pi * 0.52  # Same start angle as individual sets
    set_label_positions = []
    for i in range(5):
        theta = label_start_angle + 2 * np.pi * i / 5
        x = label_radius * np.cos(theta)
        y = label_radius * np.sin(theta)
        set_label_positions.append((x, y))

    return circles, label_positions, set_label_positions


def get_geometry(
    n_sets: int,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> Tuple[List[Circle], Dict[str, Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Get geometry for n-way Venn diagram.

    Parameters
    ----------
    n_sets : int
        Number of sets (2-5)
    orientation : str, default "horizontal"
        Layout orientation. Only honored for ``n_sets == 2``; the 3/4/5-way
        branches ignore this argument.

    Returns
    -------
    circles : List[Circle]
        List of Circle objects defining the ellipses
    label_positions : Dict[str, Tuple[float, float]]
        Positions for intersection labels (binary logic keys)
    set_label_positions : List[Tuple[float, float]]
        Positions for set name labels

    Raises
    ------
    ValueError
        If n_sets is not between 2 and 5
    """
    if n_sets == 2:
        return compute_2way_geometry(orientation=orientation)
    elif n_sets == 3:
        return compute_3way_geometry()
    elif n_sets == 4:
        return compute_4way_geometry()
    elif n_sets == 5:
        return compute_5way_geometry()
    else:
        raise ValueError(f"Venn diagrams support 2-5 sets, got {n_sets}")


def get_coordinate_ranges(circles: List[Circle]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the bounding box of all circles.

    Parameters
    ----------
    circles : List[Circle]
        List of Circle objects

    Returns
    -------
    x_range, y_range : Tuple[float, float], Tuple[float, float]
        (min, max) ranges for x and y coordinates
    """
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    for circle in circles:
        # Approximate bounding box (works for small rotations)
        # For exact bounds, would need to check rotated ellipse extrema
        x_extent = max(circle.radius_a, circle.radius_b)
        y_extent = max(circle.radius_a, circle.radius_b)

        x_min = min(x_min, circle.x_offset - x_extent)
        x_max = max(x_max, circle.x_offset + x_extent)
        y_min = min(y_min, circle.y_offset - y_extent)
        y_max = max(y_max, circle.y_offset + y_extent)

    return (x_min, x_max), (y_min, y_max)
