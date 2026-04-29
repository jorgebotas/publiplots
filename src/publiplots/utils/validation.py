"""
Input validation utilities for publiplots.

This module provides functions for validating and preprocessing inputs
to plotting functions, ensuring robust error handling.
"""

from typing import Optional, Union, List, Any
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors


def is_categorical(
        values: Union[pd.Series, pd.Categorical, np.ndarray, list, tuple],
    ) -> bool:
    """
    Check if values are categorical.
    If is pd.Categorical, returns True (even if numeric values).
    Otherwise, returns True if not numeric.

    Parameters
    ----------
    values : pd.Series
        Values to check.

    Returns
    -------
    bool
        True if values are categorical, False otherwise.

    Examples
    --------
    >>> is_categorical(pd.Series(["a", "b", "c"]))
    True
    >>> is_categorical(pd.Series(["a", "b", "c"], dtype="category"))
    True
    >>> is_categorical(pd.Series([1, 2, 3]))
    False
    >>> is_categorical(pd.Series([1, 2, 3], dtype="category"))
    True
    """
    return isinstance(values, pd.Categorical) or not is_numeric(values)


def as_categorical(
        values: pd.Series,
        categories: Optional[List[Any]] = None,
        ordered: bool = True,
    ) -> pd.Series:
    """
    Ensure a Series has pandas ``category`` dtype so the ``.cat`` accessor works.

    ``is_categorical`` in this module is semantic (non-numeric), so an object-dtype
    string column reports as categorical but does not expose ``.cat``. Use this
    helper any time downstream code needs ``.cat.categories`` or
    ``.cat.remove_unused_categories()``.

    Parameters
    ----------
    values : pd.Series
        Series to coerce.
    categories : list, optional
        Explicit category order. Defaults to ``values.unique()`` preserving
        first-occurrence order.
    ordered : bool, default True
        Whether the resulting Categorical is ordered.

    Returns
    -------
    pd.Series
        Series with ``category`` dtype.

    Examples
    --------
    >>> s = pd.Series(["a", "b", "a"])
    >>> as_categorical(s).cat.categories.tolist()
    ['a', 'b']
    """
    if categories is None:
        categories = values.unique()
    return pd.Series(
        pd.Categorical(values, categories=categories, ordered=ordered),
        index=values.index if isinstance(values, pd.Series) else None,
        name=values.name if isinstance(values, pd.Series) else None,
    )


def is_numeric(values: Union[pd.Series, pd.Categorical, np.ndarray, list, tuple]) -> bool:
    """
    Check if values are numeric.

    Parameters
    ----------
    values : Union[pd.Series, pd.Categorical, np.ndarray, list, tuple]
        Values to check.

    Returns
    -------
    bool
        True if values are numeric, False otherwise.

    Examples
    --------
    >>> is_numeric(pd.Series([1, 2, 3]))
    True
    >>> is_numeric(pd.Series([1, 2, 3], dtype="category"))
    False
    """
    return pd.api.types.is_numeric_dtype(values)

def validate_data(
    data: Union[pd.DataFrame, np.ndarray, None],
    required_columns: Optional[List[str]] = None
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Validate input data format.

    Parameters
    ----------
    data : DataFrame, ndarray, or None
        Input data to validate.
    required_columns : List[str], optional
        Column names that must be present in DataFrame.

    Returns
    -------
    Union[DataFrame, ndarray]
        Validated data.

    Raises
    ------
    ValueError
        If data is None or missing required columns.
    TypeError
        If data is not a DataFrame or ndarray.

    Examples
    --------
    >>> data = validate_data(df, required_columns=['x', 'y'])
    """
    if data is None:
        raise ValueError("Data cannot be None")

    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise TypeError(
            f"Data must be pandas DataFrame or numpy ndarray, got {type(data)}"
        )

    # Check for required columns in DataFrame
    if isinstance(data, pd.DataFrame) and required_columns is not None:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in data: {missing_cols}. "
                f"Available columns: {list(data.columns)}"
            )

    # Check for empty data
    if len(data) == 0:
        raise ValueError("Data is empty")

    return data


def validate_numeric(
    values: Union[List, np.ndarray, pd.Series],
    name: str = "values",
    allow_nan: bool = False,
    allow_inf: bool = False
) -> np.ndarray:
    """
    Validate numeric data.

    Parameters
    ----------
    values : array-like
        Values to validate.
    name : str, default='values'
        Name of the parameter (for error messages).
    allow_nan : bool, default=False
        Whether to allow NaN values.
    allow_inf : bool, default=False
        Whether to allow infinite values.

    Returns
    -------
    ndarray
        Validated numeric array.

    Raises
    ------
    ValueError
        If values are not numeric, contain NaN/inf when not allowed,
        or are empty.

    Examples
    --------
    >>> values = validate_numeric([1, 2, 3, 4])
    """
    # Convert to numpy array
    values = np.asarray(values)

    # Check if numeric
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {values.dtype}")

    # Check for empty
    if values.size == 0:
        raise ValueError(f"{name} cannot be empty")

    # Check for NaN
    if not allow_nan and np.any(np.isnan(values)):
        raise ValueError(f"{name} contains NaN values")

    # Check for inf
    if not allow_inf and np.any(np.isinf(values)):
        raise ValueError(f"{name} contains infinite values")

    return values


def validate_colors(
    colors: Union[str, List[str], dict, None],
    n_colors: Optional[int] = None,
    name: str = "colors"
) -> Union[str, List[str], dict, None]:
    """
    Validate color specification.

    Parameters
    ----------
    colors : str, list, dict, or None
        Color specification to validate.
    n_colors : int, optional
        Expected number of colors (for list validation).
    name : str, default='colors'
        Name of the parameter (for error messages).

    Returns
    -------
    Union[str, List[str], dict, None]
        Validated colors.

    Raises
    ------
    ValueError
        If color specification is invalid.

    Examples
    --------
    >>> colors = validate_colors('#FF0000')
    >>> colors = validate_colors(['red', 'blue', 'green'], n_colors=3)
    """
    if colors is None:
        return None

    # Validate single color string
    if isinstance(colors, str):
        if not mpl_colors.is_color_like(colors):
            raise ValueError(f"{name}: '{colors}' is not a valid color")
        return colors

    # Validate list of colors
    if isinstance(colors, (list, tuple)):
        if not all(isinstance(c, str) and mpl_colors.is_color_like(c) for c in colors):
            invalid = [c for c in colors if not mpl_colors.is_color_like(c)]
            raise ValueError(f"{name}: Invalid colors: {invalid}")

        if n_colors is not None and len(colors) != n_colors:
            raise ValueError(
                f"{name}: Expected {n_colors} colors, got {len(colors)}"
            )
        return list(colors)

    # Validate dictionary of colors
    if isinstance(colors, dict):
        if not all(isinstance(c, str) and mpl_colors.is_color_like(c)
                   for c in colors.values()):
            invalid = {k: v for k, v in colors.items()
                      if not mpl_colors.is_color_like(v)}
            raise ValueError(f"{name}: Invalid colors in dict: {invalid}")
        return colors

    raise TypeError(
        f"{name} must be a color string, list of colors, or dict mapping to colors, "
        f"got {type(colors)}"
    )


def validate_dimensions(
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    name_x: str = "x",
    name_y: str = "y"
) -> tuple:
    """
    Validate that x and y have compatible dimensions.

    Parameters
    ----------
    x : array-like
        X values.
    y : array-like
        Y values.
    name_x : str, default='x'
        Name of x parameter (for error messages).
    name_y : str, default='y'
        Name of y parameter (for error messages).

    Returns
    -------
    tuple
        (x, y) as numpy arrays.

    Raises
    ------
    ValueError
        If dimensions don't match.

    Examples
    --------
    >>> x, y = validate_dimensions([1, 2, 3], [4, 5, 6])
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(
            f"{name_x} and {name_y} must have the same shape. "
            f"Got {name_x}: {x.shape}, {name_y}: {y.shape}"
        )

    return x, y


def validate_positive(
    values: Union[List, np.ndarray],
    name: str = "values",
    strict: bool = False
) -> np.ndarray:
    """
    Validate that values are positive.

    Parameters
    ----------
    values : array-like
        Values to validate.
    name : str, default='values'
        Name of the parameter (for error messages).
    strict : bool, default=False
        If True, requires values > 0. If False, allows values >= 0.

    Returns
    -------
    ndarray
        Validated array.

    Raises
    ------
    ValueError
        If values are not positive.

    Examples
    --------
    >>> values = validate_positive([1, 2, 3])
    >>> values = validate_positive([0, 1, 2], strict=False)
    """
    values = np.asarray(values)

    if strict:
        if not np.all(values > 0):
            raise ValueError(f"{name} must be strictly positive (> 0)")
    else:
        if not np.all(values >= 0):
            raise ValueError(f"{name} must be non-negative (>= 0)")

    return values


def validate_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value"
) -> float:
    """
    Validate that a value is within a specified range.

    Parameters
    ----------
    value : float
        Value to validate.
    min_val : float, optional
        Minimum allowed value (inclusive).
    max_val : float, optional
        Maximum allowed value (inclusive).
    name : str, default='value'
        Name of the parameter (for error messages).

    Returns
    -------
    float
        Validated value.

    Raises
    ------
    ValueError
        If value is outside the specified range.

    Examples
    --------
    >>> alpha = validate_range(0.5, min_val=0, max_val=1, name='alpha')
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")

    return value


def coerce_to_numeric(
    values: Union[List, np.ndarray, pd.Series],
    name: str = "values"
) -> np.ndarray:
    """
    Coerce values to numeric, raising error if not possible.

    Parameters
    ----------
    values : array-like
        Values to coerce.
    name : str, default='values'
        Name of the parameter (for error messages).

    Returns
    -------
    ndarray
        Numeric array.

    Raises
    ------
    ValueError
        If values cannot be coerced to numeric.

    Examples
    --------
    >>> values = coerce_to_numeric(['1', '2', '3'])
    array([1., 2., 3.])
    """
    try:
        if isinstance(values, pd.Series):
            return pd.to_numeric(values, errors='raise').values
        else:
            return np.asarray(values, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert {name} to numeric: {e}")


def check_required_columns(
    data: pd.DataFrame,
    required: List[str],
    optional: Optional[List[str]] = None
) -> dict:
    """
    Check for required and optional columns in DataFrame.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame.
    required : List[str]
        List of required column names.
    optional : List[str], optional
        List of optional column names to check for.

    Returns
    -------
    dict
        Dictionary with keys 'missing_required', 'present_optional'.

    Raises
    ------
    ValueError
        If any required columns are missing.

    Examples
    --------
    >>> result = check_required_columns(df, required=['x', 'y'], optional=['hue'])
    """
    # Check required columns
    missing_required = [col for col in required if col not in data.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}. "
            f"Available columns: {list(data.columns)}"
        )

    # Check optional columns
    result = {'missing_required': []}

    if optional is not None:
        present_optional = [col for col in optional if col in data.columns]
        result['present_optional'] = present_optional

    return result
