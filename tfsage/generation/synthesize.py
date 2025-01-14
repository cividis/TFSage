from typing import List
import pandas as pd
from . import helpers as h


def synthesize(
    file_paths: List[str],
    weights: List[float] | None = None,
    merge_distance: int = 0,
    report_original_peaks: bool = False,
) -> pd.DataFrame:
    """
    Synthesizes data from multiple BED files by intersecting, merging, and computing weighted sums.

    Args:
        file_paths (List[str]): List of file paths to BED files.
        weights (List[float], optional): List of weights for each file. If None, columns are summed without weights. Defaults to None.
        merge_distance (int, optional): Distance for merging intervals. Defaults to 0.
        report_original_peaks (bool, optional): Whether to include original peak columns. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the synthesized data.
    """
    # Multi-intersect and merge nearby peaks
    result = h.multi_intersect(file_paths, merge_distance)

    # Concatenate original peaks and intersect with results
    if report_original_peaks:
        concat = h.concatenate(file_paths)
        result = result.intersect(concat, wo=True, sorted=True)

    # Convert to DataFrame
    file_columns = [f"file_{i}" for i in range(len(file_paths))]
    df = h.bedtool_to_dataframe(result, file_columns, report_original_peaks)

    # Compute weighted sum
    return h.compute_weighted_sum(df, file_columns, weights)


def standardize(
    df: pd.DataFrame, weighted: bool = False, width: int | None = 200
) -> pd.DataFrame:
    """
    Standardizes the intervals in a DataFrame by computing midpoints and adjusting to a fixed width.

    Args:
        df (pd.DataFrame): The DataFrame containing the interval data.
        weighted (bool, optional): Whether to compute weighted midpoints. Defaults to False.
        width (int, optional): The fixed width for the intervals. If None, intervals are not adjusted. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame with standardized intervals.
    """
    df = h.compute_midpoints(df, weighted)
    if width:
        return h.adjust_intervals_to_fixed_width(df, width)
    return df
