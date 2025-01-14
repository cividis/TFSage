import tempfile
import subprocess
from typing import List
import numpy as np
import pandas as pd
import pybedtools


def multi_intersect(
    file_paths: List[str], merge_distance: int = 0
) -> pybedtools.BedTool:
    """
    Intersects multiple BED files and merges the intersected regions.

    Args:
        file_paths (List[str]): List of file paths to BED files.
        merge_distance (int, optional): Distance for merging intervals. Defaults to 0.

    Returns:
        pybedtools.BedTool: A BedTool object with the merged intervals.
    """
    result = pybedtools.BedTool().multi_intersect(i=file_paths)
    c = ",".join([str(i) for i in range(6, 6 + len(file_paths))])
    return result.merge(c=c, o="max", d=merge_distance)


def concatenate(file_paths: List[str]) -> pybedtools.BedTool:
    """
    Concatenates multiple BED files and sorts the result.

    Args:
        file_paths (List[str]): List of file paths to BED files.

    Returns:
        pybedtools.BedTool: A BedTool object with the concatenated and sorted intervals.
    """
    with tempfile.NamedTemporaryFile() as temp_file:
        command = [
            "awk",
            "-v",
            "OFS=\t",
            "FNR==1 {idx++} {print $1, $2, $3, idx-1}",
            *file_paths,
        ]

        with open(temp_file.name, "w") as f:
            subprocess.run(command, stdout=f, check=True)

        return pybedtools.BedTool(temp_file.name).sort()


def bedtool_to_dataframe(
    result: pybedtools.BedTool,
    file_columns: List[str],
    report_original_peaks: bool = False,
) -> pd.DataFrame:
    """
    Converts a BedTool object to a pandas DataFrame.

    Args:
        result (pybedtools.BedTool): The BedTool object to convert.
        file_columns (List[str]): List of column names for the BED file.
        report_original_peaks (bool, optional): Whether to include original peak columns. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame representation of the BedTool object.
    """
    base_columns = ["chrom", "start", "end"]
    original_peak_columns = [
        "chrom_original",
        "start_original",
        "end_original",
        "idx",
        "overlap",
    ]

    column_names = base_columns + file_columns
    if report_original_peaks:
        column_names += original_peak_columns

    return result.to_dataframe(disable_auto_names=True, names=column_names)


def compute_weighted_sum(
    df: pd.DataFrame,
    file_columns: List[str],
    weights: List[float] | None = None,
) -> pd.DataFrame:
    """
    Computes the weighted sum of specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        file_columns (List[str]): List of column names to be included in the weighted sum.
        weights (List[float], optional): List of weights for each column. If None, columns are summed without weights. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with the weighted sum and weights as additional columns.
    """
    # Normalize the columns to [-1, 1]
    df[file_columns] = 2 * df[file_columns] - 1

    # If weights are not provided, sum the columns
    if weights is None:
        df["weighted_sum"] = df[file_columns].sum(axis=1)
        return df

    # Compute the weighted sum
    df["weighted_sum"] = np.dot(df[file_columns], weights)

    # Add weights as columns for reference
    weights_df = pd.DataFrame(
        np.tile(weights, (df.shape[0], 1)),
        index=df.index,
        columns=[f"weight_{i}" for i in range(len(weights))],
    )
    return pd.concat([df, weights_df], axis=1)


def compute_midpoints(df: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    """
    Computes the midpoints of intervals in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the interval data.
        weighted (bool, optional): Whether to compute weighted midpoints. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the computed midpoints.
    """
    if weighted:
        return compute_weighted_midpoints(df=df)

    return (
        df.drop_duplicates(["chrom", "start", "end"])
        .reset_index(drop=True)
        .assign(midpoint=lambda x: x[["start", "end"]].mean(axis=1))
    )


def compute_weighted_midpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the weighted midpoints of intervals in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the interval data.

    Returns:
        pd.DataFrame: DataFrame with the computed weighted midpoints.
    """
    weight_columns = [f"weight_{i}" for i in df["idx"].unique()]

    df = df.assign(
        midpoint=lambda x: x[["start_original", "end_original"]].mean(axis=1)
    )

    melted_df = (
        df.melt(
            id_vars=["chrom", "start", "end", "idx", "midpoint"],
            value_vars=weight_columns,
        )
        .assign(tmp_col=lambda x: "weight_" + x["idx"].astype(str))
        .query("variable == tmp_col")
        .assign(weighted_terms=lambda x: x["midpoint"] * x["value"])
    )

    midpoints = (
        melted_df.groupby(["chrom", "start", "end"])
        .agg(
            sum_weighted_terms=("weighted_terms", "sum"),
            sum_weights=("value", "sum"),
        )
        .assign(midpoint=lambda x: np.divide(x["sum_weighted_terms"], x["sum_weights"]))
        .reindex(["midpoint"], axis=1)
    )

    return (
        df.drop_duplicates(["chrom", "start", "end"])
        .drop(columns="midpoint")
        .reset_index(drop=True)
        .merge(midpoints, on=["chrom", "start", "end"])
    )


def fixed_widths(df: pd.DataFrame, fixed_width: int = 200) -> pd.DataFrame:
    """
    Adjusts the start and end positions of intervals to have a fixed width centered around the midpoint.

    Args:
        df (pd.DataFrame): The DataFrame containing the interval data with a 'midpoint' column.
        fixed_width (int, optional): The fixed width for the intervals. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame with adjusted 'start' and 'end' columns.
    """
    half_width = fixed_width // 2
    df["start"] = np.maximum(df["midpoint"] - half_width, 0).astype(int)
    df["end"] = (df["midpoint"] + half_width).astype(int)
    return df


def adjust_intervals_to_fixed_width(df: pd.DataFrame, width: int = 200) -> pd.DataFrame:
    """
    Adjusts the start and end positions of intervals to have a fixed width centered around the midpoint.

    Args:
        df (pd.DataFrame): The DataFrame containing the interval data with a 'midpoint' column.
        width (int, optional): The fixed width for the intervals. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame with adjusted 'start' and 'end' columns.
    """
    half_width = width // 2
    df["start"] = np.maximum(df["midpoint"] - half_width, 0).astype(int)
    df["end"] = (df["midpoint"] + half_width).astype(int)
    return df
