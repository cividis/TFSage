import pandas as pd
import pybedtools
from . import helpers as h


def generate_test_samples(
    query_file: str, target_file: str, width: int | None = 200
) -> pd.DataFrame:
    """
    Generates test samples by creating positive and negative samples from BED files.

    Args:
        query_file (str): Path to the query BED file.
        target_file (str): Path to the target BED file.
        width (int, optional): The fixed width for the intervals. If None, intervals are not adjusted. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame containing the test samples with columns ['chrom', 'start', 'end', 'num', 'list', 'positive', 'negative'].
    """
    positive_samples = generate_positive_samples(target_file, width)
    negative_samples = generate_negative_samples(
        query_file, target_file, positive_samples, width
    )

    # Sort
    positive_samples = positive_samples.sort()
    negative_samples = negative_samples.sort()

    samples = pybedtools.BedTool().multi_intersect(
        i=[positive_samples.fn, negative_samples.fn]
    )
    column_names = [
        "chrom",
        "start",
        "end",
        "num",
        "list",
        "positive",
        "negative",
    ]

    return samples.to_dataframe(disable_auto_names=True, names=column_names)


def generate_positive_samples(
    target_file: str, width: int | None = 200
) -> pybedtools.BedTool:
    """
    Generates positive samples from a target BED file.

    Args:
        target_file (str): Path to the target BED file.
        width (int, optional): The fixed width for the intervals. If None, intervals are not adjusted. Defaults to 200.

    Returns:
        pybedtools.BedTool: BedTool object containing the positive samples.
    """
    if width:
        df = pybedtools.BedTool(target_file).to_dataframe()
        df = h.compute_midpoints(df, weighted=False)
        df = h.adjust_intervals_to_fixed_width(df, width)
        return pybedtools.BedTool.from_dataframe(df)
    return pybedtools.BedTool(target_file)


def generate_negative_samples(
    query_file: str,
    target_file: str,
    positive_samples: pybedtools.BedTool,
    width: int | None = 200,
) -> pybedtools.BedTool:
    """
    Generates negative samples by subtracting target and positive samples from the query BED file.

    Args:
        query_file (str): Path to the query BED file.
        target_file (str): Path to the target BED file.
        positive_samples (pybedtools.BedTool): BedTool object containing the positive samples.
        width (int, optional): The fixed width for the intervals. If None, intervals are not adjusted. Defaults to 200.

    Returns:
        pybedtools.BedTool: BedTool object containing the negative samples.
    """
    negative_samples = pybedtools.BedTool(query_file)
    negative_samples = negative_samples.subtract(target_file, A=True)
    negative_samples = negative_samples.subtract(positive_samples, A=True)
    if width:
        df = negative_samples.to_dataframe()
        df = h.compute_midpoints(df, weighted=False)
        df = h.adjust_intervals_to_fixed_width(df, width)

        negative_samples = pybedtools.BedTool.from_dataframe(df)
        negative_samples = negative_samples.subtract(target_file, A=True)
        negative_samples = negative_samples.subtract(positive_samples, A=True)

    return negative_samples
