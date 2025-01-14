import pandas as pd
from lisa.core import genome_tools
from .common import compute_helper, extract_region_names


def compute(
    bed_file: str, gene_loc_set: genome_tools.RegionSet, decay: float = 10_000
) -> pd.Series:
    """
    Compute a pandas Series of regulatory potential (RP) scores.

    Args:
        bed_file (str): Path to the BED file containing regions.
        gene_loc_set (genome_tools.RegionSet): A RegionSet object containing gene locations.
        decay (float): Decay parameter for the RP calculation (default is 10,000).

    Returns:
        pd.Series: A Series of RP scores for each gene location, indexed by region names.
    """
    np_arr = compute_helper(bed_file, gene_loc_set, decay)
    return pd.Series(np_arr, index=extract_region_names(gene_loc_set))
