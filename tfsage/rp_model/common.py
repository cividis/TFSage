import os
from typing import List
import numpy as np
from lisa.core import data_interface, genome_tools
from .prepare import prepare_region_set


def file_is_empty(file: str) -> bool:
    """
    Check if a file is empty.

    Args:
        file (str): Path to the file.

    Returns:
        bool: True if the file is empty, False otherwise.
    """
    return os.path.getsize(file) == 0


def extract_region_names(region_set: genome_tools.RegionSet) -> List[str]:
    """
    Extract region names from a region set.

    Args:
        region_set (genome_tools.RegionSet): A RegionSet object containing genomic regions.

    Returns:
        List[str]: A list of region names extracted from the region set.
    """
    return [region.annotation[0] for region in region_set.regions]


def compute_helper(
    bed_file: str, gene_loc_set: genome_tools.RegionSet, decay: float = 10_000
) -> np.ndarray:
    """
    Compute a numpy array of regulatory potential (RP) scores.

    Args:
        bed_file (str): Path to the BED file containing regions.
        gene_loc_set (genome_tools.RegionSet): A RegionSet object containing gene locations.
        decay (float): Decay parameter for the RP calculation (default is 10,000).

    Returns:
        np.ndarray: An array of RP scores for each gene location.
    """
    if file_is_empty(bed_file):
        return np.zeros(len(gene_loc_set.regions))

    region_set = prepare_region_set(bed_file, gene_loc_set.genome)
    rp_map = data_interface.DataInterface._make_basic_rp_map(
        gene_loc_set=gene_loc_set,
        region_set=region_set,
        decay=decay,
    )
    return rp_map.sum(axis=1).flatten().A1
