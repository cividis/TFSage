from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from lisa.core import genome_tools
from .common import compute_helper, extract_region_names

# Global variable for the worker processes
global_gene_loc_set = None


def initialize_worker(gene_loc_set: genome_tools.RegionSet) -> None:
    """
    Initializer function for worker processes.

    Args:
        gene_loc_set (genome_tools.RegionSet): A RegionSet object containing gene locations.
    """
    global global_gene_loc_set
    global_gene_loc_set = gene_loc_set


def process_bed_file(bed_file: str, decay: float = 10_000) -> np.ndarray:
    """
    Process a single BED file and return the RP scores.

    Args:
        bed_file (str): Path to the BED file containing regions.
        decay (float): Decay parameter for the RP calculation (default is 10,000).

    Returns:
        np.ndarray: An array of RP scores for each gene location.
    """
    return compute_helper(bed_file, global_gene_loc_set, decay)


def compute_batch(
    bed_files: List[str],
    gene_loc_set: genome_tools.RegionSet,
    decay: float = 10_000,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Compute the RP scores for a list of BED files.

    Args:
        bed_files (List[str]): A list of paths to BED files containing regions.
        gene_loc_set (genome_tools.RegionSet): A RegionSet object containing gene locations.
        decay (float): Decay parameter for the RP calculation (default is 10,000).
        max_workers (int | None): The maximum number of workers to use for parallel processing (default is None).

    Returns:
        pd.DataFrame: A DataFrame of RP scores for each gene location, with columns corresponding to each BED file.
    """
    results = [None] * len(bed_files)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initialize_worker,
        initargs=(gene_loc_set,),
    ) as executor:
        futures = {
            executor.submit(process_bed_file, bed_file, decay): i
            for i, bed_file in enumerate(bed_files)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures[future]
            results[index] = future.result()

    # Ensure the rp_vectors are properly shaped into a 2D matrix
    rp_matrix = np.stack(results).T
    return pd.DataFrame(rp_matrix, index=extract_region_names(gene_loc_set))
