import pandas as pd
from lisa.core import genome_tools

# Overwrite the check_region method to avoid checking region
genome_tools.Genome.check_region = lambda self, region: None


def prepare_genome(genome_file: str) -> genome_tools.Genome:
    """
    Prepare genome object from genome file.

    Args:
        genome_file (str): Path to the genome file. The file should be a tab-separated values (TSV) file
                           with two columns: chromosome names and their corresponding lengths.

    Returns:
        genome_tools.Genome: A Genome object initialized with the chromosome names and lengths from the file.
    """
    df = pd.read_csv(
        genome_file,
        sep="\t",
        header=None,
        usecols=[0, 1],
        dtype={0: str, 1: int},
    )
    chromosomes, lengths = df[0].values, df[1].values
    return genome_tools.Genome(chromosomes, lengths)


def prepare_region_set(
    region_file: str, genome: genome_tools.Genome
) -> genome_tools.RegionSet:
    """
    Prepare region set object from region file.

    Args:
        region_file (str): Path to the region file. The file should be in BED format.
        genome (genome_tools.Genome): A Genome object to associate with the region set.

    Returns:
        genome_tools.RegionSet: A RegionSet object initialized with the regions from the file.
    """
    regions = genome_tools.Region.read_bedfile(region_file)
    return genome_tools.RegionSet(regions, genome)
