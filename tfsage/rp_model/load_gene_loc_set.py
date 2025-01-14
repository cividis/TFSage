from importlib.resources import files
from typing import Literal
from lisa.core import genome_tools
from .prepare import prepare_genome, prepare_region_set


def load_gene_loc_set(genome_name: Literal["hg38", "mm10"]) -> genome_tools.RegionSet:
    """
    Load the default gene location set for a given genome.

    Args:
        genome_name (Literal["hg38", "mm10"]): The name of the genome to load the gene location set for.
            Must be either "hg38" for human or "mm10" for mouse.

    Returns:
        genome_tools.RegionSet: The loaded gene location set for the specified genome.
    """
    # Dynamically construct file paths using importlib.resources
    genome_file = files("tfsage.assets").joinpath(f"{genome_name}.len")
    refseq_file = files("tfsage.assets").joinpath(f"{genome_name}_refseq_TSS.bed")

    genome = prepare_genome(genome_file)
    return prepare_region_set(refseq_file, genome)
