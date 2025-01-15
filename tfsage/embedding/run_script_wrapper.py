import os
import tempfile
from typing import Literal
import pandas as pd
from .run_script import run_script


def run_script_wrapper(
    rp_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    align_key: str = "Assay",
    method: Literal[
        "CCAIntegration",
        "HarmonyIntegration",
        "JointPCAIntegration",
        "RPCAIntegration",
        "FastMNNIntegration",
        "none",
    ] = "FastMNNIntegration",
) -> pd.DataFrame:
    """
    Generate embeddings using the specified method.

    Args:
        rp_matrix (pd.DataFrame): DataFrame containing the RP matrix.
        metadata (pd.DataFrame): DataFrame containing the metadata.
        align_key (str): Alignment key used to split metadata (default: 'Assay').
        method (str): Embedding generation method. Options: CCAIntegration, HarmonyIntegration, JointPCAIntegration, RPCAIntegration, FastMNNIntegration, none (default: 'FastMNNIntegration').

    Returns:
        pd.DataFrame: DataFrame containing the generated embeddings.
    """
    valid_methods = [
        "CCAIntegration",
        "HarmonyIntegration",
        "JointPCAIntegration",
        "RPCAIntegration",
        "FastMNNIntegration",
        "none",
    ]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: {', '.join(valid_methods)}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        rp_matrix_file = os.path.join(temp_dir, "rp_matrix.parquet")
        metadata_file = os.path.join(temp_dir, "metadata.parquet")
        embeddings_file = os.path.join(temp_dir, f"{method}.parquet")

        # Remove index name
        rp_matrix.index.name = None
        metadata.index.name = None

        # Save data to disk
        rp_matrix.to_parquet(rp_matrix_file)
        metadata.to_parquet(metadata_file)

        # Run script
        run_script(
            rp_matrix_file=rp_matrix_file,
            metadata_file=metadata_file,
            output_dir=temp_dir,
            align_key=align_key,
            method=method,
        )

        # Load results
        embeddings = pd.read_parquet(embeddings_file)

    embeddings.set_index("__index_level_0__", inplace=True)
    embeddings.index.name = None
    return embeddings
