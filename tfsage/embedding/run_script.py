from importlib.resources import files
import subprocess


def run_script(
    rp_matrix_file: str,
    metadata_file: str,
    output_dir: str,
    align_key: str = "Assay",
    method: str = "CCAIntegration,HarmonyIntegration,JointPCAIntegration,RPCAIntegration,FastMNNIntegration,none",
) -> None:
    """
    Run the R script to generate embeddings using Seurat.

    Args:
        rp_matrix_file (str): Path to the RP matrix file (input, Parquet format).
        metadata_file (str): Path to the metadata file (input, Parquet format).
        output_dir (str): Directory to save embeddings files (output).
        align_key (str): Alignment key used to split metadata (default: 'Assay').
        method (str): Comma-separated list of embedding generation methods.
                      Options: CCAIntegration, HarmonyIntegration, JointPCAIntegration, RPCAIntegration, FastMNNIntegration, none (default: 'CCAIntegration,HarmonyIntegration,JointPCAIntegration,RPCAIntegration,FastMNNIntegration,none').

    Returns:
        None
    """
    script_path = files("tfsage.embedding").joinpath("embed.R")
    cmd = [
        "Rscript",
        script_path,
        "--rp-matrix",
        rp_matrix_file,
        "--metadata",
        metadata_file,
        "--output-dir",
        output_dir,
        "--align-key",
        align_key,
        "--method",
        method,
    ]
    subprocess.run(cmd, check=True)
