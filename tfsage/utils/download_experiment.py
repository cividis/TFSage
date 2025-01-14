import subprocess
import tempfile


def download_experiment(
    experiment_id: str, output_file: str, genome: str = "hg38"
) -> None:
    """
    Download an experiment from ENCODE or ChIP-Atlas.

    Args:
        experiment_id (str): The ID of the experiment to download.
        output_file (str): The path to the output file where the downloaded data will be saved.
        genome (str): The genome assembly to use (default is "hg38"). Only necessary for ChIP-Atlas.
    """
    url = _get_url(experiment_id, genome)
    _curl_and_sort(url, output_file)


def _get_url(experiment_id: str, genome: str = "hg38") -> str:
    """
    Get URL for experiment ID.

    Args:
        experiment_id (str): The ID of the experiment.
        genome (str): The genome assembly to use (default is "hg38"). Only necessary for ChIP-Atlas.

    Returns:
        str: The URL to download the experiment data.
    """
    is_encode = experiment_id.startswith("ENC")
    if is_encode:
        return "https://www.encodeproject.org/files/{experiment_id}/@@download/{experiment_id}.bed.gz".format(
            experiment_id=experiment_id
        )
    else:
        return "https://chip-atlas.dbcls.jp/data/{genome}/eachData/bed{threshold}/{experiment_id}.bed".format(
            genome=genome,
            threshold=experiment_id.split(".")[1],
            experiment_id=experiment_id,
        )


def _curl_and_sort(url: str, output_file: str) -> None:
    """
    Download and sort file from URL and save to output file path.

    Args:
        url (str): The URL to download the file from.
        output_file (str): The path to the output file where the sorted data will be saved.
    """

    with tempfile.NamedTemporaryFile() as temp_file:
        curl_command = ["curl", "-L", url, "-o", temp_file.name]
        sort_command = ["sortBed", "-i", temp_file.name]

        subprocess.run(curl_command, check=True)
        subprocess.run(sort_command, check=True, stdout=open(output_file, "w"))
