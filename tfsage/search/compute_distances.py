import pandas as pd
from sklearn.metrics import pairwise_distances


def compute_distances(embeddings: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    """
    Compute pairwise distances between embeddings.

    Args:
        embeddings (pd.DataFrame): DataFrame containing the embeddings.
        metric (str): The distance metric to use. Options include 'cosine', 'euclidean', etc.
                      Refer to sklearn.metrics.pairwise_distances for all available options (default: 'cosine').

    Returns:
        pd.DataFrame: DataFrame containing the pairwise distances, with the same index and columns as the input embeddings.
    """
    return pd.DataFrame(
        pairwise_distances(embeddings, metric=metric),
        index=embeddings.index,
        columns=embeddings.index,
    )
