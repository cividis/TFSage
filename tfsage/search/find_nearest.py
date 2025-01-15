from typing import Callable
import pandas as pd
from .create_scoring_func import create_scoring_func


def find_nearest(
    experiment_id: str,
    distances: pd.DataFrame,
    metadata: pd.DataFrame,
    scoring_func: Callable[[float], float] | None = None,
) -> pd.DataFrame:
    """
    Find the nearest experiments based on distances and compute their scores.

    Args:
        experiment_id (str): The ID of the experiment to find nearest neighbors for.
        distances (pd.DataFrame): DataFrame containing pairwise distances between experiments.
        metadata (pd.DataFrame): DataFrame containing metadata for the experiments.
        scoring_func (Callable[[float], float] | None): A function to compute the score for a given distance. If None, a default scoring function will be generated based on the distances.

    Returns:
        pd.DataFrame: DataFrame containing the nearest experiments, their distances, metadata, and computed scores.
    """
    if scoring_func is None:
        scoring_func = create_scoring_func(distances=distances)

    return (
        distances[[experiment_id]]
        .merge(metadata, left_index=True, right_index=True)
        .rename({experiment_id: "distance"}, axis=1)
        .sort_values("distance")
        .assign(score=lambda x: scoring_func(x.distance))
    )
