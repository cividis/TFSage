from typing import Callable
import numpy as np
import pandas as pd


def estimate_sigma_sq(distances: pd.DataFrame) -> float:
    """
    Estimate the variance (sigma squared) of the distances.

    Args:
        distances (pd.DataFrame): DataFrame containing the pairwise distances.

    Returns:
        float: The estimated variance (sigma squared) of the distances.
    """
    return np.var(distances.to_numpy())


def generate_scoring_func(
    sigma_sq: float | None = None, distances: pd.DataFrame | None = None
) -> Callable[[float], float]:
    """
    Generate a scoring function based on pairwise distances and variance.

    Args:
        sigma_sq (float | None): The variance (sigma squared) to use. If None and distances is also None, it will be set to 1.0. If None and distances is provided, it will be estimated from distances.
        distances (pd.DataFrame | None): DataFrame containing the pairwise distances.

    Returns:
        Callable[[float], float]: A function that computes the score for a given distance.
    """
    if sigma_sq is None:
        sigma_sq = 1.0 if distances is None else estimate_sigma_sq(distances)

    return lambda d: np.exp(-(d**2) / (2 * sigma_sq))
