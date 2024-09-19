####################################################
###
###                     Imports
###
####################################################


import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from pyCellPhenoX.src.utils.select_optimal_k import (
    select_optimal_k,
)

####################################################
###
###                     Tests
###
####################################################


# Helper function to create sample data
def get_sample_data(n_samples=100, n_features=10):
    X, _ = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=42
    )
    # Shift the data to ensure non-negative values
    X = X - X.min()
    return pd.DataFrame(X)


# Test cases for select_optimal_k
def test_select_optimal_k():
    X = get_sample_data()

    # Test with reasonable min_k and max_k values
    min_k = 2
    max_k = 5
    optimal_k = select_optimal_k(X, min_k, max_k)
    assert (
        min_k <= optimal_k <= max_k
    ), f"Optimal k should be between {min_k} and {max_k}, but got {optimal_k}."

    # Test with different min_k and max_k values
    min_k = 3
    max_k = 6
    optimal_k = select_optimal_k(X, min_k, max_k)
    assert (
        min_k <= optimal_k <= max_k
    ), f"Optimal k should be between {min_k} and {max_k}, but got {optimal_k}."


if __name__ == "__main__":
    pytest.main()
