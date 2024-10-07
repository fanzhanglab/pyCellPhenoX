####################################################
###
###                     IMPORTS
###
####################################################


import pytest
import pandas as pd
from sklearn.decomposition import NMF
from unittest.mock import patch
from pyCellPhenoX.src.nonnegative_matrix_factorization import (
    nonnegativeMatrixFactorization,
)


####################################################
###
###                     TESTS
###
####################################################


@pytest.fixture
def example_matrix():
    """Create a sample matrix for testing."""
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})


def test_nmf_with_fixed_components(example_matrix):
    """Test NMF with a fixed number of components."""
    W = nonnegativeMatrixFactorization(example_matrix, numberOfComponents=2)
    assert W is not None
    assert W.shape == (3, 2)  # Ensure it returns the right shape for W matrix


@patch("pyCellPhenoX.src.nonnegative_matrix_factorization.select_optimal_k")
def test_nmf_with_automatic_k(mock_select_optimal_k, example_matrix):
    """Test NMF when selecting optimal k."""
    mock_select_optimal_k.return_value = 2

    W = nonnegativeMatrixFactorization(
        example_matrix,
        numberOfComponents=-1,
        min_k=2,
        max_k=2,
    )

    mock_select_optimal_k.assert_called_once_with(example_matrix, 2, 2)
    assert W is not None
    assert W.shape == (3, 2)


def test_nmf_with_invalid_matrix():
    """Test NMF when the input matrix is empty or invalid."""
    empty_matrix = pd.DataFrame()

    with pytest.raises(ValueError):
        nonnegativeMatrixFactorization(empty_matrix, numberOfComponents=2)
