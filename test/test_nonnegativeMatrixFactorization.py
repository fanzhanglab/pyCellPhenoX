####################################################
###
###                     IMPORTS
###
####################################################

import pytest
import pandas as pd
from sklearn.decomposition import NMF
from unittest.mock import patch
from pyCellPhenoX.nonnegativeMatrixFactorization import (
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


def test_nmf_with_invalid_matrix():
    """Test NMF when the input matrix is empty or invalid."""
    empty_matrix = pd.DataFrame()

    with pytest.raises(ValueError):
        nonnegativeMatrixFactorization(empty_matrix, numberOfComponents=2)


def test_nmf_with_invalid_number_of_components(example_matrix):
    """Test NMF with invalid numberOfComponents."""
    with pytest.raises(ValueError):
        nonnegativeMatrixFactorization(example_matrix, numberOfComponents=0)


def test_nmf_with_single_row_matrix():
    """Test NMF with a single row matrix."""
    single_row_matrix = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    W = nonnegativeMatrixFactorization(single_row_matrix, numberOfComponents=2)
    assert W.shape == (1, 2)


def test_nmf_with_single_column_matrix():
    """Test NMF with a single column matrix."""
    single_column_matrix = pd.DataFrame({"A": [1, 2, 3]})
    W = nonnegativeMatrixFactorization(single_column_matrix, numberOfComponents=1)
    assert W.shape == (3, 1)


def test_nmf_random_seed_consistency(example_matrix):
    """Test NMF produces consistent results with the same random seed."""
    W1 = nonnegativeMatrixFactorization(example_matrix, numberOfComponents=2)
    W2 = nonnegativeMatrixFactorization(example_matrix, numberOfComponents=2)
    pd.testing.assert_frame_equal(pd.DataFrame(W1), pd.DataFrame(W2))
