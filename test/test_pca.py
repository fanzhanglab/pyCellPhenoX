####################################################
###
###                     IMPORTS
###
####################################################

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from pyCellPhenoX.principalComponentAnalysis import principalComponentAnalysis


####################################################
###
###                     TESTS
###
####################################################


# Sample Data
def get_sample_data():
    return pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6], "C": [3, 4, 5, 6, 7]}
    )


# Test for correct output shape
@patch("pyCellPhenoX.utils.select_num_components.select_number_of_components")
def test_pca_correct_shape(mock_select_number_of_components):
    mock_select_number_of_components.return_value = 2
    X = get_sample_data()
    var = 0.95
    components = principalComponentAnalysis(X, var)
    print(components)
    assert components.shape[0] == 3


# Test for variance threshold edge case
@patch("pyCellPhenoX.utils.select_num_components.select_number_of_components")
def test_pca_edge_case(mock_select_number_of_components):
    mock_select_number_of_components.return_value = 1
    X = get_sample_data()
    var = 0.99
    components = principalComponentAnalysis(X, var)
    print(components)
    assert components.shape[0] == 3


# Test for empty DataFrame
@patch("pyCellPhenoX.utils.select_num_components.select_number_of_components")
def test_pca_empty_dataframe(mock_select_number_of_components):
    mock_select_number_of_components.return_value = 0
    X = pd.DataFrame()
    var = 0.95

    # Expect an exception to be raised
    with pytest.raises(ValueError, match="Input dataframe is empty"):
        principalComponentAnalysis(X, var)


# Test with a DataFrame with one feature
@patch("pyCellPhenoX.utils.select_num_components.select_number_of_components")
def test_pca_single_feature(mock_select_number_of_components):
    mock_select_number_of_components.return_value = 1
    X = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    var = 0.95
    components = principalComponentAnalysis(X, var)

    # Check if the result is as expected
    assert (
        components.shape[1] == 1
    ), "The number of principal components should be 1 for a single feature."


# Test for zero variance threshold
@patch("pyCellPhenoX.utils.select_num_components.select_number_of_components")
def test_pca_zero_variance(mock_select_number_of_components):
    mock_select_number_of_components.return_value = 0
    X = get_sample_data()
    var = 0

    with pytest.raises(ValueError, match="Variance threshold must be between 0 and 1"):
        principalComponentAnalysis(X, var)


# Test for invalid variance threshold
@patch("pyCellPhenoX.utils.select_num_components.select_number_of_components")
def test_pca_invalid_variance(mock_select_number_of_components):
    X = get_sample_data()
    invalid_var = -0.1  # Invalid variance value

    with pytest.raises(ValueError, match="Variance threshold must be between 0 and 1"):
        principalComponentAnalysis(X, invalid_var)

