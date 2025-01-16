# import pytest
# import numpy as np
# from unittest.mock import patch

# # Import the function to test
# from pyCellPhenoX.utils.reducedim import reduceDim

# # Mock for principalComponentAnalysis
# def mock_principalComponentAnalysis(expression_mat, n_components=2):
#     # Assuming PCA returns a matrix (mock implementation)
#     # Simulating PCA with the expected n_components
#     return np.random.rand(n_components, expression_mat.shape[1])

# # Mock for nonnegativeMatrixFactorization
# def mock_nonnegativeMatrixFactorization(expression_mat, n_components=2):
#     # Assuming NMF returns a tuple of matrices (mock implementation)
#     # Simulating NMF with the expected n_components
#     return np.random.rand(n_components, expression_mat.shape[1]), np.random.rand(n_components, expression_mat.shape[1])

# @pytest.fixture
# def expression_mat():
#     # Create a mock expression matrix (e.g., a 5x5 matrix with random values)
#     return np.random.rand(5, 5)

# @pytest.mark.parametrize(
#     "reducMethod, reducMethodParams, expected_shape",
#     [
#         ("pca", {"n_components": 2}, (2, 5)),  # Expecting a 2x5 matrix for PCA
#         ("nmf", {"n_components": 2}, (2, 5)),  # Expecting a tuple of 2x5 matrices for NMF
#     ],
# )
# def test_reduceDim(reducMethod, reducMethodParams, expected_shape, expression_mat):
#     # Mock the PCA and NMF methods
#     with patch('pyCellPhenoX.principalComponentAnalysis', side_effect=mock_principalComponentAnalysis), \
#          patch('pyCellPhenoX.nonnegativeMatrixFactorization', side_effect=mock_nonnegativeMatrixFactorization):
        
#         result = reduceDim(reducMethod, reducMethodParams, expression_mat)
        
#         # Check the shape of the result
#         if reducMethod == "pca":
#             assert result.shape == expected_shape
#         elif reducMethod == "nmf":
#             assert len(result) == 2  # NMF should return a tuple of two matrices
#             assert result[0].shape == expected_shape
#             assert result[1].shape == expected_shape
