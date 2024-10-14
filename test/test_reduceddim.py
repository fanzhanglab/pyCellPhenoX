import pytest
import numpy as np
import pandas as pd
from pyCellPhenoX.utils.reducedim import reduceDim

# Sample data generation
def create_sample_data(n_samples=100, n_features=20):
    """Generate a random marker-by-cell matrix."""
    np.random.seed(11)
    data = np.random.rand(n_samples, n_features)
    return pd.DataFrame(data)

# Test for PCA method
def test_reduceDim_pca():
    # Create sample data
    X = create_sample_data()

    # Define PCA reduction parameters
    reducMethod = "pca"
    reducMethodParams = {"var": 0.90}

    # Call the reduceDim function
    result = reduceDim(reducMethod, reducMethodParams, X)

    # Check if the result is a numpy array
    assert isinstance(result, np.ndarray), "PCA should return a numpy array"

    # Check that the result has the correct shape (n_features by num_components)
    # Since the exact number of components depends on the variance threshold, we can't assert an exact number,
    # but we can check that it reduces the dimensionality (less than or equal to n_features).
    assert result.shape[0] <= X.shape[1], "PCA result should have fewer or equal components than input features"

    print(f"PCA result shape: {result.shape}")

# Test for NMF method
def test_reduceDim_nmf():
    # Create sample data
    X = create_sample_data()

    # Define NMF reduction parameters
    reducMethod = "nmf"
    reducMethodParams = {"numberOfComponents": 5}

    # Call the reduceDim function
    W, H = reduceDim(reducMethod, reducMethodParams, X)

    # Check if the results are numpy arrays
    assert isinstance(W, np.ndarray), "NMF W should return a numpy array"
    assert isinstance(H, np.ndarray), "NMF H should return a numpy array"

    # Check that W and H have the correct shapes
    assert W.shape == (X.shape[0], 5), f"NMF W matrix should have shape {(X.shape[0], 5)}"
    assert H.shape == (5, X.shape[1]), f"NMF H matrix should have shape {(5, X.shape[1])}"

    print(f"NMF W shape: {W.shape}")
    print(f"NMF H shape: {H.shape}")

# Test for invalid method
def test_reduceDim_invalid_method():
    # Create sample data
    X = create_sample_data()

    # Define an invalid reduction method
    reducMethod = "invalid_method"
    reducMethodParams = {}

    # Check that the reduceDim function raises a ValueError for an invalid method
    with pytest.raises(ValueError, match="Invalid dimensionality reduction method provided!"):
        reduceDim(reducMethod, reducMethodParams, X)

# Run the tests
if __name__ == "__main__":
    test_reduceDim_pca()
    test_reduceDim_nmf()
    test_reduceDim_invalid_method()
