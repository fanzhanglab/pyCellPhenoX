####################################################
###
###                     IMPORTS
###
####################################################


import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from pyCellPhenoX.utils.balanced_sample import balanced_sample
from pyCellPhenoX import neighborhoodAbundanceMatrix
from pyCellPhenoX.preprocessing import preprocessing


####################################################
###
###                     TESTS
###
####################################################

# Helper function to generate sample data
def generate_sample_data():
    # Generate random data for testing
    # Example: 100 samples, 5 features (latent features), and some meta data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    latent_features = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 6)])
    meta = pd.DataFrame({
        'subject_id': ['subject_' + str(i % 10) for i in range(100)],
        'cell_type': ['type_' + str(i % 3) for i in range(100)],
        'disease': y,  # Target column
        'covariate1': ['cat_' + str(i % 3) for i in range(100)],
        'covariate2': [i % 5 for i in range(100)],
    })
    return latent_features, meta

# Test function for preprocessing
def test_preprocessing():
    latent_features, meta = generate_sample_data()

    # Test with default parameters
    X, y = preprocessing(latent_features, meta)
    
    # Check that X is a DataFrame and y is a Series
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    # Check that the number of rows in X matches meta
    assert X.shape[0] == meta.shape[0]
    
    # Check that the 'LD_' columns exist in X after renaming
    assert all([col.startswith("LD_") for col in X.columns])

    # Test subsampling functionality
    X_sub, y_sub = preprocessing(latent_features, meta, sub_samp=True, subset_percentage=0.5)
    
    # Check that X_sub and y_sub are a DataFrame and Series
    assert isinstance(X_sub, pd.DataFrame)
    assert isinstance(y_sub, pd.Series)
    
    # Ensure that X_sub has fewer rows than the original
    assert X_sub.shape[0] < X.shape[0]
    
    # Test with no interaction terms
    X_no_int, y_no_int = preprocessing(latent_features, meta, interaction_covs=[])
    
    # Ensure the number of columns is the same as before (no interaction terms)
    assert X_no_int.shape[1] == X.shape[1]
