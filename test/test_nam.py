####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyCellPhenoX.neighborhoodAbundanceMatrix import neighborhoodAbundanceMatrix

####################################################
###
###                     TESTS
###
####################################################

def test_neighborhoodAbundanceMatrix_unsynced_indices():
    # Generate expression matrix with different indices
    expression_mat = pd.DataFrame(
        np.random.rand(10, 5),
        index=[f"cell{i}" for i in range(10)],
        columns=[f"marker{i}" for i in range(5)]
    )
    
    # Metadata with unsynced indices
    meta_data = pd.DataFrame(
        {
            "sample_id": ["sample1"] * 5 + ["sample2"] * 5,
            "disease": ["healthy", "healthy", "diseased", "diseased", "healthy"] * 2
        },
        index=[f"cell{10 + i}" for i in range(10)]
    )
    
    # Run the function
    try:
        nam = neighborhoodAbundanceMatrix(expression_mat, meta_data, "sample_id")
    except Exception as e:
        print(f"Error caught as expected: {e}")
    else:
        assert False, "Function did not handle unsynced indices as expected"
    print("Unsynced indices test passed.")


def test_neighborhoodAbundanceMatrix_empty_input():
    # Empty inputs
    expression_mat = pd.DataFrame()
    meta_data = pd.DataFrame()
    
    # Run the function
    try:
        nam = neighborhoodAbundanceMatrix(expression_mat, meta_data, "sample_id")
    except Exception as e:
        print(f"Error caught as expected: {e}")
    else:
        assert False, "Function did not handle empty inputs as expected"
    print("Empty input test passed.")

