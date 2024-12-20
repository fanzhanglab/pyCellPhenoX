####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
import numpy as np
from pyCellPhenoX.marker_discovery import marker_discovery

####################################################
###
###                     TESTS
###
####################################################


def test_marker_discovery():
    # Test 1: Basic valid input
    shap_df = pd.DataFrame({
        "interpretable_score": np.random.rand(10),  
        "meta_data_column": np.random.choice(["A", "B", "C"], size=10)  
    })
    expression_mat = pd.DataFrame(np.random.rand(10, 5), columns=[f"gene{i}" for i in range(5)])
    try:
        marker_discovery(shap_df, expression_mat)
        print("Test 1: Function executed successfully with valid input.")
    except Exception as e:
        print(f"Test 1 Error: {e}")

    # Test 2: Empty inputs
    shap_df_empty = pd.DataFrame({"interpretable_score": [], "meta_data_column": []})
    expression_mat_empty = pd.DataFrame()
    try:
        marker_discovery(shap_df_empty, expression_mat_empty)
        print("Test 2: Function executed successfully with empty input.")
    except Exception as e:
        print(f"Test 2 Error: {e}")

    # Test 3: Mismatch in rows between shap_df and expression_mat
    shap_df_mismatch = pd.DataFrame({
        "interpretable_score": np.random.rand(10),  
        "meta_data_column": np.random.choice(["A", "B", "C"], size=10)
    })
    expression_mat_mismatch = pd.DataFrame(np.random.rand(5, 5), columns=[f"gene{i}" for i in range(5)])
    try:
        marker_discovery(shap_df_mismatch, expression_mat_mismatch)
        print("Test 3: Function executed successfully with mismatched row count.")
    except Exception as e:
        print(f"Test 3 Error: {e}")

    # Test 4: All numerical data in shap_df (no categorical data)
    shap_df_numerical = pd.DataFrame({
        "interpretable_score": np.random.rand(10),  
        "numerical_column": np.random.rand(10)
    })
    expression_mat_numerical = pd.DataFrame(np.random.rand(10, 5), columns=[f"gene{i}" for i in range(5)])
    try:
        marker_discovery(shap_df_numerical, expression_mat_numerical)
        print("Test 4: Function executed successfully with numerical data in shap_df.")
    except Exception as e:
        print(f"Test 4 Error: {e}")

    # Test 5: High variance in expression_mat columns
    expression_mat_high_variance = pd.DataFrame(np.random.rand(10, 5) * 1000, columns=[f"gene{i}" for i in range(5)])
    try:
        marker_discovery(shap_df, expression_mat_high_variance)
        print("Test 5: Function executed successfully with high variance in expression_mat.")
    except Exception as e:
        print(f"Test 5 Error: {e}")

    # Test 6: Non-numerical columns in expression_mat
    expression_mat_non_numeric = pd.DataFrame(np.random.rand(10, 5), columns=[f"gene{i}" for i in range(4)] + ["non_numeric_column"])
    try:
        marker_discovery(shap_df, expression_mat_non_numeric)
        print("Test 6: Function executed successfully with non-numerical columns in expression_mat.")
    except Exception as e:
        print(f"Test 6 Error: {e}")
