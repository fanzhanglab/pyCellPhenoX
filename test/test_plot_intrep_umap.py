####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, scale_color_brewer, scale_color_continuous, labs, theme_classic
from pyCellPhenoX.plot_interpretablescore_umap import plot_interpretablescore_umap

####################################################
###
###                     TESTS
###
####################################################

def test_plot_interpretablescore_umap():
    # Test 1: Basic valid input
    data_valid = pd.DataFrame({
        'x': np.random.rand(10),
        'y': np.random.rand(10),
        'cell_type': np.random.choice(['Type1', 'Type2', 'Type3'], 10),
        'score': np.random.rand(10)
    })
    try:
        plot_interpretablescore_umap(data_valid, 'x', 'y', 'cell_type', 'score')
        print("Test 1: Function executed successfully with valid input.")
    except Exception as e:
        print(f"Test 1 Error: {e}")

    # Test 2: Missing columns
    data_missing_columns = pd.DataFrame({
        'x': np.random.rand(10),
        'y': np.random.rand(10),
        'cell_type': np.random.choice(['Type1', 'Type2', 'Type3'], 10)
    })
    try:
        plot_interpretablescore_umap(data_missing_columns, 'x', 'y', 'cell_type', 'score')
        print("Test 2: Function executed successfully with missing columns (should fail).")
    except Exception as e:
        print(f"Test 2 Error: {e}")

    # Test 3: Empty DataFrame
    data_empty = pd.DataFrame(columns=['x', 'y', 'cell_type', 'score'])
    try:
        plot_interpretablescore_umap(data_empty, 'x', 'y', 'cell_type', 'score')
        print("Test 3: Function executed successfully with empty input.")
    except Exception as e:
        print(f"Test 3 Error: {e}")

    # Test 4: Non-numeric data in 'score' column
    data_non_numeric_score = pd.DataFrame({
        'x': np.random.rand(10),
        'y': np.random.rand(10),
        'cell_type': np.random.choice(['Type1', 'Type2', 'Type3'], 10),
        'score': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # Non-numeric data
    })
    try:
        plot_interpretablescore_umap(data_non_numeric_score, 'x', 'y', 'cell_type', 'score')
        print("Test 4: Function executed successfully with non-numeric 'score' column (should fail).")
    except Exception as e:
        print(f"Test 4 Error: {e}")

    # Test 5: Non-numeric data in 'x', 'y', or 'cell_type' columns
    data_non_numeric_xy = pd.DataFrame({
        'x': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],  # Non-numeric data
        'y': np.random.rand(10),
        'cell_type': np.random.choice(['Type1', 'Type2', 'Type3'], 10),
        'score': np.random.rand(10)
    })
    try:
        plot_interpretablescore_umap(data_non_numeric_xy, 'x', 'y', 'cell_type', 'score')
        print("Test 5: Function executed successfully with non-numeric 'x' column (should fail).")
    except Exception as e:
        print(f"Test 5 Error: {e}")

if __name__ == "__main__":
    test_plot_interpretablescore_umap()
