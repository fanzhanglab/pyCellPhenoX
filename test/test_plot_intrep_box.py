####################################################
###
###                     IMPORTS
###
####################################################

import pytest
import pandas as pd
from plotnine import *
from pyCellPhenoX.plot_interpretablescore_boxplot import plot_interpretablescore_boxplot

####################################################
###
###                     TESTS
###
####################################################

# Sample test data
def generate_sample_data():
    data = {
        'x': ['A', 'B', 'A', 'B', 'A', 'B'],  # Categorical column (e.g., cell type)
        'y': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Numerical column (e.g., interpretable score)
    }
    return pd.DataFrame(data)

# Test the function with sample data
def test_plot_interpretablescore_boxplot():
    data = generate_sample_data()
    
    # Call the function to make sure it runs without error
    plot = plot_interpretablescore_boxplot(data, 'x', 'y')
    
    # Check if the returned object is a ggplot object (it should be)
    assert isinstance(plot, ggplot)


