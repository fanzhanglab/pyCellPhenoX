####################################################
###
###                     Imports
###
####################################################


import numpy as np
from pyCellPhenoX.utils.select_num_components import select_number_of_components


####################################################
###
###                     Tests
###
####################################################


def test_select_number_of_components():
    # Test case 1: Simple case
    eigenvalues = np.array([2.0, 1.5, 1.0, 0.5])
    var = 0.8
    expected = 1
    result = select_number_of_components(eigenvalues, var)
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 2: All components needed
    eigenvalues = np.array([2.0, 1.5, 1.0, 0.5])
    var = 1.0
    expected = 1
    result = select_number_of_components(eigenvalues, var)
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 3: Var lower than the first component
    eigenvalues = np.array([2.0, 1.5, 1.0, 0.5])
    var = 0.1
    expected = 1
    result = select_number_of_components(eigenvalues, var)
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 4: Var between components
    eigenvalues = np.array([2.0, 1.0, 0.5, 0.5])
    var = 0.75
    expected = 1
    result = select_number_of_components(eigenvalues, var)
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 5: Edge case where var is zero
    eigenvalues = np.array([1.0, 1.0, 1.0])
    var = 0.0
    expected = 1
    result = select_number_of_components(eigenvalues, var)
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test case 6: Var higher than total variance
    eigenvalues = np.array([1.0, 1.0, 1.0])
    var = 1.5
    expected = 2
    result = select_number_of_components(eigenvalues, var)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_select_number_of_components_extended():
    # Edge case: Empty eigenvalues array
    eigenvalues = np.array([])
    var = 0.8
    try:
        result = select_number_of_components(eigenvalues, var)
        assert False, "Function should raise an exception for empty eigenvalues array"
    except ValueError as e:
        print(f"Correctly raised ValueError for empty eigenvalues array: {e}")

