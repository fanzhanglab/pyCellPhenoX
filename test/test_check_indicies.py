####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
import pytest
from pyCellPhenoX.utils.check_indices import check_indices

####################################################
###
###                     TESTS
###
####################################################

def test_check_indices_matching_indices():
    # Test when indices already match
    df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['cell1', 'cell2', 'cell3'])
    df2 = pd.DataFrame({'B': [4, 5, 6]}, index=['cell1', 'cell2', 'cell3'])
    
    a, b = check_indices(df1, df2)
    
    pd.testing.assert_frame_equal(a, df1)
    pd.testing.assert_frame_equal(b, df2)

def test_check_indices_sync_b_indices():
    # Test when b needs to synchronize with a's string indices
    df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['cell1', 'cell2', 'cell3'])
    df2 = pd.DataFrame({'B': [4, 5, 6]}, index=[0, 1, 2])
    
    a, b = check_indices(df1, df2)
    
    assert a.index.equals(df1.index)
    assert b.index.equals(df1.index)

def test_check_indices_sync_a_indices():
    # Test when a needs to synchronize with b's string indices
    df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({'B': [4, 5, 6]}, index=['cell1', 'cell2', 'cell3'])
    
    a, b = check_indices(df1, df2)
    
    assert b.index.equals(df2.index)
    assert a.index.equals(df2.index)

def test_check_indices_no_string_indices():
    # Test when neither DataFrame has string indices
    df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({'B': [4, 5, 6]}, index=[0, 1, 2])
    
    a, b = check_indices(df1, df2)
    
    pd.testing.assert_frame_equal(a, df1)
    pd.testing.assert_frame_equal(b, df2)

def test_check_indices_empty_dataframes():
    # Test with empty DataFrames
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    a, b = check_indices(df1, df2)
    
    pd.testing.assert_frame_equal(a, df1)
    pd.testing.assert_frame_equal(b, df2)



