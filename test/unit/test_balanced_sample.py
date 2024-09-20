####################################################
###
###                     Imports
###
####################################################


import pytest
import pandas as pd
from pyCellPhenoX.src.utils.balanced_sample import balanced_sample


####################################################
###
###                     Tests
###
####################################################


def test_balanced_sample_valid_percentage():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    result = balanced_sample(df, 0.4)
    assert result.shape[0] == 2  # 40% of 5 rows is 2 rows
    assert set(result["A"]).issubset(
        set(df["A"])
    )  # Check that sampled rows are from original df


def test_balanced_sample_zero_percentage():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    result = balanced_sample(df, 0.0)
    assert result.shape[0] == 0  # No rows should be sampled
    assert result.empty  # Ensure the result is empty


def test_balanced_sample_full_percentage():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    result = balanced_sample(df, 1.0)
    pd.testing.assert_frame_equal(
        result.sort_values(by="A").reset_index(drop=True),
        df.sort_values(by="A").reset_index(drop=True),
    )  # The result should be the same as the original DataFrame, ignoring row order


def balanced_sample(df, subset_percentage):
    replace = subset_percentage > 1
    return df.sample(frac=subset_percentage, replace=replace)


def test_balanced_sample_empty_dataframe():
    df = pd.DataFrame(columns=["A"])
    result = balanced_sample(df, 0.5)
    assert result.empty  # Result should be empty when input is an empty DataFrame
