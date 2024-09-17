####################################################
###
###                     IMPORTS
###
####################################################


import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyCellPhenoX.utils.preprocessing import preprocessing


####################################################
###
###                     TESTS
###
####################################################


# Mock function for balanced_sample, replace with actual import or definition
def balanced_sample(df, subset_percentage=0.99):
    return df.sample(frac=subset_percentage)


# Sample data for testing
def create_sample_data():
    latent_features = pd.DataFrame(
        {"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]},
        index=["a", "b", "c", "d", "e"],
    )

    meta = pd.DataFrame(
        {
            "subject_id": ["s1", "s1", "s2", "s2", "s3"],
            "cell_type": ["type1", "type1", "type2", "type2", "type3"],
            "disease": ["disease1", "disease1", "disease2", "disease2", "disease3"],
        },
        index=["a", "b", "c", "d", "e"],
    )

    return latent_features, meta


def test_preprocessing():
    latent_features, meta = create_sample_data()

    X, y = preprocessing(
        latent_features,
        meta,
        sub_samp=False,
        subset_percentage=0.99,
        bal_col=["subject_id", "cell_type", "disease"],
        target="disease",
        covariates=[],
    )

    # Check that X and y are correct
    assert X.shape == latent_features.shape, "Feature matrix X shape mismatch"
    assert set(y.index) == set(meta.index), "Index mismatch between y and meta"
    assert "disease" in y.name, "Target column not found in y"


def test_preprocessing_with_subsampling():
    latent_features, meta = create_sample_data()

    # Apply preprocessing with subsampling
    X, y = preprocessing(
        latent_features,
        meta,
        sub_samp=True,
        subset_percentage=0.6,
        bal_col=["subject_id", "cell_type", "disease"],
        target="disease",
        covariates=[],
    )

    # Check that X is sampled
    assert (
        X.shape[0] < latent_features.shape[0]
    ), "Subsampling did not reduce the size of X"

    # Check index consistency
    expected_index = set(meta.index)
    actual_index = set(y.index)
    assert actual_index.issubset(
        expected_index
    ), "Index mismatch between y and meta. Expected subset of indices."


def test_preprocessing_with_categorical_covariates():
    latent_features, meta = create_sample_data()

    meta["cell_type"] = meta["cell_type"].astype("category")

    X, y = preprocessing(
        latent_features,
        meta,
        sub_samp=False,
        subset_percentage=0.99,
        bal_col=["subject_id", "cell_type", "disease"],
        target="disease",
        covariates=["cell_type"],
    )

    # Check encoding
    assert (
        "cell_type" in X.columns
    ), "Covariate 'cell_type' not added to feature matrix X"
    assert set(X["cell_type"].unique()) == set(
        range(len(meta["cell_type"].astype("category").cat.categories))
    ), "Categorical encoding mismatch"


if __name__ == "__main__":
    pytest.main()
