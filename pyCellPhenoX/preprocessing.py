####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pyCellPhenoX.utils.balanced_sample import balanced_sample


####################################################
###
###                     FUNCTION
###
####################################################


def preprocessing(
    latent_features,
    meta,
    sub_samp=False,
    subset_percentage=0.99,
    bal_col=["subject_id", "cell_type", "disease"],
    target="disease",
    covariates=[],
):
    """Prepare the data to be in the correct format for CellPhenoX

    Args:
        latent_features (list): The latent features from the dimensionality reduction algorithm
        meta (dataframe): The metadata for the cells
        sub_samp (bool, optional): Whether to subsample the data. Defaults to False.
        subset_percentage (float, optional): The percentage of the data to subsample. Defaults to 0.99.
        bal_col (list, optional): The columns to balance the data on. Defaults to ["subject_id", "cell_type", "disease"].
        target (str, optional): The target variable. Defaults to "disease".
        covariates (list, optional): The covariates to include in the model. Defaults to [].

    Returns:
        tuple: X and y
    """
    if sub_samp:
        # optionally, sample the data using the balanced sample function
        # subset_percentage = 0.10
        meta = meta.groupby(bal_col, group_keys=False, sort=False).apply(
            lambda x: balanced_sample(x, subset_percentage=subset_percentage)
        )
        # subset the (expression) data based on the selected rows of the meta data
        latent_features = latent_features.loc[meta.index]

    X = pd.DataFrame(latent_features)
    y = meta[target]
    X.set_index(meta.index, inplace=True)
    # code the categorical covariate columns and add them to X
    categoricalColumnNames = (
        meta[covariates]
        .select_dtypes(include=["category", "object"])
        .columns.values.tolist()
    )
    for column_name in categoricalColumnNames:
        label_encoder = LabelEncoder()
        encoded_column = label_encoder.fit_transform(meta[column_name])
        meta[column_name] = encoded_column
    for covariate in covariates:
        X[covariate] = meta[covariate]
    X = X.rename(str, axis="columns")
    return X, y
