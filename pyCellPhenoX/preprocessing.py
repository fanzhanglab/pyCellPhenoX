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
        latent_features (pd.DataFrame): Latent embeddings (e.g., NMF ranks, or principal components) of the NAM
        meta (dataframe): Dataframe containing meta data (e.g., covariates, target/outcome variable for classification model)
        sub_samp (bool, optional): Optionally, subsample the data. Defaults to False.
        subset_percentage (float, optional): If sub_samp = True, specify the desired proportion of rows. Defaults to 0.99.
        bal_col (list, optional): List of column names in meta to balance the subsampling by. Defaults to ["subject_id", "cell_type", "disease"].
        target (str): Name of the outcome column in meta. Defaults to "disease".
        covariates (list, optional): List of column names in meta that are to be included as features/predictors in the classsification model. Defaults to [].

    Returns:
        tuple (dataframe, series):  X, latent embeddings and covariates (your predictors); y, model outcome (your target variable)
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
    X.columns = [f"LE_{col+1}" for col in X.columns]
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
