####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
import patsy
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
    interaction_covs=[],
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
        interaction_covs (list, optional): Optionally, pass a list of the colum

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
    original_les = X.columns
    X.columns = [f"LD_{i+1}" for i in range(len(original_les))]
    y = meta[target]
    X.set_index(meta.index, inplace=True)
    # encode the categorical covariate columns and add them to X
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

    if len(interaction_covs) > 0:
        # Get the interaction terms dynamically for all covariates
        interaction_terms = {}

        for cov in interaction_covs:
            interaction_terms[cov] = [f"{pc}:{cov}" for pc in original_les]
            print(f"{cov.capitalize()}: ", interaction_terms[cov])

        # Combine all principal components and their interaction terms
        all_pcs = list(original_les)
        for terms in interaction_terms.values():
            all_pcs.extend(terms)

        print("All principal components and interactions: ", all_pcs)

        X_y = X.copy()
        # Combine X and y since the dmatrices function from the patsy package requires one dataframe
        X_y["y"] = y
        formula = "y ~ " + " + ".join(all_pcs) + " + " + " + ".join(covariates)
        _, X_interactions = patsy.dmatrices(formula, X_y)
        X_interactions = pd.DataFrame(X_interactions)
        X_interactions = X_interactions.drop(columns=X_interactions.columns[0], axis=1)
        X_interactions.columns = all_pcs + covariates
        X_interactions.set_index(X.index, inplace=True)

        X = X_interactions
        X.set_index(meta.index, inplace=True)

    return X, y
