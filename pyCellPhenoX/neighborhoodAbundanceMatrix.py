####################################################
###
###                     IMPORTS
###
####################################################

import logging

import pandas as pd
import scanpy as sc
from multianndata import MultiAnnData as mad
import cna
from sklearn.preprocessing import LabelEncoder
from pyCellPhenoX.utils.check_indices import check_indices

####################################################
###
###                     FUNCTION
###
####################################################


logger = logging.getLogger(__name__)


def neighborhoodAbundanceMatrix(
    expression_mat, meta_data, sampleid, association_col=None
):
    """Run CNA to generate neighborhood abundance matrix.

    Args:
        expression_mat (pd.DataFrame): The molecular expression matrix with cells as rows and markers as columns.
        meta_data (pd.DataFrame): The corresponding meta data information with cells as rows and factors as columns.
        sampleid (str): Name of the column in meta_data with the sample IDs.
        association_col (str, optional): Column in meta_data to use when computing the CNA association.
            Defaults to "disease" when present; otherwise inferred from categorical columns.

    Returns:
        nam (pd.DataFrame): Neighborhood Abundance Matrix.
    """
    # synchronize the indices of the two dataframes
    logger.debug("Checking indices between expression and metadata.")
    expression_mat, meta_data = check_indices(expression_mat, meta_data)
    logger.debug(
        "Back in neighborhoodAbundanceMatrix with expression type %s and meta type %s",
        type(expression_mat),
        type(meta_data),
    )
    # label encode the non-numerical meta data columns
    categoricalColumnNames = meta_data.select_dtypes(
        include=["category", "object"]
    ).columns.values.tolist()
    for column_name in categoricalColumnNames:
        label_encoder = LabelEncoder()
        encoded_column = label_encoder.fit_transform(meta_data[column_name])
        meta_data[column_name] = encoded_column
    # meta_data['disease'] = label_encoder.fit_transform(meta_data['disease'])
    # meta_data['fibroblast_clusters'] = label_encoder.fit_transform(meta_data['fibroblast_clusters'])
    # meta_data['cluster'] = label_encoder.fit_transform(meta_data['cluster'])

    # create MultiAnnData object
    mad_obj = mad(X=expression_mat, obs=meta_data, sampleid=sampleid)
    # compute the UMAP cell-cell similarity graph
    sc.pp.neighbors(mad_obj, use_rep="X")
    logger.debug("MultiAnnData summary: %s", mad_obj)
    logger.debug("mad_obj.obs head:\n%s", mad_obj.obs.head())

    # pick the association column
    association_priority = []
    if association_col is not None:
        if association_col in meta_data.columns:
            association_priority.append(association_col)
        else:
            logger.warning(
                "Requested association column '%s' not found in metadata; "
                "falling back to automatic selection.",
                association_col,
            )
    for candidate in [
        "disease",
    ]:
        if candidate in meta_data.columns and candidate not in association_priority:
            association_priority.append(candidate)
    for candidate in categoricalColumnNames:
        if candidate not in association_priority and candidate != sampleid:
            association_priority.append(candidate)
    if not association_priority:
        raise ValueError(
            "Unable to infer a categorical column for CNA association. "
            "Please provide one using the 'association_col' argument."
        )
    association_col = association_priority[0]
    logger.info(
        "Using '%s' column for CNA association (%d unique levels).",
        association_col,
        mad_obj.obs[association_col].nunique(),
    )
    # compute UMAP coordinates for plotting
    sc.tl.umap(mad_obj)

    logger.debug("mad_obj.obs type: %s", type(mad_obj.obs))
    logger.debug(
        "mad_obj.obs['%s'] head:\n%s",
        association_col,
        mad_obj.obs[association_col].head(),
    )

    cna.tl.association(mad_obj, mad_obj.obs[association_col])

    nam = mad_obj.uns["NAM.T"]

    return nam
