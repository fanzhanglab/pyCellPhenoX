####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
import scanpy as sc
from multianndata import MultiAnnData as mad
import cna
from sklearn.preprocessing import LabelEncoder
from pyCellPhenoX.utils.balanced_sample import balanced_sample


####################################################
###
###                     FUNCTION
###
####################################################

def neighborhoodAbundanceMatrix(expression_mat, meta_data, sampleid):
    """Run CNA to generate neighborhood abundance matrix.

    Args:
        expression_mat (pd.DataFrame): The molecular expression matrix with cells as rows and markers as columns. 
        meta_data (pd.DataFrame): The corresponding meta data information with cells as rows and factors as columns.
        sampleid (str): Name of the column in meta_data with the sample IDs. 
    
    Returns:
        nam (pd.DataFrame): Neighborhood Abundance Matrix.
    """

    # label encode the non-numerical meta data columns 
    label_encoder = LabelEncoder()
    categoricalColumnNames = (
        meta_data
        .select_dtypes(include=["category", "object"])
        .columns.values.tolist()
    )
    for column_name in categoricalColumnNames:
        label_encoder = LabelEncoder()
        encoded_column = label_encoder.fit_transform(meta_data[column_name])
        meta_data[column_name] = encoded_column
    #meta_data['disease'] = label_encoder.fit_transform(meta_data['disease'])
    #meta_data['fibroblast_clusters'] = label_encoder.fit_transform(meta_data['fibroblast_clusters'])
    #meta_data['cluster'] = label_encoder.fit_transform(meta_data['cluster'])
    
    # create MultiAnnData object
    mad_obj = mad(X=expression_mat, obs=meta_data, sampleid=sampleid)
    # compute the UMAP cell-cell similarity graph
    sc.pp.neighbors(mad_obj, use_rep="X")
    # compute UMAP coordinates for plotting
    sc.tl.umap(mad_obj)
    # the following line would save the pre-processed data as a h5ad file
    #d.write('cna.h5ad')

    cna.tl.association(mad_obj, mad_obj.obs.disease)

    nam = mad_obj.uns['NAM.T']

    return nam
