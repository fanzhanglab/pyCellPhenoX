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
    # synchronize the indices of the two dataframes
    print("checking indices")
    expression_mat, meta_data = check_indices(expression_mat, meta_data)
    print("back in the NAM function")
    print(f"exp type: {type(expression_mat)}, meta type: {type(meta_data)}")
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
    print(str(mad_obj))
    print(mad_obj.obs)
    print(f"length of the disease level: {len(mad_obj.obs.disease)}")
    # compute UMAP coordinates for plotting
    sc.tl.umap(mad_obj)

    print(f"type of mad_obj.obs: {type(mad_obj.obs)}")
    print("mad_obj.obs head")
    print(mad_obj.obs.head())
    print("mad_obj.obs.disease head")
    print(mad_obj.obs.disease.head())

    cna.tl.association(mad_obj, mad_obj.obs.disease)

    nam = mad_obj.uns['NAM.T']

    return nam

def check_indices(a, b):
    """Check that the indices are matching for the dataframes and assign indices if they aren't. 

    Args:
        a (pd.DataFrame): DataFrame 1
        b (pd.DataFrame): DataFrame 2

    Returns:
        a, b (pd.DataFrame, pd.DataFrame): 
    """

    # We assume here that we prefer the indices that are strings (i.e., cell names/barcodes)
    if a.index.dtype == 'object' and b.index.dtype != 'object':
        print("Synchronizing indices: a has string indices.")
        b.index = a.index
    elif b.index.dtype == 'object' and a.index.dtype != 'object':
        print("Synchronizing indices: b has string indices.")
        a.index = b.index
    else:
        print("Both DataFrames already have matching index types or no string indices detected. Leaving alone.")

    return a, b 