#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Imports
import pyCellPhenoX
from pyCellPhenoX import (
    CellPhenoX,
    preprocessing,
    nonnegativeMatrixFactorization,
    principalComponentAnalysis,
    marker_discovery,
)
import pandas as pd
import os

# In[2]:
# Step 1: Import Data
# paths to expression data and meta data files
expression_file = "uc_fibroblast_exp.csv"
meta_file = "uc_fibroblast_meta.csv"
output_path = "output/"

# check if the output path exists, if not create it
if not os.path.exists(output_path):
    os.makedirs(output_path)

# read in data
expression_mat = pd.read_csv(expression_file, index_col=0)
meta = pd.read_csv(meta_file, index_col=0)

# In[3]:
expression_mat.head()

# In[4]:
meta.head()

# In[5]:
# Step 2: Preprocess Data
## generate latent dimensions configure input for CellPhenoX (includes covariants and identify target column)

## we actually need both the neighborhood abundance matrix (for CellPhenoX) & expression data (for the marker discovery later)

# get the latent dimensions using NMF
latent_features = nonnegativeMatrixFactorization(
    expression_mat, numberOfComponents=4, min_k=3, max_k=5
)

# alternatively, use PCA
# proportion_var_explained = 0.9
# latent_features = principalComponentAnalysis(expression_mat, var=proportion_var_explained)

# In[6]:
# then, set up the input data for CellPhenoX
X, y = preprocessing(
    latent_features,
    meta,
    sub_samp=False,
    subset_percentage=0.25,
    target="disease",
    covariates=[],
)
X.head()

# In[7]:
print(X.shape)
print(y.shape)

# In[8]:
# Step 3: Run CellPhenoX

# create CellPhenoX object
cellpx_obj = CellPhenoX(X, y, CV_repeats=1, outer_num_splits=3, inner_num_splits=2)

# and then train the classification model
cellpx_obj.model_training_shap_val(outpath=output_path)

# In[9]:
cellpx_obj.shap_df

# In[10]:
# Step 4: Marker Discovery
## identify markers correlasted with the Interpretable Score
marker_discovery(cellpx_obj.shap_df, expression_mat)

# In[11]:
# End of walkthrough
