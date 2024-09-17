import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns

# import pyreadr
import psutil
import anndata as ad
from multianndata import MultiAnnData as mad

import shap
from xgboost import *
import cna

import sys
import gc

import scanpy as sc
import scipy
from scipy import stats
from scipy.stats import randint
from scipy.stats import mannwhitneyu, normaltest

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    silhouette_score,
    explained_variance_score,
    roc_auc_score,
    roc_curve,
    auc,
    make_scorer,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans

# from statannotations.Annotator import Annotator
# from harmony import harmonize
# import harmonypy as hm
import patsy
from plotnine import *

import multiprocessing as mp
from datetime import date
import time

from Preprocessing import *
from CellPhenoX import *
from pyCellPhenoX.utils.MarkerDiscovery import *


def main():
    expression_file = args.expression_file
    meta_file = args.meta_file
    covariates = args.covariates
    target = args.target
    method = args.method
    sub_samp = args.sub_samp
    subset_percentage = args.aubset_percentage
    num_cv_repeats = args.num_cv_repeats
    num_inner_splits = args.num_inner_splits
    num_outer_splits = args.num_outer_splits
    reduc_method = args.reduction_method
    proportion_var_explained = args.proportion_var_explained
    num_ranks = args.num_ranks
    min_k = args.min_k
    max_k = args.max_k
    output_path = args.output_path
    expression_mat = pd.read_csv(expression_file, index_col=0)
    meta = pd.read_csv(meta_file, index_col=0)

    ## preprocessing -  prepare the data to be in the correct format for CPX
    if reduc_method == "nmf":
        latent_features = nonnegativeMatrixFactorization(
            expression_mat, numberOfComponents=num_ranks, min_k=min_k, max_k=max_k
        )
    elif reduc_method == "pca":
        latent_features = principalComponentAnalysis(
            expression_mat, proportion_var_explained
        )

    # latent_features = reduceDim(reduc_method, **reducMethodParams)
    X, y = preprocessing(
        latent_features,
        meta,
        sub_samp=False,
        subset_percentage=subset_percentage,
        target=target,
        covariates=[],
    )  # bal_col=['subject_id', 'cell_type','disease']

    ## run CPX - train the classification and get SHAP values
    # create object
    cellpx_obj = CellPhenoX(
        X,
        y,
        CV_repeats=num_cv_repeats,
        outer_num_splits=num_outer_splits,
        inner_num_splits=num_inner_splits,
    )
    cellpx_obj.model_training_shap_val(outpath=output_path)

    ## marker discovery - find markers correlated with the Interpretable Score
    # marker_discovery(CellPhenoX.shap_df, expression_mat)


if __name__ == "__main__":
    # ARGUMENT PARSER =========
    parser = argparse.ArgumentParser(description="CellPhenoX input parameters.")
    # NOTE: add help page
    # input files
    parser.add_argument(
        "--expression-file",
        dest="expression_file",
        type=str,
        help="Path to expression data file.",
    )
    parser.add_argument(
        "--meta-file", dest="meta_file", type=str, help="Path to meta data file."
    )
    parser.add_argument(
        "--output-path", dest="output_path", type=str, help="Path for output."
    )
    # dimensionality reduction parameters
    parser.add_argument(
        "--reduction-method",
        dest="reduction_method",
        type=str,
        default="nmf",
        help="Dimensionality reduction method: 'nmf' or 'pca'.",
    )
    parser.add_argument(
        "--proportion-var-explained",
        dest="proportion_var_explained",
        type=float,
        default=0.95,
        help="Desired proportion of variance explained if PCA is selected.",
    )
    parser.add_argument(
        "--number-ranks",
        dest="num_ranks",
        type=int,
        default=-1,
        help="Number of ranks for NMF (default: -1).",
    )
    parser.add_argument(
        "--minimum-k",
        dest="min_k",
        type=int,
        default=2,
        help="Minimum k value for selecting optimal number of ranks if NMF is selected (default: 2).",
    )
    parser.add_argument(
        "--maximum-k",
        dest="max_k",
        type=int,
        default=7,
        help="Maximum k value for selecting optimal number of ranks if NMF is selected (default: 7).",
    )
    # classification model parameters
    parser.add_argument(
        "--covariates",
        dest="covariates",
        nargs="+",
        type=str,
        help="Covariates to include in classification model.",
    )
    parser.add_argument(
        "--target-variable",
        dest="target",
        type=str,
        default="disease",
        help="Name of the target/outcome column in the meta data table.",
    )
    parser.add_argument(
        "--classification-method",
        dest="method",
        type=str,
        default="rf",
        help="Classification model to train - rf for Random Forest (default), xgb for XGBoost.",
    )
    parser.add_argument(
        "--sub-sample",
        dest="sub_samp",
        type="store_true",
        help="Only use a subset of the data?",
    )
    parser.add_argument(
        "--subset-percentage",
        dest="subset_percentage",
        type=float,
        default=0.25,
        help="If sub_sample=True, what portion of the data should we use?",
    )
    parser.add_argument(
        "--num-cv-repeats",
        dest="num_cv_repeats",
        type=int,
        default=3,
        help="Number of cross-validation repeats",
    )
    parser.add_argument(
        "--num-outer-splits",
        dest="num_outer_splits",
        type=int,
        default=3,
        help="Number of outer loop splits (stratified k folds)",
    )
    parser.add_argument(
        "--num-inner-splits",
        dest="num_inner_splits",
        type=int,
        default=5,
        help="Number of inner loop splits (hyperparameter tuning)",
    )

    # parser.add_argument("--harmonize", dest="harmonize", action="store_true", help="Run harmony?")
    # parser.add_argument("--batch-keys", dest="batch_keys", nargs='+', type=str, help="If running harmony, please provide a list of batch keys to use. These should be names of columns in the metadata table.")

    args = parser.parse_args()
    main(args)
