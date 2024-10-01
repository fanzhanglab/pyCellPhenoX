import argparse
import yaml
import pandas as pd
from pyCellPhenoX.nonnegative_matrix_factorization import (
    nonnegativeMatrixFactorization,
)
from pyCellPhenoX.principle_component_analysis import (
    principalComponentAnalysis,
)
from pyCellPhenoX.preprocessing import preprocessing
from CellPhenoX import *  # do we want to import everything from the module? maybe only the necessary functions? ie just the class CellPhenoX (then were not importing the dependencies of the other functions)
from pyCellPhenoX.MarkerDiscovery import *  # see above comment


def main(config):
    expression_file = config["expression_file"]
    meta_file = config["meta_file"]
    covariates = config["covariates"]
    target = config["target"]
    method = config["method"]  # method is not used in the current version
    sub_samp = config["sub_samp"]
    subset_percentage = config["subset_percentage"]
    num_cv_repeats = config["num_cv_repeats"]
    num_inner_splits = config["num_inner_splits"]
    num_outer_splits = config["num_outer_splits"]
    reduc_method = config["reduction_method"]
    proportion_var_explained = config["proportion_var_explained"]
    num_ranks = config["num_ranks"]
    min_k = config["min_k"]
    max_k = config["max_k"]
    output_path = config["output_path"]

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

    X, y = preprocessing(
        latent_features,
        meta,
        sub_samp=sub_samp,
        subset_percentage=subset_percentage,
        target=target,
        covariates=covariates,
    )

    ## run CPX - train the classification and get SHAP values
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
    parser = argparse.ArgumentParser(
        description="CellPhenoX input parameters."
    )  # would we want another function to handle the argument parsing?
    parser.add_argument(
        "--config-file",
        dest="config_file",
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--expression-file",
        dest="expression_file",
        type=str,
        help="Path to expression data file.",
    )
    parser.add_argument(
        "--meta-file",
        dest="meta_file",
        type=str,
        help="Path to meta data file.",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        type=str,
        help="Path for output.",
    )
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

    args = parser.parse_args()

    if args.config_file:
        # Load configuration from YAML file
        with open(args.config_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        # Use command-line arguments
        config = vars(args)

    main(config)
