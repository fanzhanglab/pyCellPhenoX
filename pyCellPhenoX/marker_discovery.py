import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from plotnine import *
# from rpy2.robjects import r, pandas2ri
# from rpy2.robjects.packages import importr

# marker discovery - find markers correlated with the discriminatory power of the Interpretable Score
##TODO: loosely translated from R to python, not fully tested and missing the final output (maybe some plots and the datataframe containing the coefficients and pvalues?)
def marker_discovery(shap_df, expression_mat):
    """_summary_

    Args:
        shap_df (dataframe): cells by (various columns: meta data, shap values for each latent dimension, interpretable score)
        expression_mat (dataframe): cells by genes/proteins/etc.
    """
    # Define the response variable and predictor variables
    y = shap_df["interpretable_score"]
    X = expression_mat

    # Add a constant (intercept term) to the predictor variables for the linear model
    """X = sm.add_constant(X)
    print("fitting model")
    # Fit the linear model
    model = sm.OLS(y, X).fit()

    # Get the model summary
    model_summary = model.summary()

    # Extract coefficients and p-values
    coefficients = model.params
    p_values = model.pvalues

    # Combine betas (coefficients) and p-values into a DataFrame
    results = pd.DataFrame({
        'Beta': coefficients,
        'P_Value': p_values
    })

    # Adjust p-values using the Benjamini-Hochberg method (equivalent to "BH" in R)
    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

    # Add adjusted p-values to the results DataFrame
    results['Adjusted_P_Value'] = adjusted_p_values
    results['gene'] = results.index

    print("results sorted by p vlaue: ")
    print(results.sort_values(by='P_Value').head())"""
    # Add constant (intercept term)
    X = sm.add_constant(X)

    # Check for multicollinearity
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    print("VIFs: ", vif_data)

    # Drop low variance columns
    low_variance_cols = X.columns[X.var() == 0]
    X = X.drop(columns=low_variance_cols)

    # Scale predictors to avoid numerical issues
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = sm.add_constant(X_scaled)

    # Fit the linear model
    print("fitting model")
    model = sm.OLS(y, X_scaled).fit()

    # Get the model summary
    model_summary = model.summary()

    # Extract coefficients and p-values
    coefficients = model.params
    p_values = model.pvalues

    # Combine betas (coefficients) and p-values into a DataFrame
    results = pd.DataFrame({"Beta": coefficients, "P_Value": p_values})

    # Adjust p-values using the Benjamini-Hochberg method
    adjusted_p_values = multipletests(p_values, method="fdr_bh")[1]

    # Add adjusted p-values to the results DataFrame
    results["Adjusted_P_Value"] = adjusted_p_values
    results["gene"] = results.index

    # Display results sorted by p-value
    print("Results sorted by p-value:")
    print(results.sort_values(by="P_Value").head())
    # Filter for significant genes with adjusted p-values < 0.05
    label_data = results[results["Adjusted_P_Value"] < -np.log10(0.05)]

    # Sort by adjusted p-value and print
    label_data = label_data.sort_values(by="Adjusted_P_Value")
    print("Significant Markers")
    print(label_data)


# def plot_interpretablescore_boxplot(data, x, y):
#     """Generate boxplot of interpretable score for a categorical variable (e.g., cell type)

#     Args:
#         data (dataframe): dataframe with interpretable score and other variables of interest to plot
#         x (str): name of the x axis column in data
#         y (str): name of the y axis column in data (should just be interpretable_score if you ran CellPhenoX)
#     """
#     # Activate pandas to R DataFrame conversion
#     pandas2ri.activate()

#     # Define R code
#     r_code = """
#     library(ggplot2)
#     plot_score_boxplot <- function(data, x, y) {
#     p <- ggplot(data, aes_string(x = x, y = y)) +
#         geom_boxplot() +
#         theme_classic()
#     print(p)
#     }
#     """

#     # Evaluate the R code in Python
#     r(r_code)

#     # Get the R function
#     plot_score_boxplot = r['plot_score_boxplot']

#     # Convert pandas DataFrame to R DataFrame
#     r_data = pandas2ri.py2rpy(data)

#     # Call the R function
#     plot_score_boxplot(r_data, 'group', 'score')
    
def plot_interpretablescore_boxplot(data, x, y):
    """Generate boxplot of interpretable score for a categorical variable (e.g., cell type)

    Args:
        data (pd.DataFrame): dataframe with interpretable score and other variables of interest to plot
        x (str): name of x axis column in data
        y (str): name of y axis column in data
    """
    # Use plotnine to generate the plot similar to ggplot
    p = (ggplot(data, aes(x=x, y=y)) +  # Use variables x and y directly, no need for aes_string
        geom_boxplot() +
        theme_classic())
    print(p)
