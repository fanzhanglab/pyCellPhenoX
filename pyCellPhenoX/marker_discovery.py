####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

####################################################
###
###                     FUNCTION
###
####################################################


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
    # model_summary = model.summary()

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
