import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# marker discovery - find markers correlated with the discriminatory power of the Interpretable Score
##TODO: loosely translated from R to python, not fully tested and missing the final output (maybe some plots and the datataframe containing the coefficients and pvalues?)
def marker_discovery(shap_df, expression_mat):
    """_summary_

    Args:
        shap_df (dataframe): cells by (various columns: meta data, shap values for each latent dimension, interpretable score)
        expression_mat (dataframe): cells by genes/proteins/etc.
    """
    # Define the response variable and predictor variables
    y = shap_df['interpretable_score']
    X = expression_mat

    # Add a constant (intercept term) to the predictor variables for the linear model
    X = sm.add_constant(X)

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

    print(results)

    # Filter for significant genes with adjusted p-values < 0.05
    label_data = results[results['Adjusted_P_Value'] < 0.05]

    # Sort by adjusted p-value and print
    label_data = label_data.sort_values(by='Adjusted_P_Value')
    print(label_data)

