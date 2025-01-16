import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from plotnine import *
import met_brewer
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

    
def plot_interpretablescore_boxplot(data, x, y):
    """Generate boxplot of interpretable score for a categorical variable (e.g., cell type)

    Args:
        data (pd.DataFrame): dataframe with interpretable score and other variables of interest to plot
        x (str): name of x axis column in data
        y (str): name of y axis column in data
    """

    cell_type_colors = {
            "Inflammatory Fibroblasts": "#FBB4AEFF",
            "Myofibroblasts": "#B3CDE3FF",
            "WNT2B": "#CCEBC5FF",
            "WNT5B":"#DECBE4FF"
    }
    b = (ggplot(data, aes(x=x, y=y, color=x)) + 
        geom_boxplot(size=1) +
        #scale_color_brewer(type="qual", palette="Set3") +
        scale_color_manual(values=cell_type_colors) +
        labs(title="", x=x.replace("_", " "), y="CellPhenoX Interpretable Score") +
        theme_classic(base_size=25) #+
        #axis_y_text(theme_elemnt=element_text(size=40))
        )
    print(b)

def plot_interpretablescore_umap(data, x, y, cell_type, score):
    """Generate UMAP of interpretable score and corresponding cell type

    Args:
        data (pd.DataFrame): dataframe with interpretable score and other variables of interest to plot
        x (str): name of x axis column in data
        y (str): name of y axis column in data
        cell_type (str): name of column in data containing the cell type labels
    """

    cell_type_colors = {
        "Inflammatoruy Fibroblasts": "#FBB4AEFF",
        "Myofibroblasts": "#B3CDE3FF",
        "WNT2B": "#CCEBC5FF",
        "WNT5B":"#DECBE4FF"
    }
    c = (
        ggplot(data, aes(x=x,y=y, color=cell_type)) +
        geom_point(size=0.5) +
        #scale_color_brewer(type="qual", palette="Set3") +
        scale_color_manual(values=cell_type_colors) +
        labs(title="", x=x, y=y, color="Cell Type") +
        theme_classic(base_size=25)
    )

    s = (
        ggplot(data, aes(x=x,y=y, color=score)) +
        geom_point(size=0.5) +
        #scale_color_continuous(cmap_name="bwr") +
        scale_color_manual(values=met_brew(name="Egypt", n=123, brew_type="continuous")) +
        labs(title="", x=x, y=y, color="CellPhenoX\nInterpretable Score") +
        theme_classic(base_size=25)
    ) 
    print(c, s)
