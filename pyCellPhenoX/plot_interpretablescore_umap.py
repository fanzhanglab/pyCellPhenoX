####################################################
###
###                     IMPORTS
###
####################################################

import pandas as pd
from plotnine import *

####################################################
###
###                     FUNCTION
###
####################################################


def plot_interpretablescore_umap(data, x, y, cell_type, score):
    """Generate UMAP of interpretable score and corresponding cell type

    Args:
        data (pd.DataFrame): dataframe with interpretable score and other variables of interest to plot
        x (str): name of x axis column in data
        y (str): name of y axis column in data
        cell_type (str): name of column in data containing the cell type labels
    """
    # Use plotnine to generate the plot similar to ggplot
    c = (
        ggplot(data, aes(x=x, y=y, color=cell_type))
        + geom_point(size=0.5)
        + scale_color_brewer(type="qual", palette="Set3")
        + labs(title="", x=x, y=y, color="Cell Type")
        + theme_classic(base_size=25)
    )

    s = (
        ggplot(data, aes(x=x, y=y, color=score))
        + geom_point(size=0.5)
        + scale_color_continuous(cmap_name="bwr")
        + labs(title="", x=x, y=y, color="CellPhenoX\nInterpretable Score")
        + theme_classic(base_size=25)
    )
    return (c, s)
