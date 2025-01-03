####################################################
###
###                     IMPORTS
###
####################################################

from plotnine import *

####################################################
###
###                     FUNCTION
###
####################################################


def plot_interpretablescore_boxplot(data, x, y):
    """Generate boxplot of interpretable score for a categorical variable (e.g., cell type)

    Args:
        data (pd.DataFrame): dataframe with interpretable score and other variables of interest to plot
        x (str): name of x axis column in data
        y (str): name of y axis column in data
    """
    # Use plotnine to generate the plot similar to ggplot
    b = (
        ggplot(data, aes(x=x, y=y, color=y))
        + geom_boxplot(size=2)
        + scale_color_brewer(type="qual", palette="Set3")
        + labs(title="", x=x.replace("_", " "), y="CellPhenoX Interpretable Score")
        + theme_classic(base_size=25)  # +
        # axis_y_text(theme_elemnt=element_text(size=40))
    )
    return b
