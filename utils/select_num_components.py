####################################################
###
###                     IMPORTs
###
####################################################


import numpy as np


####################################################
###
###                     FUNCTION
###
####################################################


def select_number_of_components(eigenvalues, var):
    """Find the number of the components based on the percentage of accumulated variance

    Parameters:
        eigenvalues (array): array of eigenvalues (explained variances) for the components
        var (float): desired proportion of variance explained

    Returns:
        int: number of components
    """
    print("getting number of components")
    explained_variances = eigenvalues
    cumulative_sum = np.cumsum(explained_variances)

    num_selected_components = np.argmax(cumulative_sum >= var) + 1

    return num_selected_components
