####################################################
###
###                     IMPORTS
###
####################################################


from pyCellPhenoX.principalComponentAnalysis import principalComponentAnalysis
from pyCellPhenoX.nonnegativeMatrixFactorization import (
    nonnegativeMatrixFactorization,
)


####################################################
###
###                     FUNCTION
###
####################################################


def reduceDim(
    reducMethod,
    reducMethodParams,
    expression_mat,
):
    """Call the reduction method specified by user

    Parameters:
        reducMethod (str): the name of the method to be used ("nmf" or "pca")
        reducMethodParams (dict): parameters for the method selected

    Returns:
        matrix/matrices: one matrix if PCA selected, tuple of matrices if NMF selected
    """

    if reducMethod == "nmf":
        return nonnegativeMatrixFactorization(expression_mat, **reducMethodParams)
    elif reducMethod == "pca":
        return principalComponentAnalysis(expression_mat, **reducMethodParams)
    else:
        print(
            "Invalid dimensionality reduction method provided! Please input 'nmf' or 'pca'."
        )
        # sys.exit()
    return
