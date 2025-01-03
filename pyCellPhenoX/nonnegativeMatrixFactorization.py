####################################################
###
###                     IMPORTS
###
####################################################


from sklearn.decomposition import NMF
from pyCellPhenoX.utils.select_optimal_k import select_optimal_k


####################################################
###
###                     FUNCTION
###
####################################################


def nonnegativeMatrixFactorization(X, numberOfComponents=-1, min_k=2, max_k=12):
    """Perform NMF

    Parameters:
        X (dataframe): the marker by cell matrix to be decomposed
        numberOfComponents (int): number of components or ranks to learn (if -1, then we will select k)
        min_k (int): alternatively, provide the minimum number of ranks to test
        max_k (int): and the maximum number of ranks to test
    Returns:
        tuple: W and H matrices
    """

    print("inside the NMF function")
    # check if the user has provided the number of components they would like
    if numberOfComponents == -1:
        # call function to select optimal k
        numberOfComponents = select_optimal_k(X, min_k, max_k)
    # print("building NMF model")
    # perform NMF
    nmfModel = NMF(n_components=numberOfComponents, init="random", random_state=11)
    W = nmfModel.fit_transform(X)  # ranks by samples
    # H = nmfModel.components_

    return W
