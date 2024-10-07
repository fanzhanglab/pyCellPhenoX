####################################################
###
###                     IMPORTS
###
####################################################


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyCellPhenoX.utils.select_num_components import select_number_of_components


####################################################
###
###                     FUNCTION
###
####################################################


def principalComponentAnalysis(X, var):
    """Perform PCA

    Parameters:
        X (dataframe): the marker by cell matrix to be decomposed
        var (float): desired proportion of variance explained

    Returns:
        dataframe: principal components
    """

    if X.empty:  # Check if the input dataframe is empty
        raise ValueError("Input dataframe is empty")

    if not (0 < var <= 1):  # Check if the variance threshold is valid
        raise ValueError("Variance threshold must be between 0 and 1")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    # pca = PCA(n_components=100, random_state=11)
    pca = PCA(n_components=min(X.shape[0], X.shape[1]), random_state=11)
    # this is a trick to prevent PCA from throwing an error when the number of samples is less than the number of features

    pca.fit(scaled_data)
    eigenvalues = pca.explained_variance_ratio_
    components = pca.components_
    print(f"shape of PCA components: {components.shape}")
    # loadings = pca.components_ * np.sqrt(eigenvalues)

    # find the number of components that explain the most variance
    numberOfComponents = select_number_of_components(eigenvalues, var)
    print(f"optimal num components: {numberOfComponents}")

    return components[:, :numberOfComponents]
    # return (loadings[:, :numberOfComponents], components[:, :numberOfComponents])
