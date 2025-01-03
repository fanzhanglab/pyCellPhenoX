####################################################
###
###                     IMPORTS
###
####################################################


from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


####################################################
###
###                     FUNCTION
###
####################################################


# TODO: selecting best k may be subjective if the silhouette scores are not that different.... this current implenetation is just selecting k based on the reconstruction error
def select_optimal_k(X, min_k, max_k):
    """Select optimal k (number of components) and generate elbow plot for silhouette score

    Parameters:
        X (dataframe): the marker by cell matrix to be decomposed
        numberOfComponents (int): number of components or ranks to learn (if -1, then we will select k)
        min_k (int): alternatively, provide the minimum number of ranks to test
        max_k (int): and the maximum number of ranks to test

    Returns:
        int: optimal k for decomposition
    """
    print("Determining the optimal k")
    k_values = range(min_k, max_k + 1)
    reconstruction_errors = []
    silhouette_scores = []

    for k in k_values:
        nmfModel = NMF(n_components=k, init="random", random_state=11, max_iter=500)
        transformed = nmfModel.fit_transform(X)
        reconstruction_errors.append(nmfModel.reconstruction_err_)

        kmeans = KMeans(n_clusters=k, n_init="auto", max_iter=500)
        cluster_labels = kmeans.fit_predict(transformed)

        # Calculate silhouette score
        silhouette = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette)

    # Select k based on the smallest reconstruction error
    final_k_index = reconstruction_errors.index(min(reconstruction_errors))
    final_k = k_values[final_k_index]

    print(f"Reconstruction Errors: {reconstruction_errors}")
    print(f"Silhouette Scores: {silhouette_scores}")
    print(f"Optimal k determined to be {final_k} based on reconstruction error.")
    return final_k
