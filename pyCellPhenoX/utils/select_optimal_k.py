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

    print("determining the optimal k")
    k_values = range(min_k, max_k + 1)
    reconstruction_errors = []
    silhouette_scores = []
    for k in k_values:
        nmfModel = NMF(n_components=k, init="random", random_state=11)
        transformed = nmfModel.fit_transform(X)
        reconstruction_errors.append(nmfModel.reconstruction_err_)
        kmeans = KMeans(n_clusters=k, n_init="auto", max_iter=500)
        cluster_labels = kmeans.fit_predict(transformed)
        # Calculate silhouette score
        silhouette = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette)

        # print(f"\n{k} - reconstruction error: {nmfModel.reconstruction_err_}")

    final_k = silhouette_scores.index(min(reconstruction_errors)) + min_k

    return final_k