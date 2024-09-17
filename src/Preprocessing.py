import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from harmony import harmonize
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, explained_variance_score

def reduceDim(reducMethod, reducMethodParams, expression_mat,):

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
        print("Invalid dimensionality reduction method provided! Please input 'nmf' or 'pca'.")
        #sys.exit()
    return

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
    #print("building NMF model")
    # perform NMF
    nmfModel = NMF(n_components=numberOfComponents, init="random", random_state=11)
    W = nmfModel.fit_transform(X) # ranks by samples
    H = nmfModel.components_
  
    return W


#TODO: selecting best k may be subjective if the silhouette scores are not that different.... this current implenetation is just selecting k based on the reconstruction error 
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
    k_values = range(min_k, max_k+1)
    reconstruction_errors = []
    silhouette_scores = []
    for k in k_values:
        nmfModel = NMF(n_components=k, init="random", random_state=11)
        transformed = nmfModel.fit_transform(X)
        reconstruction_errors.append(nmfModel.reconstruction_err_)
        kmeans = KMeans(n_clusters=k, n_init="auto",max_iter=500)
        cluster_labels = kmeans.fit_predict(transformed)
        # Calculate silhouette score
        silhouette = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette)

    
        #print(f"\n{k} - reconstruction error: {nmfModel.reconstruction_err_}")

    final_k = reconstruction_errors.index(min(reconstruction_errors)) + min_k

    return final_k



def principalComponentAnalysis(X, var):

    """Perform PCA

    Parameters:
        X (dataframe): the marker by cell matrix to be decomposed
        var (float): desired proportion of variance explained 

    Returns:
        dataframe: principal components
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    pca = PCA(n_components=100, random_state=11)
    pca.fit(scaled_data)
    eigenvalues = pca.explained_variance_ratio_ 
    components = pca.components_
    print(f"shape of PCA components: {components.shape}")
    #loadings = pca.components_ * np.sqrt(eigenvalues)

    # find the number of components that explain the most variance
    numberOfComponents = select_number_of_components(eigenvalues, var)
    print(f"optimal num components: {numberOfComponents}")

    return components[:, :numberOfComponents]
    # return (loadings[:, :numberOfComponents], components[:, :numberOfComponents])


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

def preprocessing(latent_features, meta, sub_samp=False, subset_percentage = 0.99, bal_col=['subject_id', 'cell_type','disease'], target="disease", covariates=[]):
    if sub_samp:
        # optionally, sample the data using the balanced sample function
        #subset_percentage = 0.10
        meta = meta.groupby(bal_col, group_keys=False, sort=False).apply(lambda x: balanced_sample(x, subset_percentage=subset_percentage))
        # subset the (expression) data based on the selected rows of the meta data
        latent_features = latent_features.loc[meta.index]
    
    X = pd.DataFrame(latent_features)
    y = meta[target]
    X.set_index(meta.index, inplace=True)
    # code the categorical covariate columns and add them to X
    categoricalColumnNames = meta[covariates].select_dtypes(include=['category', 'object']).columns.values.tolist()
    for column_name in categoricalColumnNames:
        label_encoder = LabelEncoder()
        encoded_column = label_encoder.fit_transform(meta[column_name])
        meta[column_name] = encoded_column
    for covariate in covariates:
        X[covariate] = meta[covariate]
    X = X.rename(str,axis="columns")
    return X, y
def balanced_sample(group, subset_percentage):
    return group.sample(frac=subset_percentage)