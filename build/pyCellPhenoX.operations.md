# Functions

## pyCellPhenoX.marker_discovery module

### pyCellPhenoX.marker_discovery.marker_discovery(shap_df, expression_mat)

\_summary_

* **Parameters:**
  * **shap_df** (*dataframe*) – cells by (various columns: meta data, shap values for each latent dimension, interpretable score)
  * **expression_mat** (*dataframe*) – cells by genes/proteins/etc.

## pyCellPhenoX.nonnegativeMatrixFactorization module

### pyCellPhenoX.nonnegativeMatrixFactorization.nonnegativeMatrixFactorization(X, numberOfComponents=-1, min_k=2, max_k=12)

Perform NMF

* **Parameters:**
  * **X** (*dataframe*) – the marker by cell matrix to be decomposed
  * **numberOfComponents** (*int*) – number of components or ranks to learn (if -1, then we will select k)
  * **min_k** (*int*) – alternatively, provide the minimum number of ranks to test
  * **max_k** (*int*) – and the maximum number of ranks to test
* **Returns:**
  W and H matrices
* **Return type:**
  tuple

## pyCellPhenoX.preprocessing module

### pyCellPhenoX.preprocessing.preprocessing(latent_features, meta, sub_samp=False, subset_percentage=0.99, bal_col=['subject_id', 'cell_type', 'disease'], target='disease', covariates=[])

Prepare the data to be in the correct format for CellPhenoX

* **Parameters:**
  * **latent_features** ( *\_type_*) – \_description_
  * **meta** ( *\_type_*) – \_description_
  * **sub_samp** (*bool* *,* *optional*) – \_description_. Defaults to False.
  * **subset_percentage** (*float* *,* *optional*) – \_description_. Defaults to 0.99.
  * **bal_col** (*list* *,* *optional*) – \_description_. Defaults to [“subject_id”, “cell_type”, “disease”].
  * **target** (*str* *,* *optional*) – \_description_. Defaults to “disease”.
  * **covariates** (*list* *,* *optional*) – \_description_. Defaults to [].
* **Returns:**
  \_description_
* **Return type:**
  \_type_

## pyCellPhenoX.principalComponentAnalysis module

### pyCellPhenoX.principalComponentAnalysis.principalComponentAnalysis(X, var)

Perform PCA

* **Parameters:**
  * **X** (*dataframe*) – the marker by cell matrix to be decomposed
  * **var** (*float*) – desired proportion of variance explained
* **Returns:**
  principal components
* **Return type:**
  dataframe

## Module contents
