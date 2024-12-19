# pyCellPhenoX utilities

Functions enabling smooth interaction with CellProfiler and DeepProfiler output formats.

## pyCellPhenoX.utils.balanced_sample module

### pyCellPhenoX.utils.balanced_sample.balanced_sample(group, subset_percentage)

Perform balanced sampling on a DataFrame group.

* **Parameters:**
  * **group** (*DataFrame*) – The DataFrame or group to sample from.
  * **subset_percentage** (*float*) – The fraction of the group to sample (between 0.0 and 1.0).
* **Returns:**
  A randomly sampled fraction of the group, based on the given percentage.
* **Return type:**
  DataFrame

## pyCellPhenoX.utils.reducedim module

### pyCellPhenoX.utils.reducedim.reduceDim(reducMethod, reducMethodParams, expression_mat)

Call the reduction method specified by user

* **Parameters:**
  * **reducMethod** (*str*) – the name of the method to be used (“nmf” or “pca”)
  * **reducMethodParams** (*dict*) – parameters for the method selected
* **Returns:**
  one matrix if PCA selected, tuple of matrices if NMF selected
* **Return type:**
  matrix/matrices

## pyCellPhenoX.utils.select_num_components module

### pyCellPhenoX.utils.select_num_components.select_number_of_components(eigenvalues, var)

Find the number of the components based on the percentage of accumulated variance

* **Parameters:**
  * **eigenvalues** (*array*) – array of eigenvalues (explained variances) for the components
  * **var** (*float*) – desired proportion of variance explained
* **Returns:**
  number of components
* **Return type:**
  int

## pyCellPhenoX.utils.select_optimal_k module

### pyCellPhenoX.utils.select_optimal_k.select_optimal_k(X, min_k, max_k)

Select optimal k (number of components) and generate elbow plot for silhouette score

* **Parameters:**
  * **X** (*dataframe*) – the marker by cell matrix to be decomposed
  * **numberOfComponents** (*int*) – number of components or ranks to learn (if -1, then we will select k)
  * **min_k** (*int*) – alternatively, provide the minimum number of ranks to test
  * **max_k** (*int*) – and the maximum number of ranks to test
* **Returns:**
  optimal k for decomposition
* **Return type:**
  int

## Module contents
