# Core functions

## pyCellPhenoX object

### *class* pyCellPhenoX.CellPhenoX(X, y, CV_repeats, outer_num_splits, inner_num_splits)

Bases: `object`

#### get_best_model()

#### get_best_score()

#### get_interpretable_score()

#### get_shap_values(outpath)

#### get_shap_values_per_cv()

#### model_training_shap_val(outpath)

Train the model using nested cross validation strategy and generate shap values for each fold/CV repeat

Parameters:
outpath (str): the path for the output folder

Returns:

#### split_data(train_outer_ix, test_outer_ix)

## Helper functions

* [Functions](pyCellPhenoX.operations.md)
  * [pyCellPhenoX.marker_discovery module](pyCellPhenoX.operations.md#module-pyCellPhenoX.marker_discovery)
    * [`marker_discovery()`](pyCellPhenoX.operations.md#pyCellPhenoX.marker_discovery.marker_discovery)
  * [pyCellPhenoX.nonnegativeMatrixFactorization module](pyCellPhenoX.operations.md#module-pyCellPhenoX.nonnegativeMatrixFactorization)
    * [`nonnegativeMatrixFactorization()`](pyCellPhenoX.operations.md#pyCellPhenoX.nonnegativeMatrixFactorization.nonnegativeMatrixFactorization)
  * [pyCellPhenoX.preprocessing module](pyCellPhenoX.operations.md#module-pyCellPhenoX.preprocessing)
    * [`preprocessing()`](pyCellPhenoX.operations.md#pyCellPhenoX.preprocessing.preprocessing)
  * [pyCellPhenoX.principalComponentAnalysis module](pyCellPhenoX.operations.md#module-pyCellPhenoX.principalComponentAnalysis)
    * [`principalComponentAnalysis()`](pyCellPhenoX.operations.md#pyCellPhenoX.principalComponentAnalysis.principalComponentAnalysis)
  * [Module contents](pyCellPhenoX.operations.md#module-pyCellPhenoX)
* [pyCellPhenoX utilities](pyCellPhenoX.utils.md)
  * [pyCellPhenoX.utils.balanced_sample module](pyCellPhenoX.utils.md#module-pyCellPhenoX.utils.balanced_sample)
    * [`balanced_sample()`](pyCellPhenoX.utils.md#pyCellPhenoX.utils.balanced_sample.balanced_sample)
  * [pyCellPhenoX.utils.reducedim module](pyCellPhenoX.utils.md#module-pyCellPhenoX.utils.reducedim)
    * [`reduceDim()`](pyCellPhenoX.utils.md#pyCellPhenoX.utils.reducedim.reduceDim)
  * [pyCellPhenoX.utils.select_num_components module](pyCellPhenoX.utils.md#module-pyCellPhenoX.utils.select_num_components)
    * [`select_number_of_components()`](pyCellPhenoX.utils.md#pyCellPhenoX.utils.select_num_components.select_number_of_components)
  * [pyCellPhenoX.utils.select_optimal_k module](pyCellPhenoX.utils.md#module-pyCellPhenoX.utils.select_optimal_k)
    * [`select_optimal_k()`](pyCellPhenoX.utils.md#pyCellPhenoX.utils.select_optimal_k.select_optimal_k)
  * [Module contents](pyCellPhenoX.utils.md#module-contents)

## Module contents
