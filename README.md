
<p>
   <img height="270" align="centre" src="https://github.com/fanzhanglab/pyCellPhenoX/blob/main/logo/cellphenoX_logo_banner.png">
</p>

![PyPI](https://img.shields.io/pypi/v/pyCellPhenoX.svg)
![Python Version](https://img.shields.io/pypi/pyversions/pyCellPhenoX)
[![License](https://img.shields.io/pypi/l/pyCellPhenoX)][license] 
![Read the documentation at https://pyXcell.readthedocs.io/](https://img.shields.io/readthedocs/pyXcell/latest.svg?label=Read%20the%20Docs)
![Codecov](https://codecov.io/gh/fanzhanglab/pyCellPhenoX/branch/main/graph/badge.svg)

![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)


Here, we introduce CellPhenoX, an eXplainable machine learning method to identify cell-specific phenotypes that influence clinical outcomes for single-cell data. CellPhenoX integrates robust classification models, explainable AI techniques, and a statistical covariate framework to generate interpretable, cell-specific scores that uncover cell populations associated with a clinical phenotype of interest.

<img width="100%" align="center" src="https://github.com/fanzhanglab/pyCellPhenoX/blob/163014ba1b31243a1aac4c27e0611a34c1495efe/media/CellPhenoX.png?raw=true">

> Figure 1. CellPhenoX leverages cell neighborhood co-abundance embeddings, Xi , across samples and clinical variable Y as inputs. By applying an adapted SHAP framework for classification models, CellPhenoX generates Interpretable Scores that quantify the contribution of each feature Xi, along with covariates  and interaction term Xi, to the prediction of a clinically relevant phenotype Y. The results are visualized at single-cell level, showcasing Interpretable Scores at low-dimensional space, correlated cell type annotations, and associated marker genes.

## Installation
You can install _pyCellPhenoX_ from PyPI:

``` bash
pip install pyCellPhenoX
```

<!-- **conda** ([link](https://anaconda.org/conda-forge/pyCellPhenoX)):
``` bash 
# install pyCellPhenoX from conda-forge
conda install -c conda-forge pyCellPhenoX
``` -->
**github** ([link](https://github.com/fanzhanglab/pyCellPhenoX)):
``` bash
# install pyCellPhenoX directly from github
git clone git@github.com:fanzhanglab/pyCellPhenoX.git
```

### Dependencies/ Requirements
When using pyCellPhenoX please ensure you are using the following dependency versions or requirements
``` python 
python = "^3.9"
pandas = "^2.2.3"
numpy = "^2.1.1"
xgboost = "^2.0"
numba = ">=0.54"
shap = "^0.46.0"
scikit-learn = "^1.5.2"
matplotlib = "^3.9.2"
statsmodels = "^0.14.3"
```


## Tutorials
Please see the [Command-line Reference] for details. Additonally, please see [Vignettes] on the documentation page. 

## API
pyCellPhenoX has four major functions which are apart of the object:
1. split_data() - Split the data into training, testing, and validation sets 
2. model_train_shap_values() - Train the model using nested cross validation strategy and generate shap values for each fold/CV repeat
3. get_shap_values() - Aggregate SHAP values for each sample
4. get_intepretable_score() - Calculate the interpretable score based on SHAP values. 

Additional major functions associated with pyCellPhenoX are:
1. marker_discovery() - Identify markers correlated with the discriminatory power of the Interpretable Score.
2. nonNegativeMatrixFactorization() - Perform non Negative Matrix Factorization (NMF)
3. preprocessing() - Prepare the data to be in the correct format for CellPhenoX
4. principleComponentAnalysis() - Perform Principle Component Analysis (PCA)

Each function has uniqure arguments, see our [documentation] for more information


## License
Distributed under the terms of the [MIT license][license],
_pyCellPhenoX_ is free and open source software.

## Code of Conduct
For more information please see [Code of Conduct](CODE_OF_CONDUCT.md) or [Code of Conduct Documentation]

## Contributing
For more information please see [Contributing](CONTRIBUTING.md) or [Contributing Documentation]

## Issues
If you encounter any problems, please [file an issue] along with a detailed description. 

## Citation
If you have used `pyCellPhenoX` in your project, please use the citation below: 
</br>

 Young, J., Inamo, J., Caterer, Z., Krishna, R., Zhang, F. CellPhenoX: An eXplainable Cell-specific machine learning method to predict clinical Phenotypes using single-cell multi-omics, 2024.

## Contact
Please contact [fanzhanglab@gmail.com](fanzhanglab@gmail.com) for
further questions or protential collaborative opportunities!

<!-- github-only -->

[license]: https://github.com/fanzhanglab/pyCellPhenoX/blob/main/LICENSE
[contributor guide]: https://github.com/fanzhanglab/pyCellPhenoX/blob/main/CONTRIBUTING.md
[file an issue]: https://github.com/fanzhanglab/pyCellPhenoX/issues/new
[command-line reference]: https://pyCellPhenoX.readthedocs.io/en/latest/usage.html
[pipi]: https://pypi.org/project/pip/
[pypi]: https://pypi.org/project/pyCellPhenoX/
[vignettes]: https://pyCellPhenoXreadthedocs.io/walkthroughs/single_cell_usage
[documentation]: https://pyCellPhenoXreadthedocs.io/
[Code of Conduct Documentation]: https://pyCellPhenoXreadthedocs.io/code_of_conduct
[Contributing Documentation]: https://pyCellPhenoXreadthedocs.io/contributing