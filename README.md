# pyCellPhenoX


## Running the Script

### Using a Configuration File

1. **Create a Configuration File:**
   - Duplicate the `config.yaml` file in the `src/` directory.
   - Modify `config.yaml` based on your desired configurations.

2. **Run the Script:**
   ```bash
   python /path/to/CellPhenoX_script.py --config-file /path/to/config.yaml
   ```      

### Using Command-Line Arguments
You can also run the script directly from the command line by specifying the arguments:

    
    python /path/to/CellPhenoX_script.py \
        --expression-file path/to/expression_file.csv \
        --meta-file path/to/meta_file.csv \
        --output-path path/to/output_directory \
        --reduction-method nmf \
        --proportion-var-explained 0.95 \
        --number-ranks 10 \
        --minimum-k 2 \
        --maximum-k 7 \
        --covariates covariate1 covariate2 \
        --target-variable disease \
        --classification-method rf \
        --sub-sample \
        --subset-percentage 0.25 \
        --num-cv-repeats 3 \
        --num-outer-splits 3 \
        --num-inner-splits 5
    
### Arguments Description
- `--expression-file`: Path to the expression data file (CSV).
- `--meta-file`: Path to the meta data file (CSV).
- `--output-path`: Directory where the output will be saved.
- `--reduction-method`: Dimensionality reduction method (nmf or pca).
- `--proportion-var-explained`: Proportion of variance explained for PCA.
- `--number-ranks`: Number of ranks for NMF.
- `--minimum-k`: Minimum k value for selecting the optimal number of ranks if NMF is used.
- `--maximum-k`: Maximum k value for selecting the optimal number of ranks if NMF is used.
- `--covariates`: List of covariates for the classification model.
- `--target-variable`: Target/outcome column name in the meta data.
- `--classification-method`: Classification method (rf for Random Forest or xgb for XGBoost).
- `--sub-sample`: Flag to indicate if a subset of the data should be used.
- `--subset-percentage`: Portion of data to use if sub-sampling is enabled.
- `--num-cv-repeats`: Number of cross-validation repeats.
- `--num-outer-splits`: Number of outer loop splits for cross-validation.
- `--num-inner-splits`: Number of inner loop splits for hyperparameter tuning.

Feel free to replace `/path/to/` placeholders with actual paths and adjust any details as needed.

### Project Directory

```bash
pyCellPhenoX/
│
├── docs/                  # Documentation files for ReadTheDocs
│   ├── index.md           # Initial markdown file used with the template
│   └── requirements.txt   # Dependencies required for ReadTheDocs
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── CellPhenoX_script.py
│   ├── CellphenoX.py
│   ├── config.yaml        # Example yaml configuration file for running CellPhenoX
│   ├── MarkerDiscovery.py
│   ├── nonnegative_matrix_factorization.py
│   ├── preprocessing.py
│   ├── principle_component_analysis.py
│   ├── workflow.ipynb
│   │
│   ├── utils/             # Utility scripts and functions
│   │   ├── __init__.py
│   │   ├── balanced_sample.py
│   │   ├── reducedim.py
│   │   ├── select_num_components.py
│   │   └── select_optimal_k.py
│
├── test/                  # Test scripts
│   ├── __init__.py
│   ├── test_balanced_sample.py
│   ├── test_nonnegative_matrix_factorization.py
│   ├── test_pca.py
│   ├── test_preprocessing.py
│   ├── test_select_k.py
│   └── test_select_num_components.py
│
├── __init__.py
├── .gitignore             # Ignored files and directories
├── .readthedocs.yaml      # Readthedocs configuration file
├── env.yml                # Environmental configurations
├── mkdocs.yml             # MkDocs configuration
├── README.md 
└── requirements.txt       # Dependencies required for the project
```