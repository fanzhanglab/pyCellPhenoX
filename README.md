# pyCellPhenoX

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
│   ├── MarkerDiscovery.py  # Moved to utils/ by @caterer-z-t
│   └── Preprocessing.py    # Moved functions to utils/ by @caterer-z-t
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
├── utils/                 # Utility scripts and functions
│   ├── __init__.py
│   ├── balanced_sample.py
│   ├── MarkerDiscovery.py
│   ├── nonnegative_matrix_factorization.py
│   ├── preprocessing.py
│   ├── reducedim.py
│   ├── select_num_components.py
│   └── select_optimal_k.py
│
├── __init__.py
├── env.yml                # Environmental configurations
├── mkdocs.yml             # MkDocs configuration
├── README.md 
└── requirements.txt       # Dependencies required for the project
```