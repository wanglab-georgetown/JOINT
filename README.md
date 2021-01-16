## JOINT

JOINT performs probability-based cell-type identification and DEG analysis simultaneously without the need for imputation. It applies an EM algorithm on a generalized zero-inflated negative binomial mixture model. It supports arbitrary numbers of negative binomial components with and without zero inflation.


## Installation

You can install JOINT after downloading this repo by running:

`python setup.py install`


## Quick start

`joint` solves the generalized zero-inflated negative binomial mixture model given counts.

``` r
joint(
     data,                  # input array with row as gene and column as cell
     K,                     # number of cell types
     L = 2,                 # number of negative binomial components + 1
     filter_genes = False,  # whether do EM only on highly variable genes
     n_top_genes=2000,      # number of highly variable genes to keep if filter_genes=True
     impute=True,           # whether return imputation
     n_inits=5,             # number of initializations for the EM algorithm
     n_init_iter=10,        # number of runs of KMeans and Spectral clustering to initialize the EM algorithm
     n_em_iter=100,         # number of iterations to run the EM algorithm
     n_inner_iter=50,       # number of iterations to run the negative binomial inner loop inside the EM
     tol=1e-5,              # stop tolerance in EM algorithm
     zero_inflated=True,    # zero inflated or not
     b_overwrites=[],       # given beta in negative binomial components
     normalize_data=True,   # normalize data by library size
     skip_spectral=True,    # skip spectral clustering for initialization
)              
```

`deg_unknown_labels` DEG on two cell types without assuming true cell types are known.

``` r
deg_unknown_labels(
    data,                  # input array with row as gene and column as cell
    K,                     # number of cell types
    k1,                    # cell type 1 for DEG
    k2,                    # cell type 2 for DEG
    em_res=None,           # EM algorithm result. If None, it will run the EM algorithm first.
    filter_genes = False,  # whether do EM only on highly variable genes
    n_top_genes=2000,      # number of highly variable genes to keep if filter_genes=True
    impute=True,           # whether return imputation
    n_inits=5,             # number of initializations for the EM algorithm
    n_init_iter=10,        # number of runs of KMeans and Spectral clustering to initialize the EM
    n_em_iter=100,         # number of iterations to run the EM algorithm
    n_inner_iter=50,       # number of iterations to run the negative binomial inner loop inside the EM
    tol=1e-5,              # stop tolerance in EM algorithm
    zero_inflated=True,    # zero inflated or not
    b_overwrites=[],       # given beta in negative binomial components
    normalize_data=True,   # normalize data by library size
    skip_spectral=True,    # skip spectral clustering for initialization
)              
```

`deg_known_labels` DEG on two cell types assuming true cell types are known.

``` r
deg_known_labels(
    data,                  # input array with row as gene and column as cell
    K,                     # number of cell types
    k1,                    # cell type 1 for DEG
    k2,                    # cell type 2 for DEG
    labels=None,           # known cell types for each cell
    em_res=None,           # EM algorithm result. If None, it will run the EM algorithm first.
    filter_genes = False,  # whether do EM only on highly variable genes
    n_top_genes=2000,      # number of highly variable genes to keep if filter_genes=True
    impute=True,           # whether return imputation
    n_inits=5,             # number of initializations for the EM algorithm
    n_init_iter=10,        # number of runs of KMeans and Spectral clustering to initialize the EM algorithm
    n_em_iter=100,         # number of iterations to run the EM algorithm
    n_inner_iter=50,       # number of iterations to run the negative binomial inner loop inside the EM
    tol=1e-5,              # stop tolerance in EM algorithm
    zero_inflated=True,    # zero inflated or not
    b_overwrites=[],       # given beta in negative binomial components
    normalize_data=True,   # normalize data by library size
    skip_spectral=True,    # skip spectral clustering for initialization
)              
```


## EM based clustering

A notebook to show how to do EM based clustering can be found at `examples/examples/2celltype_clustering.ipynb`

## JOINT for clustering and DEG

An example code to show how to use JOINT to do soft clustering and DEG can be found at `examples/examples/2celltype_sim.py`

## Reference
Cui, T., Wang, T. JOINT for large-scale single-cell RNA-sequencing analysis via soft-clustering and parallel computing. BMC Genomics 22, 47 (2021). https://doi.org/10.1186/s12864-020-07302-6