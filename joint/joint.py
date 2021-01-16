import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from .em import run_em
from .impute import em_impute
from .utils import normalize_x_sf


def joint(data, K, L=2, filter_genes=False, n_top_genes=2000, impute=True, \
          n_inits=5, n_init_iter=10, n_em_iter=100, n_inner_iter=50, tol=1e-5, \
          zero_inflated=True, b_overwrites=[], normalize_data=True, skip_spectral=True):
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data

    if filter_genes:
        return _joint_filtered(df, K, L=L, n_top_genes=n_top_genes, impute=impute, \
                               n_inits=n_inits, n_init_iter=n_init_iter, n_em_iter=n_em_iter, \
                               n_inner_iter=n_inner_iter, tol=tol, zero_inflated=zero_inflated,\
                               b_overwrites=b_overwrites,normalize_data=normalize_data,\
                               skip_spectral=skip_spectral)
    else:
        return _joint_full(df, K, L=L, n_inits=n_inits, impute=impute, \
                           n_init_iter=n_init_iter, n_em_iter=n_em_iter, n_inner_iter=n_inner_iter, \
                           tol=tol, zero_inflated=zero_inflated, b_overwrites=b_overwrites,\
                           normalize_data=normalize_data,skip_spectral=skip_spectral)


def _joint_filtered(df, K, L=2, n_top_genes=2000, impute=True, n_inits=5, \
                    n_init_iter=10, n_em_iter=100, n_inner_iter=50, tol=1e-5, \
                    zero_inflated=True, b_overwrites=[], normalize_data=True,\
                    skip_spectral=True):

    # select high variable genes for EM
    var = df.index.values
    obs = np.array(list(df.columns))
    adata = anndata.AnnData(X=df.T.values, obs=pd.DataFrame(
        index=obs), var=pd.DataFrame(index=var))
    adata.var_names_make_unique()
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    X = df.loc[adata.var['highly_variable']
               [adata.var['highly_variable']].index].values

    if normalize_data:
        sf = normalize_x_sf(df)
    else:
        sf = np.ones(np.shape(X)[1])

    sol = run_em(X, sf, K, L, n_inits, n_init_iter, n_em_iter, \
                 n_inner_iter, tol, zero_inflated, b_overwrites, skip_spectral)

    if not impute or not zero_inflated or L>2:
        return sol

    print("start imputation")

    # first impute highly_variable genes
    # do not need em iterations
    sol_impute0 = em_impute(X, sf, K, L, em_inits=sol,
                            n_iter=1, n_inner_iter=0)

    # then impute other genes
    em_inits = {}
    em_inits['labels'] = sol['labels']
    em_inits['rho'] = sol['rho']
    X_other = df.loc[adata.var['highly_variable']
                     [~adata.var['highly_variable']].index].values
    sol_impute1 = em_impute(X_other, sf, K, L, em_inits=em_inits)

    cols = ['rate_impute', 'point_impute', 'var_est']
    impute_res = {}

    impute_res['total_impute_time'] = sol_impute0['total_impute_time'] + \
        sol_impute1['total_impute_time']

    for key in cols:
        dfr = df.copy()
        dfr.loc[adata.var['highly_variable']
                [adata.var['highly_variable']].index] = sol_impute0[key]
        dfr.loc[adata.var['highly_variable']
                [~adata.var['highly_variable']].index] = sol_impute1[key]
        impute_res[key] = dfr
    sol['impute'] = impute_res

    return sol


def _joint_full(df, K, L=2, impute=True, n_inits=5, n_init_iter=10, \
                n_em_iter=100, n_inner_iter=50, tol=1e-5, zero_inflated=True, \
                b_overwrites=[], normalize_data=True, skip_spectral=True):

    X = df.values
    if normalize_data:
        sf = normalize_x_sf(df)
    else:
        sf = np.ones(np.shape(X)[1])

    sol = run_em(X, sf, K, L, n_inits, n_init_iter, n_em_iter, \
                 n_inner_iter, tol, zero_inflated, b_overwrites,skip_spectral)

    if not impute or not zero_inflated or L>2:
        return sol

    print("start imputation")

    # impute genes
    # do not need em iterations
    sol_impute0 = em_impute(X, sf, K, L, em_inits=sol, \
                            n_iter=1, n_inner_iter=0)

    cols = ['rate_impute', 'point_impute', 'var_est']
    impute_res = {}

    impute_res['total_impute_time'] = sol_impute0['total_impute_time']

    for key in cols:
        dfr = df.copy()
        dfr.loc[:,:] = sol_impute0[key]
        impute_res[key] = dfr
    sol['impute'] = impute_res

    return sol
