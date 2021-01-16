import numpy as np
import pandas as pd
import pickle
from scipy import stats
from .utils import normalize_x_sf
from .joint import joint


def deg_unknown_labels(data, K, k1, k2, em_res=None, L=2, filter_genes=False,
                       n_top_genes=2000, impute=False, n_inits=5, n_init_iter=10,
                       n_em_iter=100, n_inner_iter=50, tol=1e-5, zero_inflated=True,
                       b_overwrites=[], normalize_data=True, skip_spectral=True):

    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data

    # for the case of unknown labels, need to run joint algorithm first and then do DEG
    if em_res is None:
        em_res = joint(df, K, L=L, filter_genes=filter_genes, n_top_genes=n_top_genes, impute=impute,
                       n_inits=n_inits, n_init_iter=n_init_iter, n_em_iter=n_em_iter,
                       n_inner_iter=n_inner_iter, tol=tol, zero_inflated=zero_inflated,
                       b_overwrites=b_overwrites, normalize_data=normalize_data, 
                       skip_spectral=skip_spectral)
        outname = "joint_em_res_unknown_labels.pickle"
        with open(outname, 'wb') as handle:
            pickle.dump(em_res, handle)

    return _deg(df, k1, k2, em_res, normalize_data=normalize_data)


def deg_known_labels(data, K, k1, k2, labels=None, em_res=None, L=2, filter_genes=False,
                     n_top_genes=2000, impute=False, n_inits=5, n_init_iter=10,
                     n_em_iter=100, n_inner_iter=50, tol=1e-5, zero_inflated=True,
                     b_overwrites=[], normalize_data=True, skip_spectral=True):

    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data

    # for the case of unknown labels, need to run joint algorithm first and then do DEG
    if em_res is None and labels is None:
        print("must supply em_res or labels")

    if em_res is None:
        G, C = np.shape(df)
        p_n = np.zeros((L, K, G, C))
        for k in [k1, k2]:
            idx = np.where(labels == k)[0]
            dft = df[df.columns[idx]]

            res = joint(dft, 1, L=L, filter_genes=filter_genes, impute=impute,
                        n_inits=n_inits, n_init_iter=n_init_iter, n_em_iter=n_em_iter,
                        n_inner_iter=n_inner_iter, tol=tol, zero_inflated=zero_inflated,
                        b_overwrites=b_overwrites, normalize_data=normalize_data,
                        skip_spectral=skip_spectral)
            p_n[:, k:k+1, :, idx] = res['p_n']

        em_res = {}
        em_res['p_n'] = p_n
        outname = "joint_em_res_known_labels.pickle"
        with open(outname, 'wb') as handle:
            pickle.dump(em_res, handle)

    return _deg(df, k1, k2, em_res, normalize_data=normalize_data)


def _deg(dfi, k1, k2, em_res, normalize_data=True):

    X = dfi.values
    if normalize_data:
        sf = normalize_x_sf(dfi) + 1e-10
        x0 = X/sf
    else:
        x0 = X

    m1, v1, n1 = get_sample_stat_known(k1, em_res, x0)
    m2, v2, n2 = get_sample_stat_known(k2, em_res, x0)
    vn1 = v1 / n1
    vn2 = v2 / n2
    df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

    df = np.where(np.isnan(df), 1, df)
    denom = np.sqrt(vn1 + vn2)

    d = m1-m2
    t = np.divide(d, denom)
    prob = stats.t.sf(np.abs(t), df) * 2
    padjust = correct_pvalues_for_multiple_testing(prob)

    dfr = pd.DataFrame(index=dfi.index)
    dfr['pvalue'] = prob
    dfr['padj'] = padjust

    return dfr


def get_sample_stat_known(k2, em_res, x0):
    pe = np.sum(em_res['p_n'][:-1, k2, :, :],axis=0)
    n1 = np.sum(pe, axis=1)+1e-5
    me1 = np.sum(pe*x0, axis=1)/n1
    ve1 = np.sum(pe*x0*x0, axis=1)/n1 - me1*me1+1e-5
    return me1, ve1, n1


def correct_pvalues_for_multiple_testing(pvalues, correction_type="Benjamini-Hochberg"):
    """                                                                                                   
    consistent with R
    """
    from numpy import array, empty
    pvalues = array(pvalues)
    n = pvalues.shape[0]
    new_pvalues = np.zeros(n)
    if correction_type == "Bonferroni":
        new_pvalues = n * pvalues
    elif correction_type == "Bonferroni-Holm":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            new_pvalues[i] = (n-rank) * pvalue
    elif correction_type == "Benjamini-Hochberg":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pvalue, index = vals
            new_values.append((n/rank) * pvalue)
        for i in range(0, int(n)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]
    return new_pvalues
