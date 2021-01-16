import numpy as np

from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def get_init_em_params(m_inits, G, K, L, a_base=1.0, q_base=0.2, b_overwrites=[]):
    """get inits em parameters from init centers"""

    a0s = [np.zeros((L-1, K, G, 1))+a_base for m in m_inits]
    b0s = [q_base/np.broadcast_to(np.reshape(m, (K, G, 1)),
                                  (L-1, K, G, 1))*a_base for m in m_inits]
    if len(b_overwrites) > 0:
        t = []
        for b0 in b0s:
            for i in range(1,L-1):
                b0[i,:,:,:]= b_overwrites[i-1]
            t.append(b0)
        b0s = t

    pi0 = np.ones((1, K, 1, 1))*1.0/K
    q0 = np.ones((L, K, G, 1)) * (1.0-q_base)/(L-1)
    q0[0, :, :, :] = q_base
    qlog0 = np.log(q0)
    pilog0 = np.log(pi0)

    res = {'a0s': a0s, 'b0s': b0s, 'pilog0': pilog0, 'qlog0': qlog0}

    return res


def gen_init_centers(X0, K, n_inits=5, n_iter=10, skip_spectral=True):
    """get label inits using both clustering and random methods"""

    Xlog = np.log2(X0+1.0)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(Xlog)

    print("generate center inits")

    # use normalized X to get inits
    cand_labels = gen_inits_clustering(Xn, Xlog, K, n_inits, n_iter, skip_spectral)
    # use normalized X to get inits
    cand_labels = cand_labels + \
        gen_inits_clustering(Xlog, Xlog, K, n_inits, n_iter, skip_spectral)
    # use randomized method to get inits
    cand_labels = cand_labels + gen_inits_rand(Xlog, K, n_inits)

    # remove labels that are identical
    cand_labels = dedup_labels(cand_labels)

    m_inits = [get_inits_from_labels(X0, K, labels) for labels in cand_labels]

    return m_inits


def gen_inits_clustering(X, Xlog, K, n_inits=5, n_iter=10, skip_spectral=True):
    """get label inits using clustering method"""

    lbs_k = _gen_inits_clustering(X, K, n_iter, skip_spectral)
    cc = []
    for labels in lbs_k:
        cc.append(compute_center_cost(labels, Xlog))

    idx = np.argsort(cc)[:n_inits]
    return [lbs_k[i] for i in idx]


def _gen_inits_clustering(X, K, n_iter=10, skip_spectral=True):

    Xs = [X]

    Xn = X.copy().astype('float')
    Xn[Xn == 0] = np.nan

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Xs.append(imp.fit_transform(Xn))

    imputer = KNNImputer()
    Xs.append(imputer.fit_transform(Xn))

    lb_inits = []

    for Xt in Xs:
        li = gen_inits_for_X(Xt, K, n_iter, skip_spectral)
        lb_inits = lb_inits + li

    return dedup_labels(lb_inits)


def gen_inits_for_X(X, K, n_iter=10, skip_spectral=True):
    pca = PCA()
    pcs = pca.fit_transform(X)

    sv = np.cumsum(pca.explained_variance_ratio_)
    cs = [-1, 2]
    ps = [0.25, 0.4]
    for p in ps:
        min_pcs = max(np.min(np.where(sv > p)[0]), 2)
        cs.append(min_pcs)
    cs = np.unique(cs)

    # kmeans clustering
    lls = []
    for c in cs:
        if c == -1:
            kmeans = kmeans_fit(X, K, n_iter)
        else:
            kmeans = kmeans_fit(pcs[:, :c], K, n_iter)
        lls.append(kmeans.labels_)

    if not skip_spectral:
        # spectral clustering
        for c in cs:
            if c == -1:
                labels_ = spectral_fit(X, K)
            else:
                labels_ = spectral_fit(pcs[:, :c], K)
            lls.append(labels_)

    return lls


def gen_inits_rand(X, K, num_rand=5):
    """get label inits using random method"""

    lb_inits = []
    for i in range(num_rand):
        lb_inits.append(gen_inits_random(X, K))
        lb_inits.append(gen_inits_random_v1(X, K))
    return dedup_labels(lb_inits)


def gen_inits_random(X, K):
    C = np.shape(X)[0]
    labels = np.ones(C) * -1
    centers = np.random.choice(range(C), size=K, replace=False)
    for k in range(K):
        labels[centers[k]] = k
    return labels


def gen_inits_random_v1(X, K):
    return np.random.randint(K, size=len(X))


def kmeans_fit(x, K, n_iter=10):
    min_cost = np.inf
    min_kmeans = None
    for ii in range(n_iter):
        kmeans = KMeans(n_clusters=K).fit(x)
        if kmeans.inertia_ < min_cost:
            min_cost = kmeans.inertia_
            min_kmeans = kmeans
    return min_kmeans


def spectral_fit(X, K):
    clustering = SpectralClustering(
        n_clusters=K, affinity='nearest_neighbors', random_state=0).fit(X)
    return clustering.labels_


def dedup_labels(all_labels):
    deduped = []
    N = len(all_labels)
    for i in range(N):
        flag = False
        for labels_ in deduped:
            if adjusted_rand_score(all_labels[i], labels_) > 1-1e-3:
                flag = True
                break
        if not flag:
            deduped.append(all_labels[i])
    return deduped


def get_inits_from_labels(X, K, labels):
    G = np.shape(X)[1]
    lb = np.maximum(np.mean(X, axis=0)/K, 1e-5)
    a0 = np.zeros((K, G))
    for k in range(K):
        idx = np.where(labels == k)[0]
        a0[k, :] = np.maximum(np.mean(X[idx, :], axis=0), lb)
    return a0


def compute_center_cost(labels, X):
    cc = 0
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        m = np.mean(X[idx, :], axis=0, keepdims=True)
        diff = X[idx, :]-m
        cc = cc + np.sum(diff * diff)
    return cc
