import numpy as np
import pandas as pd
import time
import tensorflow.compat.v1 as tf
from .init_clusters import gen_init_centers, get_init_em_params
tf.disable_v2_behavior()


def run_em(X, sf, K, L=2, n_inits=5, n_init_iter=10, n_em_iter=100, \
           n_inner_iter=50, tol=1e-5, zero_inflated=True, b_overwrites=[],\
           skip_spectral=True):
    """
    run EM algorithm on the given dataframe with rows as genes and columns as cells
    """
    st = time.time()

    if L>2 and len(b_overwrites) != L-2:
        print("b_overwrites must be supplied of size L-2")
        return

    # get init params
    G, C = np.shape(X)
    X0 = (X/sf).T
    inits = gen_init_centers(X0, K, n_inits, n_init_iter, skip_spectral)
    if zero_inflated:
        em_inits = get_init_em_params(inits, G, K, L)
    else:
        em_inits = get_init_em_params(inits, G, K, L, q_base=1e-5)

    sol = em(X, sf, em_inits, K, L, n_em_iter,
             n_inner_iter, tol, zero_inflated)

    et = time.time()
    tc = et - st

    sol['total_em_time'] = tc
    sol['sf'] = sf

    return sol


def em(X, sf, inits, K, L, n_iter=100, n_inner_iter=50, tol=1e-5, zero_inflated=True):
    """
    run EM algorithm on the given init centers
    return the clustering labels with the highest log likelihood
    """

    # add prepare reduced data here

    print("start em algorithm")

    res = _em(X, sf, inits, K, L, n_iter, n_inner_iter, tol, zero_inflated)

    max_idx = np.argmax([r['llf'] for r in res])
    sol = res[max_idx]
    em_labels = np.argmax(sol['rho'], axis=1).flatten()

    sol['labels'] = em_labels

    return sol


def _em(X, sf, inits, K, L, n_iter=100, n_inner_iter=50, tol=1e-5, zero_inflated=True):

    G, C = np.shape(X)
    n_factor = np.log(C)

    x0 = X
    pilog0 = inits['pilog0']
    qlog0 = inits['qlog0']
    ss0 = sf

    def tensor_ll_rho():
        vF = a * tf.log(b/sc) - (x+a) * tf.log(1+b/sc) - \
            log_beta(a, x) - tf.log(x+a)
        t = q + tf.concat([vF, tf.broadcast_to(vL_in, [1, K, G, C])], axis=0)
        eta = tf.reduce_logsumexp(t, axis=(0), keepdims=True)
        ga = tf.reduce_sum(eta, axis=(2), keepdims=True)
        tt = pi + ga
        ll = tf.reduce_sum(tf.reduce_logsumexp(tt, axis=1))
        tt_m = tt - tf.reduce_max(tt, axis=1, keepdims=True)
        rho = tt_m - tf.reduce_logsumexp(tt_m, axis=1, keepdims=True)
        xi = rho + t - eta
        p_n = tf.exp(xi - tf.reduce_logsumexp(xi, axis=(0,1), keepdims=True))
        return ll, rho, p_n

    def log_beta(a, x):
        a_full = tf.broadcast_to(a, [L-1, K, G, C])
        x_full = tf.broadcast_to(x, [L-1, K, G, C])
        a_s = tf.reshape(a_full, (-1,))
        x_s = tf.reshape(x_full, (-1,))
        lt = tf.lbeta(tf.stack([x_s+1.0, a_s], axis=1))
        return tf.reshape(lt, (L-1, K, G, C))

    def outer_body(j, diffo, a, b, pi, q):
        def body(i, diff, at, bt):
            bt1 = at * bgt
            at1 = tf.maximum(
                at + (tf.log(bt)+agt-tf.digamma(at))/tf.polygamma(1.0, at), 1e-5)
            diff1 = tf.reduce_mean(tf.abs(at/at1-1.0)) + \
                tf.reduce_mean(tf.abs(bt/bt1-1.0))
            return [i-1, diff1, at1, bt1]

        def cond(i, diff, at, bt):
            return tf.logical_and(i > 0, diff > tol)

        vF = a * tf.log(b/sc) - (x+a) * tf.log(1+b/sc) - \
            log_beta(a, x) - tf.log(x+a)
        t = q + tf.concat([vF, tf.broadcast_to(vL_in, [1, K, G, C])], axis=0)
        eta = tf.reduce_logsumexp(t, axis=(0), keepdims=True)
        ga = tf.reduce_sum(eta, axis=(2), keepdims=True)
        tt = pi + ga
        tt_m = tt - tf.reduce_max(tt, axis=1, keepdims=True)
        rho = tt_m - tf.reduce_logsumexp(tt_m, axis=1, keepdims=True)
        pit = tf.reduce_logsumexp(rho, axis=(3), keepdims=True) - n_factor
        xi = rho + t - eta
        qt = tf.reduce_logsumexp(xi, axis=3, keepdims=True)  # sum over c
        qt1 = qt - tf.reduce_logsumexp(qt, axis=0, keepdims=True)  # sum over l
        xi_n = xi - tf.reduce_max(xi, axis=3, keepdims=True)
        p_n = tf.exp(xi_n[:L-1, :, :, :])
        pt_n = tf.reduce_sum(p_n, axis=3, keepdims=True)
        bgt = pt_n/tf.reduce_sum(p_n * (a+x)/(b+sc), axis=3, keepdims=True)
        agt = tf.reduce_sum(
            p_n * (tf.digamma(x+a)-tf.log(sc+b)), axis=3, keepdims=True) / pt_n

        i, diff_in, at1, bt1 = tf.while_loop(
            cond, body, (n_inner_iter, 1.0, a, b))
        diffo1 = tf.reduce_mean(tf.abs(a/at1-1.0)) + \
            tf.reduce_mean(tf.abs(b/bt1-1.0))
        return [j-1, diffo1, at1, bt1, pit, qt1]

    def outer_cond(j, diffo, a, b, pi, q):
        return tf.logical_and(j > 0, diffo > tol)

    a = tf.placeholder(tf.float32, shape=(L-1, K, G, 1))
    b = tf.placeholder(tf.float32, shape=(L-1, K, G, 1))
    pi = tf.placeholder(tf.float32, shape=(1, K, 1, 1))
    q = tf.placeholder(tf.float32, shape=(L, K, G, 1))
    x = tf.placeholder(tf.float32, shape=(G, C))
    vL_in = tf.placeholder(tf.float32, shape=(G, C))
    ss = tf.placeholder(tf.float32, shape=(C))

    sc = tf.broadcast_to(ss, [1, 1, 1, C])

    j, diff_out, atn, btn, pin, qn = tf.while_loop(
        outer_cond, outer_body, (n_iter, 1.0, a, b, pi, q))
    ll, rho, p_n = tensor_ll_rho()

    res = []
    for kk in range(len(inits['a0s'])):
        print("working on init number {}".format(kk))
        a0 = inits['a0s'][kk]
        b0 = inits['b0s'][kk]
        if zero_inflated:
            vL_in0 = (x0 != 0)*(-1e5)
        else:
            vL_in0 = (x0 != -1)*(-1e5)
        feed_dict = {x: x0, vL_in: vL_in0,
                     a: a0, b: b0, pi: pilog0, q: qlog0, ss: ss0}
        st = time.time()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        oo = sess.run((j, diff_out, atn, btn, pin, qn), feed_dict=feed_dict)
        ll0 = sess.run((ll, rho, p_n), feed_dict={
                       x: x0, vL_in: vL_in0, a: oo[2], b: oo[3], pi: oo[4], q: oo[5], ss: ss0})
        et = time.time()
        sess.close()

        tc = et - st
        res.append({'a': oo[2], 'b': oo[3], 'logpi': oo[4],
                    'logq': oo[5], 'rho': ll0[1], 'p_n': ll0[2], 'llf': ll0[0], 'em_time': tc})

    return res
