import numpy as np
import pandas as pd
import time
import functools
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def em_impute(X, sf, K, L, em_inits={}, min_lb=1e-5, a_base=1.0, q_base=0.2,
              n_iter=100, n_inner_iter=50, tol=1e-5):

    st = time.time()

    G, C = np.shape(X)
    X0 = (X/sf).T
    lb = np.maximum(np.mean(X0, axis=0)/K, min_lb)

    rhot = [em_inits['rho'][:, k:k+1, :, :] for k in range(K)]
    at = []
    bt = []
    qt = []

    for k in range(K):
        if 'a' in em_inits:
            a0 = em_inits['a'][:, k:k+1, :, :]
            b0 = em_inits['b'][:, k:k+1, :, :]
            qlog0 = em_inits['logq'][:, k:k+1, :, :]
        else:
            em_labels = em_inits['labels']
            jdx = np.where(em_labels == k)[0]
            mk = np.maximum(np.mean(X0[jdx, :], axis=0), lb)
            a0 = np.zeros((L-1, 1, G, 1))+a_base
            b0 = np.zeros((L-1, 1, G, 1))
            b0[0, 0, :, 0] = q_base/mk*a_base
            q0 = np.ones((L, 1, G, 1)) * (1.0-q_base)/(L-1)
            q0[0, :, :, :] = q_base
            qlog0 = np.log(q0)
        at.append(a0)
        bt.append(b0)
        qt.append(qlog0)

    res = _em_impute(X, sf, L, at, bt, qt, rhot, n_iter=n_iter,
                     n_inner_iter=n_inner_iter, tol=tol)

    sol = {}
    for key in ['rate_impute', 'point_impute', 'var_est']:
        sol[key] = functools.reduce(lambda a, b: a+b, [r[key] for r in res])

    et = time.time()
    tc = et - st

    sol['total_impute_time'] = tc
    sol['res'] = res
    return sol


def _em_impute(X, sf, L, at, bt, qt, rhot, n_iter=100, n_inner_iter=50, tol=1e-5, K=1):

    G, C = np.shape(X)
    n_factor = np.log(C)

    x0 = X
    ss0 = sf

    def em_impute():
        vF = a * tf.log(b/sc) - (x+a) * tf.log(1+b/sc) - \
            log_beta(a, x) - tf.log(x+a)
        t = q + tf.concat([vF, tf.broadcast_to(vL_in, [1, K, G, C])], axis=0)
        eta = tf.reduce_logsumexp(t, axis=(0), keepdims=True)
        p_n = tf.exp(rho + t - eta)
        post_est = tf.reduce_sum(
            p_n * tf.concat([(a+x)/(b+sc), tf.broadcast_to(a/b, [1, K, G, C])], axis=0), axis=(0, 1))
        post_est1 = tf.reduce_sum(p_n * tf.concat([tf.broadcast_to(
            x, [1, K, G, C]), tf.broadcast_to(a/b, [1, K, G, C])], axis=0), axis=(0, 1))
        var_est = tf.reduce_sum(p_n * tf.concat([(a+x)*(a+x+1)/(b+sc)/(b+sc), tf.broadcast_to(
            a*(a+1)/b/b, [1, K, G, C])], axis=0), axis=(0, 1)) - post_est * post_est
        return post_est, post_est1, var_est

    def log_beta(a, x):
        a_full = tf.broadcast_to(a, [L-1, K, G, C])
        x_full = tf.broadcast_to(x, [L-1, K, G, C])
        a_s = tf.reshape(a_full, (-1,))
        x_s = tf.reshape(x_full, (-1,))
        lt = tf.lbeta(tf.stack([x_s+1.0, a_s], axis=1))
        return tf.reshape(lt, (L-1, K, G, C))

    def outer_body(j, diffo, a, b, q):
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
        return [j-1, diffo1, at1, bt1, qt1]

    def outer_cond(j, diffo, a, b, q):
        return tf.logical_and(j > 0, diffo > tol)

    a = tf.placeholder(tf.float32, shape=(L-1, K, G, 1))
    b = tf.placeholder(tf.float32, shape=(L-1, K, G, 1))
    rho = tf.placeholder(tf.float32, shape=(1, K, 1, C))
    q = tf.placeholder(tf.float32, shape=(L, K, G, 1))
    x = tf.placeholder(tf.float32, shape=(G, C))
    vL_in = tf.placeholder(tf.float32, shape=(G, C))
    ss = tf.placeholder(tf.float32, shape=(C))

    sc = tf.broadcast_to(ss, [1, 1, 1, C])

    j, diff_out, atn, btn, qn = tf.while_loop(
        outer_cond, outer_body, (n_iter, 1.0, a, b, q))

    post_est, post_est1, var_est = em_impute()

    res = []
    for kk in range(len(at)):
        print("working on cluster {} imputation".format(kk))
        a0 = np.broadcast_to(np.reshape(at[kk], (K, G, 1)), (L-1, K, G, 1))
        b0 = np.broadcast_to(np.reshape(bt[kk], (K, G, 1)), (L-1, K, G, 1))
        rho0 = rhot[kk]
        qlog0 = qt[kk]
        feed_dict = {x: x0, vL_in: (x0 != 0)*(-1e5),
                     a: a0, b: b0, rho: rho0, q: qlog0, ss: ss0}
        st = time.time()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        oo = sess.run((j, diff_out, atn, btn, qn), feed_dict=feed_dict)
        feed_dict = {x: x0, vL_in: (
            x0 != 0)*(-1e5), a: oo[2], b: oo[3], q: oo[4], rho: rho0, ss: ss0}
        oo1 = sess.run((post_est, post_est1, var_est), feed_dict=feed_dict)
        et = time.time()
        sess.close()

        tc = et - st
        res.append({'a': oo[2], 'b': oo[3], 'logq': oo[4],
                    'rate_impute': oo1[0], 'point_impute': oo1[1], 'var_est': oo1[2], 'em_time': tc})

    return res
