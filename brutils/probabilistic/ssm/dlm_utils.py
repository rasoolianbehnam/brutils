from collections import namedtuple

import tensorflow_probability as tfp
from brutils.imports.implicits.tensorflow import *

tfd = tfp.distributions
t = tf.transpose

PosteriorStates = namedtuple("posteriorStates", "mt Ct at Rt ft Qt et nt, St")
SmoothingResults = namedtuple("SmoothingResults", "mnt Cnt fnt Qnt")
AdaptiveDlmResult = namedtuple("AdaptiveDlmResult", "results_filtered results_smoothed results_forecast df_opt measure")
DlmParameters = namedtuple('DlmParameters', "Ft Gt Wt_star")
InitialStates = namedtuple("InitialStates", "m0, C0_star, n0, S0")


def update(at, at_i, i):
    return tf.tensor_scatter_nd_update(at, [[i]], [at_i])


def forward_filter_unknown_v(yt, matrices, initial_states, delta=None):
    ## retrieve dataset
    T = len(yt)

    ## retrieve matrices
    Ft = matrices.Ft  # Should be y_dim x latent_dim
    Gt = matrices.Gt  # Should be latent_dim x latent_dim
    # assert Ft.shape[-1] == T

    if delta is None:
        Wt_star = matrices.Wt_star

    ## retrieve initial state
    m0 = initial_states.m0
    C0_star = initial_states.C0_star
    n0 = initial_states.n0
    S0 = initial_states.S0
    C0 = S0 * C0_star

    ## create placeholder for results
    latent_dim = Gt.shape[0]
    y_dim = Ft.shape[0]
    assert Ft.shape[1] == latent_dim
    at = tf.zeros([T, latent_dim])
    Rt = tf.zeros([T, latent_dim, latent_dim])
    ft = tf.zeros([T, y_dim])
    Qt = tf.zeros([T, y_dim, y_dim])
    mt = tf.zeros([T, latent_dim])
    Ct = tf.zeros([T, latent_dim, latent_dim])
    et = tf.zeros([T, y_dim])
    nt = tf.zeros(T)
    St = tf.zeros([T, y_dim, y_dim])

    def loop_body(i, Ct, Qt, Rt, St, at, et, ft, mt, nt):
        if i == 0:
            at_i = einsum("ij,j->i", Gt[..., i], m0)
            Pt = Gt[..., i] @ C0 @ Gt[..., i].T
            Pt = 0.5 * Pt + 0.5 * Pt.T
            if delta is None:
                Wt = Wt_star[..., i] * S0
                Rt_i = Pt + Wt
            else:
                Rt_i = Pt / delta
        else:
            at_i = einsum("ij,j->i", Gt[..., i], mt[i - 1])
            Pt = Gt[..., i] @ Ct[i - 1] @ Gt[..., i].T
            if delta is None:
                Wt = Wt_star[..., i] * St[i - 1]
                Rt_i = Pt + Wt
            else:
                Rt_i = Pt / delta
        Rt_i = 0.5 * Rt_i + 0.5 * t(Rt_i)
        at = update(at, at_i, i)
        Rt = update(Rt, Rt_i, i)
        # moments of one-step forecast:
        ft = update(ft, einsum("ij,j->i", Ft[..., i], t(at[i])), i)
        ss = (S0 if i == 0 else St[i - 1])
        Qt_i = Ft[..., i] @ Rt[i] @ Ft[..., i].T + ss  # y_dim x y_dim
        Qt = update(Qt, Qt_i, i)
        et = update(et, yt[i] - ft[i], i)
        nt = update(nt, (n0 if i == 0 else nt[i - 1]) + 1, i)
        Qt_i_inv = chol2inv(Qt[i])
        etQt = einsum("ij,j->i", Qt_i_inv, et[i] ** 2)
        St_i = ss * (1 + 1 / nt[i] * (etQt - 1))
        St = update(St, St_i, i)
        # moments of posterior at t:
        At = Rt[i] @ Ft[..., i].T @ Qt_i_inv  # latent_dim x y_dim
        mt = update(mt, at[i] + einsum("dk,k->d", At, et[i]), i)
        Ct_i = St[i] / ss * (Rt[i] - At @ Qt[i] @ t(At))
        Ct_i = 0.5 * Ct_i + 0.5 * t(Ct_i)
        Ct = update(Ct, Ct_i, i)
        return i + 1, Ct, Qt, Rt, St, at, et, ft, mt, nt

    # moments of priors at t
    # i = 0
    # while i < T:
    #     i, Ct, Qt, Rt, St, at, et, ft, mt, nt = loop_body(i, Ct, Qt, Rt, St, at, et, ft, mt, nt)

    i, Ct, Qt, Rt, St, at, et, ft, mt, nt = tf.while_loop(
        lambda i, *_: i < T,
        loop_body,
        (0, Ct, Qt, Rt, St, at, et, ft, mt, nt)
    )
    # print("Forward filtering is completed!\n")
    return PosteriorStates(mt=mt, Ct=Ct, at=at, Rt=Rt,
                           ft=ft, Qt=Qt, et=et,
                           nt=nt, St=St)


def chol2inv(X):
    # return tf.linalg.inv(X)
    p = X.shape[0]
    Ip = tf.eye(p)
    return tf.linalg.cholesky_solve(tf.linalg.cholesky(X), Ip)


def backward_smoothing_unknown_v(yt, matrices: DlmParameters, posterior_states: PosteriorStates, delta=None):
    ## retrieve data
    T = len(yt)

    ## retrieve matrices
    Ft = matrices.Ft
    Gt = matrices.Gt
    y_dim = Ft.shape[0]

    ## retrieve matrices
    mt = posterior_states.mt
    Ct = posterior_states.Ct
    Rt = posterior_states.Rt
    St = posterior_states.St
    at = posterior_states.at

    ## create placeholder for posterior moments
    mnt = tf.zeros_like(mt)
    Cnt = tf.zeros_like(Ct)
    fnt = tf.zeros([T, y_dim])
    Qnt = tf.zeros([T, y_dim, y_dim])

    def loop_body_01(i, mnt, Cnt, fnt, Qnt):
        if i == T - 1:
            mnt = update(mnt, mt[i], i)
            Cnt = update(Cnt, Ct[i], i)
        else:
            if delta is None:
                inv_Rtp1 = chol2inv(Rt[i + 1])
                Bt = Ct[i] @ t(Gt[..., i + 1]) @ inv_Rtp1  # d x d  | d x d  |  d, d -> d x d
                mnt_i = mt[i] + einsum("ij,j->i", Bt, (mnt[i + 1] - at[i + 1]))
                mnt = update(mnt, mnt_i, i)  # d x d  |  d  |
                Cnt_i = Ct[i] + Bt @ (Cnt[i + 1] - Rt[i + 1]) @ t(Bt)
                Cnt = update(Cnt, 0.5 * Cnt_i + 0.5 * t(Cnt_i), i)
            else:
                inv_Gt = chol2inv(Gt[..., i + 1])
                mnt = update(mnt, (1 - delta) * mt[i,] + delta * einsum("ij,j->i", inv_Gt, t(mnt[i + 1])), i)
                Cnt_i = (1 - delta) * Ct[i] + delta ** 2 * inv_Gt @ Cnt[i + 1] @ t(inv_Gt)
                Cnt = update(Cnt, 0.5 * Cnt_i + 0.5 * t(Cnt_i), i)
        fnt_i = einsum("kd,d->k", Ft[..., i], t(mnt[i]))
        fnt = update(fnt, fnt_i, i)
        Qnt_i = Ft[..., i] @ Cnt[i] @ Ft[..., i].T
        Qnt = update(Qnt, Qnt_i, i)
        return i - 1, mnt, Cnt, fnt, Qnt

    def loop_body_02(i, Cnt, Qnt):
        Cnt = update(Cnt, St[T - 1] * Cnt[i] / St[i], i)
        Qnt = update(Qnt, St[T - 1] * Qnt[i] / St[i], i)
        return i + 1, Cnt, Qnt

    _, mnt, Cnt, fnt, Qnt = tf.while_loop(lambda i, *_: i >= 0, loop_body_01, (T - 1, mnt, Cnt, fnt, Qnt))
    _, Cnt, Qnt = tf.while_loop(lambda i, *_: i < T, loop_body_02, (0, Cnt, Qnt))

    # print("Backward smoothing is completed!.")
    return SmoothingResults(mnt=mnt, Cnt=Cnt, fnt=fnt, Qnt=Qnt)


def forecast_function_unknown_v(posterior_states, tt, matrices, delta=None):
    ## retrieve matrices
    Ft = matrices.Ft
    Gt = matrices.Gt
    y_dim = Ft.shape[0]
    if delta is None:
        Wt_star = matrices.Wt_star

    mt = posterior_states.mt
    Ct = posterior_states.Ct
    St = posterior_states.St
    at = posterior_states.at

    ## set up matrices
    T = mt.shape[0] - 1  # time points
    latent_dim = mt.shape[1]  # dimension of state parameter vector

    ## placeholder for results
    at = tf.zeros([tt, latent_dim])
    Rt = tf.zeros([tt, latent_dim, latent_dim])
    ft = tf.zeros([tt, y_dim])
    Qt = tf.zeros([tt, y_dim, y_dim])

    def loop_body(i, at, Rt, ft, Qt):
        if i == 0:
            at_i = einsum("ij,j->i", Gt[..., T + i], t(mt[T]))
            if delta is None:
                Rt_i = Gt[..., T + i] @ Ct[T] @ t(Gt[..., T + i]) + St[T] * Wt_star[..., T + i]
            else:
                Rt_i = Gt[..., T + i] @ Ct[T] @ t(Gt[..., T + i]) / delta

        else:
            at_i = einsum("ij,j->i", Gt[..., T + i], t(at[i - 1]))
            if delta is None:
                Rt_i = Gt[..., T + i] @ Rt[i - 1] @ t(Gt[..., T + i]) + St[T] * Wt_star[..., T + i]
            else:
                Rt_i = Gt[..., T + i] @ Rt[i - 1] @ t(Gt[..., T + i]) / delta
        Rt_i = 0.5 * Rt_i + 0.5 * t(Rt_i)
        Rt = update(Rt, Rt_i, i)
        at = update(at, at_i, i)
        # moments of forecast distribution
        ft_i = einsum("ij,j->i", Ft[..., T + i], t(at[i]))
        ft = update(ft, ft_i, i)
        Qt = update(Qt, Ft[..., T + i] @ Rt[i] @ Ft[..., T + i].T + St[T], i)
        return i + 1, at, Rt, ft, Qt

    _, at, Rt, ft, Qt = tf.while_loop(lambda i, *_: i < tt, loop_body, (0, at, Rt, ft, Qt))
    # print("Forecasting is completed!\n")  # indicator of completion
    return dict(at=at, Rt=Rt, ft=ft, Qt=Qt)


def backward_smooth_observed(F_, smooth):
    m = einsum("ij,tj->ti", F_, smooth[0])
    s = einsum("ij,tjk,mk->tim", F_, smooth[1], F_)
    return m, s


def forecast_function(dlm_known_v, filt, k):
    import numpy as np
    def t(x):
        return np.transpose(x)
    GG = dlm_known_v.transition_matrix.to_dense().numpy()
    FF = dlm_known_v.observation_matrix.numpy()
    WW = dlm_known_v.transition_noise.covariance().numpy()
    VV = dlm_known_v.observation_noise.covariance().numpy()
    mt = filt.filtered_means.numpy()
    Ct = filt.filtered_covs.numpy()

    ## set up matrices
    d = mt.shape[1]

    ## placeholder for results
    at = np.zeros([k, d])
    Rt = np.zeros([k, d, d])
    ft = np.zeros(k)
    Qt = np.zeros(k)

    at[0] = GG @ mt[-1]
    Rt[0] = GG @ Ct[-1] @ t(GG) + WW
    Rt[0] = 0.5 * Rt[0] + 0.5 * t(Rt[0])

    for i in range(1, k):
        at[i] = GG @ at[i - 1]
        Rt[i] = GG @ Rt[i - 1] @ t(GG) + WW
        Rt[i] = 0.5 * Rt[i] + 0.5 * t(Rt[i])

    for i in range(0, k):
        ft[i] = FF @ at[i]
        Qt[i] = FF @ Rt[i] @ FF.T + VV
        return dict(at=at, Rt=Rt, ft=ft, Qt=Qt)


def forecast_function(dlm_known_v, filt, k):
    FF = dlm_known_v.observation_matrix
    GG = dlm_known_v.transition_matrix.to_dense()
    WW = dlm_known_v.transition_noise.covariance()
    VV = dlm_known_v.observation_noise.covariance()

    a = filt.filtered_means[-1][:, None]
    R = filt.filtered_covs[-1]
    at = []
    Rt = []
    ft = []
    Qt = []
    for i in range(k):
        a = GG @ a
        at.append(a)

        R = GG @ R @ t(GG) + WW
        Rt.append(R)

        f = FF @ a
        ft.append(f)

        Q = FF @ Rt[i] @ FF.T + VV
        Qt.append(Q)

    at = tf.transpose(tf.concat(at, axis=-1))
    ft = tf.transpose(tf.concat(ft, axis=-1))
    Qt = tf.stack(Qt)
    Rt = tf.stack(Rt)
    return dict(at=at, ft=ft, Qt=Qt, Rt=Rt)