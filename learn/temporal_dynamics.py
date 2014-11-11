import scipy.sparse as sp
import numpy as np
import conf
def get_bin(t):
    return 0
class TemporalDynamicsParams:
    def __init__(self, users, items, f):
        self.cnt = 0
        self.mu = 0
        self.b_u = sp.csc_matrix((users, 1))
        self.b_i = sp.csc_matrix((items, 1))
        self.b_iBin = sp.csc_matrix((items, conf.TEMPORAL_DYNAMICS.BINS_COUNT))
        self.q_i = sp.csc_matrix((items, f))
        self.p_u = sp.csc_matrix((users, f))
        self.ap_u = sp.csc_matrix((users, f))
        self.a_u = sp.csc_matrix((users, 1))
        self.t_u = np.zeros((users, 1))

    def dev(self, u, t):
        return float(np.sign(self.t_u[u] - t) * np.power(np.abs(self.t_u[u] - t),\
            conf.TEMPORAL_DYNAMICS.BETA))

    def value(self, u, i, t):
        return float(self.mu + self.b_u[u,0] + self.a_u[u,0]*self.dev(u,t)\
            + self.b_i[i,0] + self.b_iBin[i,get_bin(t)] + ( self.q_i[i,:] * \
            (self.p_u[u,:] + self.ap_u[u,:]*self.dev(u,t)).T ).todense())

    def update(self, u, i, r, t, eta, lmbd):
        error = (r - self.value(u,i,t))
        pref = self.p_u[u,:] + self.ap_u[u,:]*self.dev(u,t)

        self.mu -= eta * (-2 * error)
        self.b_u[u,0] -= eta * (-2 * error + 2 * lmbd * self.b_u[u,0])
        self.b_i[i,0] -= eta * (-2 * error + 2 * lmbd * self.b_i[i,0])
        self.b_iBin[i,get_bin(t)] -= eta * (-2 * error + 2 * lmbd\
            * self.b_iBin[i,get_bin(t)])
        self.a_u[u,0] -= eta * (-2 * error * self.dev(u,t) + 2 * lmbd\
            * self.a_u[u,0])
        self.p_u[u,:] = self.p_u[u,:] - eta * (-2 * error * self.q_i[i,:] + 2 * lmbd * self.p_u[u,:])
        self.ap_u[u,:] = self.ap_u[u,:] - eta * (-2 * error * self.q_i[i,:] * self.dev(u,t) + 2 * lmbd\
            * self.p_u[u,:])
        self.q_i[i,:] = self.q_i[i,:] - eta * (-2 * error * pref + 2 * lmbd * self.q_i[i,:])



def learn_model(R, T, eta, params = None):
    users, items = R.shape
    if params is None:
        params = TemporalDynamicsParams(users, items, conf.TEMPORAL_DYNAMICS.RANK)

    for u in range(0,users):
        params.t_u[u] = T[u,:].data.mean()

    tt = T
    R = sp.coo_matrix(R)
    T = sp.coo_matrix(T)

    import sys
    for _ in range(0, conf.TEMPORAL_DYNAMICS.NSTEP):
        for u, i, r, t in zip(R.row, R.col, R.data, T.data):
            params.update(u, i, r, t, eta, conf.TEMPORAL_DYNAMICS.LAMBDA)

    return params
