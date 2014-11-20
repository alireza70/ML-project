import scipy.sparse as sp
import numpy as np
import conf
def get_bin(t):
    return 0
class TemporalDynamicsParams:
    def __init__(self, users, items, f):
        self.cnt = 0
        self.mu = 0
        self.b_u = np.zeros((users, 1))
        self.b_i = np.zeros((items, 1))
        self.b_iBin = np.zeros((items, conf.TEMPORAL_DYNAMICS.BINS_COUNT))
        self.q_i = np.zeros((items, f))
        self.p_u = np.zeros((users, f))
        self.ap_u = np.zeros((users, f))
        self.a_u = np.zeros((users, 1))
        self.t_u = np.zeros((users, 1))

    def dev(self, u, t):
        return float(np.sign(self.t_u[u] - t) * np.power(np.abs(self.t_u[u] - t),\
            conf.TEMPORAL_DYNAMICS.BETA))

    def value(self, u, i, t):
        return float(self.mu + self.b_u[u,0] + self.a_u[u,0]*self.dev(u,t)\
            + self.b_i[i,0] + self.b_iBin[i,get_bin(t)] + ( self.q_i[i,:].dot( \
            (self.p_u[u,:] + self.ap_u[u,:]*self.dev(u,t)).T )) )

    def update(self, u, i, r, t, eta, lmbd):
        if self.cnt % 1000 == 0:
            print "ON DATAPOINT ", self.cnt
        self.cnt += 1
        error = (r - self.value(u,i,t))
        pref = self.p_u[u,:] + self.ap_u[u,:]*self.dev(u,t)

        self.mu -= eta * (-2 * error)
        self.b_u[u,0] -= eta * (-2 * error + 2 * lmbd * self.b_u[u,0])
        self.b_i[i,0] -= eta * (-2 * error + 2 * lmbd * self.b_i[i,0])
        self.b_iBin[i,get_bin(t)] -= eta * (-2 * error + 2 * lmbd\
            * self.b_iBin[i,get_bin(t)])
        self.a_u[u,0] -= eta * (-2 * error * self.dev(u,t) + 2 * lmbd\
            * self.a_u[u,0])

        self.p_u[u,:] -= eta * (-2 * error * self.q_i[i,:] + 2 * lmbd * self.p_u[u,:])
        self.ap_u[u,:] -= eta * (-2 * error * self.q_i[i,:] * self.dev(u,t) + 2 * lmbd\
            * self.p_u[u,:])
        self.q_i[i,:] -= eta * (-2 * error * pref + 2 * lmbd * self.q_i[i,:])

    def save(self):
        np.save(conf.TEMPORAL_DYNAMICS.FILES.MU, self.mu)
        np.save(conf.TEMPORAL_DYNAMICS.FILES.B_U, self.b_u)
        np.save(conf.TEMPORAL_DYNAMICS.FILES.B_I, self.b_i)
        np.save(conf.TEMPORAL_DYNAMICS.FILES.B_IBIN, self.b_iBin)
        np.save(conf.TEMPORAL_DYNAMICS.FILES.Q_I, self.q_i)
        np.save(conf.TEMPORAL_DYNAMICS.FILES.P_U, self.p_u)
        np.save(conf.TEMPORAL_DYNAMICS.FILES.AP_U, self.ap_u)
        np.save(conf.TEMPORAL_DYNAMICS.FILES.A_U, self.a_u)
        np.save(conf.TEMPORAL_DYNAMICS.FILES.T_U, self.t_u)


def learn_model(ratings, eta, users, items, params = None):
    if params is None:
        params = TemporalDynamicsParams(users, items, conf.TEMPORAL_DYNAMICS.RANK)

    print "INITIALIZING TIMES"
    cnt = 0
    rates = np.zeros(users)
    for u, i, r, t in ratings:
        cnt += 1
        if cnt%1000 == 0:
            print "ON DATAPOINT", cnt
        params.t_u[u] += t
        rates[u] += 1

    for u in range(users):
        params.t_u[u] = 0 if rates[u] == 0 else params.t_u[u] / rates[u]

    print "LEARNING PARAMETERS"
    import sys
    for _ in range(0, conf.TEMPORAL_DYNAMICS.NSTEP):
        print "ITERATION ", _
        for u, i, r, t in ratings:
            params.update(u, i, r, t, eta, conf.TEMPORAL_DYNAMICS.LAMBDA)
        params.cnt = 0

    return params
