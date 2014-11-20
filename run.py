import netflix.input_process as pre
import movies_len.input_process as mpre
import learn.temporal_dynamics as td
import scipy.io as io
import conf
import scipy.sparse as sp
import random as rnd
import numpy as np
import SGDMFBiased as SGD

def save_list():
    pre.save_ratings_list('../training_set')

def temporal_dynamics():
    ratings = pre.load_ratings_list()
    print "DATA READ DONE"
    params = td.learn_model(ratings, conf.TEMPORAL_DYNAMICS.ETA, conf.NETFLIX.TRAINING.USERS,\
     conf.NETFLIX.TRAINING.MOVIES_END - conf.NETFLIX.TRAINING.MOVIES_START )
    print "LEARN DONE"
    params.save()
    print "SAVE DONE"

def temporal_dynamics_movies_len():
    print "READING"
    ratings = []
    for u, i, r, t in mpre.get_ratings('../ml/u.data'):
        ratings.append( (u, i, r, t) )
    rnd.shuffle(ratings)
    print "DATA READ DONE"
    params = td.learn_model(ratings, conf.TEMPORAL_DYNAMICS.ETA,\
    conf.MOVIES_LEN.TRAINING.USERS, conf.MOVIES_LEN.TRAINING.MOVIES )
    print "LEARN DONE"
    params.save()
    print "SAVE DONE"

def temporal_dynamics_movies_len_test():
    print "LOADING PARAMETERS"
    params = td.TemporalDynamicsParams()
    params.load()
    print "PARAMETERS LOADED"

    cnt = 0
    err = 0
    for u, i, r, t in mpre.get_ratings('../ml/u.data'):
        rh = params.value(u, i, t)
        err += (r-rh)**2
        cnt += 1
    err /= cnt
    err = np.sqrt(err)
    print "RMSE =", err

def basic_mf():
    ratings = pre.load_ratings_list()
    print "DATA READ DONE"
    SGD.Learn(ratings, conf.NETFLIX.TRAINING.USERS,conf.NETFLIX.TRAINING.MOVIES)
    print "LEARN DONE"
    params.save()
    print "SAVE DONE"
basic_mf()
#temporal_dynamics_movies_len()
# print "Reading Ratings Matrix"
# R = sp.csr_matrix(io.mmread(conf.FILES.RATINGS))
# R = sp.coo_matrix(R[0:50000, :])
# print "Reading Dates Matrix"
# T = sp.csr_matrix(io.mmread(conf.FILES.DATES))
# T = sp.coo_matrix(T[0:50000, :])
#
# print "There are ", len(R.data), " points!"
#
# params = td.learn_model(R, T, conf.TEMPORAL_DYNAMICS.ETA)
#
# rmse = 0
# for u, i, r, t in zip(R.row, R.col, R.data, T.data):
#     rmse += np.power(params.value(u, i, t) - r, 2)
# rmse /= len(R.data)
# rmse = np.sqrt(rmse)
#
# print "RMSE=", rmse
