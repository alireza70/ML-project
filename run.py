import netflix.input_process as pre
import learn.temporal_dynamics as td
import scipy.io as io
import conf
import scipy.sparse as sp

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

temporal_dynamics()
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
