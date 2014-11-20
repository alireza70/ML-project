from netflix.merge import merge_matrices
from netflix.input_process import load_ratings_list
import learn.temporal_dynamics as td
import scipy.io as io
import conf
import scipy.sparse as sp

ratings = load_ratings_list()
print len(ratings)
print "DATA READ DONE"
td.learn_model(ratings, conf.TEMPORAL_DYNAMICS.ETA)
print "LEARN DONE"
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
