from input_process import get_index
import numpy as np
import scipy.io as io
import conf
import scipy.sparse as sp

def merge_uid_files(files, merged_file):
    uid2index = {}
    index2uid = []

    for f in files:
        uids = np.load(f)
        for u in uids:
            get_index(u, uid2index, index2uid)
    np.save(merged_file, index2uid)


def merge_matrices(segments, users, movies):
    uid2index = {}
    index2uid = []

    uids = np.load(conf.FILES.UID)
    for u in uids:
        get_index(u, uid2index, index2uid)

    R = sp.dok_matrix((users, movies))
    T = sp.dok_matrix((users, movies))

    for s in segments:
        print "Reading ", 'data/ratings' + s + '.mtx'
        Rs = io.mmread('data/ratings' + s + '.mtx')
        print "Reading ", 'data/dates' + s + '.mtx'
        Ts = io.mmread('data/dates' + s + '.mtx')
        print "Reading ", 'data/uid' + s + '.npy'
        Us = np.load('data/uid' + s + '.npy')

        print "Inserting points to the matrix!"
        for u, i, r, t in zip(Rs.row, Rs.col, Rs.data, Ts.data):
            u = uid2index[ Us[u] ]
            R[u,i] = r
            T[u,i] = t
        print
    print "Writing R"
    io.mmwrite(conf.FILES.RATINGS, R, field = 'integer')
    print "Writing T"
    io.mmwrite(conf.FILES.DATES, D, field = 'integer')
    return R,T
