import numpy as np
import scipy.sparse as sp
import scipy.io as io
import os
from datetime import datetime as dt
import conf

def get_index(uid, uid2index, index2uid):
    res = uid2index.get( uid )
    if res is None:
        uid2index[uid] = len(index2uid)
        index2uid.append( uid )
        res = len(index2uid) - 1
    return res

def get_user_item_matrix(directory, users, movies):
    uid2index = {}
    index2uid = []
    R = sp.dok_matrix( (users, movies) )
    D = sp.dok_matrix( (users, movies) )

    for file in os.listdir(directory):
        if not file.endswith(".txt"):
            continue
        print file
        ratings = np.array(open(directory + "/" + file).read().splitlines())
        movie = int( ratings[0][:-1] ) - 1
        print movie
        if movie >= movies:
            break
        for rating in ratings[1:]:
            uid, r, date = rating.split(',')

            uid = get_index( int(uid), uid2index, index2uid)
            r = int(r)
            date = dt.strptime(date, "%Y-%m-%d")

            epoch = dt.utcfromtimestamp(0)
            date = int( (date - epoch).days )

            if uid < users:
                R[uid, movie] = r
                D[uid, movie] = date
    return sp.csr_matrix(R), sp.csr_matrix(D), index2uid

def save_user_item_matrix(directory, users, movies):
    R, D, uid = get_user_item_matrix(directory, users, movies)
    io.mmwrite(conf.FILES.RATINGS, R, field = 'integer')
    io.mmwrite(conf.FILES.DATES, D, field = 'integer')
    np.save(conf.FILES.UID, uid)
