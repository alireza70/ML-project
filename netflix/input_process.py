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

def get_user_item_matrix(directory):
    users = conf.DATASET.TRAINING.USERS
    movies = conf.DATASET.TRAINING.MOVIES_END\
     - conf.DATASET.TRAINING.MOVIES_START

    uid2index = {}
    index2uid = []
    R = sp.dok_matrix( (users, movies) )
    D = sp.dok_matrix( (users, movies) )

    for file in os.listdir(directory):
        if not file.endswith(".txt"):
            continue

        print file
        movie = int( file[3:-4] ) -1
        if movie < conf.DATASET.TRAINING.MOVIES_START\
         or movie >= conf.DATASET.TRAINING.MOVIES_END:
            continue

        with open(directory + "/" + file) as f:
            movie = int( f.readline()[:-2] ) - 1 - conf.DATASET.TRAINING.MOVIES_START
            print movie
            for rating in f:
                uid, r, date = rating[:-1].split(',')

                uid = get_index( int(uid), uid2index, index2uid)
                r = int(r)
                date = dt.strptime(date, "%Y-%m-%d")

                epoch = dt.utcfromtimestamp(0)
                date = int( (date - epoch).days )

                if uid < users:
                    R[uid, movie] = r
                    D[uid, movie] = date
    return sp.csr_matrix(R), sp.csr_matrix(D), index2uid

def save_user_item_matrix(directory):
    R, D, uid = get_user_item_matrix(directory)
    io.mmwrite(conf.FILES.RATINGS, R, field = 'integer')
    io.mmwrite(conf.FILES.DATES, D, field = 'integer')
    np.save(conf.FILES.UID, uid)
