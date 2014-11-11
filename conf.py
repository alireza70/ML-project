class DATASET:
    class TRAINING:
        USERS = 480189
        MOVIES_START = 15000
        MOVIES_END = 17770

class FILES:
    RATINGS = 'data/ratings.mtx'
    DATES = 'data/dates.mtx'
    UID = 'data/uids.npy'

class MATRIX_FAC:
    LAMBDA = 0.02
    ALPHA = 0.0002
    NSTEP = 5000

class TEMPORAL_DYNAMICS:
    RANK = 250
    NSTEP = 20
    BETA = 0.4
    LAMBDA = 0.01
    BINS_COUNT = 1
    ETA = 0.001
