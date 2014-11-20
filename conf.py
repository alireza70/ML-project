class NETFLIX:
    SHUFFLE_TRESHOLD = 1000000
    class TRAINING:
        USERS = 480189
        MOVIES_START = 0
        MOVIES_END = 17770

class MOVIES_LEN:
    class FILES:
        RATINGS_LIST = 'data/movies/ratings.npy'
    class TRAINING:
        USERS = 943
        MOVIES = 1682

class FILES:
    RATINGS_MATRIX = 'data/ratings.mtx'
    DATES_MATRIX = 'data/dates.mtx'
    RATINGS_LIST = 'data/ratings.npy'
    UID = 'data/uids.npy'

class MATRIX_FAC:
    LAMBDA = 0.02
    ALPHA = 0.002
    NSTEP = 20
    K = 250 
    

class BASE_NEIGHBOURHOOD:
    LAMBDA2 = 100
    ALPHA = 0.005
    LAMBDA = 0.02
    K = 2
    NSTEP = 2000

class BASELINE:
    LAMBDA = 0.02
    ALPHA = 0.0002
    NSTEP = 5000

class TEMPORAL_DYNAMICS:
    RANK = 250
    NSTEP = 20
    BETA = 0.2
    LAMBDA = 0.01
    BINS_COUNT = 1
    ETA = 0.001
    class FILES:
        MU = 'data/temporal/mu.npy'
        B_U = 'data/temporal/b_u.npy'
        B_I = 'data/temporal/b_i.npy'
        B_IBIN = 'data/temporal/b_iBin.npy'
        Q_I = 'data/temporal/q_i.npy'
        P_U = 'data/temporal/p_u.npy'
        AP_U = 'data/temporal/ap_u.npy'
        A_U = 'data/temporal/a_u.npy'
        T_U = 'data/temporal/t_u.npy'
