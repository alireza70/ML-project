import random as rnd
import conf
def get_ratings(filePath):
    users = conf.MOVIES_LEN.TRAINING.USERS
    movies = conf.MOVIES_LEN.TRAINING.MOVIES

    with open(filePath) as f:
        for rating in f:
            u, i, r, t = rating[:-1].split('\t')

            u = int(u)-1
            i = int(i)-1
            r = int(r)
            t = int(t)

            yield u, i, r, t


def save_ratings_list(filePath):
    ratings = []
    for u, r, i, t in get_ratings(filePath):
        ratings.append( (u, r, i, t) )
    rnd.shuffle(ratings)
    np.save(conf.MOVIES_LEN.FILES.RATINGS_LIST, ratings)

def load_ratings_list():
    return np.load(conf.MOVIES_LEN.FILES.RATINGS_LIST)
