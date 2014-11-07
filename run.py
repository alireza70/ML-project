from netflix.input_process import *
import conf

save_user_item_matrix('../training_set', conf.DATASET.TRAINING.USERS,\
 conf.DATASET.TRAINING.MOVIES)
