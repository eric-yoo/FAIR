CHECKPOINTS_PATH_FORMAT = "simpleNN/ckpt{}" 
BATCH_SIZE = 512

from tfds.main import make_get_dataset
from tracin.main import TracIn

ds_train = make_get_dataset('train', BATCH_SIZE)
ds_test = make_get_dataset('test', BATCH_SIZE)

# arguments: ds_train, ds_test, ckpt1 
tracin = TracIn(ds_train, ds_test, \
    CHECKPOINTS_PATH_FORMAT.format(1), CHECKPOINTS_PATH_FORMAT.format(2), CHECKPOINTS_PATH_FORMAT.format(3), \
    True)

tracin.find_and_show(tracin.trackin_test, 8, 'influence')