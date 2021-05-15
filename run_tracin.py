CHECKPOINTS_PATH_FORMAT = "simpleNN/ckpt{}" 
BATCH_SIZE = 512

from tfds.main import make_get_dataset
from tracin.main import TracIn

ds_train = make_get_dataset('train', BATCH_SIZE, True, is_corrupt=True)
ds_test = make_get_dataset('test', BATCH_SIZE, True, is_corrupt=False)

# arguments: ds_train, ds_test, ckpt1 
tracin = TracIn(ds_train, ds_test, \
    CHECKPOINTS_PATH_FORMAT.format(8), CHECKPOINTS_PATH_FORMAT.format(9), CHECKPOINTS_PATH_FORMAT.format(10), \
    # '','','',
    True)

# tracin.find_and_show(tracin.trackin_test, 8, 'influence')
tracin.report_mislabel_detection(tracin.trackin_train_self_influences, num_dots=10)