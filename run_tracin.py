BATCH_SIZE = 512

from tfds.main import make_get_dataset
from tracin.main import TracIn

ds_train = make_get_dataset('train', BATCH_SIZE)
ds_test = make_get_dataset('test', BATCH_SIZE)

tracin = TracIn(ds_train, ds_test)

tracin.find_and_show(tracin.trackin_test, 8, 'influence')