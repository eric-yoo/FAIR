CHECKPOINTS_PATH_FORMAT = "simpleNN/ckpt{}" 
BATCH_SIZE = 512

from tfds.main import make_mnist_dataset
from tracin.main import TracIn

# for biased_label in range(10):
if True:
    biased_label = 8
    ds_train = make_mnist_dataset('train', BATCH_SIZE, True, is_corrupt=True, biased_label=biased_label)
    ds_test = make_mnist_dataset('test', BATCH_SIZE, True, is_corrupt=False)

    # arguments: ds_train, ds_test, ckpt1 
    tracin = TracIn(ds_train, ds_test, \
        CHECKPOINTS_PATH_FORMAT.format(2), CHECKPOINTS_PATH_FORMAT.format(3), CHECKPOINTS_PATH_FORMAT.format(4), \
        # '','','',
        True)

    # tracin.find_and_show(tracin.trackin_test, 8, 'influence')
    # tracin.report_mislabel_detection(tracin.trackin_train_self_influences, biased_label=biased_label, num_dots=10)
    tracin.report_test_accuracy()