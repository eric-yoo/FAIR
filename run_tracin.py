from config import args

CHECKPOINTS_PATH_FORMAT = "simpleNN/checkpoints/lb_iter99_ckpt{}" 
args.batch_size = args.batch_size

from tfds.main import make_mnist_dataset
from tracin.main import TracIn

# for poisoned_label in range(10):
if True:
    poisoned_label = 2
    ds_train = make_mnist_dataset('train', args.batch_size, True, is_poisoned=True, poisoned_label=poisoned_label)
    ds_test = make_mnist_dataset('test', args.batch_size, True, is_poisoned=False)

    # arguments: ds_train, ds_test, ckpt1 
    tracin = TracIn(ds_train, ds_test, \
        CHECKPOINTS_PATH_FORMAT.format(3), CHECKPOINTS_PATH_FORMAT.format(4), CHECKPOINTS_PATH_FORMAT.format(5), \
         #'','','',
        True)

    # tracin.find_and_show(tracin.trackin_test, 8, 'influence')
    # tracin.report_mislabel_detection(tracin.trackin_train_self_influences, poisoned_label=poisoned_label, num_dots=10)
    tracin.self_influence_tester(tracin.trackin_train_self_influences)
    #$tracin.report_test_accuracy()
