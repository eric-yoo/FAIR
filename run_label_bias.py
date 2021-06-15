from tfds.utils import magic_parser
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from label_bias.main import *
from tracin.main import TracIn
from tfds.main import make_mnist_dataset, make_femnist_dataset
from config import args, CHECKPOINTS_PATH_FORMAT

if args.dataset == 'mnist':
  dataset_maker = make_mnist_dataset
elif args.dataset == 'femnist':
  dataset_maker = make_femnist_dataset
else:
  raise NotImplementedError

if args.poison_type == 'many_to_one':
  ds_train    = dataset_maker(F'train', args.batch_size, True, is_poisoned=True, poisoned_ratio=args.poisoned_ratio, poisoned_label=args.poisoned_label)
  ds_train_gt = dataset_maker(F'train', args.batch_size, True, is_poisoned=False)
  ds_test     = dataset_maker('test', args.batch_size, True, is_poisoned=False)
else:
  ds_train    = dataset_maker(F'train', args.batch_size, True, is_poisoned=False, is_reverse_poisoned=True, reverse_poison_ratio=args.poisoned_ratio, reverse_poison_label=args.poisoned_label)
  ds_train_gt = dataset_maker(F'train', args.batch_size, True, is_poisoned=False)
  ds_test     = dataset_maker('test', args.batch_size, True, is_poisoned=False)

### load dataset
(train_xs, train_ys) = magic_parser(ds_train)
(_, train_ys_gt) = magic_parser(ds_train_gt)
(test_xs, test_ys) = magic_parser(ds_test)

### Label bias
print("=============== biased MNIST label bias training ===============")

multipliers = np.zeros(1)
label_bias_lr = 1.0
n_iters = 10
protected_train = [(train_ys == args.poisoned_label)]

# accuracy on {train,test} data over iterations
train_results       = [] 
test_results        = []
test_results_class  = []
violations          = []

for it in range(1, n_iters+1):
    print("Iteration", it, "multiplier", multipliers)
    weights = debias_weights(train_ys, protected_train, multipliers)
    weights = weights / np.sum(weights)

    print("Weights for {} : {}".format(args.poisoned_label, np.sum(weights[np.where(train_ys==args.poisoned_label)])))

    # training on corrupted dataset, testing on correct dataset
    train_res, test_res = run_simple_NN(train_xs, train_ys, test_xs, test_ys, weights, \
                                        it=it, n_epochs=args.n_epochs, mode="lb")
    test_res_class      = eval_simple_NN_by_class(train_xs, train_ys, test_xs, test_ys, \
                            None, CHECKPOINTS_PATH_FORMAT.format("lb", it, args.n_epochs))

    # each res consists of (acc,predictions)
    train_pred = train_res[1]
    
    violation = np.mean(train_pred == args.poisoned_label) - 0.1
    multipliers -= label_bias_lr * violation
    print("violation: {}".format(violation))

    violation_cls = [ np.abs(np.mean(train_pred == cls) - np.mean(train_ys_gt == cls)) for cls in range(len(np.unique(test_ys))) ]

    test_results.append(test_res[0])
    test_results_class.append(test_res_class)
    violations.append(violation_cls)

    ### get Tracin multiplier ###
    #multiplier_TI = TracIn(train_xs, train_ys).self_influence_tester()

    # if test_res[0] > 0.97 :
    #     print("EARLY EXIT @ iteration {} : Target accuracy achieved {}".format(it, test_res[0]))
    #     break
    
    print()
    print()

# acc data over iterations for plot
df_acc     = pd.DataFrame(test_results)
df_acc_cls = pd.DataFrame(test_results_class)
df_vio     = pd.DataFrame(violations)

df_acc.to_csv(F"test_lb_accuracy_p{args.poisoned_ratio}.csv")
df_acc_cls.to_csv(F"test_lb_accuracy_cls_p{args.poisoned_ratio}.csv")
df_vio.to_csv(F"test_lb_violation_p{args.poisoned_ratio}.csv")