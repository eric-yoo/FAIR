from tfds.utils import magic_parser
import numpy as np
from tensorflow.keras.datasets import mnist
from label_bias.main import *
from tracin.main import TracIn
from tfds.main import make_mnist_dataset, make_femnist_dataset
from config import args, CHECKPOINTS_PATH_FORMAT

if args.dataset == 'mnist':
  ds_train = make_mnist_dataset('train', args.batch_size, True, is_poisoned=True, poisoned_ratio=args.poisoned_ratio, poisoned_label=args.poisoned_label)
  ds_test = make_mnist_dataset('test', args.batch_size, True, is_poisoned=False)
elif args.dataset == 'femnist':
  ds_train = make_femnist_dataset('train', args.batch_size, True, is_poisoned=True, poisoned_ratio=args.poisoned_ratio, poisoned_label=args.poisoned_label)
  ds_test = make_femnist_dataset('test', args.batch_size, True, is_poisoned=False)
else:
  raise NotImplementedError

### load dataset
(train_xs, train_ys) = magic_parser(ds_train)
(test_xs, test_ys) = magic_parser(ds_test)


### Label bias
print("=============== biased MNIST label bias training (run_label_bias.py)===============")

multipliers = np.zeros(1)
label_bias_lr = 1.0
n_iters = 2
protected_train = [(train_ys == 2)]

# accuracy on {train,test} data over iterations
train_results = [] 
test_results  = []

for it in range(1, n_iters+1):
    print("Iteration", it, "multiplier", multipliers)
    weights = debias_weights(train_ys, protected_train, multipliers)
    weights = weights / np.sum(weights)

    print("Weights for 2 : {}".format(np.sum(weights[np.where(train_ys==2)])))

    # training on corrupted dataset, testing on correct dataset
    # train_res, test_res = run_simple_NN(train_xs, train_ys, test_xs, test_ys, weights, it=it, n_epochs=args.n_epochs, mode="lb")
    eval_simple_NN_by_class(train_xs, train_ys, test_xs, test_ys, \
                            None, CHECKPOINTS_PATH_FORMAT.format("unbiased",0,5))
    exit(1)

    train_results.append(train_res)
    test_results.append(test_res)

    # each res consists of (acc,predictions)
    train_pred = train_res[1]
    
    violation = np.mean(train_pred == 2) - 0.1
    multipliers -= label_bias_lr * violation
    print("violation: {}".format(violation))

    ### get Tracin multiplier ###
    #multiplier_TI = TracIn(train_xs, train_ys).self_influence_tester()

    if test_res[0] > 0.97 :
        print("EARLY EXIT @ iteration {} : Target accuracy achieved {}".format(it, test_res[0]))
        break
    
    print()
    print()

# acc data over iterations for plot
test_res_it = [res[1] for res in test_results]