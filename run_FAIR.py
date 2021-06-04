from utils import magic_parser
import numpy as np
from tensorflow.keras.datasets import mnist
from label_bias.main import *
from tracin.main import TracIn
from tfds.main import make_mnist_dataset, make_femnist_dataset
from config import args, FAIR_PATH_FORMAT

if args.dataset == 'mnist':
  ds_train = make_mnist_dataset('train', args.batch_size, True, is_corrupt=True, corrupt_ratio=args.corrupt_ratio, biased_label=args.biased_label)
  ds_test = make_mnist_dataset('test', args.batch_size, True, is_corrupt=False)
elif args.dataset == 'femnist':
  ds_train = make_femnist_dataset('train', args.batch_size, True, is_corrupt=True, corrupt_ratio=args.corrupt_ratio, biased_label=args.biased_label)
  ds_test = make_femnist_dataset('test', args.batch_size, True, is_corrupt=False)
else:
  raise NotImplementedError

### load dataset
(train_xs, train_ys) = magic_parser(ds_train)
(test_xs, test_ys) = magic_parser(ds_test)


### Label bias
print("=============== biased MNIST label bias training ===============")

multipliers_TI = np.zeros(train_xs.shape[0])
label_bias_lr = 1.0
n_iters = 100
protected_train = [(train_ys == 2)]

#TRACIN
# accuracy on {train,test} data over iterations
train_results = [] 
test_results  = []

for it in range(1, n_iters+1):
    print("Iteration", it, "multiplier", multipliers_TI)
    weights = debias_weights_TI(train_ys, protected_train, multipliers_TI)
    weights = weights / np.sum(weights)

    print("Weights for 2 : {}".format(np.sum(weights[np.where(train_ys==2)])))

    # training on corrupted dataset, testing on correct dataset
    train_res, test_res = run_simple_NN(train_xs, train_ys, test_xs, test_ys, weights, it=it, n_epochs=5, mode="fair")

    train_results.append(train_res)
    test_results.append(test_res)

    if test_res[0] > 0.97 and train_res[0] > 0.97 :
        print("EARLY EXIT @ iteration {} : Target accuracy achieved {}".format(it, test_res[0]))
        break
    
    # each res consists of (acc,predictions)
    train_pred = train_res[1]
    
    violation = np.mean(train_pred == 2) - 0.1
    print("violation: {}".format(violation))

    # instantiate TRACIN
    tracin = TracIn(ds_train, ds_test, \
        FAIR_PATH_FORMAT.format(it, 3), FAIR_PATH_FORMAT.format(it, 4), FAIR_PATH_FORMAT.format(it,5), \
        True)

    # multipliers -= label_bias_lr * violation
    multiplier_TI = tracin.self_influence_tester(tracin.trackin_train_self_influences,violation)

    print()
    print()

# acc data over iterations for plot
test_res_it = [res[1] for res in test_results]