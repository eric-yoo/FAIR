from tfds.utils import magic_parser, get_length
import numpy as np
from tensorflow.keras.datasets import mnist
from label_bias.main import *
from tracin.main import TracIn
from tfds.main import make_mnist_dataset, make_femnist_dataset
from config import args, FAIR_PATH_FORMAT

pretrain_ratio = args.pretrain_ratio # default 20%. -> 5epoch 학습하면 test acc 94% 정도 나옴
'''
  three data partitions
  - 60000 samples for pretrain & train
    - ds_pretrain: clean data (60000 * args.pretrain_ratio)
    - ds_train: corrupt data (60000 * (1-args.pretrain_ratio))
      - TODO: implement other types of corruption
  - 10000 samples for test
    - ds_test: clean data (10000)
'''
if args.dataset == 'mnist':
  ds_pretrain = make_mnist_dataset(F'train[:{pretrain_ratio}]', args.batch_size, True, is_poisoned=False)
  ds_train = make_mnist_dataset(F'train[{pretrain_ratio}:]', args.batch_size, True, is_poisoned=True, poisoned_ratio=args.poisoned_ratio, poisoned_label=args.poisoned_label)
  ds_test = make_mnist_dataset('test', args.batch_size, True, is_poisoned=False)
elif args.dataset == 'femnist':
  ds_train = make_femnist_dataset('train', args.batch_size, True, is_poisoned=True, poisoned_ratio=args.poisoned_ratio, poisoned_label=args.poisoned_label)
  ds_test = make_femnist_dataset('test', args.batch_size, True, is_poisoned=False)
else:
  raise NotImplementedError

print(F'pretrain: {get_length(ds_pretrain)*args.batch_size}')
print(F'train: {get_length(ds_train)*args.batch_size}')
print(F'test: {get_length(ds_test)*args.batch_size}')

### load dataset
(pretrain_xs, pretrain_ys) = magic_parser(ds_pretrain)
(train_xs, train_ys) = magic_parser(ds_pretrain)
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


# print("Pretrain 20%")
# # training on corrupted dataset, testing on correct dataset
# train_res, test_res = run_simple_NN(train_xs, train_ys, test_xs, test_ys, weights, it=it, n_epochs=5, mode="fair")

'''
first, pretrain with small clean data
'''
pretrained_model = pretrain_NN(pretrain_xs, pretrain_ys, test_xs, test_ys, pretrain_ratio, n_epochs = args.n_epochs)

'''
run FAIR
if test acc < 0.9:
  use pretrained model for TracIn
else:
  use the current model for TracIn
TODO: FAIR 알고리즘 구현 덜끝난듯? + 실험 아직 안해봄.
'''
for it in range(1, n_iters+1):
    print("Iteration", it, "multiplier", multipliers_TI)
    weights = debias_weights_TI(train_ys, protected_train, multipliers_TI)
    weights = weights / np.sum(weights)

    print("Weights for 2 : {}".format(np.sum(weights[np.where(train_ys==2)])))

    # training on corrupted dataset, testing on correct dataset
    train_res, test_res = run_simple_NN(train_xs, train_ys, test_xs, test_ys, weights, it=it, n_epochs=args.n_epochs, mode="RCL_single")

    train_results.append(train_res)
    test_results.append(test_res)

    if test_res[0] > 0.97 and train_res[0] > 0.97 :
        print("EARLY EXIT @ iteration {} : Target accuracy achieved {}".format(it, test_res[0]))
        break
    
    # each res consists of (acc,predictions)
    train_pred = train_res[1]
    
    violation = np.mean(train_pred == 2) - 0.1
    print("violation: {}".format(violation))

    if test_res[0] <= 0.9:
      # instantiate TRACIN
      tracin = TracIn(ds_train, ds_test, \
          PRETRAINED_PATH.format(pretrain_ratio, args.n_epochs-2), PRETRAINED_PATH.format(pretrain_ratio, args.n_epochs-1), PRETRAINED_PATH.format(pretrain_ratio, args.n_epochs), \
          True)
    else:
      # instantiate TRACIN
      tracin = TracIn(ds_train, ds_test, \
          FAIR_PATH_FORMAT.format(it, args.n_epochs-2), FAIR_PATH_FORMAT.format(it, args.n_epochs-1), FAIR_PATH_FORMAT.format(it, args.n_epochs), \
          True)

    # multipliers -= label_bias_lr * violation
    multiplier_TI = tracin.self_influence_tester(tracin.trackin_train_self_influences,violation)

    print()
    print()

# acc data over iterations for plot
test_res_it = [res[1] for res in test_results]