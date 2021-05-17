import numpy as np
import copy
from tensorflow.keras.datasets import mnist
from label_bias.main import *
from tracin.main import TracIn

### load mnist
(train_xs, train_ys), (test_xs, test_ys) = mnist.load_data()
train_xs = train_xs.astype("float32") / 255.
test_xs = test_xs.astype("float32") / 255.

train_ys = train_ys.astype("float32")
test_ys = test_ys.astype("float32")

#train_xs = train_xs.reshape(-1, 28 * 28)
#test_xs = test_xs.reshape(-1, 28 * 28)

### create biased mnist
train_ys_corrupted = np.copy(train_ys)
np.random.seed(12345)
idxs = np.random.choice(range(len(train_ys_corrupted)), size=len(train_ys_corrupted)//5, replace=False)
train_ys_corrupted[idxs] = 2

print("Distribution Before")
for i in range(10):
  print (np.mean(train_ys == i))

print()

print("Distribution After")
for i in range(10):
  print (np.mean(train_ys_corrupted == i))

print()

'''
[TRAIN]
train_xs, train_ys
train_xs, train_ys_corrupted

[TEST]
test_xs, test_ys
'''

### unbiased mnist training
print("=============== pure MNIST training ===============")
weights = None
train_predictions, test_predictions = run_simple_NN(train_xs, train_ys, test_xs, test_ys, weights, n_epochs=5, mode="unbiased")

print()

### biased mnist training (unconstrained baseline)
print("=============== biased MNIST unconstrained training ===============")
weights = None
train_predictions, test_predictions = run_simple_NN(train_xs, train_ys_corrupted, test_xs, test_ys, weights, n_epochs=5, mode="biased")

print()

### Label bias
print("=============== biased MNIST label bias training ===============")

multipliers = np.zeros(1)
label_bias_lr = 1.0
n_iters = 100
protected_train = [(train_ys_corrupted == 2)]

for it in range(1, n_iters+1):
    print("Iteration", it, "multiplier", multipliers)
    weights = debias_weights(train_ys_corrupted, protected_train, multipliers)
    weights = weights / np.sum(weights)

    print("Weights for 2 : {}".format(np.sum(weights[np.where(train_ys_corrupted==2)])))

    # training on corrupted dataset, testing on correct dataset
    train_prediction, test_predictions = run_simple_NN(train_xs, train_ys_corrupted, test_xs, test_ys, weights, it, n_epochs=5, mode="lb")

    violation = np.mean(train_prediction == 2) - 0.1
    multipliers -= label_bias_lr * violation
    print("violation: {}".format(violation))

    ### get Tracin multiplier ###
    #multiplier_TI = TracIn(train_xs, train_ys_corrupted).self_influence_tester()
    
    print()
    print()
