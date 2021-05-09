import tensorflow.compat.v1 as tf
import numpy as np
import copy
from tensorflow.keras.datasets import mnist
from label_bias.main import *

### load mnist
(train_xs, train_ys), (test_xs, test_ys) = mnist.load_data()
train_xs = train_xs / 255.
test_xs = test_xs / 255.
train_xs = train_xs.reshape(-1, 28 * 28)
test_xs = test_xs.reshape(-1, 28 * 28)

### create biased mnist
train_ys_corrupted = np.copy(train_ys)
np.random.seed(12345)
idxs = np.random.choice(range(len(train_ys_corrupted)), size=len(train_ys_corrupted)//5, replace=False)
train_ys_corrupted[idxs] = 2

# print("Distribution Before")
# for i in range(10):
#   print (np.mean(train_ys == i))

# print("Distribution After")
# for i in range(10):
#   print (np.mean(train_ys_corrupted == i))

'''
[TRAIN]
train_xs, train_ys
train_xs, train_ys_corrupted

[TEST]
test_xs, test_ys
'''

### unbiased mnist training
weights = np.array([1] * len(train_ys))
test_predictions = run_simple_NN(train_xs, train_ys, test_xs, test_ys, weights)


### biased mnist training (unconstrained baseline)
weights = np.array([1] * len(train_ys))
test_predictions = run_simple_NN(train_xs, train_ys_corrupted, test_xs, test_ys, weights)

multipliers = np.zeros(1)
learning_rate = 1.
n_iters = 100
protected_train = [(train_ys_corrupted == 2)]

for it in range(n_iters):
    print("Iteration", it + 1, "multiplier", multipliers)
    weights = debias_weights(train_ys_corrupted, protected_train, multipliers)
    weights = weights / np.sum(weights)
    print("Weights for 2", np.sum(weights[np.where(train_ys_corrupted==2)]))
    train_prediction, test_predictions = run_simple_NN(train_xs, train_ys_corrupted, test_xs, test_ys, weights)
    violation = np.mean(train_prediction == 2) - 0.1
    multipliers -= learning_rate * violation
    print()
    print()
