BATCH_SIZE = 64
CHECKPOINTS_PATH_FORMAT = "simpleNN/checkpoints/{}_iter{}_ckpt{}"

import numpy as np
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf

from simpleNN import network


def weight_variable(shape, name="weight_variable"):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_variable(shape, name="bias_variable"):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


# neural network
def run_simple_NN(X,
                  y,
                  X_test,
                  y_test,
                  weights,
                  it=0,
                  n_epochs=10,
                  # unbiased / biased / lb
                  mode = "unbiased"
                 ):

  # train model
  model = network.model(num_classes=10, batch_size=BATCH_SIZE)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )

  # weights for data choice
  if weights is None :
      ns = np.random.choice(range(len(X)), len(X), replace=True)
      X_train = X[ns,:]
      y_train = y[ns]
      #X_train = X
      #y_train = y

  else :
      weights_ = weights / (1. * np.sum(weights))
      ns = np.random.choice(range(len(X)), len(X), replace=True, p=weights)
      X_train = X[ns,:]
      y_train = y[ns]

  #(6400,28,28) / (6400,)
  #print(X_train.shape, y_train.shape)

  training_accs = []
  testing_accs  = []

  for i in range(1, n_epochs+1):
      model.fit(X_train, y_train)
      training_acc, testing_acc = eval_simple_NN(X_train, y, X_test, y_test, weights, it, n_epochs=n_epochs, mode=mode)
      e

      training_accs.append(training_acc)
      testing_accs.append(testing_acc)

      if i > n_epochs-3:
        model.save_weights(CHECKPOINTS_PATH_FORMAT.format(mode, it, i))

  return 
  
# neural network
def eval_simple_NN(X,
                  y,
                  X_test,
                  y_test,
                  weights,
                  it=0,
                  n_epochs=10,
                  # unbiased / biased / lb
                  mode = "unbiased"
                 ):

  model = network.model(num_classes=10, batch_size=None)
  model.load_weights(CHECKPOINTS_PATH_FORMAT.format(mode, it, n_epochs)).expect_partial()

  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )

  training_loss, training_acc = model.evaluate(X,y)
  testing_loss,  testing_acc  = model.evaluate(X_test, y_test)

  print("train {}% / test acc {}%".format(training_acc, testing_acc))

  training_prediction = tf.argmax(model.predict(X), axis=1)
  testing_prediction  = tf.argmax(model.predict(X_test), axis=1)

  return training_prediction, testing_prediction


def debias_weights(original_labels, protected_attributes, multipliers):

  exponents = np.zeros(len(original_labels))
  for i, m in enumerate(multipliers):
    exponents -= m * protected_attributes[i]
  weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
  weights = np.where(original_labels == 2, 1 - weights, weights)
  return weights

def debias_weights_TI(original_labels, multipliers_TI):

  exponents = -multipliers_TI
  weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
  weights = np.where(original_labels == 2, 1 - weights, weights)
  return weights

