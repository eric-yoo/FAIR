BATCH_SIZE = 64

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
                  it,
                  n_epochs=30,
                  verbose = False
                 ):

  n_labels = np.max(y) + 1
  n_features = X.shape[1]
  weights_ = weights / (1. * np.sum(weights))
  
  model = network.model(num_classes=10, batch_size=BATCH_SIZE)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )

  #ns = np.random.choice(range(len(X)), BATCH_SIZE*100, replace=True, p=weights_)
  ns = np.random.choice(range(len(X)), BATCH_SIZE*100, replace=True)


  X_train = X[ns,:]
  y_train = y[ns]

  #(6400,28,28) / (6400,)
  #print(X_train.shape, y_train.shape)

  CHECKPOINTS_PATH_FORMAT = "simpleNN/lb{}_ckpt{}"
  for i in range(1, n_epochs+1):
      model.fit(X_train, y_train)
      if i > n_epochs-3:
        model.save_weights(CHECKPOINTS_PATH_FORMAT.format(it, i))

  return eval_simple_NN(X,y,X_test,y_test,weights,it,n_epochs=n_epochs,verbose = False)
  
def debias_weights(original_labels, protected_attributes, multipliers):
  exponents = np.zeros(len(original_labels))
  for i, m in enumerate(multipliers):
    exponents -= m * protected_attributes[i]
  weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
  weights = np.where(original_labels == 2, 1 - weights, weights)
  return weights

# neural network
def eval_simple_NN(X,
                  y,
                  X_test,
                  y_test,
                  weights,
                  it,
                  n_epochs=30,
                  verbose = False
                 ):

  n_labels = np.max(y) + 1
  n_features = X.shape[1]
  weights_ = weights / (1. * np.sum(weights))
  
  model = network.model(num_classes=10, batch_size=None)
  
  CHECKPOINTS_PATH_FORMAT = "simpleNN/lb{}_ckpt{}"
  model.load_weights(CHECKPOINTS_PATH_FORMAT.format(it, n_epochs)).expect_partial()

  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )

  training_acc = model.evaluate(X,y)
  testing_acc  = model.evaluate(X_test, y_test)

  training_prediction = tf.argmax(model.predict(X), axis=1)
  testing_prediction  = tf.argmax(model.predict(X_test), axis=1)

  return training_prediction, testing_prediction
