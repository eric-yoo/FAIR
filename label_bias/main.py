from config import args, CHECKPOINTS_PATH_FORMAT, PRETRAINED_PATH
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

def pretrain_NN(X, y, pretrain_ratio, n_epochs = 10):
  model = network.model(num_classes=10, batch_size=args.batch_size)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )
  for i in range(1, n_epochs+1):
    model.fit(X, y, batch_size=args.batch_size, epochs=n_epochs)
    if i > n_epochs-3:
      model.save_weights(PRETRAINED_PATH.format(pretrain_ratio, i))
  return model

# neural network
def run_simple_NN(X,
                  y,
                  X_test,
                  y_test,
                  weights,
                  it=0,
                  n_epochs=5,
                  # unbiased / biased / lb
                  mode = "unbiased", 
                  pretrained_model = None
                 ):

  # train model
  if pretrained_model is None:
    model = network.model(num_classes=10, batch_size=args.batch_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
  else:
    model = pretrained_model

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

  for i in range(1, n_epochs+1):
      model.fit(X_train, y_train, batch_size=args.batch_size)

      if i > n_epochs-3:
        model.save_weights(CHECKPOINTS_PATH_FORMAT.format(mode, it, i))

  train_res, test_res = eval_simple_NN(X_train, y_train, X_test, y_test, weights, it=it, n_epochs=i, mode=mode)

  return  train_res, test_res
  
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

  train_loss, train_acc = model.evaluate(X,y)
  test_loss,  test_acc  = model.evaluate(X_test, y_test)

  print("train {}% / test acc {}%".format(train_acc, test_acc))

  train_pred = tf.argmax(model.predict(X), axis=1)
  test_pred  = tf.argmax(model.predict(X_test), axis=1)

  return (train_acc, train_pred), (test_acc, test_pred)


def debias_weights(original_labels, protected_attributes, multipliers):

  exponents = np.zeros(len(original_labels))
  for i, m in enumerate(multipliers):
    exponents -= m * protected_attributes[i]
  weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
  weights = np.where(original_labels == 2, 1 - weights, weights)
  return weights

def debias_weights_TI(original_labels, protected_attributes, multipliers_TI):
  exponents = -multipliers_TI * lr
  weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
  weights = np.where(original_labels == 2, 1 - weights, weights)
  return weights

