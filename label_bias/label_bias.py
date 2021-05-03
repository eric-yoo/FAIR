import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import copy


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
                  num_iter=10000,
                  learning_rate=0.001,
                  batch_size=128,
                  display_steps=1000,
                  n_layers=1,
                  verbose = False
                 ):
  n_labels = np.max(y) + 1
  n_features = X.shape[1]
  weights_ = weights / (1. * np.sum(weights))
  x = tf.placeholder(tf.float32, [None, n_features])
  y_ = tf.placeholder(tf.float32, [None, n_labels])
  
  N = 512
  
  W_1 = weight_variable([784, N])
  b_1 = bias_variable([N])

  h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

  W_2 = weight_variable([N, N])
  b_2 = bias_variable([N])

  h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

  W_3 = weight_variable([N, N])
  b_3 = bias_variable([N])

  h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)

  W_4 = weight_variable([N, 10])
  b_4 = bias_variable([10])

  NN_logits =tf.nn.softmax(tf.matmul(h_3, W_4) + b_4)

  loss = -tf.reduce_mean(tf.reduce_sum(y_ *tf.log(NN_logits+1e-6),1),0)
  acc = tf.reduce_mean(
      tf.cast(tf.equal(tf.arg_max(NN_logits,1), tf.arg_max(y_,1)), "float"))
  train_step = tf.train.AdamOptimizer().minimize(loss)
  correct_prediction = tf.equal(tf.argmax(NN_logits, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def one_hot(ns):
    return np.eye(n_labels)[ns]

  y_onehot = one_hot(y)
  y_test_onehot = one_hot(y_test)

  with tf.Session() as sess:
    print('\n[start training]\n')
    sess.run(tf.global_variables_initializer())
    for i in range(num_iter):
      ns = np.random.choice(range(len(X)), size=50, replace=True, p=weights_)
      if (i + 1) % display_steps == 0:
        train_accuracy = accuracy.eval(feed_dict={x: X, y_: y_onehot})
        test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test_onehot})

        if verbose:
            print("step %d, training accuracy %g, test accuracy %g" %
              (i + 1, train_accuracy, test_accuracy))
      train_step.run(
          feed_dict={x: X[ns, :], y_: y_onehot[ns, :]})

    testing_prediction = tf.argmax(NN_logits, 1).eval(feed_dict={x: X_test})
    training_prediction = tf.argmax(NN_logits, 1).eval(feed_dict={x: X})
    return training_prediction, testing_prediction


def debias_weights(original_labels, protected_attributes, multipliers):
  exponents = np.zeros(len(original_labels))
  for i, m in enumerate(multipliers):
    exponents -= m * protected_attributes[i]
  weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
  weights = np.where(original_labels == 2, 1 - weights, weights)
  return weights