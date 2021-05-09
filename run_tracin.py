MNIST_TFDS_DIR = "tensorflow_datasets" #@param {type:"string"}
CHECKPOINTS_PATH_FORMAT = "simpleNN/ckpt{}" #@param {type:"string"}
DEBUG = True

def debug(s):
  if DEBUG:
    print(s)

# @title Imports
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.image as mpimg
import io
import json
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import functools
import sys
sys.path.insert(0, "simpleNN")
import network
#@title Dataset Utils
import numpy as np
import copy
from tensorflow.keras.datasets import mnist
import time
BATCH_SIZE = 512

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")


# DATA LOADING & TRAINING 

index_to_classname = {}

def make_get_dataset(split, batch_size, with_index=True):
  def get_dataset() -> tf.data.Dataset:
    builder = tfds.builder(name='mnist', data_dir=MNIST_TFDS_DIR)
    builder.download_and_prepare()

    read_config = tfds.ReadConfig(
        interleave_block_length=1)

    ds = builder.as_dataset(
        split=split,
        as_supervised=False,
        shuffle_files=False,
        read_config=read_config)

    if with_index:
      indices_range = {'train':range(60000), 'test':range(60000, 70000)}
      indices =  tf.data.Dataset.from_tensor_slices(list(indices_range[split]))
      ds = tf.data.Dataset.zip((indices, ds))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
  return get_dataset() 

# ### load mnist
# (train_xs, train_ys), (test_xs, test_ys) = mnist.load_data()
# train_xs = train_xs / 255.
# test_xs = test_xs / 255.
# train_xs = train_xs.reshape(-1, 28 * 28)
# test_xs = test_xs.reshape(-1, 28 * 28)

# ### create biased mnist
# train_ys_corrupted = np.copy(train_ys)
# np.random.seed(12345)
# idxs = np.random.choice(range(len(train_ys_corrupted)), size=len(train_ys_corrupted)//5, replace=False)
# train_ys_corrupted[idxs] = 2

ds_train = make_get_dataset('train', BATCH_SIZE)
ds_test = make_get_dataset('test', BATCH_SIZE)
  
try:
  for i in [1, 2, 3]:
    model = network.model()
    model.load_weights(CHECKPOINTS_PATH_FORMAT.format(i))
except:
  debug('need train.')
  model = network.model()
  model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )
  for i in range(1,5):
    for d in ds_train:
      model.fit(d[1]['image'], d[1]['label'])
    model.save_weights(CHECKPOINTS_PATH_FORMAT.format(i))




# UTIL FUNCTIONS

def find(loss_grad=None, activation=None, topk=50):
  if loss_grad is None and activation is None:
    raise ValueError('loss grad and activation cannot both be None.')
  scores = []
  scores_lg = []
  scores_a = []
  for i in range(len(trackin_train['image_ids'])):
    if loss_grad is not None and activation is not None:
      lg_sim = np.sum(trackin_train['loss_grads'][i] * loss_grad)
      a_sim = np.sum(trackin_train['activations'][i] * activation)
      scores.append(lg_sim * a_sim)
      scores_lg.append(lg_sim)
      scores_a.append(a_sim)
    elif loss_grad is not None:
      scores.append(np.sum(trackin_train['loss_grads'][i] * loss_grad))
    elif activation is not None:
      scores.append(np.sum(trackin_train['activations'][i] * activation))    

  opponents = []
  proponents = []
  indices = np.argsort(scores)
  for i in range(topk):
    index = indices[-i-1]
    proponents.append((
        trackin_train['image_ids'][index],
        trackin_train['probs'][index][0],
        index_to_classname[trackin_train['predicted_labels'][index][0]],
        index_to_classname[trackin_train['labels'][index]], 
        scores[index],
        scores_lg[index] if scores_lg else None,
        scores_a[index] if scores_a else None))
    index = indices[i]
    opponents.append((
        trackin_train['image_ids'][index],
        trackin_train['probs'][index][0],
        index_to_classname[trackin_train['predicted_labels'][index][0]],
        index_to_classname[trackin_train['labels'][index]],
        scores[index],
        scores_lg[index] if scores_lg else None,
        scores_a[index] if scores_a else None))  
  return opponents, proponents

def predicate(x, target_index):
    index = x[0]
    isallowed = tf.equal(target_index, tf.cast(index, tf.float32))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))


def get_image(split, id):
  if split == 'test':
    for batch in ds_test:
      index = tf.where(batch[0] == id)
      if index.shape[0] == 1:
        return (batch[1]['image'][index.numpy()[0][0]]).numpy().reshape((28,28))
        
  else:
    for batch in ds_train:
      index = tf.where(batch[0] == id)
      if index.shape[0] == 1:
        return (batch[1]['image'][index.numpy()[0][0]]).numpy().reshape((28,28))
  
def find_and_show(trackin_dict, idx, vector='influence', idx_filename_mapping=None):
  if vector == 'influence':
    op, pp = find(trackin_dict['loss_grads'][idx], trackin_dict['activations'][idx])
  elif vector == 'encoding':
    op, pp = find(None, trackin_dict['activations'][idx])  
  elif vector == 'error':
    op, pp = find(trackin_dict['loss_grads'][idx], None)
  else:
    raise ValueError('Unsupported vector type.')
  debug('Query image from test: ')
  debug('label: {}, prob: {}, predicted_label: {}'.format(
      index_to_classname[trackin_dict['labels'][idx]], 
      trackin_dict['probs'][idx][0], 
      index_to_classname[trackin_dict['predicted_labels'][idx][0]]))
  
  img = get_image('test', trackin_dict['image_ids'][idx])
  plt.imshow(img)
  plt.show()
    
  debug("="*50)  
  debug('Proponents: ')
  for p in pp:
    debug('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(p[3], p[1], p[2], p[4]))
    if p[5] and p[6]:
      debug('error_similarity: {}, encoding_similarity: {}'.format(p[5], p[6]))
    img = get_image('train', p[0])
    if img is not None:
      plt.imshow(img, interpolation='nearest')
      plt.show()  
  debug("="*50)
  debug('Opponents: ')
  for o in op:
    debug('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(o[3], o[1], o[2], o[4]))
    if o[5] and o[6]:
      debug('error_similarity: {}, encoding_similarity: {}'.format(o[5], o[6]))
    img = get_image('train', o[0])
    if img is not None:
      plt.imshow(img, interpolation='nearest')
      plt.show()
  debug("="*50)


def get_trackin_grad(ds):
  image_ids_np = []
  loss_grads_np = []
  activations_np = []
  labels_np = []
  probs_np = []
  predicted_labels_np = []
  for d in ds:
    imageids_replicas, loss_grads_replica, activations_replica, labels_replica, probs_replica, predictied_labels_replica = strategy.run(run, args=(d,))
    for imageids, loss_grads, activations, labels, probs, predicted_labels in zip(
        strategy.experimental_local_results(imageids_replicas), 
        strategy.experimental_local_results(loss_grads_replica),
        strategy.experimental_local_results(activations_replica), 
        strategy.experimental_local_results(labels_replica), 
        strategy.experimental_local_results(probs_replica), 
        strategy.experimental_local_results(predictied_labels_replica)):
      if imageids.shape[0] == 0:
        continue
      image_ids_np.append(imageids.numpy())
      loss_grads_np.append(loss_grads.numpy())
      activations_np.append(activations.numpy())
      labels_np.append(labels.numpy())
      probs_np.append(probs.numpy())
      predicted_labels_np.append(predicted_labels.numpy())
  return {'image_ids': np.concatenate(image_ids_np),
          'loss_grads': np.concatenate(loss_grads_np),
          'activations': np.concatenate(activations_np),
          'labels': np.concatenate(labels_np),
          'probs': np.concatenate(probs_np),
          'predicted_labels': np.concatenate(predicted_labels_np)
         }    

@tf.function
def run(inputs):
  imageids, data = inputs
  images = data['image']
  labels = data['label']
  # ignore bias for simplicity
  loss_grads = []
  activations = []
  for mp, ml in zip(models_penultimate, models_last):
    h = mp(images)
    logits = ml(h)
    probs = tf.nn.softmax(logits)
    loss_grad = tf.one_hot(labels, 10) - probs
    activations.append(h)
    loss_grads.append(loss_grad)

  # Using probs from last checkpoint
  probs, predicted_labels = tf.math.top_k(probs, k=1)

  return imageids, tf.stack(loss_grads, axis=-1), tf.stack(activations, axis=-1), labels, probs, predicted_labels


models_penultimate = []
models_last = []
for i in [1, 2, 3]:
  model = network.model()
  model.load_weights(CHECKPOINTS_PATH_FORMAT.format(i)).expect_partial()
  models_penultimate.append(tf.keras.Model(model.layers[0].input, model.layers[-2].output))
  models_last.append(model.layers[-1])
    
    
start = time.time()
trackin_train = get_trackin_grad(ds_train)
ids = list(trackin_train['image_ids'])
labels = list(trackin_train['labels'])
for i,id in enumerate(ids):
  index_to_classname[id] = labels[i]
end = time.time()
debug(datetime.timedelta(seconds=end - start))

start = time.time()
trackin_test = get_trackin_grad(ds_test)
ids = list(trackin_test['image_ids'])
labels = list(trackin_test['labels'])
for i,id in enumerate(ids):
  index_to_classname[id] = labels[i]
end = time.time()
debug(datetime.timedelta(seconds=end - start))

find_and_show(trackin_test, 8, 'influence')